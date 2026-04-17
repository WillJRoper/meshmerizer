#ifndef MESHMERIZER_ADAPTIVE_CPP_POISSON_PIPELINE_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_POISSON_PIPELINE_HPP_

/**
 * @file poisson_pipeline.hpp
 * @brief Combined particles-to-mesh pipeline.
 *
 * This header provides `run_full_pipeline`, a single entry point that
 * takes raw particle data (positions + smoothing lengths) and returns
 * a triangle mesh.  The steps are:
 *
 *   1. Build top-level cells and query particle contributors.
 *   2. Refine the adaptive octree.
 *   3. Solve QEF vertices + normals on active leaf cells.
 *   4. Assign B-spline DOF indices on the **same** adaptive leaves.
 *   5. Build the spatial hash for DOF lookup.
 *   6. Splat QEF normals into the B-spline vector field V.
 *   7. Enumerate stencils and assemble the Poisson RHS.
 *   8. Accumulate screening weights from QEF positions.
 *   9. Solve the screened Poisson system with PCG.
 *  10. Evaluate the indicator function chi at leaf-cell corners.
 *  11. Compute the isovalue from QEF positions.
 *  12. Extract the isosurface via Marching Cubes.
 *
 * The key design choice is that the Poisson reconstruction operates
 * on the **adaptive octree** produced by refinement — not a separate
 * uniform grid.  This preserves resolution where the particle data
 * demands it and avoids wasting memory on empty regions.
 *
 * References:
 *   - Kazhdan, Bolitho & Hoppe, "Poisson Surface Reconstruction",
 *     SGP 2006 (basis splatting, RHS assembly).
 *   - Kazhdan & Hoppe, "Screened Poisson Surface Reconstruction",
 *     ToG 2013 (screening term, cascadic solver).
 *   - Lorensen & Cline, "Marching Cubes", SIGGRAPH 1987
 *     (isosurface extraction).
 */

#include "bounding_box.hpp"
#include "faces.hpp"
#include "mesh.hpp"
#include "morton.hpp"
#include "octree_cell.hpp"
#include "particle.hpp"
#include "particle_grid.hpp"
#include "poisson_basis.hpp"
#include "poisson_mc.hpp"
#include "poisson_rhs.hpp"
#include "poisson_solver.hpp"
#include "poisson_stencil.hpp"
#include "progress_bar.hpp"
#include "qef.hpp"
#include "vector3d.hpp"

#include <cstdint>
#include <vector>

/**
 * @brief Result of the full pipeline.
 *
 * Contains the output triangle mesh (vertices + face indices).
 */
struct PipelineResult {
    /// Output vertex positions.
    std::vector<Vector3d> vertices;

    /// Output triangle face indices (3 per triangle).
    std::vector<std::array<std::uint32_t, 3>> triangles;

    /// Isovalue used for the Marching Cubes extraction.
    double isovalue;

    /// Number of QEF vertices produced before Poisson.
    std::size_t n_qef_vertices;

    /// Whether the PCG solver converged.
    bool solver_converged;

    /// Number of PCG iterations taken.
    std::size_t solver_iterations;

    /// Final PCG residual norm.
    double solver_residual;
};

/**
 * @brief Run the complete particles-to-mesh pipeline.
 *
 * This function orchestrates the entire adaptive meshing pipeline
 * in C++, from raw particle data to a triangle mesh.  No
 * intermediate state is returned to Python — the octree, QEF
 * vertices, B-spline DOFs, Poisson solution, and MC extraction
 * are all computed internally.
 *
 * @param positions       Particle positions (N particles).
 * @param smoothing_lengths Particle smoothing lengths (N).
 * @param domain          Domain bounding box.
 * @param base_resolution Number of top-level cells per axis.
 * @param isovalue        Density isovalue for octree refinement
 *                        (determines which cells are "active").
 * @param max_depth       Maximum octree refinement depth.
 * @param screening_weight Poisson screening weight alpha
 *                        (higher = tighter fit to data points,
 *                        lower = smoother surface).
 * @param max_iters       Maximum PCG iterations.
 * @param tol             PCG relative residual tolerance.
 * @return PipelineResult containing the output mesh.
 */
inline PipelineResult run_full_pipeline(
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    double isovalue,
    std::uint32_t max_depth,
    double screening_weight,
    std::size_t max_iters,
    double tol) {

    PipelineResult result;

    // ================================================================
    // Step 1: Build top-level cells and query contributors.
    // ================================================================
    // The TopLevelParticleGrid spatially bins particles so that
    // contributor queries for each cell are O(local) rather than
    // O(N).  This is identical to the logic in
    // run_octree_pipeline_py.

    TopLevelParticleGrid grid(domain, base_resolution);
    grid.insert_particles(positions);
    grid.compute_bin_max_h(smoothing_lengths);

    std::vector<OctreeCell> top_cells =
        create_top_level_cells(domain, base_resolution);

    std::vector<OctreeCell> initial_cells;
    initial_cells.reserve(top_cells.size());
    std::vector<std::size_t> initial_contributors;

    ProgressBar contrib_bar(
        "Contributor query", top_cells.size());

    for (std::size_t ci = 0; ci < top_cells.size(); ++ci) {
        OctreeCell cell = top_cells[ci];

        std::uint32_t sx = 0, sy = 0, sz = 0;
        std::uint32_t ex = 0, ey = 0, ez = 0;
        grid.contributor_bin_span(
            cell.bounds, smoothing_lengths,
            sx, sy, sz, ex, ey, ez);

        const std::int64_t begin =
            static_cast<std::int64_t>(
                initial_contributors.size());

        for (std::uint32_t ix = sx; ix <= ex; ++ix) {
            for (std::uint32_t iy = sy; iy <= ey; ++iy) {
                for (std::uint32_t iz = sz;
                     iz <= ez; ++iz) {
                    const TopLevelBin &bin =
                        grid.bins[grid.flatten_index(
                            ix, iy, iz)];
                    for (std::size_t pi :
                         bin.particle_indices) {
                        if (particle_support_overlaps_box(
                                positions[pi],
                                smoothing_lengths[pi],
                                cell.bounds)) {
                            initial_contributors.push_back(
                                pi);
                        }
                    }
                }
            }
        }

        const std::int64_t end =
            static_cast<std::int64_t>(
                initial_contributors.size());
        cell.contributor_begin = begin;
        cell.contributor_end = end;
        initial_cells.push_back(cell);
        contrib_bar.tick();
    }
    contrib_bar.finish();

    // ================================================================
    // Step 2: Refine the adaptive octree.
    // ================================================================
    // refine_octree subdivides cells where the density field
    // crosses the isovalue, producing an adaptive leaf set that
    // concentrates resolution where the surface is.

    auto [all_cells, all_contributors] = refine_octree(
        std::move(initial_cells),
        std::move(initial_contributors),
        positions,
        smoothing_lengths,
        isovalue,
        max_depth,
        domain,
        base_resolution);

    // ================================================================
    // Step 3: Solve QEF vertices + normals on active leaves.
    // ================================================================
    // Each active leaf cell gets a Quadric Error Function vertex
    // positioned at the best-fit intersection of Hermite data
    // from the density field.  The normal is the density gradient
    // at that position.

    std::vector<MeshVertex> qef_vertices =
        solve_all_leaf_vertices(
            all_cells, all_contributors, positions,
            smoothing_lengths, isovalue);

    result.n_qef_vertices = qef_vertices.size();

    if (qef_vertices.empty()) {
        // No active cells — return empty mesh.
        result.isovalue = 0.0;
        result.solver_converged = true;
        result.solver_iterations = 0;
        result.solver_residual = 0.0;
        return result;
    }

    // Separate QEF vertices into position and normal arrays
    // for the Poisson pipeline.  The Poisson solver treats
    // QEF vertices as the oriented point cloud.
    std::vector<Vector3d> qef_positions(qef_vertices.size());
    std::vector<Vector3d> qef_normals(qef_vertices.size());
    for (std::size_t i = 0; i < qef_vertices.size(); ++i) {
        qef_positions[i] = qef_vertices[i].position;
        qef_normals[i] = qef_vertices[i].normal;
    }

    // Free the MeshVertex vector — we only need the separated
    // arrays from here on.
    { std::vector<MeshVertex>().swap(qef_vertices); }

    // Sanitize normals: remove zero-length normals and
    // re-normalize to unit length.  QEF solving can produce
    // degenerate normals from ill-conditioned cells.
    std::vector<bool> valid(qef_positions.size(), true);
    std::size_t n_valid = 0;
    for (std::size_t i = 0; i < qef_normals.size(); ++i) {
        const double len = std::sqrt(
            qef_normals[i].x * qef_normals[i].x +
            qef_normals[i].y * qef_normals[i].y +
            qef_normals[i].z * qef_normals[i].z);
        if (len > 1e-12) {
            const double inv = 1.0 / len;
            qef_normals[i].x *= inv;
            qef_normals[i].y *= inv;
            qef_normals[i].z *= inv;
            valid[i] = true;
            ++n_valid;
        } else {
            valid[i] = false;
        }
    }

    // Compact valid entries if any were removed.
    if (n_valid < qef_positions.size()) {
        std::size_t write = 0;
        for (std::size_t i = 0; i < qef_positions.size();
             ++i) {
            if (valid[i]) {
                qef_positions[write] = qef_positions[i];
                qef_normals[write] = qef_normals[i];
                ++write;
            }
        }
        qef_positions.resize(write);
        qef_normals.resize(write);
    }

    if (qef_positions.empty()) {
        result.isovalue = 0.0;
        result.solver_converged = true;
        result.solver_iterations = 0;
        result.solver_residual = 0.0;
        return result;
    }

    const std::size_t n_samples = qef_positions.size();

    // ================================================================
    // Step 4: Assign B-spline DOF indices on adaptive leaves.
    // ================================================================
    // Each leaf cell at max_depth gets one degree-of-freedom (DOF)
    // for the degree-2 B-spline basis.  We restrict DOFs to
    // max_depth leaves only because the Galerkin Laplacian stencil
    // assumes uniform cell widths — mixing depths breaks operator
    // symmetry and causes PCG divergence.  Max_depth leaves form
    // a dense shell around the surface (from refinement) plus
    // 2:1-balance neighbors, providing enough interior/exterior
    // coverage for the indicator function.

    std::vector<std::int64_t> cell_to_dof;
    std::vector<std::size_t> dof_to_cell;
    assign_dof_indices(all_cells, cell_to_dof, dof_to_cell,
                       static_cast<int>(max_depth));
    const std::size_t n_dofs = dof_to_cell.size();

    if (n_dofs == 0) {
        result.isovalue = 0.0;
        result.solver_converged = true;
        result.solver_iterations = 0;
        result.solver_residual = 0.0;
        return result;
    }

    // ================================================================
    // Step 5: Build spatial hash for DOF lookup.
    // ================================================================
    // PoissonLeafHash provides O(1) lookup of which leaf cell
    // contains a given point, used by find_overlapping_dofs to
    // evaluate B-spline basis functions at arbitrary positions.

    PoissonLeafHash hash;
    hash.build(all_cells, domain, max_depth, base_resolution);

    // ================================================================
    // Step 6: Splat QEF normals into B-spline vector field.
    // ================================================================
    // Each QEF sample's unit normal is distributed into the
    // overlapping B-spline DOFs weighted by the degree-2 basis
    // value (SGP06 Sec 3).

    std::vector<Vector3d> v_field;
    splat_normals(
        qef_positions.data(), qef_normals.data(),
        n_samples, hash, all_cells, cell_to_dof,
        dof_to_cell, n_dofs, base_resolution, v_field);

    // ================================================================
    // Step 7: Enumerate stencils and assemble Poisson RHS.
    // ================================================================
    // The RHS b_i = sum_j V_j . G_ij where G_ij is the gradient
    // inner product between B-spline basis functions i and j.

    std::vector<std::size_t> stencil_offsets;
    std::vector<std::int64_t> stencil_neighbors;
    std::vector<int> stencil_depth_deltas;
    enumerate_stencils(
        all_cells, cell_to_dof, dof_to_cell,
        domain, base_resolution, max_depth,
        stencil_offsets, stencil_neighbors,
        &stencil_depth_deltas);

    std::vector<double> rhs;
    compute_rhs(
        v_field, all_cells, cell_to_dof, dof_to_cell,
        stencil_offsets, stencil_neighbors, stencil_depth_deltas,
        n_dofs, rhs);

    // Free the vector field — no longer needed after RHS assembly.
    { std::vector<Vector3d>().swap(v_field); }

    // ================================================================
    // Step 8: Accumulate screening from QEF positions.
    // ================================================================
    // The screening term (ToG13 Sec 4.3) adds a data-fitting
    // constraint alpha * sum_s B_i(s) * B_j(s) to the operator,
    // pulling the solution toward the data points.

    ScreeningData screening;
    accumulate_screening(
        qef_positions.data(), n_samples,
        screening_weight,
        hash, all_cells, cell_to_dof, dof_to_cell,
        n_dofs, base_resolution, screening);

    // ================================================================
    // Step 9: Solve the screened Poisson system with PCG.
    // ================================================================
    // The system is (L + alpha * W) x = b where L is the
    // Laplacian and W is the screening matrix.  PCG with Jacobi
    // preconditioning converges rapidly for well-conditioned
    // systems.

    std::vector<double> solution;
    solution.assign(n_dofs, 0.0);

    SolverResult solver = solve_pcg(
        rhs, all_cells, cell_to_dof, dof_to_cell,
        stencil_offsets, stencil_neighbors,
        stencil_depth_deltas,
        screening, n_dofs, max_iters, tol, solution);

    result.solver_converged = solver.converged;
    result.solver_iterations = solver.iterations;
    result.solver_residual = solver.residual_norm;

    // Free stencil data — no longer needed after solve.
    { std::vector<std::size_t>().swap(stencil_offsets); }
    { std::vector<std::int64_t>().swap(stencil_neighbors); }

    // ================================================================
    // Step 10: Evaluate chi at leaf-cell corners.
    // ================================================================
    // chi(p) = sum_j x_j B_j(p) evaluated at the 8 corners of
    // each leaf cell.  A global corner cache ensures adjacent
    // cells sharing a corner see the same chi value.

    std::vector<std::array<double, 8>> corner_values;
    std::vector<VirtualCell> virtual_cells;
    std::size_t n_leaves = 0;
    for (const auto &c : all_cells) {
        if (c.is_leaf) {
            ++n_leaves;
        }
    }

    evaluate_chi_at_corners(
        solution, all_cells, cell_to_dof, hash,
        base_resolution, max_depth, n_leaves,
        domain, corner_values, virtual_cells);

    // ================================================================
    // Step 11: Compute isovalue from QEF positions.
    // ================================================================
    // The isovalue is the mean chi at the QEF sample positions.
    // This places the level set at a representative value of the
    // indicator function at the data points.

    result.isovalue = compute_isovalue(
        qef_positions.data(), n_samples,
        solution, all_cells, cell_to_dof, hash,
        base_resolution);

    // Free the solution vector — chi corners have been evaluated
    // and virtual cell corners have been computed.
    { std::vector<double>().swap(solution); }

    // ================================================================
    // Step 12: Extract isosurface via Marching Cubes.
    // ================================================================
    // Classic MC (Lorensen & Cline 1987) with collision-free edge
    // caching to ensure shared vertices across adjacent cells.
    // Virtual boundary cells close the mesh at the boundary of
    // the fine-leaf region (see poisson_mc.hpp for details).

    extract_isosurface(
        all_cells, corner_values, virtual_cells,
        result.isovalue,
        domain, base_resolution, max_depth,
        result.vertices, result.triangles);

    return result;
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_POISSON_PIPELINE_HPP_
