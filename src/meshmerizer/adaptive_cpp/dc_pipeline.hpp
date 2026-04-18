#ifndef MESHMERIZER_ADAPTIVE_CPP_DC_PIPELINE_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_DC_PIPELINE_HPP_

/**
 * @file dc_pipeline.hpp
 * @brief Combined particles-to-mesh pipeline using Dual Contouring.
 *
 * This header provides `run_dc_pipeline`, a single entry point that
 * takes raw particle data (positions + smoothing lengths) and returns
 * a triangle mesh.  The steps are:
 *
 *   1. Build top-level cells and query particle contributors.
 *   2. Refine the adaptive octree (BFS + 2:1 balance).
 *   3. Solve QEF vertices + normals on active leaf cells.
 *   4. Build spatial index for leaf cell neighbor lookups.
 *   5. Generate dual contour faces from sign-changing edges.
 *
 * Unlike the Poisson pipeline, this approach directly connects the
 * QEF vertices (one per active leaf) using the primal edge topology
 * of the octree.  For each primal edge where the indicator function
 * changes sign, the four leaf cells sharing that edge contribute a
 * quad (split into two triangles).
 *
 * References:
 *   - Ju, Losasso, Schaefer & Warren, "Dual Contouring of Hermite
 *     Data", SIGGRAPH 2002.
 *   - Schaefer & Warren, "Dual Marching Cubes: Primal Contouring
 *     of Dual Grids", Pacific Graphics 2004 (adaptive handling).
 */

#include "bounding_box.hpp"
#include "faces.hpp"
#include "mesh.hpp"
#include "morton.hpp"
#include "octree_cell.hpp"
#include "particle.hpp"
#include "particle_grid.hpp"
#include "progress_bar.hpp"
#include "qef.hpp"
#include "vector3d.hpp"

#include <array>
#include <cstdint>
#include <vector>

/**
 * @brief Result of the Dual Contouring pipeline.
 *
 * Contains the output triangle mesh (vertices + face indices)
 * and metadata about the octree.
 */
struct DCPipelineResult {
    /// Output vertex positions.
    std::vector<Vector3d> vertices;

    /// Output triangle face indices (3 per triangle).
    std::vector<std::array<std::uint32_t, 3>> triangles;

    /// Isovalue used for surface extraction.
    double isovalue;

    /// Number of QEF vertices (one per active leaf cell).
    std::size_t n_qef_vertices;
};

/**
 * @brief Run the complete particles-to-mesh pipeline using Dual
 *        Contouring.
 *
 * This function orchestrates the entire adaptive meshing pipeline
 * in C++, from raw particle data to a triangle mesh.  No
 * intermediate state is returned to Python.
 *
 * @param positions         Particle positions (N particles).
 * @param smoothing_lengths Particle smoothing lengths (N).
 * @param domain            Domain bounding box.
 * @param base_resolution   Number of top-level cells per axis.
 * @param isovalue          Density isovalue for octree refinement
 *                          (determines which cells are "active").
 * @param max_depth         Maximum octree refinement depth.
 * @return DCPipelineResult containing the output mesh.
 */
inline DCPipelineResult run_dc_pipeline(
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    double isovalue,
    std::uint32_t max_depth) {

    DCPipelineResult result;
    result.isovalue = isovalue;

    // ================================================================
    // Step 1: Build top-level cells and query contributors.
    // ================================================================
    // The TopLevelParticleGrid spatially bins particles so that
    // contributor queries for each cell are O(local) rather than
    // O(N).

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
    // from the density field.

    std::vector<MeshVertex> qef_vertices =
        solve_all_leaf_vertices(
            all_cells, all_contributors, positions,
            smoothing_lengths, isovalue);

    result.n_qef_vertices = qef_vertices.size();

    if (qef_vertices.empty()) {
        return result;
    }

    // ================================================================
    // Step 4: Build spatial index for leaf cell lookups.
    // ================================================================
    // The LeafSpatialIndex hashes leaf cells by their quantized
    // min-corner so that find_leaf_at(ix, iy, iz) is O(1).

    LeafSpatialIndex spatial_index;
    spatial_index.build(all_cells, domain, max_depth,
                        base_resolution);

    // ================================================================
    // Step 5: Generate dual contour faces.
    // ================================================================
    // For each primal edge of the finest-resolution grid where the
    // indicator function changes sign, emit a quad (two triangles)
    // connecting the QEF vertices of the four leaf cells that share
    // that edge.  On an adaptive grid, some of these four cells
    // may be the same coarser cell, producing degenerate quads that
    // are handled by emit_quad.

    std::vector<MeshTriangle> dc_triangles =
        generate_dual_contour_faces(
            all_cells, qef_vertices, spatial_index,
            max_depth, base_resolution, isovalue);

    // ================================================================
    // Step 6: Convert to output format.
    // ================================================================
    // Copy MeshVertex positions and MeshTriangle indices into the
    // result vectors.

    result.vertices.reserve(qef_vertices.size());
    for (const auto &v : qef_vertices) {
        result.vertices.push_back(v.position);
    }

    result.triangles.reserve(dc_triangles.size());
    for (const auto &tri : dc_triangles) {
        result.triangles.push_back(
            {static_cast<std::uint32_t>(tri.vertex_indices[0]),
             static_cast<std::uint32_t>(tri.vertex_indices[1]),
             static_cast<std::uint32_t>(tri.vertex_indices[2])});
    }

    return result;
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_DC_PIPELINE_HPP_
