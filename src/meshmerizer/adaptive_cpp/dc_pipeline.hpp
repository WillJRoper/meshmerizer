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
 * This approach directly connects the QEF vertices (one per active leaf)
 * using the primal edge topology
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
#include "adaptive_solid.hpp"
#include "edge_subdiv.hpp"
#include "faces.hpp"
#include "mesh.hpp"
#include "morton.hpp"
#include "octree_cell.hpp"
#include "particle.hpp"
#include "particle_grid.hpp"
#include "progress_bar.hpp"
#include "qef.hpp"
#include "vector3d.hpp"
#include "vertex_adjacency.hpp"
#include "vertex_smooth.hpp"

VertexAdjacency build_triangle_mesh_adjacency(
    std::size_t n_vertices,
    const std::vector<MeshTriangle> &triangles);

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
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
 * @param smoothing_iterations Number of Laplacian smoothing
 *                          iterations (0 = disabled).
 * @param smoothing_strength   Smoothing lambda in (0, 1].
 * @param max_edge_ratio    Maximum edge length as a multiple of
 *                          local cell size for gap filling.
 *                          Default 1.5.  Always active.
 * @param minimum_usable_hermite_samples Minimum number of usable Hermite
 *                          samples required before a corner-crossing cell is
 *                          allowed to stop refining.
 * @param max_qef_rms_residual_ratio Maximum allowed RMS QEF plane residual as
 *                          a fraction of the local cell radius.
 * @param min_normal_alignment_threshold Minimum allowed alignment between
 *                          usable Hermite normals and their mean direction.
 * @param min_feature_thickness Minimum physical thickness to preserve via
 *                          adaptive solid opening. 0 disables the regularizer.
 * @param pre_thickening_radius Optional outward leaf-space thickening radius
 *                          applied before the opening stage.
 * @return DCPipelineResult containing the output mesh.
 */
inline DCPipelineResult run_dc_pipeline(
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    double isovalue,
    std::uint32_t max_depth,
    std::uint32_t smoothing_iterations,
    double smoothing_strength,
    double max_edge_ratio,
    std::uint32_t minimum_usable_hermite_samples = 3U,
    double max_qef_rms_residual_ratio = 0.1,
    double min_normal_alignment_threshold = 0.97,
    double min_feature_thickness = 0.0,
    double pre_thickening_radius = 0.0) {

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
        "Building", "run_dc_pipeline", top_cells.size());

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
        base_resolution,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold);

    if (min_feature_thickness > 0.0) {
        const Vector3d domain_extent = domain.extent();
        const double min_domain_extent = std::min(
            domain_extent.x,
            std::min(domain_extent.y, domain_extent.z));
        const double finest_leaf_size =
            min_domain_extent /
            static_cast<double>(base_resolution * (1U << max_depth));
        const double minimum_resolvable_thickness = 2.0 * finest_leaf_size;

        double effective_min_feature_thickness = min_feature_thickness;
        if (effective_min_feature_thickness < minimum_resolvable_thickness) {
            meshmerizer_log_detail::print_status(
                "Regularization",
                "run_dc_pipeline",
                "requested min_feature_thickness=%.6g is below the resolvable "
                "minimum %.6g at base_resolution=%u and max_depth=%u; "
                "clamping to %.6g\n",
                min_feature_thickness,
                minimum_resolvable_thickness,
                base_resolution,
                max_depth,
                minimum_resolvable_thickness);
            effective_min_feature_thickness = minimum_resolvable_thickness;
        }

        const double opening_radius = 0.5 * effective_min_feature_thickness;
        meshmerizer_log_detail::print_status(
            "Regularization",
            "run_dc_pipeline",
            "min_feature_thickness=%.6g (opening radius=%.6g)\n",
            effective_min_feature_thickness,
            opening_radius);

        const double max_surface_leaf_size = opening_radius;
        std::size_t regularization_refine_pass = 0U;
        while (true) {
            ++regularization_refine_pass;
            meshmerizer_log_detail::print_status(
                "Regularization",
                "run_dc_pipeline",
                "pass %zu: target_leaf_size<=%.6g (total_cells=%zu)\n",
                regularization_refine_pass,
                max_surface_leaf_size,
                all_cells.size());
            if (!refine_surface_band_cells(
                    all_cells, all_contributors, positions,
                    smoothing_lengths, isovalue, max_depth,
                    max_surface_leaf_size,
                    minimum_usable_hermite_samples,
                    max_qef_rms_residual_ratio,
                    min_normal_alignment_threshold)) {
                meshmerizer_log_detail::print_status(
                    "Regularization",
                    "run_dc_pipeline",
                    "pass %zu: no further surface-band refinement required\n",
                    regularization_refine_pass);
                break;
            }
        }

        meshmerizer_log_detail::print_status(
            "Regularization",
            "run_dc_pipeline",
            "targeted refinement complete; starting final balance "
            "(total_cells=%zu)\n",
            all_cells.size());
        balance_octree(
            all_cells, all_contributors, positions, smoothing_lengths,
            isovalue, domain, base_resolution, max_depth);

        LeafSpatialIndex solid_spatial_index;
        solid_spatial_index.build(all_cells, domain, max_depth, base_resolution);
        std::vector<OccupiedSolidLeaf> solid_leaves =
            classify_occupied_solid_leaves(
                all_cells, all_contributors, positions,
                smoothing_lengths, solid_spatial_index,
                isovalue, max_depth);
        std::vector<std::uint8_t> inside_mask = build_inside_mask(solid_leaves);

        if (pre_thickening_radius > 0.0) {
            const double thickening_leaf_size_target =
                std::max(pre_thickening_radius * 0.5, finest_leaf_size);
            std::size_t thickening_refine_pass = 0U;

            while (true) {
                const std::vector<double> thickening_distance =
                    compute_outside_distance_from_inside_mask(
                        solid_leaves, inside_mask);
                ++thickening_refine_pass;
                meshmerizer_log_detail::print_status(
                    "Regularization",
                    "run_dc_pipeline",
                    "pre-thickening pass %zu: target_leaf_size<=%.6g "
                    "(radius=%.6g, total_cells=%zu)\n",
                    thickening_refine_pass,
                    thickening_leaf_size_target,
                    pre_thickening_radius,
                    all_cells.size());
                if (!refine_thickening_band_cells(
                        all_cells, all_contributors, positions,
                        smoothing_lengths, isovalue, max_depth,
                        thickening_leaf_size_target, solid_leaves,
                        inside_mask, thickening_distance,
                        pre_thickening_radius,
                        minimum_usable_hermite_samples,
                        max_qef_rms_residual_ratio,
                        min_normal_alignment_threshold)) {
                    meshmerizer_log_detail::print_status(
                        "Regularization",
                        "run_dc_pipeline",
                        "pre-thickening pass %zu: no further growth-band "
                        "refinement required\n",
                        thickening_refine_pass);
                    break;
                }

                meshmerizer_log_detail::print_status(
                    "Regularization",
                    "run_dc_pipeline",
                    "pre-thickening: balancing octree after growth-band "
                    "refinement (total_cells=%zu)\n",
                    all_cells.size());
                balance_octree(
                    all_cells, all_contributors, positions, smoothing_lengths,
                    isovalue, domain, base_resolution, max_depth);
                solid_spatial_index.build(
                    all_cells, domain, max_depth, base_resolution);
                solid_leaves = classify_occupied_solid_leaves(
                    all_cells, all_contributors, positions,
                    smoothing_lengths, solid_spatial_index,
                    isovalue, max_depth);
                inside_mask = build_inside_mask(solid_leaves);
            }

            const std::vector<double> thickening_distance =
                compute_outside_distance_from_inside_mask(
                    solid_leaves, inside_mask);
            inside_mask = dilate_inside_mask(
                inside_mask, thickening_distance, pre_thickening_radius);
            meshmerizer_log_detail::print_status(
                "Regularization",
                "run_dc_pipeline",
                "pre-thickening: applied outward radius %.6g\n",
                pre_thickening_radius);
        }

        const std::vector<double> clearance =
            compute_inside_clearance(solid_leaves, inside_mask);
        const std::vector<std::uint8_t> eroded_inside =
            erode_occupied_solid_leaves(
                inside_mask, clearance, opening_radius);
        const std::vector<double> dilation_distance =
            compute_distance_to_eroded_solid(solid_leaves, eroded_inside);
        std::vector<std::uint8_t> opened_inside =
            dilate_eroded_solid_leaves(
                eroded_inside, dilation_distance, opening_radius);
        fill_small_opened_cavities(solid_leaves, opened_inside);
        prune_small_opened_components(solid_leaves, opened_inside);
        suppress_opened_edge_contacts(
            solid_leaves, all_cells, solid_spatial_index, opened_inside);

        auto log_opened_count = [&]() {
            return static_cast<std::size_t>(std::count(
                opened_inside.begin(), opened_inside.end(),
                static_cast<std::uint8_t>(1U)));
        };

        meshmerizer_log_detail::print_status(
            "Regularization",
            "run_dc_pipeline",
            "extracting opened blocky surface (opened_inside=%zu)\n",
            log_opened_count());

        OpenedSurfaceMesh opened_surface = generate_opened_surface_mesh(
            solid_leaves, opened_inside, all_cells, solid_spatial_index,
            domain, base_resolution, max_depth);
        if (resolve_opened_edge_ambiguities(
                solid_leaves, all_cells, solid_spatial_index,
                opened_inside, opened_surface)) {
            meshmerizer_log_detail::print_status(
                "Regularization",
                "run_dc_pipeline",
                "re-extracting opened blocky surface after ambiguity cleanup "
                "(opened_inside=%zu)\n",
                log_opened_count());
            opened_surface = generate_opened_surface_mesh(
                solid_leaves, opened_inside, all_cells, solid_spatial_index,
                domain, base_resolution, max_depth);
        }

        meshmerizer_log_detail::print_status(
            "Regularization",
            "run_dc_pipeline",
            "opened surface extraction done (%zu vertices, %zu triangles)\n",
            opened_surface.vertices.size(),
            opened_surface.triangles.size());

        if (smoothing_iterations > 0 && !opened_surface.vertices.empty()) {
            meshmerizer_log_detail::print_status(
                "Cleaning",
                "run_dc_pipeline",
                "%u smoothing iterations, lambda=%.2f (%zu vertices)\n",
                smoothing_iterations, smoothing_strength,
                opened_surface.vertices.size());
            VertexAdjacency adjacency = build_triangle_mesh_adjacency(
                opened_surface.vertices.size(),
                opened_surface.triangles);
            laplacian_smooth_vertices(
                opened_surface.vertices, adjacency,
                smoothing_iterations, smoothing_strength);
            meshmerizer_log_detail::print_status(
                "Cleaning", "run_dc_pipeline", "smoothing done\n");
        }

        std::vector<MeshTriangle> &opened_triangles = opened_surface.triangles;
        std::vector<MeshVertex> &opened_vertices = opened_surface.vertices;

        if (max_edge_ratio > 0.0 && !opened_triangles.empty()) {
            const std::size_t tris_before = opened_triangles.size();
            const std::size_t verts_before = opened_vertices.size();
            meshmerizer_log_detail::print_status(
                "Meshing",
                "run_dc_pipeline",
                "gap filling ratio=%.2f (%zu triangles, %zu vertices)\n",
                max_edge_ratio, tris_before, verts_before);
            std::vector<double> cell_sizes(opened_vertices.size(), opening_radius);
            subdivide_long_edges(
                opened_vertices, opened_triangles,
                cell_sizes, max_edge_ratio);
            meshmerizer_log_detail::print_status(
                "Meshing",
                "run_dc_pipeline",
                "gap filling done (+%zu vertices, +%zu triangles)\n",
                opened_vertices.size() - verts_before,
                opened_triangles.size() - tris_before);
        }

        std::size_t opened_inside_count = 0U;
        for (std::uint8_t flag : opened_inside) {
            opened_inside_count += flag != 0U ? 1U : 0U;
        }
        meshmerizer_log_detail::print_status(
            "Regularization",
            "run_dc_pipeline",
            "leaves=%zu opened_inside=%zu surface_vertices=%zu "
            "surface_triangles=%zu qef_vertices=%zu\n",
            solid_leaves.size(),
            opened_inside_count,
            opened_vertices.size(),
            opened_triangles.size(),
            static_cast<std::size_t>(0));

        meshmerizer_log_detail::print_status(
            "Regularization",
            "run_dc_pipeline",
            "using opened-solid blocky extraction with existing smoothing\n");

        print_octree_structure_summary(all_cells);

        result.n_qef_vertices = 0U;
        result.vertices.reserve(opened_vertices.size());
        for (const auto &v : opened_vertices) {
            result.vertices.push_back(v.position);
        }
        result.triangles.reserve(opened_triangles.size());
        for (const auto &tri : opened_triangles) {
            result.triangles.push_back(
                {static_cast<std::uint32_t>(tri.vertex_indices[0]),
                 static_cast<std::uint32_t>(tri.vertex_indices[1]),
                 static_cast<std::uint32_t>(tri.vertex_indices[2])});
        }
        return result;
    }

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
    // Step 4b: Retroactively activate incident cells missing QEF vertices.
    // ================================================================
    // A leaf can be incident to a sign-changing fine-grid edge even if its
    // own corner sign mask is uniform. Those leaves still need representative
    // vertices; otherwise face generation drops the corresponding quads and
    // produces square holes.

    while (refine_zero_sample_incident_cells(
        all_cells, all_contributors, positions, smoothing_lengths,
        spatial_index, max_depth, base_resolution, isovalue,
        domain)) {
        spatial_index.build(all_cells, domain, max_depth, base_resolution);
        qef_vertices = solve_all_leaf_vertices(
            all_cells, all_contributors, positions,
            smoothing_lengths, isovalue);
    }
    spatial_index.build(all_cells, domain, max_depth, base_resolution);

    activate_missing_incident_cells(
        all_cells, qef_vertices, all_contributors,
        positions, smoothing_lengths, spatial_index,
        max_depth, base_resolution, isovalue);

    // ================================================================
    // Step 4c: Optional Laplacian smoothing of QEF vertices.
    // ================================================================
    // If smoothing is enabled (iterations > 0), build a vertex
    // adjacency structure from the octree leaf connectivity and
    // apply iterative Laplacian smoothing to regularize vertex
    // positions.  This reduces jitter and improves triangle
    // quality, especially in regions with noisy Hermite data.
    // Smoothing runs BEFORE face generation so that the DC faces
    // connect the smoothed positions.

    if (smoothing_iterations > 0) {
        meshmerizer_log_detail::print_status(
            "Cleaning",
            "run_dc_pipeline",
            "%u smoothing iterations, lambda=%.2f (%zu vertices)\n",
            smoothing_iterations, smoothing_strength,
            qef_vertices.size());
        VertexAdjacency adjacency = build_vertex_adjacency(
            all_cells, spatial_index, max_depth,
            qef_vertices.size());
        laplacian_smooth_vertices(
            qef_vertices, adjacency,
            smoothing_iterations, smoothing_strength);
        meshmerizer_log_detail::print_status(
            "Cleaning", "run_dc_pipeline", "smoothing done\n");
    }

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
    // Step 5b: Gap filling via edge subdivision.
    // ================================================================
    // Identify triangle edges that are longer than max_edge_ratio
    // times the local cell size, insert intermediate vertices by
    // linear interpolation, and re-triangulate affected faces.
    // This fills gaps caused by large depth transitions or poorly
    // placed QEF vertices.  Always active (default ratio = 1.5).

    if (max_edge_ratio > 0.0 && !dc_triangles.empty()) {
        const std::size_t tris_before = dc_triangles.size();
        const std::size_t verts_before = qef_vertices.size();
        meshmerizer_log_detail::print_status(
            "Meshing",
            "run_dc_pipeline",
            "gap filling ratio=%.2f (%zu triangles, %zu vertices)\n",
            max_edge_ratio, tris_before, verts_before);
        std::vector<double> cell_sizes =
            compute_vertex_cell_sizes(
                all_cells, qef_vertices.size());
        subdivide_long_edges(
            qef_vertices, dc_triangles,
            cell_sizes, max_edge_ratio);
        const std::size_t new_verts =
            qef_vertices.size() - verts_before;
        const std::size_t new_tris =
            dc_triangles.size() - tris_before;
        meshmerizer_log_detail::print_status(
            "Meshing",
            "run_dc_pipeline",
            "gap filling done (+%zu vertices, +%zu triangles)\n",
            new_verts, new_tris);
    }

    print_octree_structure_summary(all_cells);

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
