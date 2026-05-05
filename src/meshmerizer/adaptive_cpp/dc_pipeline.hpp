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
#include "cancellation.hpp"
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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

inline double elapsed_seconds_since(
    const std::chrono::steady_clock::time_point &start_time) {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - start_time)
        .count();
}

enum class OccupiedSolidCacheRebuildMode : std::uint8_t {
    kLegacyPass = 0U,
};

inline void rebuild_occupied_solid_classification_cache(
    const std::vector<OctreeCell> &all_cells,
    const std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const LeafSpatialIndex &solid_spatial_index,
    double isovalue,
    std::uint32_t max_depth,
    OccupiedSolidClassificationCache &classification_cache,
    const std::vector<std::uint8_t> &dirty_cells,
    OccupiedSolidCacheRebuildMode rebuild_mode) {
    switch (rebuild_mode) {
        case OccupiedSolidCacheRebuildMode::kLegacyPass:
        default:
            update_occupied_solid_classification_cache(
                all_cells,
                all_contributors,
                positions,
                smoothing_lengths,
                solid_spatial_index,
                isovalue,
                max_depth,
                classification_cache,
                &dirty_cells);
            break;
    }
}

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
    std::uint32_t worker_count,
    std::uint32_t smoothing_iterations,
    double smoothing_strength,
    double max_edge_ratio,
    std::uint32_t minimum_usable_hermite_samples = 3U,
    double max_qef_rms_residual_ratio = 0.1,
    double min_normal_alignment_threshold = 0.97,
    double min_feature_thickness = 0.0,
    double pre_thickening_radius = 0.0,
    double table_cadence_seconds = 10.0) {

    DCPipelineResult result;
    result.isovalue = isovalue;
    const auto pipeline_start = std::chrono::steady_clock::now();

    // ================================================================
    // Step 1: Build top-level cells and query contributors.
    // ================================================================
    // The TopLevelParticleGrid spatially bins particles so that
    // contributor queries for each cell are O(local) rather than
    // O(N).

    TopLevelParticleGrid grid(domain, base_resolution);
    const auto contributor_query_start = std::chrono::steady_clock::now();
    grid.insert_particles(positions);
    grid.compute_bin_max_h(smoothing_lengths);

    std::vector<OctreeCell> top_cells =
        create_top_level_cells(domain, base_resolution);

    std::vector<OctreeCell> initial_cells;
    initial_cells.reserve(top_cells.size());
    std::vector<std::size_t> initial_contributors;
    std::vector<std::vector<std::size_t>> top_cell_contributors(
        top_cells.size());

    ProgressBar contrib_bar(
        "Building", "run_dc_pipeline", top_cells.size());

    const std::int64_t top_cell_count =
        static_cast<std::int64_t>(top_cells.size());
#pragma omp parallel for schedule(dynamic)
    for (std::int64_t ci = 0; ci < top_cell_count; ++ci) {
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                static_cast<std::size_t>(ci))) {
            contrib_bar.tick();
            continue;
        }
        const OctreeCell &cell = top_cells[static_cast<std::size_t>(ci)];
        std::vector<std::size_t> &cell_contributors =
            top_cell_contributors[static_cast<std::size_t>(ci)];

        std::uint32_t sx = 0, sy = 0, sz = 0;
        std::uint32_t ex = 0, ey = 0, ez = 0;
        grid.contributor_bin_span(
            cell.bounds, smoothing_lengths,
            sx, sy, sz, ex, ey, ez);

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
                            cell_contributors.push_back(pi);
                        }
                    }
                }
            }
        }
        contrib_bar.tick();
    }
    contrib_bar.finish();

    std::vector<std::size_t> contributor_offsets(top_cells.size() + 1U, 0U);
    for (std::size_t ci = 0; ci < top_cells.size(); ++ci) {
        contributor_offsets[ci + 1U] =
            contributor_offsets[ci] + top_cell_contributors[ci].size();
    }
    initial_contributors.reserve(contributor_offsets.back());
    for (std::size_t ci = 0; ci < top_cells.size(); ++ci) {
        OctreeCell cell = top_cells[ci];
        const std::size_t begin = contributor_offsets[ci];
        const std::size_t end = contributor_offsets[ci + 1U];
        cell.contributor_begin = static_cast<std::int64_t>(begin);
        cell.contributor_end = static_cast<std::int64_t>(end);
        initial_cells.push_back(cell);
        const std::vector<std::size_t> &cell_contributors =
            top_cell_contributors[ci];
        initial_contributors.insert(
            initial_contributors.end(),
            cell_contributors.begin(),
            cell_contributors.end());
    }
    meshmerizer_log_detail::print_debug_status(
        "Timing",
        "run_dc_pipeline",
        "Contributor query: %.3f s\n",
        elapsed_seconds_since(contributor_query_start));

    // ================================================================
    // Step 2: Refine the adaptive octree.
    // ================================================================
    // refine_octree subdivides cells where the density field
    // crosses the isovalue, producing an adaptive leaf set that
    // concentrates resolution where the surface is.

    const Vector3d domain_extent = domain.extent();
    const double min_domain_extent = std::min(
        domain_extent.x,
        std::min(domain_extent.y, domain_extent.z));
    const double finest_leaf_size =
        min_domain_extent /
        static_cast<double>(base_resolution * (1U << max_depth));
    double effective_min_feature_thickness = min_feature_thickness;
    double opening_radius = 0.0;
    double max_surface_leaf_size = 0.0;
    if (min_feature_thickness > 0.0) {
        const double minimum_resolvable_thickness = 2.0 * finest_leaf_size;
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
        opening_radius = 0.5 * effective_min_feature_thickness;
        max_surface_leaf_size = opening_radius;
    }

    const auto refine_start = std::chrono::steady_clock::now();
    auto [all_cells, all_contributors] = refine_octree(
        std::move(initial_cells),
        std::move(initial_contributors),
        positions,
        smoothing_lengths,
        isovalue,
        max_depth,
        domain,
        base_resolution,
        worker_count,
        table_cadence_seconds,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
        max_surface_leaf_size);
    meshmerizer_log_detail::print_debug_status(
        "Timing",
        "run_dc_pipeline",
        "Initial octree refinement (+surface band): %.3f s\n",
        elapsed_seconds_since(refine_start));

    if (min_feature_thickness > 0.0) {
        meshmerizer_log_detail::print_debug_status(
            "Regularization",
            "run_dc_pipeline",
            "min_feature_thickness=%.6g (opening radius=%.6g)\n",
            effective_min_feature_thickness,
            opening_radius);

        meshmerizer_log_detail::print_status(
            "Regularization",
            "run_dc_pipeline",
            "building solid leaf index (%zu cells)\n",
            all_cells.size());
        LeafSpatialIndex solid_spatial_index;
        solid_spatial_index.build(all_cells, domain, max_depth, base_resolution);
        OccupiedSolidClassificationCache classification_cache;
        constexpr OccupiedSolidCacheRebuildMode occupied_solid_rebuild_mode =
            OccupiedSolidCacheRebuildMode::kLegacyPass;
        std::vector<std::uint8_t> dirty_cells(all_cells.size(), 1U);
        std::vector<std::uint8_t> regularization_inside_mask_by_cell;
        meshmerizer_log_detail::print_status(
            "Regularization",
            "run_dc_pipeline",
            "updating occupied solid classification cache\n");
        const auto initial_solid_classify_start =
            std::chrono::steady_clock::now();
        rebuild_occupied_solid_classification_cache(
            all_cells,
            all_contributors,
            positions,
            smoothing_lengths,
            solid_spatial_index,
            isovalue,
            max_depth,
            classification_cache,
            dirty_cells,
            occupied_solid_rebuild_mode);
        meshmerizer_log_detail::print_status(
            "Timing",
            "run_dc_pipeline",
            "initial solid classification took %.3f s\n",
            elapsed_seconds_since(initial_solid_classify_start));

        if (pre_thickening_radius > 0.0) {
            const double thickening_leaf_size_target =
                std::max(pre_thickening_radius * 0.5, finest_leaf_size);

            // Frontier-seeded thickening closure.
            //
            // The old B2 path seeded kClassify for every eligible leaf and
            // allowed classify/distance/refine tasks to grow from there. On
            // large trees this created millions of classify tasks before the
            // queue had a useful frontier. Until the distance-front task graph
            // is fully cell-side-car driven, seed the closure from the bounded
            // outside growth band computed from the current solid mask. This
            // keeps task work proportional to the pre-thickening band rather
            // than the whole leaf set, while still using the task queue for
            // the structural refinement work.
            {
                const auto thickening_frontier_start =
                    std::chrono::steady_clock::now();
                meshmerizer_log_detail::print_debug_status(
                    "Regularization",
                    "run_dc_pipeline",
                    "frontier thickening closure: target_leaf_size<=%.6g "
                    "(radius=%.6g, total_cells=%zu)\n",
                    thickening_leaf_size_target,
                    pre_thickening_radius,
                    all_cells.size());
                const auto seed_distance_start = std::chrono::steady_clock::now();
                const std::vector<double> seed_distance =
                    compute_outside_distance_from_classification_cache(
                        all_cells, classification_cache,
                        pre_thickening_radius + 2.0 * thickening_leaf_size_target);
                meshmerizer_log_detail::print_status(
                    "Timing",
                    "run_dc_pipeline",
                    "frontier seed distance took %.3f s\n",
                    elapsed_seconds_since(seed_distance_start));
                dirty_cells.assign(all_cells.size(), 0U);
                std::vector<std::uint8_t> closure_inside_flags;
                std::vector<double> closure_center_values;
                std::vector<std::uint8_t> closure_occupancy_states;
                std::vector<std::array<std::size_t, 6>> closure_face_neighbors;
                const bool thickening_refined = refine_thickening_band_cells(
                    all_cells, all_contributors, positions,
                    smoothing_lengths, isovalue, max_depth,
                    domain, base_resolution, worker_count,
                    thickening_leaf_size_target,
                    classification_cache, seed_distance,
                    pre_thickening_radius,
                    table_cadence_seconds,
                    minimum_usable_hermite_samples,
                    max_qef_rms_residual_ratio,
                    min_normal_alignment_threshold,
                    &dirty_cells,
                    &closure_inside_flags,
                    &closure_center_values,
                    &closure_occupancy_states,
                    &closure_face_neighbors);
                if (!thickening_refined) {
                    meshmerizer_log_detail::print_status(
                        "Regularization",
                        "run_dc_pipeline",
                        "frontier thickening skipped: no eligible leaf cells "
                        "inside the growth band\n");
                }
                meshmerizer_log_detail::print_debug_status(
                    "Timing",
                    "run_dc_pipeline",
                    "Frontier thickening closure: %.3f s (total_cells=%zu)\n",
                    elapsed_seconds_since(thickening_frontier_start),
                    all_cells.size());
                meshmerizer_log_detail::print_status(
                    "Regularization",
                    "run_dc_pipeline",
                    "materializing occupied solid classification cache after pre-thickening\n");
                const auto rebuild_solid_classify_start =
                    std::chrono::steady_clock::now();
                if (thickening_refined) {
                    solid_spatial_index.build(
                        all_cells, domain, max_depth, base_resolution);
                    const bool exported_state_matches_tree =
                        closure_inside_flags.size() == all_cells.size() &&
                        closure_center_values.size() == all_cells.size() &&
                        closure_occupancy_states.size() == all_cells.size() &&
                        closure_face_neighbors.size() == all_cells.size();
                    if (!exported_state_matches_tree) {
                        meshmerizer_log_detail::print_status(
                            "Regularization",
                            "run_dc_pipeline",
                            "closure solid-state export size mismatch after pre-thickening "
                            "(cells=%zu, inside=%zu, center=%zu, occupancy=%zu, neighbors=%zu); "
                            "falling back to full cache rebuild\n",
                            all_cells.size(),
                            closure_inside_flags.size(),
                            closure_center_values.size(),
                            closure_occupancy_states.size(),
                            closure_face_neighbors.size());
                        dirty_cells.assign(all_cells.size(), 1U);
                        rebuild_occupied_solid_classification_cache(
                            all_cells,
                            all_contributors,
                            positions,
                            smoothing_lengths,
                            solid_spatial_index,
                            isovalue,
                            max_depth,
                            classification_cache,
                            dirty_cells,
                            occupied_solid_rebuild_mode);
                    } else {
                        merge_occupied_solid_cache_from_closure_state(
                            all_cells,
                            solid_spatial_index,
                            max_depth,
                            classification_cache,
                            std::move(closure_inside_flags),
                            std::move(closure_center_values),
                            std::move(closure_occupancy_states),
                            std::move(closure_face_neighbors),
                            dirty_cells);
                    }
                }
                if (classification_cache.inside_flags.size() != all_cells.size() ||
                    classification_cache.center_values.size() != all_cells.size() ||
                    classification_cache.occupancy_states.size() != all_cells.size() ||
                    classification_cache.face_neighbor_cell_indices.size() !=
                        all_cells.size()) {
                    throw std::runtime_error(
                        "post-thickening solid-state cache size mismatch");
                }
                meshmerizer_log_detail::print_status(
                    "Timing",
                    "run_dc_pipeline",
                    "post-thickening solid-state materialization took %.3f s\n",
                    elapsed_seconds_since(rebuild_solid_classify_start));
            }

            const auto final_thickening_distance_start =
                std::chrono::steady_clock::now();
            const std::vector<double> thickening_distance =
                compute_outside_distance_from_classification_cache(
                    all_cells, classification_cache,
                    pre_thickening_radius);
            meshmerizer_log_detail::print_status(
                "Timing",
                "run_dc_pipeline",
                "Final pre-thickening distance: %.3f s\n",
                elapsed_seconds_since(final_thickening_distance_start));
            const auto pre_thickening_dilate_start =
                std::chrono::steady_clock::now();
            regularization_inside_mask_by_cell =
                build_inside_mask_from_classification_cache(
                    all_cells, classification_cache);
            const std::vector<std::uint8_t> dilated_inside_mask_by_cell =
                dilate_inside_cell_mask(
                    all_cells,
                    regularization_inside_mask_by_cell,
                    thickening_distance,
                    pre_thickening_radius);
            meshmerizer_log_detail::print_status(
                "Timing",
                "run_dc_pipeline",
                "pre-thickening dilation took %.3f s\n",
                elapsed_seconds_since(pre_thickening_dilate_start));
            meshmerizer_log_detail::print_debug_status(
                "Regularization",
                "run_dc_pipeline",
                "pre-thickening: applied outward radius %.6g\n",
                pre_thickening_radius);
        }

        OccupiedSolidExtractionView extraction_view =
            build_occupied_solid_extraction_view(
                all_cells,
                classification_cache,
                std::move(regularization_inside_mask_by_cell));
        std::vector<OccupiedSolidLeaf> &solid_leaves =
            extraction_view.solid_leaves;
        std::vector<std::uint8_t> &inside_mask =
            extraction_view.inside_mask;

        const auto morphology_start = std::chrono::steady_clock::now();
        const std::vector<std::uint8_t> &inside_mask_by_cell =
            extraction_view.inside_mask_by_cell;
        const auto clearance_start = std::chrono::steady_clock::now();
        const std::vector<double> clearance_by_cell =
            compute_inside_clearance_from_cell_mask(
                all_cells, classification_cache, inside_mask_by_cell,
                opening_radius);
        meshmerizer_log_detail::print_status(
            "Timing",
            "run_dc_pipeline",
            "inside clearance took %.3f s\n",
            elapsed_seconds_since(clearance_start));
        const auto erosion_start = std::chrono::steady_clock::now();
        const std::vector<std::uint8_t> eroded_inside_by_cell =
            erode_occupied_solid_cells(
                all_cells, inside_mask_by_cell, clearance_by_cell,
                opening_radius);
        meshmerizer_log_detail::print_status(
            "Timing",
            "run_dc_pipeline",
            "erosion took %.3f s\n",
            elapsed_seconds_since(erosion_start));
        const auto dilation_distance_start =
            std::chrono::steady_clock::now();
        const std::vector<double> dilation_distance_by_cell =
            compute_distance_to_eroded_solid_from_cell_mask(
                all_cells, classification_cache, eroded_inside_by_cell,
                opening_radius);
        meshmerizer_log_detail::print_status(
            "Timing",
            "run_dc_pipeline",
            "eroded-solid distance took %.3f s\n",
            elapsed_seconds_since(dilation_distance_start));
        const auto dilation_start = std::chrono::steady_clock::now();
        const std::vector<std::uint8_t> opened_inside_by_cell =
            dilate_eroded_solid_cells(
                all_cells, eroded_inside_by_cell, dilation_distance_by_cell,
                opening_radius);
        std::vector<std::uint8_t> opened_inside =
            build_leaf_mask_from_cell_mask(
                solid_leaves, opened_inside_by_cell);
        meshmerizer_log_detail::print_status(
            "Timing",
            "run_dc_pipeline",
            "eroded-solid dilation took %.3f s\n",
            elapsed_seconds_since(dilation_start));
        const auto cleanup_start = std::chrono::steady_clock::now();
        fill_small_opened_cavities(solid_leaves, opened_inside);
        prune_small_opened_components(solid_leaves, opened_inside);
        suppress_opened_edge_contacts(
            solid_leaves, all_cells, solid_spatial_index, opened_inside);
        meshmerizer_log_detail::print_status(
            "Timing",
            "run_dc_pipeline",
            "opened-solid cleanup took %.3f s\n",
            elapsed_seconds_since(cleanup_start));
        meshmerizer_log_detail::print_status(
            "Timing",
            "run_dc_pipeline",
            "Regularization morphology: %.3f s\n",
            elapsed_seconds_since(morphology_start));

        auto log_opened_count = [&]() {
            return static_cast<std::size_t>(std::count(
                opened_inside.begin(), opened_inside.end(),
                static_cast<std::uint8_t>(1U)));
        };

        meshmerizer_log_detail::print_debug_status(
            "Regularization",
            "run_dc_pipeline",
            "extracting opened blocky surface (opened_inside=%zu)\n",
            log_opened_count());

        const auto opened_surface_start = std::chrono::steady_clock::now();
        OpenedSurfaceMesh opened_surface = generate_opened_surface_mesh(
            solid_leaves, opened_inside, all_cells, solid_spatial_index,
            domain, base_resolution, max_depth);
        meshmerizer_log_detail::print_status(
            "Timing",
            "run_dc_pipeline",
            "opened surface extraction pass 1 took %.3f s\n",
            elapsed_seconds_since(opened_surface_start));
        if (resolve_opened_edge_ambiguities(
                solid_leaves, all_cells, solid_spatial_index,
                opened_inside, opened_surface)) {
            meshmerizer_log_detail::print_debug_status(
                "Regularization",
                "run_dc_pipeline",
                "re-extracting opened blocky surface after ambiguity cleanup "
                "(opened_inside=%zu)\n",
                log_opened_count());
            const auto opened_surface_reextract_start =
                std::chrono::steady_clock::now();
            opened_surface = generate_opened_surface_mesh(
                solid_leaves, opened_inside, all_cells, solid_spatial_index,
                domain, base_resolution, max_depth);
            meshmerizer_log_detail::print_status(
                "Timing",
                "run_dc_pipeline",
                "opened surface extraction pass 2 took %.3f s\n",
                elapsed_seconds_since(opened_surface_reextract_start));
        }
        meshmerizer_log_detail::print_status(
            "Timing",
            "run_dc_pipeline",
            "Opened surface extraction: %.3f s\n",
            elapsed_seconds_since(opened_surface_start));

        meshmerizer_log_detail::print_status(
            "Regularization",
            "run_dc_pipeline",
            "opened surface extraction done (%zu vertices, %zu triangles)\n",
            opened_surface.vertices.size(),
            opened_surface.triangles.size());

        if (smoothing_iterations > 0 && !opened_surface.vertices.empty()) {
            const auto smoothing_start = std::chrono::steady_clock::now();
            meshmerizer_log_detail::print_debug_status(
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
            meshmerizer_log_detail::print_debug_status(
                "Cleaning", "run_dc_pipeline", "smoothing done\n");
            meshmerizer_log_detail::print_debug_status(
                "Timing",
                "run_dc_pipeline",
                "Opened-surface smoothing: %.3f s\n",
                elapsed_seconds_since(smoothing_start));
        }

        std::vector<MeshTriangle> &opened_triangles = opened_surface.triangles;
        std::vector<MeshVertex> &opened_vertices = opened_surface.vertices;

        if (max_edge_ratio > 0.0 && !opened_triangles.empty()) {
            const auto gap_fill_start = std::chrono::steady_clock::now();
            const std::size_t tris_before = opened_triangles.size();
            const std::size_t verts_before = opened_vertices.size();
            meshmerizer_log_detail::print_debug_status(
                "Meshing",
                "run_dc_pipeline",
                "gap filling ratio=%.2f (%zu triangles, %zu vertices)\n",
                max_edge_ratio, tris_before, verts_before);
            std::vector<double> cell_sizes(opened_vertices.size(), opening_radius);
            subdivide_long_edges(
                opened_vertices, opened_triangles,
                cell_sizes, max_edge_ratio);
            meshmerizer_log_detail::print_debug_status(
                "Meshing",
                "run_dc_pipeline",
                "gap filling done (+%zu vertices, +%zu triangles)\n",
                opened_vertices.size() - verts_before,
                opened_triangles.size() - tris_before);
            meshmerizer_log_detail::print_debug_status(
                "Timing",
                "run_dc_pipeline",
                "Opened-surface gap filling: %.3f s\n",
                elapsed_seconds_since(gap_fill_start));
        }

        std::size_t opened_inside_count = 0U;
        for (std::uint8_t flag : opened_inside) {
            opened_inside_count += flag != 0U ? 1U : 0U;
        }
        meshmerizer_log_detail::print_debug_status(
            "Regularization",
            "run_dc_pipeline",
            "leaves=%zu opened_inside=%zu surface_vertices=%zu "
            "surface_triangles=%zu qef_vertices=%zu\n",
            solid_leaves.size(),
            opened_inside_count,
            opened_vertices.size(),
            opened_triangles.size(),
            static_cast<std::size_t>(0));

        meshmerizer_log_detail::print_debug_status(
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

    const auto qef_solve_start = std::chrono::steady_clock::now();
    std::vector<MeshVertex> qef_vertices =
        solve_all_leaf_vertices(
            all_cells, all_contributors, positions,
            smoothing_lengths, isovalue);
    meshmerizer_log_detail::print_debug_status(
        "Timing",
        "run_dc_pipeline",
        "QEF vertex solve: %.3f s\n",
        elapsed_seconds_since(qef_solve_start));

    result.n_qef_vertices = qef_vertices.size();

    if (qef_vertices.empty()) {
        return result;
    }

    // ================================================================
    // Step 4: Build spatial index for leaf cell lookups.
    // ================================================================
    // The LeafSpatialIndex hashes leaf cells by their quantized
    // min-corner so that find_leaf_at(ix, iy, iz) is O(1).

    const auto spatial_index_build_start = std::chrono::steady_clock::now();
    LeafSpatialIndex spatial_index;
    spatial_index.build(all_cells, domain, max_depth,
                        base_resolution);
    meshmerizer_log_detail::print_debug_status(
        "Timing",
        "run_dc_pipeline",
        "Leaf spatial index build: %.3f s\n",
        elapsed_seconds_since(spatial_index_build_start));

    // ================================================================
    // Step 4b: Retroactively activate incident cells missing QEF vertices.
    // ================================================================
    // A leaf can be incident to a sign-changing fine-grid edge even if its
    // own corner sign mask is uniform. Those leaves still need representative
    // vertices; otherwise face generation drops the corresponding quads and
    // produces square holes.

    const auto incident_repair_start = std::chrono::steady_clock::now();
    while (refine_zero_sample_incident_cells(
        all_cells, all_contributors, positions, smoothing_lengths,
        spatial_index, max_depth, base_resolution, isovalue,
        domain, table_cadence_seconds)) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(
            qef_vertices.size() + all_cells.size());
        spatial_index.build(all_cells, domain, max_depth, base_resolution);
        qef_vertices = solve_all_leaf_vertices(
            all_cells, all_contributors, positions,
            smoothing_lengths, isovalue);
    }
    spatial_index.build(all_cells, domain, max_depth, base_resolution);
    meshmerizer_log_detail::print_debug_status(
        "Timing",
        "run_dc_pipeline",
        "Incident cell repair/activation prep: %.3f s\n",
        elapsed_seconds_since(incident_repair_start));

    const auto incident_activation_start = std::chrono::steady_clock::now();
    activate_missing_incident_cells(
        all_cells, qef_vertices, all_contributors,
        positions, smoothing_lengths, spatial_index,
        max_depth, base_resolution, isovalue);
    meshmerizer_log_detail::print_debug_status(
        "Timing",
        "run_dc_pipeline",
        "Incident cell activation: %.3f s\n",
        elapsed_seconds_since(incident_activation_start));

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
        const auto qef_smoothing_start = std::chrono::steady_clock::now();
        meshmerizer_log_detail::print_debug_status(
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
        meshmerizer_log_detail::print_debug_status(
            "Cleaning", "run_dc_pipeline", "smoothing done\n");
        meshmerizer_log_detail::print_debug_status(
            "Timing",
            "run_dc_pipeline",
            "QEF smoothing: %.3f s\n",
            elapsed_seconds_since(qef_smoothing_start));
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

    const auto face_generation_start = std::chrono::steady_clock::now();
    std::vector<MeshTriangle> dc_triangles =
        generate_dual_contour_faces(
            all_cells, qef_vertices, spatial_index,
            max_depth, base_resolution, isovalue);
    meshmerizer_log_detail::print_debug_status(
        "Timing",
        "run_dc_pipeline",
        "Dual contour face generation: %.3f s\n",
        elapsed_seconds_since(face_generation_start));

    // ================================================================
    // Step 5b: Gap filling via edge subdivision.
    // ================================================================
    // Identify triangle edges that are longer than max_edge_ratio
    // times the local cell size, insert intermediate vertices by
    // linear interpolation, and re-triangulate affected faces.
    // This fills gaps caused by large depth transitions or poorly
    // placed QEF vertices.  Always active (default ratio = 1.5).

    if (max_edge_ratio > 0.0 && !dc_triangles.empty()) {
        const auto gap_fill_start = std::chrono::steady_clock::now();
        const std::size_t tris_before = dc_triangles.size();
        const std::size_t verts_before = qef_vertices.size();
        meshmerizer_log_detail::print_debug_status(
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
        meshmerizer_log_detail::print_debug_status(
            "Meshing",
            "run_dc_pipeline",
            "gap filling done (+%zu vertices, +%zu triangles)\n",
            new_verts, new_tris);
        meshmerizer_log_detail::print_debug_status(
            "Timing",
            "run_dc_pipeline",
            "Dual contour gap filling: %.3f s\n",
            elapsed_seconds_since(gap_fill_start));
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

    meshmerizer_log_detail::print_debug_status(
        "Timing",
        "run_dc_pipeline",
        "Total C++ reconstruction core: %.3f s\n",
        elapsed_seconds_since(pipeline_start));

    return result;
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_DC_PIPELINE_HPP_
