/**
 * @file adaptive_solid.hpp
 * @brief Adaptive opened-solid regularization helpers.
 *
 * This header implements the topology-regularization path used to remove thin
 * or fragile features before final surface extraction. The core workflow is:
 *
 * 1. classify octree leaves as inside, outside, or boundary-adjacent,
 * 2. optionally refine a narrow surface band to improve local resolution,
 * 3. compute erosion and dilation distances on the occupied solid, and
 * 4. extract an opened, print-friendly surface from the modified topology.
 *
 * The data structures below capture the per-leaf state reused across those
 * phases. Dense comments are especially important here because topology edits
 * depend on invariants shared between several passes.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_ADAPTIVE_SOLID_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_ADAPTIVE_SOLID_HPP_

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <queue>
#include <span>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cancellation.hpp"
#include "faces.hpp"
#include "octree_cell.hpp"
#include "refinement_closure.hpp"
#include "refinement_work_queue.hpp"

/**
 * @brief Occupancy classification used by the opened-solid pipeline.
 */
enum class OccupancyState : std::uint8_t {
    kOutside = 0U,
    kInside = 1U,
    kBoundaryInside = 2U,
    kBoundaryOutside = 3U,
};

/**
 * @brief Compact per-leaf record for occupied-solid operations.
 *
 * Each record stores the minimum state needed to run morphology and reopened
 * surface extraction on octree leaves without repeatedly rewalking the full
 * adaptive tree.
 */
struct OccupiedSolidLeaf {
    /** Index into the global ``all_cells`` array. */
    std::size_t cell_index;
    /** Scalar field value sampled at the leaf centre. */
    double center_value;
    /** Leaf edge length in domain units. */
    double cell_size;
    /** Octree depth of the leaf. */
    std::uint32_t depth;
    /** Occupancy classification assigned to this leaf. */
    OccupancyState occupancy;
    /** Neighbor leaf indices for the six axis-aligned faces, or -1 if absent. */
    std::array<std::int64_t, 6> face_neighbor_leaf_indices;
};

/**
 * @brief Cached classification arrays aligned with the full octree.
 *
 * These vectors let later topology passes reuse occupancy decisions without
 * repeatedly walking and reclassifying the whole tree.
 */
struct OccupiedSolidClassificationCache {
    /** Boolean inside/outside flags for all cells. */
    std::vector<std::uint8_t> inside_flags;
    /** Scalar field values sampled at cell centres. */
    std::vector<double> center_values;
    /** Occupancy state values for all cells. */
    std::vector<std::uint8_t> occupancy_states;
    /** Face-neighbor lookup into the global cell array. */
    std::vector<std::array<std::size_t, 6>> face_neighbor_cell_indices;
};

/**
 * @brief Compact extraction views derived from the full-tree classification cache.
 *
 * Phase 5 groundwork: keep cell-local material state authoritative on the full
 * octree, and build compact leaf-only views explicitly at the
 * morphology/extraction boundary.
 */
struct OccupiedSolidExtractionView {
    /** Compact occupied-solid leaf records used by morphology/extraction. */
    std::vector<OccupiedSolidLeaf> solid_leaves;
    /** Reverse lookup from global cell index to ``solid_leaves`` index. */
    std::vector<std::int64_t> cell_to_leaf_index;
    /** Full-tree inside mask aligned with ``all_cells``. */
    std::vector<std::uint8_t> inside_mask_by_cell;
    /** Leaf-compact inside mask aligned with ``solid_leaves``. */
    std::vector<std::uint8_t> inside_mask;
};

/**
 * @brief Return whether one cached cell classification is part of the solid.
 *
 * @param occupancy Cached occupancy state stored in the full-tree side-car.
 * @return ``true`` when the cell is inside or boundary-inside.
 */
inline bool occupied_solid_cache_is_inside(std::uint8_t occupancy) {
    const auto state = static_cast<OccupancyState>(occupancy);
    return state == OccupancyState::kInside ||
           state == OccupancyState::kBoundaryInside;
}

/**
 * @brief Boundary sample emitted from the opened solid.
 *
 * These samples act as the reopened analogue of classic Hermite samples: they
 * provide a boundary position and outward-facing normal that later QEF solves
 * can use to place representative mesh vertices.
 */
struct OpenedBoundarySample {
    /** Sample position on the opened boundary. */
    Vector3d position;
    /** Outward-facing normal at the sample position. */
    Vector3d outward_normal;
    /** Index of the source occupied-solid leaf. */
    std::size_t leaf_index;
};

/**
 * @brief Build a reverse lookup from global cell index to occupied-solid leaf.
 *
 * @param all_cells Complete octree cell array.
 * @param solid_leaves Occupied-solid leaf records.
 * @return Vector mapping cell index to leaf index, or ``-1`` when absent.
 */
inline std::vector<std::int64_t> build_opened_cell_to_leaf_index(
    const std::vector<OctreeCell> &all_cells,
    const std::vector<OccupiedSolidLeaf> &solid_leaves) {
    std::vector<std::int64_t> cell_to_leaf_index(all_cells.size(), -1);
    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        cell_to_leaf_index[solid_leaves[leaf_index].cell_index] =
            static_cast<std::int64_t>(leaf_index);
    }
    return cell_to_leaf_index;
}

/**
 * @brief Return the contributor slice bounds for one cell.
 *
 * @param cell Cell whose contributor range is requested.
 * @param all_contributors Global flat contributor array.
 * @return Pair ``(begin, end)`` into ``all_contributors``.
 */
inline std::pair<std::size_t, std::size_t> cell_contributor_bounds(
    const OctreeCell &cell,
    const std::vector<std::size_t> &all_contributors) {
    if (cell.contributor_begin < 0 ||
        cell.contributor_end <= cell.contributor_begin) {
        return {0U, 0U};
    }
    const auto begin_idx = static_cast<std::size_t>(cell.contributor_begin);
    const auto end_idx = static_cast<std::size_t>(cell.contributor_end);
    if (end_idx > all_contributors.size()) {
        return {0U, 0U};
    }
    return {begin_idx, end_idx};
}

/**
 * @brief Return the contributor span for one cell.
 *
 * @param cell Cell whose contributor span is requested.
 * @param all_contributors Global flat contributor array.
 * @return View over the cell's contributor indices.
 */
inline std::span<const std::size_t> cell_contributor_span(
    const OctreeCell &cell,
    const std::vector<std::size_t> &all_contributors) {
    const auto [begin_idx, end_idx] =
        cell_contributor_bounds(cell, all_contributors);
    return std::span<const std::size_t>(
        all_contributors.data() + begin_idx,
        end_idx - begin_idx);
}

/**
 * @brief Return the edge length of a cubic octree cell.
 *
 * @param cell Input cell.
 * @return Cell edge length in domain units.
 */
inline double cell_edge_length(const OctreeCell &cell) {
    return cell.bounds.max.x - cell.bounds.min.x;
}

/**
 * @brief Refine surface-adjacent leaves until they satisfy a target size.
 *
 * This is the first major regularization step. It focuses extra resolution in
 * a narrow band around the implicit surface so later morphological operations
 * can measure thickness on cells that are small enough to be meaningful.
 *
 * Cells are re-sampled immediately after splitting so the queue always contains
 * leaves whose current corner data and contributor ranges are valid.
 *
 * This refinement is intentionally local: it improves the fidelity of the
 * topology-regularization stage near the implicit surface without forcing the
 * whole octree to refine to the same size threshold.
 *
 * @param all_cells Complete octree cell array to update in place.
 * @param all_contributors Global flat contributor array to update in place.
 * @param positions Particle positions.
 * @param smoothing_lengths Per-particle support radii.
 * @param isovalue Scalar field threshold for surface detection.
 * @param max_depth Maximum octree refinement depth.
 * @param max_surface_leaf_size Largest acceptable surface-band leaf size.
 * @param minimum_usable_hermite_samples Minimum usable Hermite sample count.
 * @param max_qef_rms_residual_ratio Maximum allowed RMS QEF residual ratio.
 * @param min_normal_alignment_threshold Minimum allowed Hermite normal
 *     alignment.
 * @return ``true`` when at least one split was performed.
 */
inline bool refine_surface_band_cells(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue,
    std::uint32_t max_depth,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    std::uint32_t worker_count,
    double max_surface_leaf_size,
    double table_cadence_seconds,
    std::uint32_t minimum_usable_hermite_samples,
    double max_qef_rms_residual_ratio,
    double min_normal_alignment_threshold) {
    if (max_surface_leaf_size <= 0.0) {
        return false;
    }

    const RefinementClosureConfig closure_config = {
        isovalue,
        max_depth,
        domain,
        base_resolution,
        worker_count,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
        table_cadence_seconds,
        "Regularization",
        "refine_surface_band_cells",
        "surface_band",
    };

    return refine_surface_band_with_closure(
        all_cells,
        all_contributors,
        positions,
        smoothing_lengths,
        closure_config,
        max_surface_leaf_size);
}

inline std::array<std::size_t, 6> face_neighbor_cells(
    const OctreeCell &cell,
    const LeafSpatialIndex &spatial_index,
    std::uint32_t max_depth) {
    std::array<std::size_t, 6> neighbors = {
        SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX,
    };

    std::uint32_t cell_x = 0U;
    std::uint32_t cell_y = 0U;
    std::uint32_t cell_z = 0U;
    morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
    const std::uint32_t span = 1U << (max_depth - cell.depth);
    const std::uint32_t ix0 = cell_x << (max_depth - cell.depth);
    const std::uint32_t iy0 = cell_y << (max_depth - cell.depth);
    const std::uint32_t iz0 = cell_z << (max_depth - cell.depth);

    const std::uint32_t cx = ix0 + span / 2U;
    const std::uint32_t cy = iy0 + span / 2U;
    const std::uint32_t cz = iz0 + span / 2U;
    const std::uint32_t fine_res = spatial_index.fine_resolution;

    if (ix0 > 0U) {
        neighbors[0] = spatial_index.find_leaf_at(ix0 - 1U, cy, cz);
    }
    if (ix0 + span < fine_res) {
        neighbors[1] = spatial_index.find_leaf_at(ix0 + span, cy, cz);
    }
    if (iy0 > 0U) {
        neighbors[2] = spatial_index.find_leaf_at(cx, iy0 - 1U, cz);
    }
    if (iy0 + span < fine_res) {
        neighbors[3] = spatial_index.find_leaf_at(cx, iy0 + span, cz);
    }
    if (iz0 > 0U) {
        neighbors[4] = spatial_index.find_leaf_at(cx, cy, iz0 - 1U);
    }
    if (iz0 + span < fine_res) {
        neighbors[5] = spatial_index.find_leaf_at(cx, cy, iz0 + span);
    }

    return neighbors;
}

inline void update_occupied_solid_classification_cache(
    const std::vector<OctreeCell> &all_cells,
    const std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const LeafSpatialIndex &spatial_index,
    double isovalue,
    std::uint32_t max_depth,
    OccupiedSolidClassificationCache &cache,
    const std::vector<std::uint8_t> *dirty_cells = nullptr,
    const std::vector<std::uint8_t> *precomputed_inside_flags = nullptr,
    const std::vector<double> *precomputed_center_values = nullptr) {
    cache.inside_flags.resize(all_cells.size(), 0U);
    cache.center_values.resize(all_cells.size(), 0.0);
    cache.occupancy_states.resize(
        all_cells.size(),
        static_cast<std::uint8_t>(OccupancyState::kOutside));
    cache.face_neighbor_cell_indices.resize(
        all_cells.size(),
        {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX});
    std::vector<std::uint8_t> occupancy_dirty;
    std::vector<std::size_t> classify_cell_indices;
    std::vector<std::size_t> occupancy_cell_indices;

    if (dirty_cells != nullptr) {
        occupancy_dirty = *dirty_cells;
        if (occupancy_dirty.size() < all_cells.size()) {
            occupancy_dirty.resize(all_cells.size(), 0U);
        }
        for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
            meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
            const OctreeCell &cell = all_cells[cell_idx];
            if (!cell.is_leaf) {
                continue;
            }
            if (cell_idx < dirty_cells->size() && (*dirty_cells)[cell_idx] != 0U) {
                classify_cell_indices.push_back(cell_idx);
            }
            if (cell_idx >= occupancy_dirty.size() || occupancy_dirty[cell_idx] == 0U) {
                continue;
            }
            const auto neighbors = face_neighbor_cells(
                cell, spatial_index, max_depth);
            for (std::size_t neighbor_idx : neighbors) {
                if (neighbor_idx == SIZE_MAX || neighbor_idx >= occupancy_dirty.size()) {
                    continue;
                }
                occupancy_dirty[neighbor_idx] = 1U;
            }
        }
        occupancy_cell_indices.reserve(classify_cell_indices.size());
        for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
            meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
            if (!all_cells[cell_idx].is_leaf) {
                continue;
            }
            if (occupancy_dirty[cell_idx] != 0U) {
                occupancy_cell_indices.push_back(cell_idx);
            }
        }
    } else {
        classify_cell_indices.reserve(all_cells.size());
        occupancy_cell_indices.reserve(all_cells.size());
        for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
            meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
            if (!all_cells[cell_idx].is_leaf) {
                continue;
            }
            classify_cell_indices.push_back(cell_idx);
            occupancy_cell_indices.push_back(cell_idx);
        }
    }

    ProgressCounter classify_counter(
        "Regularization", "classify_occupied_solid_leaves", "cells", 1000);

#pragma omp parallel for schedule(dynamic)
    for (std::int64_t active_idx = 0;
         active_idx < static_cast<std::int64_t>(classify_cell_indices.size());
         ++active_idx) {
        const std::size_t cell_idx =
            classify_cell_indices[static_cast<std::size_t>(active_idx)];
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                cell_idx)) {
            classify_counter.tick();
            continue;
        }
        classify_counter.tick();
        const OctreeCell &cell = all_cells[cell_idx];

        const bool has_precomputed_value =
            precomputed_inside_flags != nullptr &&
            precomputed_center_values != nullptr &&
            cell_idx < precomputed_inside_flags->size() &&
            cell_idx < precomputed_center_values->size();
        if (has_precomputed_value) {
            cache.center_values[cell_idx] = (*precomputed_center_values)[cell_idx];
            cache.inside_flags[cell_idx] = (*precomputed_inside_flags)[cell_idx];
            continue;
        }

        const std::span<const std::size_t> contributors =
            cell_contributor_span(cell, all_contributors);
        const double center_value = evaluate_field_at_point(
            cell.bounds.center(), contributors, positions, smoothing_lengths);
        cache.center_values[cell_idx] = center_value;

        std::size_t inside_corner_count = 0;
        for (double corner_value : cell.corner_values) {
            if (corner_value >= isovalue) {
                ++inside_corner_count;
            }
        }

        const bool inside =
            center_value >= isovalue || inside_corner_count >= 4U;
        cache.inside_flags[cell_idx] = inside ? 1U : 0U;
    }

    classify_counter.finish();

    ProgressCounter occupancy_counter(
        "Regularization", "classify_occupied_solid_leaves", "cells", 1000);

#pragma omp parallel for schedule(dynamic)
    for (std::int64_t active_idx = 0;
         active_idx < static_cast<std::int64_t>(occupancy_cell_indices.size());
         ++active_idx) {
        const std::size_t cell_idx =
            occupancy_cell_indices[static_cast<std::size_t>(active_idx)];
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                cell_idx)) {
            occupancy_counter.tick();
            continue;
        }
        occupancy_counter.tick();
        const OctreeCell &cell = all_cells[cell_idx];

        const bool inside = cache.inside_flags[cell_idx] != 0U;
        OccupancyState occupancy = OccupancyState::kOutside;
        bool touches_opposite = false;
        const auto neighbors = face_neighbor_cells(
            cell, spatial_index, max_depth);
        cache.face_neighbor_cell_indices[cell_idx] = neighbors;
        for (std::size_t neighbor_idx : neighbors) {
            if (neighbor_idx == SIZE_MAX || neighbor_idx >= all_cells.size()) {
                continue;
            }
            if (cache.inside_flags[neighbor_idx] !=
                cache.inside_flags[cell_idx]) {
                touches_opposite = true;
                break;
            }
        }

        occupancy = inside ? OccupancyState::kInside
                           : OccupancyState::kOutside;
        if (touches_opposite) {
            occupancy = inside ? OccupancyState::kBoundaryInside
                               : OccupancyState::kBoundaryOutside;
        }
        cache.occupancy_states[cell_idx] =
            static_cast<std::uint8_t>(occupancy);
    }
    occupancy_counter.finish();

    for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
        if (all_cells[cell_idx].is_leaf) {
            continue;
        }
        cache.inside_flags[cell_idx] = 0U;
        cache.center_values[cell_idx] = 0.0;
        cache.occupancy_states[cell_idx] =
            static_cast<std::uint8_t>(OccupancyState::kOutside);
        cache.face_neighbor_cell_indices[cell_idx] = {
            SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
    }
}

template <typename ProcessCell>
inline void run_static_leaf_queue_epoch(
    const std::vector<std::size_t> &cell_indices,
    std::uint32_t worker_count,
    ProcessCell process_cell) {
    if (cell_indices.empty()) {
        return;
    }

    RefinementWorkQueue queue;
    const std::uint32_t active_workers = std::max(1U, worker_count);
    queue.initialize(active_workers);
    for (std::size_t ordinal = 0; ordinal < cell_indices.size(); ++ordinal) {
        queue.push(
            {cell_indices[ordinal], 0U, 0U, RefinementTaskKind::kRefine},
            static_cast<std::uint32_t>(ordinal % active_workers));
    }
    queue.capture_initial_queue_size();

    std::exception_ptr worker_error;
    std::mutex error_mutex;
    std::vector<std::thread> workers;
    workers.reserve(active_workers);
    for (std::uint32_t worker_id = 0; worker_id < active_workers; ++worker_id) {
        workers.emplace_back([&, worker_id]() {
            RefinementTask task;
            while (queue.pop(worker_id, task)) {
                try {
                    process_cell(task.cell_index);
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (worker_error == nullptr) {
                            worker_error = std::current_exception();
                        }
                    }
                    queue.task_done();
                    queue.shutdown();
                    return;
                }
                queue.task_done();
                queue.try_shutdown_if_idle();
            }
        });
    }

    for (std::thread &worker : workers) {
        worker.join();
    }
    if (worker_error != nullptr) {
        std::rethrow_exception(worker_error);
    }
}

inline void update_occupied_solid_classification_cache_with_queue(
    const std::vector<OctreeCell> &all_cells,
    const std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const LeafSpatialIndex &spatial_index,
    double isovalue,
    std::uint32_t max_depth,
    std::uint32_t worker_count,
    OccupiedSolidClassificationCache &cache,
    const std::vector<std::uint8_t> *dirty_cells = nullptr,
    const std::vector<std::uint8_t> *precomputed_inside_flags = nullptr,
    const std::vector<double> *precomputed_center_values = nullptr) {
    cache.inside_flags.resize(all_cells.size(), 0U);
    cache.center_values.resize(all_cells.size(), 0.0);
    cache.occupancy_states.resize(
        all_cells.size(),
        static_cast<std::uint8_t>(OccupancyState::kOutside));
    cache.face_neighbor_cell_indices.resize(
        all_cells.size(),
        {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX});
    std::vector<std::uint8_t> occupancy_dirty;
    std::vector<std::size_t> classify_cell_indices;
    std::vector<std::size_t> occupancy_cell_indices;

    if (dirty_cells != nullptr) {
        occupancy_dirty = *dirty_cells;
        if (occupancy_dirty.size() < all_cells.size()) {
            occupancy_dirty.resize(all_cells.size(), 0U);
        }
        for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
            meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
            const OctreeCell &cell = all_cells[cell_idx];
            if (!cell.is_leaf) {
                continue;
            }
            if (cell_idx < dirty_cells->size() && (*dirty_cells)[cell_idx] != 0U) {
                classify_cell_indices.push_back(cell_idx);
            }
            if (cell_idx >= occupancy_dirty.size() || occupancy_dirty[cell_idx] == 0U) {
                continue;
            }
            const auto neighbors = face_neighbor_cells(
                cell, spatial_index, max_depth);
            for (std::size_t neighbor_idx : neighbors) {
                if (neighbor_idx == SIZE_MAX || neighbor_idx >= occupancy_dirty.size()) {
                    continue;
                }
                occupancy_dirty[neighbor_idx] = 1U;
            }
        }
        occupancy_cell_indices.reserve(classify_cell_indices.size());
        for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
            meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
            if (!all_cells[cell_idx].is_leaf) {
                continue;
            }
            if (occupancy_dirty[cell_idx] != 0U) {
                occupancy_cell_indices.push_back(cell_idx);
            }
        }
    } else {
        classify_cell_indices.reserve(all_cells.size());
        occupancy_cell_indices.reserve(all_cells.size());
        for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
            meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
            if (!all_cells[cell_idx].is_leaf) {
                continue;
            }
            classify_cell_indices.push_back(cell_idx);
            occupancy_cell_indices.push_back(cell_idx);
        }
    }

    run_static_leaf_queue_epoch(
        classify_cell_indices,
        worker_count,
        [&](std::size_t cell_idx) {
            if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                    cell_idx)) {
                return;
            }
            const OctreeCell &cell = all_cells[cell_idx];

            const bool has_precomputed_value =
                precomputed_inside_flags != nullptr &&
                precomputed_center_values != nullptr &&
                cell_idx < precomputed_inside_flags->size() &&
                cell_idx < precomputed_center_values->size();
            if (has_precomputed_value) {
                cache.center_values[cell_idx] = (*precomputed_center_values)[cell_idx];
                cache.inside_flags[cell_idx] = (*precomputed_inside_flags)[cell_idx];
                return;
            }

            const std::span<const std::size_t> contributors =
                cell_contributor_span(cell, all_contributors);
            const double center_value = evaluate_field_at_point(
                cell.bounds.center(), contributors, positions, smoothing_lengths);
            cache.center_values[cell_idx] = center_value;

            std::size_t inside_corner_count = 0;
            for (double corner_value : cell.corner_values) {
                if (corner_value >= isovalue) {
                    ++inside_corner_count;
                }
            }

            const bool inside =
                center_value >= isovalue || inside_corner_count >= 4U;
            cache.inside_flags[cell_idx] = inside ? 1U : 0U;
        });

    run_static_leaf_queue_epoch(
        occupancy_cell_indices,
        worker_count,
        [&](std::size_t cell_idx) {
            if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                    cell_idx)) {
                return;
            }
            const OctreeCell &cell = all_cells[cell_idx];

            const bool inside = cache.inside_flags[cell_idx] != 0U;
            OccupancyState occupancy = OccupancyState::kOutside;
            bool touches_opposite = false;
            const auto neighbors = face_neighbor_cells(
                cell, spatial_index, max_depth);
            cache.face_neighbor_cell_indices[cell_idx] = neighbors;
            for (std::size_t neighbor_idx : neighbors) {
                if (neighbor_idx == SIZE_MAX || neighbor_idx >= all_cells.size()) {
                    continue;
                }
                if (cache.inside_flags[neighbor_idx] !=
                    cache.inside_flags[cell_idx]) {
                    touches_opposite = true;
                    break;
                }
            }

            occupancy = inside ? OccupancyState::kInside
                               : OccupancyState::kOutside;
            if (touches_opposite) {
                occupancy = inside ? OccupancyState::kBoundaryInside
                                   : OccupancyState::kBoundaryOutside;
            }
            cache.occupancy_states[cell_idx] =
                static_cast<std::uint8_t>(occupancy);
        });

    for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
        if (all_cells[cell_idx].is_leaf) {
            continue;
        }
        cache.inside_flags[cell_idx] = 0U;
        cache.center_values[cell_idx] = 0.0;
        cache.occupancy_states[cell_idx] =
            static_cast<std::uint8_t>(OccupancyState::kOutside);
        cache.face_neighbor_cell_indices[cell_idx] = {
            SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
    }
}

inline std::vector<OccupiedSolidLeaf> build_occupied_solid_leaves_from_cache(
    const std::vector<OctreeCell> &all_cells,
    const OccupiedSolidClassificationCache &cache,
    std::vector<std::int64_t> *out_cell_to_leaf_index = nullptr) {
    std::vector<OccupiedSolidLeaf> solid_leaves;
    solid_leaves.reserve(all_cells.size());
    std::vector<std::int64_t> cell_to_leaf_index(all_cells.size(), -1);

    for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
        const OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf) {
            continue;
        }
        solid_leaves.push_back({
            cell_idx,
            cache.center_values[cell_idx],
            cell_edge_length(cell),
            cell.depth,
            static_cast<OccupancyState>(cache.occupancy_states[cell_idx]),
            {-1, -1, -1, -1, -1, -1},
        });
        cell_to_leaf_index[cell_idx] =
            static_cast<std::int64_t>(solid_leaves.size() - 1U);
    }

    ProgressCounter neighbor_counter(
        "Regularization", "classify_occupied_solid_leaves", "leaves", 1000);

#pragma omp parallel for schedule(dynamic)
    for (std::int64_t leaf_index = 0;
         leaf_index < static_cast<std::int64_t>(solid_leaves.size());
         ++leaf_index) {
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                static_cast<std::size_t>(leaf_index))) {
            neighbor_counter.tick();
            continue;
        }
        OccupiedSolidLeaf &leaf =
            solid_leaves[static_cast<std::size_t>(leaf_index)];
        neighbor_counter.tick();
        const auto &neighbors = cache.face_neighbor_cell_indices[leaf.cell_index];
        for (std::size_t face = 0; face < neighbors.size(); ++face) {
            const std::size_t neighbor_cell_index = neighbors[face];
            if (neighbor_cell_index == SIZE_MAX ||
                neighbor_cell_index >= cell_to_leaf_index.size()) {
                continue;
            }
            leaf.face_neighbor_leaf_indices[face] =
                cell_to_leaf_index[neighbor_cell_index];
        }
    }
    neighbor_counter.finish();

    if (out_cell_to_leaf_index != nullptr) {
        *out_cell_to_leaf_index = std::move(cell_to_leaf_index);
    }

    return solid_leaves;
}

inline std::vector<OccupiedSolidLeaf> classify_occupied_solid_leaves(
    const std::vector<OctreeCell> &all_cells,
    const std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const LeafSpatialIndex &spatial_index,
    double isovalue,
    std::uint32_t max_depth,
    OccupiedSolidClassificationCache *cache = nullptr,
    const std::vector<std::uint8_t> *dirty_cells = nullptr) {
    OccupiedSolidClassificationCache local_cache;
    OccupiedSolidClassificationCache &cache_ref =
        cache == nullptr ? local_cache : *cache;
    update_occupied_solid_classification_cache(
        all_cells,
        all_contributors,
        positions,
        smoothing_lengths,
        spatial_index,
        isovalue,
        max_depth,
        cache_ref,
        dirty_cells);
    return build_occupied_solid_leaves_from_cache(all_cells, cache_ref);
}

inline void merge_occupied_solid_cache_from_closure_state(
    const std::vector<OctreeCell> &all_cells,
    const LeafSpatialIndex &spatial_index,
    std::uint32_t max_depth,
    OccupiedSolidClassificationCache &cache,
    std::vector<std::uint8_t> inside_flags,
    std::vector<double> center_values,
    std::vector<std::uint8_t> occupancy_states,
    const std::vector<std::uint8_t> &dirty_cells) {
    cache.inside_flags = std::move(inside_flags);
    cache.center_values = std::move(center_values);
    cache.occupancy_states = std::move(occupancy_states);
    if (cache.inside_flags.size() < all_cells.size()) {
        cache.inside_flags.resize(all_cells.size(), 0U);
    }
    if (cache.center_values.size() < all_cells.size()) {
        cache.center_values.resize(all_cells.size(), 0.0);
    }
    if (cache.occupancy_states.size() < all_cells.size()) {
        cache.occupancy_states.resize(
            all_cells.size(),
            static_cast<std::uint8_t>(OccupancyState::kOutside));
    }
    cache.face_neighbor_cell_indices.resize(
        all_cells.size(),
        {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX});

    std::vector<std::uint8_t> active_mask(all_cells.size(), 0U);
    std::vector<std::size_t> active_leaf_indices;
    active_leaf_indices.reserve(dirty_cells.size() * 2U);

    auto activate_leaf = [&](std::size_t cell_idx) {
        if (cell_idx >= all_cells.size() || !all_cells[cell_idx].is_leaf ||
            active_mask[cell_idx] != 0U) {
            return;
        }
        active_mask[cell_idx] = 1U;
        active_leaf_indices.push_back(cell_idx);
    };

    for (std::size_t dirty_idx = 0; dirty_idx < dirty_cells.size(); ++dirty_idx) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(dirty_idx);
        if (dirty_cells[dirty_idx] == 0U || dirty_idx >= all_cells.size()) {
            continue;
        }
        if (!all_cells[dirty_idx].is_leaf) {
            cache.face_neighbor_cell_indices[dirty_idx] = {
                SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
            cache.inside_flags[dirty_idx] = 0U;
            cache.center_values[dirty_idx] = 0.0;
            cache.occupancy_states[dirty_idx] =
                static_cast<std::uint8_t>(OccupancyState::kOutside);
            continue;
        }
        activate_leaf(dirty_idx);
        const auto neighbors = face_neighbor_cells(
            all_cells[dirty_idx], spatial_index, max_depth);
        cache.face_neighbor_cell_indices[dirty_idx] = neighbors;
        for (std::size_t neighbor_idx : neighbors) {
            if (neighbor_idx == SIZE_MAX || neighbor_idx >= all_cells.size()) {
                continue;
            }
            activate_leaf(neighbor_idx);
        }
    }

    ProgressCounter neighbor_counter(
        "Regularization",
        "merge_occupied_solid_cache_from_closure_state",
        "cells",
        1000);
#pragma omp parallel for schedule(dynamic)
    for (std::int64_t active_idx = 0;
         active_idx < static_cast<std::int64_t>(active_leaf_indices.size());
         ++active_idx) {
        const std::size_t idx =
            active_leaf_indices[static_cast<std::size_t>(active_idx)];
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(idx)) {
            neighbor_counter.tick();
            continue;
        }
        neighbor_counter.tick();
        cache.face_neighbor_cell_indices[idx] = face_neighbor_cells(
            all_cells[idx], spatial_index, max_depth);
    }
    neighbor_counter.finish();
}

inline std::vector<std::uint8_t> build_inside_mask_from_classification_cache(
    const std::vector<OctreeCell> &all_cells,
    const OccupiedSolidClassificationCache &classification_cache) {
    std::vector<std::uint8_t> inside_mask(all_cells.size(), 0U);
    ProgressCounter mask_counter(
        "Regularization",
        "build_inside_mask_from_classification_cache",
        "cells",
        1000);
    for (std::size_t cell_index = 0; cell_index < all_cells.size(); ++cell_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_index);
        mask_counter.tick();
        if (!all_cells[cell_index].is_leaf) {
            continue;
        }
        inside_mask[cell_index] = occupied_solid_cache_is_inside(
                                      classification_cache.occupancy_states[
                                          cell_index])
                                      ? 1U
                                      : 0U;
    }
    mask_counter.finish();
    return inside_mask;
}

inline std::vector<std::uint8_t> build_leaf_mask_from_cell_mask(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &cell_mask);

/**
 * @brief Build compact morphology/extraction views from the classification cache.
 *
 * This is the current queue boundary between full-tree side-cars and legacy
 * compact leaf views. Later phases should shrink the amount of work done here,
 * but for now this helper makes that boundary explicit and single-shot.
 */
inline OccupiedSolidExtractionView build_occupied_solid_extraction_view(
    const std::vector<OctreeCell> &all_cells,
    const OccupiedSolidClassificationCache &classification_cache,
    std::vector<std::uint8_t> inside_mask_by_cell = {}) {
    OccupiedSolidExtractionView view;
    view.solid_leaves =
        build_occupied_solid_leaves_from_cache(
            all_cells,
            classification_cache,
            &view.cell_to_leaf_index);
    if (inside_mask_by_cell.size() != all_cells.size()) {
        inside_mask_by_cell = build_inside_mask_from_classification_cache(
            all_cells, classification_cache);
    }
    view.inside_mask_by_cell = std::move(inside_mask_by_cell);
    view.inside_mask = build_leaf_mask_from_cell_mask(
        view.solid_leaves, view.inside_mask_by_cell);
    return view;
}

inline std::vector<std::uint8_t> build_leaf_mask_from_cell_mask(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &cell_mask) {
    std::vector<std::uint8_t> inside_mask(solid_leaves.size(), 0U);
    ProgressCounter mask_counter(
        "Regularization",
        "build_leaf_mask_from_cell_mask",
        "leaves",
        1000);
    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(leaf_index);
        mask_counter.tick();
        const std::size_t cell_index = solid_leaves[leaf_index].cell_index;
        if (cell_index < cell_mask.size()) {
            inside_mask[leaf_index] = cell_mask[cell_index];
        }
    }
    mask_counter.finish();
    return inside_mask;
}

inline std::vector<double> project_leaf_scalars_from_cell_state(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<double> &cell_values,
    double default_value) {
    std::vector<double> leaf_values(solid_leaves.size(), default_value);
    ProgressCounter value_counter(
        "Regularization",
        "project_leaf_scalars_from_cell_state",
        "leaves",
        1000);
    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(leaf_index);
        value_counter.tick();
        const std::size_t cell_index = solid_leaves[leaf_index].cell_index;
        if (cell_index < cell_values.size()) {
            leaf_values[leaf_index] = cell_values[cell_index];
        }
    }
    value_counter.finish();
    return leaf_values;
}

inline std::vector<double> compute_outside_distance_from_classification_cache(
    const std::vector<OctreeCell> &all_cells,
    const OccupiedSolidClassificationCache &classification_cache,
    double max_distance = std::numeric_limits<double>::infinity()) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> distance_from_inside(all_cells.size(), inf);
    using QueueEntry = std::pair<double, std::size_t>;
    std::priority_queue<
        QueueEntry,
        std::vector<QueueEntry>,
        std::greater<QueueEntry>> queue;

    ProgressCounter seed_counter(
        "Regularization",
        "compute_outside_distance_from_classification_cache",
        "cells",
        1000);
    for (std::size_t cell_index = 0; cell_index < all_cells.size(); ++cell_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_index);
        seed_counter.tick();
        if (!all_cells[cell_index].is_leaf ||
            !occupied_solid_cache_is_inside(
                classification_cache.occupancy_states[cell_index])) {
            continue;
        }

        bool is_boundary_seed = false;
        for (std::size_t neighbor_cell_index :
             classification_cache.face_neighbor_cell_indices[cell_index]) {
            if (neighbor_cell_index == SIZE_MAX ||
                neighbor_cell_index >= all_cells.size()) {
                is_boundary_seed = true;
                break;
            }
            if (!all_cells[neighbor_cell_index].is_leaf ||
                !occupied_solid_cache_is_inside(
                    classification_cache.occupancy_states[neighbor_cell_index])) {
                is_boundary_seed = true;
                break;
            }
        }

        if (is_boundary_seed) {
            distance_from_inside[cell_index] = 0.0;
            queue.push({0.0, cell_index});
        }
    }
    seed_counter.finish();

    ProgressCounter wavefront_counter(
        "Regularization",
        "compute_outside_distance_from_classification_cache",
        "queue pops",
        10000);
    while (!queue.empty()) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(queue.size());
        wavefront_counter.tick();
        const auto [distance, cell_index] = queue.top();
        queue.pop();
        if (distance > distance_from_inside[cell_index]) {
            continue;
        }
        if (distance > max_distance) {
            continue;
        }

        for (std::size_t neighbor_cell_index :
             classification_cache.face_neighbor_cell_indices[cell_index]) {
            if (neighbor_cell_index == SIZE_MAX ||
                neighbor_cell_index >= all_cells.size()) {
                continue;
            }
            if (!all_cells[neighbor_cell_index].is_leaf ||
                occupied_solid_cache_is_inside(
                    classification_cache.occupancy_states[neighbor_cell_index])) {
                continue;
            }

            const double source_size = std::max(
                std::min(cell_edge_length(all_cells[cell_index]), max_distance),
                0.0);
            const double neighbor_size = std::max(
                std::min(
                    cell_edge_length(all_cells[neighbor_cell_index]),
                    max_distance),
                0.0);
            const double edge_cost = 0.5 * (source_size + neighbor_size);
            const double candidate = distance + edge_cost;
            if (candidate > max_distance) {
                continue;
            }
            if (candidate < distance_from_inside[neighbor_cell_index]) {
                distance_from_inside[neighbor_cell_index] = candidate;
                queue.push({candidate, neighbor_cell_index});
            }
        }
    }
    wavefront_counter.finish();

    return distance_from_inside;
}

inline std::vector<std::uint8_t> dilate_inside_cell_mask(
    const std::vector<OctreeCell> &all_cells,
    const std::vector<std::uint8_t> &inside_mask_by_cell,
    const std::vector<double> &distance_from_inside_by_cell,
    double dilation_radius) {
    std::vector<std::uint8_t> dilated_inside(all_cells.size(), 0U);

    ProgressCounter dilate_counter(
        "Regularization",
        "dilate_inside_cell_mask",
        "cells",
        1000);
#pragma omp parallel for schedule(static)
    for (std::size_t cell_index = 0; cell_index < all_cells.size(); ++cell_index) {
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                cell_index)) {
            dilate_counter.tick();
            continue;
        }
        dilate_counter.tick();
        if (!all_cells[cell_index].is_leaf) {
            continue;
        }
        dilated_inside[cell_index] =
            (inside_mask_by_cell[cell_index] != 0U ||
             distance_from_inside_by_cell[cell_index] <= dilation_radius)
                ? 1U
                : 0U;
    }
    dilate_counter.finish();

    return dilated_inside;
}

inline bool refine_thickening_band_cells(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue,
    std::uint32_t max_depth,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    std::uint32_t worker_count,
    double target_leaf_size,
    const OccupiedSolidClassificationCache &classification_cache,
    const std::vector<double> &distance_from_inside_by_cell,
    double thickening_radius,
    double table_cadence_seconds,
    std::uint32_t minimum_usable_hermite_samples,
    double max_qef_rms_residual_ratio,
    double min_normal_alignment_threshold,
    std::vector<std::uint8_t> *dirty_cells = nullptr,
    std::vector<std::uint8_t> *classified_inside_flags = nullptr,
    std::vector<double> *classified_center_values = nullptr,
    std::vector<std::uint8_t> *classified_occupancy_states = nullptr) {
    if (target_leaf_size <= 0.0 || thickening_radius <= 0.0) {
        return false;
    }

    const double refinement_halo_radius =
        thickening_radius + 2.0 * target_leaf_size;

    const auto target_depth_for_cell =
        [&](const OctreeCell &cell) -> std::uint32_t {
            std::uint32_t target_depth = cell.depth;
            double size = cell_edge_length(cell);
            while (target_depth < max_depth && size > target_leaf_size) {
                size *= 0.5;
                ++target_depth;
            }
            return target_depth;
        };

    std::queue<std::size_t> cells_to_visit;
    std::vector<std::size_t> queued_cell_indices;
    const std::uint32_t kNoTargetDepth =
        std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> queued_target_depths(
        all_cells.size(), kNoTargetDepth);
    std::vector<std::uint8_t> enqueued_cells(all_cells.size(), 0U);

    auto enqueue_cell = [&](std::size_t cell_index, std::uint32_t target_depth) {
        if (cell_index >= all_cells.size()) {
            return;
        }
        if (cell_index >= queued_target_depths.size()) {
            queued_target_depths.resize(all_cells.size(), kNoTargetDepth);
        }
        if (cell_index >= enqueued_cells.size()) {
            enqueued_cells.resize(all_cells.size(), 0U);
        }
        if (enqueued_cells[cell_index] == 0U) {
            cells_to_visit.push(cell_index);
            queued_cell_indices.push_back(cell_index);
            enqueued_cells[cell_index] = 1U;
            queued_target_depths[cell_index] = target_depth;
            return;
        }
        if (queued_target_depths[cell_index] == kNoTargetDepth) {
            queued_target_depths[cell_index] = target_depth;
            return;
        }
        queued_target_depths[cell_index] = std::max(
            queued_target_depths[cell_index], target_depth);
    };

    for (std::size_t cell_index = 0; cell_index < all_cells.size(); ++cell_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_index);
        const OctreeCell &cell = all_cells[cell_index];
        if (!cell.is_leaf) {
            continue;
        }
        if (!std::isfinite(distance_from_inside_by_cell[cell_index]) ||
            distance_from_inside_by_cell[cell_index] > refinement_halo_radius) {
            continue;
        }
        const OccupancyState occupancy = static_cast<OccupancyState>(
            classification_cache.occupancy_states[cell_index]);
        if (occupancy != OccupancyState::kBoundaryInside) {
            continue;
        }
        if (cell.depth >= max_depth || cell_edge_length(cell) <= target_leaf_size) {
            continue;
        }
        enqueue_cell(cell_index, target_depth_for_cell(cell));
    }

    // Build seed vectors with correct ordinal mapping for the closure.
    std::vector<std::size_t> closure_seed_indices;
    std::vector<std::uint32_t> closure_seed_depths;
    closure_seed_indices.reserve(queued_cell_indices.size());
    closure_seed_depths.reserve(queued_cell_indices.size());
    for (std::size_t idx : queued_cell_indices) {
        if (idx >= queued_target_depths.size()) {
            continue;
        }
        const std::uint32_t td = queued_target_depths[idx];
        if (td == kNoTargetDepth) {
            continue;
        }
        closure_seed_indices.push_back(idx);
        closure_seed_depths.push_back(td);
    }

    if (cells_to_visit.empty()) {
        meshmerizer_log_detail::print_debug_status(
            "Regularization",
            "refine_thickening_band_cells",
            "no growth-band splits needed (leaf_size_target=%.6g, radius=%.6g, halo_radius=%.6g, total_cells=%zu)\n",
            target_leaf_size,
            thickening_radius,
            refinement_halo_radius,
            all_cells.size());
        return false;
    }

    meshmerizer_log_detail::print_debug_status(
        "Regularization",
        "refine_thickening_band_cells",
        "starting from %zu growth-band cells to reach leaf_size<=%.6g (radius=%.6g, halo_radius=%.6g, total_cells_before=%zu)\n",
        cells_to_visit.size(),
        target_leaf_size,
        thickening_radius,
        refinement_halo_radius,
        all_cells.size());

    const RefinementClosureConfig closure_config = {
        isovalue,
        max_depth,
        domain,
        base_resolution,
        worker_count,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
        table_cadence_seconds,
        "Regularization",
        "refine_thickening_band_cells",
        "thickening_band",
        /*surface_band_target_leaf_size=*/0.0,
        /*thickening_band_target_leaf_size=*/target_leaf_size,
        /*thickening_radius=*/thickening_radius,
    };

    return refine_thickening_band_with_closure(
        all_cells,
        all_contributors,
        positions,
        smoothing_lengths,
        closure_config,
        target_leaf_size,
        closure_seed_indices,
        closure_seed_depths,
        &classification_cache.inside_flags,
        &classification_cache.center_values,
        &classification_cache.occupancy_states,
        dirty_cells,
        classified_inside_flags,
        classified_center_values,
        classified_occupancy_states);
}

inline bool thickening_band_is_fully_refined(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &inside_mask,
    const std::vector<double> &distance_from_inside,
    double target_leaf_size,
    double thickening_radius) {
    if (target_leaf_size <= 0.0 || thickening_radius <= 0.0) {
        return true;
    }

    ProgressCounter check_counter(
        "Regularization", "thickening_band_is_fully_refined", "leaves", 1000);
    std::size_t unresolved_count = 0U;
    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(leaf_index);
        check_counter.tick();
        if (inside_mask[leaf_index] != 0U) {
            continue;
        }
        if (distance_from_inside[leaf_index] > thickening_radius) {
            continue;
        }
        if (solid_leaves[leaf_index].cell_size > target_leaf_size) {
            ++unresolved_count;
            break;
        }
    }
    check_counter.finish();

    meshmerizer_log_detail::print_debug_status(
        "Regularization",
        "thickening_band_is_fully_refined",
        "unresolved strict-band leaves=%zu (target_leaf_size=%.6g, radius=%.6g)\n",
        unresolved_count,
        target_leaf_size,
        thickening_radius);

    return unresolved_count == 0U;
}

/**
 * @brief B2 incremental single-pass thickening-band refinement.
 *
 * Replaces the outer classify→distance→refine→reclassify loop with a single
 * closure run seeded by kClassify tasks for all current leaf cells.  The
 * incremental chain
 *   kClassify → kDistanceUpdate → kRefine → kClassify(children) → ...
 * converges without any outer iteration.
 *
 * @param all_cells    Flat cell array updated in place.
 * @param all_contributors Flat contributor array updated in place.
 * @param positions    Particle positions.
 * @param smoothing_lengths Per-particle support radii.
 * @param isovalue     Field isovalue for classification.
 * @param max_depth    Maximum octree depth.
 * @param domain       Domain bounding box.
 * @param base_resolution Top-level cell count per axis.
 * @param worker_count Number of worker threads.
 * @param target_leaf_size Target leaf size for the thickening band.
 * @param thickening_radius Outward thickening radius.
 * @param table_cadence_seconds Status table cadence.
 * @param minimum_usable_hermite_samples Min Hermite samples for QEF.
 * @param max_qef_rms_residual_ratio Max QEF residual ratio.
 * @param min_normal_alignment_threshold Min Hermite normal alignment.
 * @param dirty_cells Optional output dirty-cell mask.
 * @return ``true`` when at least one new cell was created.
 */
inline bool refine_thickening_band_incremental(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue,
    std::uint32_t max_depth,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    std::uint32_t worker_count,
    double target_leaf_size,
    double thickening_radius,
    double table_cadence_seconds,
    std::uint32_t minimum_usable_hermite_samples,
    double max_qef_rms_residual_ratio,
    double min_normal_alignment_threshold,
    std::vector<std::uint8_t> *dirty_cells = nullptr) {
    if (target_leaf_size <= 0.0 || thickening_radius <= 0.0) {
        return false;
    }

    // Thread both thickening parameters into the config so the closure
    // workers can drive the kClassify → kDistanceUpdate → kRefine chain.
    const RefinementClosureConfig closure_config = {
        isovalue,
        max_depth,
        domain,
        base_resolution,
        worker_count,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
        table_cadence_seconds,
        "Regularization",
        "refine_thickening_band_incremental",
        "thickening_incremental",
        /*surface_band_target_leaf_size=*/0.0,
        /*thickening_band_target_leaf_size=*/target_leaf_size,
        /*thickening_radius=*/thickening_radius,
    };

    return refine_thickening_band_with_closure(
        all_cells,
        all_contributors,
        positions,
        smoothing_lengths,
        closure_config,
        target_leaf_size,
        /*seed_cell_indices=*/{},
        /*seed_target_depths=*/{},
        /*initial_inside_flags=*/nullptr,
        /*initial_center_values=*/nullptr,
        /*initial_occupancy_states=*/nullptr,
        dirty_cells);
}

inline std::vector<double> compute_inside_clearance_from_cell_mask(
    const std::vector<OctreeCell> &all_cells,
    const OccupiedSolidClassificationCache &classification_cache,
    const std::vector<std::uint8_t> &inside_mask_by_cell,
    double max_distance = std::numeric_limits<double>::infinity()) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> clearance(all_cells.size(), inf);
    using QueueEntry = std::pair<double, std::size_t>;
    std::priority_queue<
        QueueEntry,
        std::vector<QueueEntry>,
        std::greater<QueueEntry>> queue;

    ProgressCounter seed_counter(
        "Regularization",
        "compute_inside_clearance_from_cell_mask",
        "cells",
        1000);
    for (std::size_t cell_index = 0; cell_index < all_cells.size(); ++cell_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_index);
        seed_counter.tick();
        if (!all_cells[cell_index].is_leaf || inside_mask_by_cell[cell_index] == 0U) {
            continue;
        }

        bool is_boundary_seed = false;
        for (std::size_t neighbor_cell_index :
             classification_cache.face_neighbor_cell_indices[cell_index]) {
            if (neighbor_cell_index == SIZE_MAX ||
                neighbor_cell_index >= all_cells.size()) {
                is_boundary_seed = true;
                break;
            }
            if (!all_cells[neighbor_cell_index].is_leaf ||
                inside_mask_by_cell[neighbor_cell_index] == 0U) {
                is_boundary_seed = true;
                break;
            }
        }

        if (is_boundary_seed) {
            clearance[cell_index] = 0.0;
            queue.push({0.0, cell_index});
        }
    }
    seed_counter.finish();

    ProgressCounter wavefront_counter(
        "Regularization",
        "compute_inside_clearance_from_cell_mask",
        "queue pops",
        10000);
    while (!queue.empty()) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(queue.size());
        wavefront_counter.tick();
        const auto [distance, cell_index] = queue.top();
        queue.pop();
        if (distance > clearance[cell_index]) {
            continue;
        }
        if (distance > max_distance) {
            continue;
        }

        for (std::size_t neighbor_cell_index :
             classification_cache.face_neighbor_cell_indices[cell_index]) {
            if (neighbor_cell_index == SIZE_MAX ||
                neighbor_cell_index >= all_cells.size()) {
                continue;
            }
            if (!all_cells[neighbor_cell_index].is_leaf ||
                inside_mask_by_cell[neighbor_cell_index] == 0U) {
                continue;
            }

            const double edge_cost = 0.5 * (
                cell_edge_length(all_cells[cell_index]) +
                cell_edge_length(all_cells[neighbor_cell_index]));
            const double candidate = distance + edge_cost;
            if (candidate > max_distance) {
                continue;
            }
            if (candidate < clearance[neighbor_cell_index]) {
                clearance[neighbor_cell_index] = candidate;
                queue.push({candidate, neighbor_cell_index});
            }
        }
    }
    wavefront_counter.finish();

    return clearance;
}

inline std::vector<std::uint8_t> erode_occupied_solid_cells(
    const std::vector<OctreeCell> &all_cells,
    const std::vector<std::uint8_t> &inside_mask_by_cell,
    const std::vector<double> &clearance_by_cell,
    double erosion_radius) {
    std::vector<std::uint8_t> kept_inside(all_cells.size(), 0U);

    ProgressCounter erode_counter(
        "Regularization", "erode_occupied_solid_cells", "cells", 1000);
#pragma omp parallel for schedule(static)
    for (std::size_t cell_index = 0; cell_index < all_cells.size(); ++cell_index) {
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                cell_index)) {
            erode_counter.tick();
            continue;
        }
        erode_counter.tick();
        if (!all_cells[cell_index].is_leaf || inside_mask_by_cell[cell_index] == 0U) {
            continue;
        }
        kept_inside[cell_index] =
            clearance_by_cell[cell_index] >= erosion_radius ? 1U : 0U;
    }
    erode_counter.finish();
    return kept_inside;
}

inline std::vector<double> compute_distance_to_eroded_solid_from_cell_mask(
    const std::vector<OctreeCell> &all_cells,
    const OccupiedSolidClassificationCache &classification_cache,
    const std::vector<std::uint8_t> &eroded_inside_by_cell,
    double max_distance = std::numeric_limits<double>::infinity()) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> distance_to_eroded(all_cells.size(), inf);
    using QueueEntry = std::pair<double, std::size_t>;
    std::priority_queue<
        QueueEntry,
        std::vector<QueueEntry>,
        std::greater<QueueEntry>> queue;

    ProgressCounter seed_counter(
        "Regularization",
        "compute_distance_to_eroded_solid_from_cell_mask",
        "cells",
        1000);
    for (std::size_t cell_index = 0; cell_index < all_cells.size(); ++cell_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_index);
        seed_counter.tick();
        if (!all_cells[cell_index].is_leaf || eroded_inside_by_cell[cell_index] == 0U) {
            continue;
        }

        bool is_boundary_seed = false;
        for (std::size_t neighbor_cell_index :
             classification_cache.face_neighbor_cell_indices[cell_index]) {
            if (neighbor_cell_index == SIZE_MAX ||
                neighbor_cell_index >= all_cells.size()) {
                is_boundary_seed = true;
                break;
            }
            if (!all_cells[neighbor_cell_index].is_leaf ||
                eroded_inside_by_cell[neighbor_cell_index] == 0U) {
                is_boundary_seed = true;
                break;
            }
        }

        if (is_boundary_seed) {
            distance_to_eroded[cell_index] = 0.0;
            queue.push({0.0, cell_index});
        }
    }
    seed_counter.finish();

    ProgressCounter wavefront_counter(
        "Regularization",
        "compute_distance_to_eroded_solid_from_cell_mask",
        "queue pops",
        10000);
    while (!queue.empty()) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(queue.size());
        wavefront_counter.tick();
        const auto [distance, cell_index] = queue.top();
        queue.pop();
        if (distance > distance_to_eroded[cell_index]) {
            continue;
        }
        if (distance > max_distance) {
            continue;
        }

        for (std::size_t neighbor_cell_index :
             classification_cache.face_neighbor_cell_indices[cell_index]) {
            if (neighbor_cell_index == SIZE_MAX ||
                neighbor_cell_index >= all_cells.size()) {
                continue;
            }
            if (!all_cells[neighbor_cell_index].is_leaf ||
                eroded_inside_by_cell[neighbor_cell_index] != 0U) {
                continue;
            }

            const double edge_cost = 0.5 * (
                cell_edge_length(all_cells[cell_index]) +
                cell_edge_length(all_cells[neighbor_cell_index]));
            const double candidate = distance + edge_cost;
            if (candidate > max_distance) {
                continue;
            }
            if (candidate < distance_to_eroded[neighbor_cell_index]) {
                distance_to_eroded[neighbor_cell_index] = candidate;
                queue.push({candidate, neighbor_cell_index});
            }
        }
    }
    wavefront_counter.finish();

    return distance_to_eroded;
}

inline std::vector<std::uint8_t> dilate_eroded_solid_cells(
    const std::vector<OctreeCell> &all_cells,
    const std::vector<std::uint8_t> &eroded_inside_by_cell,
    const std::vector<double> &distance_to_eroded_by_cell,
    double dilation_radius) {
    std::vector<std::uint8_t> opened_inside(all_cells.size(), 0U);

    ProgressCounter dilate_counter(
        "Regularization", "dilate_eroded_solid_cells", "cells", 1000);
#pragma omp parallel for schedule(static)
    for (std::size_t cell_index = 0; cell_index < all_cells.size(); ++cell_index) {
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                cell_index)) {
            dilate_counter.tick();
            continue;
        }
        dilate_counter.tick();
        if (!all_cells[cell_index].is_leaf) {
            continue;
        }
        opened_inside[cell_index] =
            (eroded_inside_by_cell[cell_index] != 0U ||
             distance_to_eroded_by_cell[cell_index] <= dilation_radius)
                ? 1U
                : 0U;
    }
    dilate_counter.finish();
    return opened_inside;
}

inline void prune_small_opened_components(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    std::vector<std::uint8_t> &opened_inside,
    double min_component_volume_ratio = 0.05) {
    if (opened_inside.empty() || min_component_volume_ratio <= 0.0) {
        return;
    }

    std::vector<std::uint8_t> visited(opened_inside.size(), 0U);
    std::vector<std::vector<std::size_t>> components;
    std::vector<double> component_volumes;

    ProgressCounter component_counter(
        "Regularization", "prune_small_opened_components", "leaves", 1000);
    for (std::size_t leaf_index = 0; leaf_index < opened_inside.size(); ++leaf_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(leaf_index);
        component_counter.tick();
        if (opened_inside[leaf_index] == 0U || visited[leaf_index] != 0U) {
            continue;
        }

        std::vector<std::size_t> component;
        std::queue<std::size_t> queue;
        queue.push(leaf_index);
        visited[leaf_index] = 1U;
        double component_volume = 0.0;

        while (!queue.empty()) {
            meshmerizer_cancel_detail::poll_for_cancellation_serial(
                component.size() + queue.size());
            const std::size_t current = queue.front();
            queue.pop();
            component.push_back(current);
            const double cell_size = solid_leaves[current].cell_size;
            component_volume += cell_size * cell_size * cell_size;

            for (std::int64_t neighbor_leaf :
                 solid_leaves[current].face_neighbor_leaf_indices) {
                if (neighbor_leaf < 0) {
                    continue;
                }
                const std::size_t neighbor =
                    static_cast<std::size_t>(neighbor_leaf);
                if (neighbor >= opened_inside.size() ||
                    opened_inside[neighbor] == 0U || visited[neighbor] != 0U) {
                    continue;
                }
                visited[neighbor] = 1U;
                queue.push(neighbor);
            }
        }

        components.push_back(std::move(component));
        component_volumes.push_back(component_volume);
    }
    component_counter.finish();

    if (components.size() <= 1U) {
        return;
    }

    const double largest_volume = *std::max_element(
        component_volumes.begin(), component_volumes.end());
    if (largest_volume <= 0.0) {
        return;
    }

    const double volume_threshold = min_component_volume_ratio * largest_volume;
    std::size_t removed_components = 0U;
    std::size_t removed_leaves = 0U;
    for (std::size_t component_index = 0; component_index < components.size();
         ++component_index) {
        if (component_volumes[component_index] >= volume_threshold) {
            continue;
        }
        ++removed_components;
        removed_leaves += components[component_index].size();
        for (std::size_t leaf_index : components[component_index]) {
            opened_inside[leaf_index] = 0U;
        }
    }

    if (removed_components > 0U) {
        meshmerizer_log_detail::print_debug_status(
            "Regularization",
            "prune_small_opened_components",
            "removed %zu small components (%zu leaves, threshold=%.6g of largest volume)\n",
            removed_components,
            removed_leaves,
            min_component_volume_ratio);
    }
}

inline void fill_small_opened_cavities(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    std::vector<std::uint8_t> &opened_inside,
    double max_cavity_volume_ratio = 0.05) {
    if (opened_inside.empty() || max_cavity_volume_ratio <= 0.0) {
        return;
    }

    double opened_volume = 0.0;
    for (std::size_t leaf_index = 0; leaf_index < opened_inside.size(); ++leaf_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(leaf_index);
        if (opened_inside[leaf_index] == 0U) {
            continue;
        }
        const double cell_size = solid_leaves[leaf_index].cell_size;
        opened_volume += cell_size * cell_size * cell_size;
    }
    if (opened_volume <= 0.0) {
        return;
    }

    struct OutsideComponent {
        std::vector<std::size_t> leaves;
        double volume = 0.0;
        bool touches_domain_boundary = false;
    };

    std::vector<std::uint8_t> visited(opened_inside.size(), 0U);
    std::vector<OutsideComponent> outside_components;
    ProgressCounter cavity_counter(
        "Regularization", "fill_small_opened_cavities", "leaves", 1000);

    for (std::size_t leaf_index = 0; leaf_index < opened_inside.size(); ++leaf_index) {
        cavity_counter.tick();
        if (opened_inside[leaf_index] != 0U || visited[leaf_index] != 0U) {
            continue;
        }

        OutsideComponent component;
        std::queue<std::size_t> queue;
        queue.push(leaf_index);
        visited[leaf_index] = 1U;

        while (!queue.empty()) {
            const std::size_t current = queue.front();
            queue.pop();
            component.leaves.push_back(current);
            const double cell_size = solid_leaves[current].cell_size;
            component.volume += cell_size * cell_size * cell_size;

            for (std::int64_t neighbor_leaf :
                 solid_leaves[current].face_neighbor_leaf_indices) {
                if (neighbor_leaf < 0) {
                    component.touches_domain_boundary = true;
                    continue;
                }

                const std::size_t neighbor =
                    static_cast<std::size_t>(neighbor_leaf);
                if (neighbor >= opened_inside.size() ||
                    opened_inside[neighbor] != 0U || visited[neighbor] != 0U) {
                    continue;
                }
                visited[neighbor] = 1U;
                queue.push(neighbor);
            }
        }

        outside_components.push_back(std::move(component));
    }
    cavity_counter.finish();

    const double max_cavity_volume = max_cavity_volume_ratio * opened_volume;
    std::size_t filled_components = 0U;
    std::size_t filled_leaves = 0U;
    for (const OutsideComponent &component : outside_components) {
        if (component.touches_domain_boundary ||
            component.volume > max_cavity_volume) {
            continue;
        }
        ++filled_components;
        filled_leaves += component.leaves.size();
        for (std::size_t leaf_index : component.leaves) {
            opened_inside[leaf_index] = 1U;
        }
    }

    if (filled_components > 0U) {
        meshmerizer_log_detail::print_debug_status(
            "Regularization",
            "fill_small_opened_cavities",
            "filled %zu enclosed cavities (%zu leaves, threshold=%.6g of opened volume)\n",
            filled_components,
            filled_leaves,
            max_cavity_volume_ratio);
    }
}

inline std::uint32_t leaf_min_fine_x(
    const OctreeCell &cell,
    const LeafSpatialIndex &spatial_index) {
    return static_cast<std::uint32_t>(std::llround(
        (cell.bounds.min.x - spatial_index.domain_min.x) *
        spatial_index.inv_cell_size_x));
}

inline std::uint32_t leaf_min_fine_y(
    const OctreeCell &cell,
    const LeafSpatialIndex &spatial_index) {
    return static_cast<std::uint32_t>(std::llround(
        (cell.bounds.min.y - spatial_index.domain_min.y) *
        spatial_index.inv_cell_size_y));
}

inline std::uint32_t leaf_min_fine_z(
    const OctreeCell &cell,
    const LeafSpatialIndex &spatial_index) {
    return static_cast<std::uint32_t>(std::llround(
        (cell.bounds.min.z - spatial_index.domain_min.z) *
        spatial_index.inv_cell_size_z));
}

inline std::uint32_t leaf_span_fine(
    const OctreeCell &cell,
    const LeafSpatialIndex &spatial_index) {
    return static_cast<std::uint32_t>(std::llround(
        (cell.bounds.max.x - cell.bounds.min.x) *
        spatial_index.inv_cell_size_x));
}

inline void suppress_opened_edge_contacts(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<OctreeCell> &all_cells,
    const LeafSpatialIndex &spatial_index,
    std::vector<std::uint8_t> &opened_inside) {
    if (opened_inside.empty()) {
        return;
    }

    struct RemovalCandidate {
        std::size_t leaf_index;
        std::uint32_t exposed_faces;
        std::uint32_t cell_volume_units;
    };

    std::vector<RemovalCandidate> candidates;
    candidates.reserve(opened_inside.size() / 16U + 1U);
    const std::vector<std::int64_t> cell_to_leaf_index =
        build_opened_cell_to_leaf_index(all_cells, solid_leaves);
    ProgressCounter contact_counter(
        "Regularization", "suppress_opened_edge_contacts", "leaves", 1000);

    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        contact_counter.tick();
        if (opened_inside[leaf_index] == 0U) {
            continue;
        }

        const OctreeCell &cell = all_cells[solid_leaves[leaf_index].cell_index];
        const std::uint32_t ix0 = leaf_min_fine_x(cell, spatial_index);
        const std::uint32_t iy0 = leaf_min_fine_y(cell, spatial_index);
        const std::uint32_t iz0 = leaf_min_fine_z(cell, spatial_index);
        const std::uint32_t span = leaf_span_fine(cell, spatial_index);
        if (span == 0U) {
            continue;
        }

        std::uint32_t exposed_faces = 0U;
        auto face_open = [&](std::int32_t dx, std::int32_t dy, std::int32_t dz) {
            const std::int64_t qx = static_cast<std::int64_t>(ix0) + dx;
            const std::int64_t qy = static_cast<std::int64_t>(iy0) + dy;
            const std::int64_t qz = static_cast<std::int64_t>(iz0) + dz;
            if (qx < 0 || qy < 0 || qz < 0 ||
                qx >= static_cast<std::int64_t>(spatial_index.fine_resolution) ||
                qy >= static_cast<std::int64_t>(spatial_index.fine_resolution) ||
                qz >= static_cast<std::int64_t>(spatial_index.fine_resolution)) {
                return true;
            }
            const std::size_t neighbor_cell = spatial_index.find_leaf_at(
                static_cast<std::uint32_t>(qx),
                static_cast<std::uint32_t>(qy),
                static_cast<std::uint32_t>(qz));
            if (neighbor_cell == SIZE_MAX ||
                neighbor_cell >= cell_to_leaf_index.size()) {
                return true;
            }
            const std::int64_t neighbor_leaf = cell_to_leaf_index[neighbor_cell];
            if (neighbor_leaf < 0) {
                return true;
            }
            return opened_inside[static_cast<std::size_t>(neighbor_leaf)] == 0U;
        };

        if (face_open(-1, 0, 0)) {
            ++exposed_faces;
        }
        if (face_open(static_cast<std::int32_t>(span), 0, 0)) {
            ++exposed_faces;
        }
        if (face_open(0, -1, 0)) {
            ++exposed_faces;
        }
        if (face_open(0, static_cast<std::int32_t>(span), 0)) {
            ++exposed_faces;
        }
        if (face_open(0, 0, -1)) {
            ++exposed_faces;
        }
        if (face_open(0, 0, static_cast<std::int32_t>(span))) {
            ++exposed_faces;
        }

        if (exposed_faces >= 5U) {
            candidates.push_back({leaf_index, exposed_faces, span * span * span});
        }
    }
    contact_counter.finish();

    if (candidates.empty()) {
        return;
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const RemovalCandidate &a, const RemovalCandidate &b) {
            if (a.exposed_faces != b.exposed_faces) {
                return a.exposed_faces > b.exposed_faces;
            }
            return a.cell_volume_units < b.cell_volume_units;
        });

    std::size_t removed = 0U;
    for (const RemovalCandidate &candidate : candidates) {
        opened_inside[candidate.leaf_index] = 0U;
        ++removed;
    }

    meshmerizer_log_detail::print_debug_status(
        "Regularization",
        "suppress_opened_edge_contacts",
        "removed %zu highly exposed leaves\n",
        removed);
}

inline std::vector<OpenedBoundarySample> generate_opened_boundary_samples(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &opened_inside,
    const std::vector<OctreeCell> &all_cells) {
    std::vector<OpenedBoundarySample> samples;
    samples.reserve(solid_leaves.size());

    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<OpenedBoundarySample>> thread_samples(
        static_cast<std::size_t>(n_threads));
    ProgressCounter sample_counter(
        "Meshing", "generate_opened_boundary_samples", "leaves", 1000);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        sample_counter.tick();
        if (opened_inside[leaf_index] == 0U) {
            continue;
        }

        std::vector<OpenedBoundarySample> &local_samples =
            thread_samples[static_cast<std::size_t>(omp_get_thread_num())];

        const BoundingBox &bounds = all_cells[solid_leaves[leaf_index].cell_index].bounds;
        const Vector3d center = bounds.center();

        for (std::size_t face = 0; face < 6U; ++face) {
            const std::int64_t neighbor_leaf_index =
                solid_leaves[leaf_index].face_neighbor_leaf_indices[face];
            const bool touches_outside =
                neighbor_leaf_index < 0 ||
                opened_inside[static_cast<std::size_t>(neighbor_leaf_index)] == 0U;
            if (!touches_outside) {
                continue;
            }

            Vector3d position = center;
            Vector3d outward_normal = {0.0, 0.0, 0.0};
            switch (face) {
                case 0:
                    position.x = bounds.min.x;
                    outward_normal.x = -1.0;
                    break;
                case 1:
                    position.x = bounds.max.x;
                    outward_normal.x = 1.0;
                    break;
                case 2:
                    position.y = bounds.min.y;
                    outward_normal.y = -1.0;
                    break;
                case 3:
                    position.y = bounds.max.y;
                    outward_normal.y = 1.0;
                    break;
                case 4:
                    position.z = bounds.min.z;
                    outward_normal.z = -1.0;
                    break;
                case 5:
                    position.z = bounds.max.z;
                    outward_normal.z = 1.0;
                    break;
                default:
                    break;
            }
            local_samples.push_back({position, outward_normal, leaf_index});
        }
    }

    sample_counter.finish();

    for (const auto &local_samples : thread_samples) {
        samples.insert(samples.end(), local_samples.begin(), local_samples.end());
    }

    return samples;
}

struct OpenedSurfaceMesh {
    std::vector<MeshVertex> vertices;
    std::vector<MeshTriangle> triangles;
    std::vector<std::uint64_t> vertex_keys;
};

struct SurfaceRegion;
struct SurfaceExtractionRuntime;
struct RegionSurfaceBuffers;

inline void unpack_surface_corner_coords(
    std::uint64_t key,
    std::uint32_t &ix,
    std::uint32_t &iy,
    std::uint32_t &iz);

inline std::uint64_t pack_surface_corner_coords(
    std::uint32_t ix,
    std::uint32_t iy,
    std::uint32_t iz);

inline std::vector<SurfaceRegion> build_surface_regions(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<OctreeCell> &all_cells);

inline void run_region_extract_surface_task(
    std::size_t region_id,
    const SurfaceExtractionRuntime &runtime,
    RegionSurfaceBuffers &out_buffers);

inline OpenedSurfaceMesh merge_region_surface_buffers(
    const std::vector<RegionSurfaceBuffers> &region_buffers);

inline bool resolve_opened_edge_ambiguities(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<OctreeCell> &all_cells,
    const LeafSpatialIndex &spatial_index,
    std::vector<std::uint8_t> &opened_inside,
    const OpenedSurfaceMesh &mesh);

struct SurfaceRegion {
    std::size_t region_id = 0U;
    std::size_t root_cell_index = 0U;
    std::uint32_t root_depth = 0U;
    std::vector<std::size_t> leaf_indices;
};

struct RegionSurfaceVertex {
    std::uint64_t key = 0U;
    Vector3d position = {0.0, 0.0, 0.0};
    Vector3d normal_sum = {0.0, 0.0, 0.0};
};

struct RegionSurfaceTriangle {
    std::size_t local_vertex_index[3] = {0U, 0U, 0U};
};

struct RegionSurfaceBuffers {
    std::size_t region_id = 0U;
    std::vector<RegionSurfaceVertex> local_vertices;
    std::vector<RegionSurfaceTriangle> local_triangles;
    std::unordered_map<std::uint64_t, std::size_t> local_vertex_lookup;
};

struct SurfaceExtractionRuntime {
    const std::vector<OccupiedSolidLeaf> *solid_leaves = nullptr;
    const std::vector<std::uint8_t> *opened_inside = nullptr;
    const std::vector<OctreeCell> *all_cells = nullptr;
    const LeafSpatialIndex *spatial_index = nullptr;
    const BoundingBox *domain = nullptr;
    const std::vector<std::int64_t> *cell_to_leaf_index = nullptr;
    const std::vector<SurfaceRegion> *regions = nullptr;
    std::uint32_t base_resolution = 0U;
    std::uint32_t max_depth = 0U;
    std::uint32_t fine_resolution = 0U;
    double fine_dx = 0.0;
    double fine_dy = 0.0;
    double fine_dz = 0.0;
};

struct SurfaceTaskGraphState {
    std::atomic<std::size_t> pending_region_tasks{0U};
    std::atomic<bool> merge_enqueued{false};
    std::atomic<bool> merge_completed{false};
};

struct SurfaceTaskGraphRuntime {
    SurfaceExtractionRuntime extraction;
    std::vector<RegionSurfaceBuffers> region_buffers;
    SurfaceTaskGraphState graph;
    OpenedSurfaceMesh merged_mesh;
    std::mutex merged_mesh_mutex;
};

struct PostRefineRegularizationResult {
    OccupiedSolidExtractionView extraction_view;
    std::vector<double> clearance_by_cell;
    std::vector<std::uint8_t> eroded_inside_by_cell;
    std::vector<double> dilation_distance_by_cell;
    std::vector<std::uint8_t> opened_inside_by_cell;
    std::vector<std::uint8_t> opened_inside;
    OpenedSurfaceMesh opened_surface;
};

struct PostRefineTaskGraphRuntime {
    const std::vector<OctreeCell> *all_cells = nullptr;
    const OccupiedSolidClassificationCache *classification_cache = nullptr;
    const LeafSpatialIndex *solid_spatial_index = nullptr;
    const BoundingBox *domain = nullptr;
    std::uint32_t base_resolution = 0U;
    std::uint32_t max_depth = 0U;
    std::uint32_t worker_count = 1U;
    double opening_radius = 0.0;
    double table_cadence_seconds = 10.0;
    std::vector<std::uint8_t> initial_inside_mask_by_cell;
    PostRefineRegularizationResult result;
    std::array<std::atomic<std::uint32_t>, 13U> dependency_counts;
    std::vector<std::uint8_t> enqueued;
    SurfaceExtractionRuntime surface_extraction;
    std::vector<std::int64_t> surface_cell_to_leaf_index;
    std::vector<SurfaceRegion> surface_regions;
    std::vector<RegionSurfaceBuffers> region_buffers;
    std::atomic<std::size_t> pending_region_tasks{0U};
    std::atomic<bool> reextract_pass{false};
    std::mutex result_mutex;
};

enum class PostRefineTaskNode : std::uint8_t {
    kBuildExtractionView = 0U,
    kComputeInsideClearance = 1U,
    kErodeOccupiedSolid = 2U,
    kComputeDistanceToEroded = 3U,
    kDilateErodedSolid = 4U,
    kProjectOpenedMaskToLeaves = 5U,
    kFillOpenedCavities = 6U,
    kPruneOpenedComponents = 7U,
    kSuppressOpenedEdgeContacts = 8U,
    kRegionExtractSurface = 9U,
    kResolveOpenedSurfaceAmbiguities = 10U,
    kReextractOpenedSurface = 11U,
    kMergeSurfaceBuffers = 12U,
    kCount = 13U,
};

inline constexpr std::size_t post_refine_task_node_count() {
    return static_cast<std::size_t>(PostRefineTaskNode::kCount);
}

inline std::size_t post_refine_task_index(PostRefineTaskNode node) {
    return static_cast<std::size_t>(node);
}

inline RefinementTaskKind post_refine_task_kind(PostRefineTaskNode node) {
    switch (node) {
        case PostRefineTaskNode::kBuildExtractionView:
            return RefinementTaskKind::kBuildExtractionView;
        case PostRefineTaskNode::kComputeInsideClearance:
            return RefinementTaskKind::kComputeInsideClearance;
        case PostRefineTaskNode::kErodeOccupiedSolid:
            return RefinementTaskKind::kErodeOccupiedSolid;
        case PostRefineTaskNode::kComputeDistanceToEroded:
            return RefinementTaskKind::kComputeDistanceToEroded;
        case PostRefineTaskNode::kDilateErodedSolid:
            return RefinementTaskKind::kDilateErodedSolid;
        case PostRefineTaskNode::kProjectOpenedMaskToLeaves:
            return RefinementTaskKind::kProjectOpenedMaskToLeaves;
        case PostRefineTaskNode::kFillOpenedCavities:
            return RefinementTaskKind::kFillOpenedCavities;
        case PostRefineTaskNode::kPruneOpenedComponents:
            return RefinementTaskKind::kPruneOpenedComponents;
        case PostRefineTaskNode::kSuppressOpenedEdgeContacts:
            return RefinementTaskKind::kSuppressOpenedEdgeContacts;
        case PostRefineTaskNode::kResolveOpenedSurfaceAmbiguities:
            return RefinementTaskKind::kResolveOpenedSurfaceAmbiguities;
        case PostRefineTaskNode::kReextractOpenedSurface:
            return RefinementTaskKind::kReextractOpenedSurface;
        case PostRefineTaskNode::kMergeSurfaceBuffers:
            return RefinementTaskKind::kMergeSurfaceBuffers;
        case PostRefineTaskNode::kRegionExtractSurface:
            return RefinementTaskKind::kRegionExtractSurface;
        case PostRefineTaskNode::kCount:
            break;
    }
    return RefinementTaskKind::kBuildExtractionView;
}

inline void enqueue_post_refine_task(
    RefinementWorkQueue &queue,
    PostRefineTaskGraphRuntime &runtime,
    PostRefineTaskNode node,
    std::size_t cell_index,
    std::uint32_t preferred_worker) {
    const std::size_t node_index = post_refine_task_index(node);
    if (node_index >= runtime.enqueued.size()) {
        return;
    }
    if (runtime.enqueued[node_index] != 0U) {
        return;
    }
    runtime.enqueued[node_index] = 1U;
    queue.push(
        {cell_index, 0U, 0U, post_refine_task_kind(node)},
        preferred_worker);
}

inline void unlock_post_refine_task(
    RefinementWorkQueue &queue,
    PostRefineTaskGraphRuntime &runtime,
    PostRefineTaskNode node,
    std::size_t cell_index,
    std::uint32_t preferred_worker) {
    const std::size_t node_index = post_refine_task_index(node);
    const std::uint32_t remaining = runtime.dependency_counts[node_index].fetch_sub(
        1U, std::memory_order_acq_rel);
    if (remaining == 1U) {
        enqueue_post_refine_task(
            queue, runtime, node, cell_index, preferred_worker);
    }
}

inline void run_build_extraction_view_task(
    PostRefineTaskGraphRuntime &runtime) {
    runtime.result.extraction_view = build_occupied_solid_extraction_view(
        *runtime.all_cells,
        *runtime.classification_cache,
        std::move(runtime.initial_inside_mask_by_cell));
}

inline void run_compute_inside_clearance_task(
    PostRefineTaskGraphRuntime &runtime) {
    runtime.result.clearance_by_cell = compute_inside_clearance_from_cell_mask(
        *runtime.all_cells,
        *runtime.classification_cache,
        runtime.result.extraction_view.inside_mask_by_cell,
        runtime.opening_radius);
}

inline void run_erode_occupied_solid_task(
    PostRefineTaskGraphRuntime &runtime) {
    runtime.result.eroded_inside_by_cell = erode_occupied_solid_cells(
        *runtime.all_cells,
        runtime.result.extraction_view.inside_mask_by_cell,
        runtime.result.clearance_by_cell,
        runtime.opening_radius);
}

inline void run_compute_distance_to_eroded_task(
    PostRefineTaskGraphRuntime &runtime) {
    runtime.result.dilation_distance_by_cell =
        compute_distance_to_eroded_solid_from_cell_mask(
            *runtime.all_cells,
            *runtime.classification_cache,
            runtime.result.eroded_inside_by_cell,
            runtime.opening_radius);
}

inline void run_dilate_eroded_solid_task(
    PostRefineTaskGraphRuntime &runtime) {
    runtime.result.opened_inside_by_cell = dilate_eroded_solid_cells(
        *runtime.all_cells,
        runtime.result.eroded_inside_by_cell,
        runtime.result.dilation_distance_by_cell,
        runtime.opening_radius);
}

inline void run_project_opened_mask_to_leaves_task(
    PostRefineTaskGraphRuntime &runtime) {
    runtime.result.opened_inside = build_leaf_mask_from_cell_mask(
        runtime.result.extraction_view.solid_leaves,
        runtime.result.opened_inside_by_cell);
}

inline void run_fill_opened_cavities_task(
    PostRefineTaskGraphRuntime &runtime) {
    fill_small_opened_cavities(
        runtime.result.extraction_view.solid_leaves,
        runtime.result.opened_inside);
}

inline void run_prune_opened_components_task(
    PostRefineTaskGraphRuntime &runtime) {
    prune_small_opened_components(
        runtime.result.extraction_view.solid_leaves,
        runtime.result.opened_inside);
}

inline void run_suppress_opened_edge_contacts_task(
    PostRefineTaskGraphRuntime &runtime) {
    suppress_opened_edge_contacts(
        runtime.result.extraction_view.solid_leaves,
        *runtime.all_cells,
        *runtime.solid_spatial_index,
        runtime.result.opened_inside);
}

inline void initialize_post_refine_task_graph(
    PostRefineTaskGraphRuntime &runtime) {
    runtime.enqueued.assign(post_refine_task_node_count(), 0U);

    runtime.dependency_counts[post_refine_task_index(
        PostRefineTaskNode::kBuildExtractionView)].store(0U);
    runtime.dependency_counts[post_refine_task_index(
        PostRefineTaskNode::kComputeInsideClearance)].store(1U);
    runtime.dependency_counts[post_refine_task_index(
        PostRefineTaskNode::kErodeOccupiedSolid)].store(1U);
    runtime.dependency_counts[post_refine_task_index(
        PostRefineTaskNode::kComputeDistanceToEroded)].store(1U);
    runtime.dependency_counts[post_refine_task_index(
        PostRefineTaskNode::kDilateErodedSolid)].store(1U);
    runtime.dependency_counts[post_refine_task_index(
        PostRefineTaskNode::kProjectOpenedMaskToLeaves)].store(1U);
    runtime.dependency_counts[post_refine_task_index(
        PostRefineTaskNode::kFillOpenedCavities)].store(1U);
    runtime.dependency_counts[post_refine_task_index(
        PostRefineTaskNode::kPruneOpenedComponents)].store(1U);
    runtime.dependency_counts[post_refine_task_index(
        PostRefineTaskNode::kSuppressOpenedEdgeContacts)].store(1U);
    runtime.dependency_counts[post_refine_task_index(
        PostRefineTaskNode::kResolveOpenedSurfaceAmbiguities)].store(1U);
    runtime.dependency_counts[post_refine_task_index(
        PostRefineTaskNode::kReextractOpenedSurface)].store(1U);
    runtime.dependency_counts[post_refine_task_index(
        PostRefineTaskNode::kMergeSurfaceBuffers)].store(1U);
    runtime.dependency_counts[post_refine_task_index(
        PostRefineTaskNode::kRegionExtractSurface)].store(1U);
}

inline void prepare_post_refine_surface_extraction_runtime(
    PostRefineTaskGraphRuntime &runtime) {
    runtime.surface_extraction.solid_leaves =
        &runtime.result.extraction_view.solid_leaves;
    runtime.surface_extraction.opened_inside = &runtime.result.opened_inside;
    runtime.surface_extraction.all_cells = runtime.all_cells;
    runtime.surface_extraction.spatial_index = runtime.solid_spatial_index;
    runtime.surface_extraction.domain = runtime.domain;
    runtime.surface_extraction.base_resolution = runtime.base_resolution;
    runtime.surface_extraction.max_depth = runtime.max_depth;
    runtime.surface_extraction.fine_resolution =
        runtime.base_resolution * (1U << runtime.max_depth);
    runtime.surface_extraction.fine_dx =
        (runtime.domain->max.x - runtime.domain->min.x) /
        static_cast<double>(runtime.surface_extraction.fine_resolution);
    runtime.surface_extraction.fine_dy =
        (runtime.domain->max.y - runtime.domain->min.y) /
        static_cast<double>(runtime.surface_extraction.fine_resolution);
    runtime.surface_extraction.fine_dz =
        (runtime.domain->max.z - runtime.domain->min.z) /
        static_cast<double>(runtime.surface_extraction.fine_resolution);
    runtime.surface_cell_to_leaf_index = build_opened_cell_to_leaf_index(
        *runtime.all_cells,
        runtime.result.extraction_view.solid_leaves);
    runtime.surface_extraction.cell_to_leaf_index =
        &runtime.surface_cell_to_leaf_index;
    runtime.surface_regions = build_surface_regions(
        runtime.result.extraction_view.solid_leaves,
        *runtime.all_cells);
    runtime.surface_extraction.regions = &runtime.surface_regions;
    runtime.region_buffers.clear();
    runtime.region_buffers.resize(runtime.surface_regions.size());
    runtime.pending_region_tasks.store(
        runtime.surface_regions.size(), std::memory_order_release);
}

inline PostRefineRegularizationResult run_post_refine_regularization_task_graph(
    const std::vector<OctreeCell> &all_cells,
    const OccupiedSolidClassificationCache &classification_cache,
    const LeafSpatialIndex &solid_spatial_index,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    std::uint32_t max_depth,
    std::uint32_t worker_count,
    double opening_radius,
    double table_cadence_seconds,
    std::vector<std::uint8_t> initial_inside_mask_by_cell) {
    if (worker_count == 0U) {
        worker_count = 1U;
    }

    PostRefineTaskGraphRuntime runtime;
    runtime.all_cells = &all_cells;
    runtime.classification_cache = &classification_cache;
    runtime.solid_spatial_index = &solid_spatial_index;
    runtime.domain = &domain;
    runtime.base_resolution = base_resolution;
    runtime.max_depth = max_depth;
    runtime.worker_count = worker_count;
    runtime.opening_radius = opening_radius;
    runtime.table_cadence_seconds = table_cadence_seconds;
    runtime.initial_inside_mask_by_cell = std::move(initial_inside_mask_by_cell);
    initialize_post_refine_task_graph(runtime);

    RefinementWorkQueue queue;
    queue.initialize(worker_count);
    enqueue_post_refine_task(
        queue,
        runtime,
        PostRefineTaskNode::kBuildExtractionView,
        0U,
        0U);
    queue.capture_initial_queue_size();

    std::exception_ptr worker_error;
    std::mutex error_mutex;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::uint32_t worker_id = 0; worker_id < worker_count; ++worker_id) {
        workers.emplace_back([&, worker_id]() {
            RefinementTask task;
            while (queue.pop(worker_id, task)) {
                try {
                    switch (task.kind) {
                        case RefinementTaskKind::kBuildExtractionView:
                            run_build_extraction_view_task(runtime);
                            unlock_post_refine_task(
                                queue,
                                runtime,
                                PostRefineTaskNode::kComputeInsideClearance,
                                0U,
                                worker_id);
                            break;
                        case RefinementTaskKind::kComputeInsideClearance:
                            run_compute_inside_clearance_task(runtime);
                            unlock_post_refine_task(
                                queue,
                                runtime,
                                PostRefineTaskNode::kErodeOccupiedSolid,
                                0U,
                                worker_id);
                            break;
                        case RefinementTaskKind::kErodeOccupiedSolid:
                            run_erode_occupied_solid_task(runtime);
                            unlock_post_refine_task(
                                queue,
                                runtime,
                                PostRefineTaskNode::kComputeDistanceToEroded,
                                0U,
                                worker_id);
                            break;
                        case RefinementTaskKind::kComputeDistanceToEroded:
                            run_compute_distance_to_eroded_task(runtime);
                            unlock_post_refine_task(
                                queue,
                                runtime,
                                PostRefineTaskNode::kDilateErodedSolid,
                                0U,
                                worker_id);
                            break;
                        case RefinementTaskKind::kDilateErodedSolid:
                            run_dilate_eroded_solid_task(runtime);
                            unlock_post_refine_task(
                                queue,
                                runtime,
                                PostRefineTaskNode::kProjectOpenedMaskToLeaves,
                                0U,
                                worker_id);
                            break;
                        case RefinementTaskKind::kProjectOpenedMaskToLeaves:
                            run_project_opened_mask_to_leaves_task(runtime);
                            unlock_post_refine_task(
                                queue,
                                runtime,
                                PostRefineTaskNode::kFillOpenedCavities,
                                0U,
                                worker_id);
                            break;
                        case RefinementTaskKind::kFillOpenedCavities:
                            run_fill_opened_cavities_task(runtime);
                            unlock_post_refine_task(
                                queue,
                                runtime,
                                PostRefineTaskNode::kPruneOpenedComponents,
                                0U,
                                worker_id);
                            break;
                        case RefinementTaskKind::kPruneOpenedComponents:
                            run_prune_opened_components_task(runtime);
                            unlock_post_refine_task(
                                queue,
                                runtime,
                                PostRefineTaskNode::kSuppressOpenedEdgeContacts,
                                0U,
                                worker_id);
                            break;
                        case RefinementTaskKind::kSuppressOpenedEdgeContacts:
                            run_suppress_opened_edge_contacts_task(runtime);
                            prepare_post_refine_surface_extraction_runtime(runtime);
                            if (runtime.surface_regions.empty()) {
                                runtime.result.opened_surface = {};
                                unlock_post_refine_task(
                                    queue,
                                    runtime,
                                    PostRefineTaskNode::kResolveOpenedSurfaceAmbiguities,
                                    0U,
                                    worker_id);
                                break;
                            }
                            for (std::size_t region_id = 0;
                                 region_id < runtime.surface_regions.size();
                                 ++region_id) {
                                queue.push(
                                    {
                                        region_id,
                                        0U,
                                        0U,
                                        RefinementTaskKind::kRegionExtractSurface,
                                    },
                                    static_cast<std::uint32_t>(region_id %
                                                               runtime.worker_count));
                            }
                            runtime.enqueued[post_refine_task_index(
                                PostRefineTaskNode::kRegionExtractSurface)] = 1U;
                            break;
                        case RefinementTaskKind::kRegionExtractSurface: {
                            const std::size_t region_id = task.cell_index;
                            run_region_extract_surface_task(
                                region_id,
                                runtime.surface_extraction,
                                runtime.region_buffers[region_id]);
                            if (runtime.pending_region_tasks.fetch_sub(
                                    1U,
                                    std::memory_order_acq_rel) == 1U) {
                                enqueue_post_refine_task(
                                    queue,
                                    runtime,
                                    PostRefineTaskNode::kMergeSurfaceBuffers,
                                    0U,
                                    0U);
                            }
                            break;
                        }
                        case RefinementTaskKind::kMergeSurfaceBuffers:
                            runtime.result.opened_surface =
                                merge_region_surface_buffers(runtime.region_buffers);
                            if (runtime.reextract_pass.load(
                                    std::memory_order_acquire)) {
                                runtime.reextract_pass.store(
                                    false, std::memory_order_release);
                            } else {
                                unlock_post_refine_task(
                                    queue,
                                    runtime,
                                    PostRefineTaskNode::kResolveOpenedSurfaceAmbiguities,
                                    0U,
                                    worker_id);
                            }
                            break;
                        case RefinementTaskKind::kResolveOpenedSurfaceAmbiguities:
                            if (resolve_opened_edge_ambiguities(
                                    runtime.result.extraction_view.solid_leaves,
                                    *runtime.all_cells,
                                    *runtime.solid_spatial_index,
                                    runtime.result.opened_inside,
                                    runtime.result.opened_surface)) {
                                enqueue_post_refine_task(
                                    queue,
                                    runtime,
                                    PostRefineTaskNode::kReextractOpenedSurface,
                                    0U,
                                    worker_id);
                            }
                            break;
                        case RefinementTaskKind::kReextractOpenedSurface:
                            runtime.reextract_pass.store(
                                true,
                                std::memory_order_release);
                            prepare_post_refine_surface_extraction_runtime(runtime);
                            runtime.enqueued[post_refine_task_index(
                                PostRefineTaskNode::kMergeSurfaceBuffers)] = 0U;
                            for (std::size_t region_id = 0;
                                 region_id < runtime.surface_regions.size();
                                 ++region_id) {
                                queue.push(
                                    {
                                        region_id,
                                        0U,
                                        0U,
                                        RefinementTaskKind::kRegionExtractSurface,
                                    },
                                    static_cast<std::uint32_t>(region_id %
                                                               runtime.worker_count));
                            }
                            break;
                        default:
                            break;
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (worker_error == nullptr) {
                        worker_error = std::current_exception();
                    }
                    queue.task_done();
                    queue.shutdown();
                    return;
                }
                queue.task_done();
                queue.try_shutdown_if_idle();
            }
        });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }
    if (worker_error != nullptr) {
        std::rethrow_exception(worker_error);
    }

    return std::move(runtime.result);
}

inline bool surface_region_contains_cell(
    const std::vector<OctreeCell> &all_cells,
    std::size_t root_cell_index,
    std::size_t cell_index) {
    if (root_cell_index >= all_cells.size() || cell_index >= all_cells.size()) {
        return false;
    }
    std::size_t current = cell_index;
    while (true) {
        if (current == root_cell_index) {
            return true;
        }
        const std::int64_t parent = all_cells[current].parent_index;
        if (parent < 0) {
            return false;
        }
        current = static_cast<std::size_t>(parent);
    }
}

inline std::vector<SurfaceRegion> build_surface_regions(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<OctreeCell> &all_cells) {
    std::unordered_map<std::size_t, std::size_t> region_lookup;
    std::vector<SurfaceRegion> regions;
    region_lookup.reserve(solid_leaves.size());

    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        std::size_t root_cell_index = solid_leaves[leaf_index].cell_index;
        while (root_cell_index < all_cells.size() &&
               all_cells[root_cell_index].parent_index >= 0) {
            root_cell_index = static_cast<std::size_t>(
                all_cells[root_cell_index].parent_index);
        }

        auto it = region_lookup.find(root_cell_index);
        if (it == region_lookup.end()) {
            const std::size_t region_id = regions.size();
            region_lookup[root_cell_index] = region_id;
            regions.push_back({
                region_id,
                root_cell_index,
                all_cells[root_cell_index].depth,
                {leaf_index},
            });
            continue;
        }

        SurfaceRegion &region = regions[it->second];
        region.leaf_indices.push_back(leaf_index);
    }

    return regions;
}

inline std::size_t region_vertex_index_for(
    const SurfaceExtractionRuntime &runtime,
    RegionSurfaceBuffers &buffers,
    std::uint32_t ix,
    std::uint32_t iy,
    std::uint32_t iz) {
    const std::uint64_t key = pack_surface_corner_coords(ix, iy, iz);
    auto existing = buffers.local_vertex_lookup.find(key);
    if (existing != buffers.local_vertex_lookup.end()) {
        return existing->second;
    }

    const Vector3d position = {
        runtime.domain->min.x + static_cast<double>(ix) * runtime.fine_dx,
        runtime.domain->min.y + static_cast<double>(iy) * runtime.fine_dy,
        runtime.domain->min.z + static_cast<double>(iz) * runtime.fine_dz,
    };
    const std::size_t vertex_index = buffers.local_vertices.size();
    buffers.local_vertices.push_back({key, position, {0.0, 0.0, 0.0}});
    buffers.local_vertex_lookup.emplace(key, vertex_index);
    return vertex_index;
}

inline void append_region_oriented_quad(
    RegionSurfaceBuffers &buffers,
    const std::array<Vector3d, 4> &positions,
    const std::array<std::size_t, 4> &indices,
    const Vector3d &outward) {
    const Vector3d e1 = {
        positions[1].x - positions[0].x,
        positions[1].y - positions[0].y,
        positions[1].z - positions[0].z,
    };
    const Vector3d e2 = {
        positions[2].x - positions[0].x,
        positions[2].y - positions[0].y,
        positions[2].z - positions[0].z,
    };
    const Vector3d cross = {
        e1.y * e2.z - e1.z * e2.y,
        e1.z * e2.x - e1.x * e2.z,
        e1.x * e2.y - e1.y * e2.x,
    };
    const double alignment =
        cross.x * outward.x + cross.y * outward.y + cross.z * outward.z;

    std::array<std::size_t, 4> oriented = indices;
    if (alignment < 0.0) {
        oriented = {indices[0], indices[3], indices[2], indices[1]};
    }

    buffers.local_triangles.push_back(
        {{oriented[0], oriented[1], oriented[2]}});
    buffers.local_triangles.push_back(
        {{oriented[0], oriented[2], oriented[3]}});

    for (std::size_t vertex_index : oriented) {
        buffers.local_vertices[vertex_index].normal_sum.x += outward.x;
        buffers.local_vertices[vertex_index].normal_sum.y += outward.y;
        buffers.local_vertices[vertex_index].normal_sum.z += outward.z;
    }
}

inline void emit_region_opened_surface(
    const SurfaceRegion &region,
    const SurfaceExtractionRuntime &runtime,
    RegionSurfaceBuffers &buffers) {
    const std::vector<OccupiedSolidLeaf> &solid_leaves = *runtime.solid_leaves;
    const std::vector<std::uint8_t> &opened_inside = *runtime.opened_inside;
    const std::vector<OctreeCell> &all_cells = *runtime.all_cells;
    const LeafSpatialIndex &spatial_index = *runtime.spatial_index;
    const std::vector<std::int64_t> &cell_to_leaf_index =
        *runtime.cell_to_leaf_index;

    for (std::size_t leaf_index : region.leaf_indices) {
        if (opened_inside[leaf_index] == 0U) {
            continue;
        }

        const std::size_t cell_index = solid_leaves[leaf_index].cell_index;
        if (!surface_region_contains_cell(all_cells, region.root_cell_index, cell_index)) {
            continue;
        }

        std::uint32_t cell_x = 0U;
        std::uint32_t cell_y = 0U;
        std::uint32_t cell_z = 0U;
        const OctreeCell &cell = all_cells[cell_index];
        morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
        const std::uint32_t span = 1U << (runtime.max_depth - cell.depth);
        const std::uint32_t ix0 = cell_x << (runtime.max_depth - cell.depth);
        const std::uint32_t iy0 = cell_y << (runtime.max_depth - cell.depth);
        const std::uint32_t iz0 = cell_z << (runtime.max_depth - cell.depth);

        for (std::uint32_t u = 0; u < span; ++u) {
            for (std::uint32_t v = 0; v < span; ++v) {
                const auto neighbor_opened = [&](std::int64_t neighbor_cell_index) {
                    if (neighbor_cell_index < 0) {
                        return false;
                    }
                    return opened_inside[static_cast<std::size_t>(neighbor_cell_index)] != 0U;
                };
                auto lookup_neighbor = [&](std::uint32_t qx,
                                           std::uint32_t qy,
                                           std::uint32_t qz) {
                    if (qx >= runtime.fine_resolution ||
                        qy >= runtime.fine_resolution ||
                        qz >= runtime.fine_resolution) {
                        return false;
                    }
                    const std::size_t neighbor_cell_index =
                        spatial_index.find_leaf_at(qx, qy, qz);
                    if (neighbor_cell_index == SIZE_MAX ||
                        neighbor_cell_index >= cell_to_leaf_index.size()) {
                        return false;
                    }
                    return neighbor_opened(cell_to_leaf_index[neighbor_cell_index]);
                };

                if (ix0 == 0U || !lookup_neighbor(ix0 - 1U, iy0 + u, iz0 + v)) {
                    const std::array<std::size_t, 4> ids = {
                        region_vertex_index_for(runtime, buffers, ix0, iy0 + u, iz0 + v),
                        region_vertex_index_for(runtime, buffers, ix0, iy0 + u + 1U, iz0 + v),
                        region_vertex_index_for(runtime, buffers, ix0, iy0 + u + 1U, iz0 + v + 1U),
                        region_vertex_index_for(runtime, buffers, ix0, iy0 + u, iz0 + v + 1U),
                    };
                    const std::array<Vector3d, 4> pos = {
                        buffers.local_vertices[ids[0]].position,
                        buffers.local_vertices[ids[1]].position,
                        buffers.local_vertices[ids[2]].position,
                        buffers.local_vertices[ids[3]].position,
                    };
                    append_region_oriented_quad(buffers, pos, ids, {-1.0, 0.0, 0.0});
                }
                if (ix0 + span >= runtime.fine_resolution ||
                    !lookup_neighbor(ix0 + span, iy0 + u, iz0 + v)) {
                    const std::array<std::size_t, 4> ids = {
                        region_vertex_index_for(runtime, buffers, ix0 + span, iy0 + u, iz0 + v),
                        region_vertex_index_for(runtime, buffers, ix0 + span, iy0 + u, iz0 + v + 1U),
                        region_vertex_index_for(runtime, buffers, ix0 + span, iy0 + u + 1U, iz0 + v + 1U),
                        region_vertex_index_for(runtime, buffers, ix0 + span, iy0 + u + 1U, iz0 + v),
                    };
                    const std::array<Vector3d, 4> pos = {
                        buffers.local_vertices[ids[0]].position,
                        buffers.local_vertices[ids[1]].position,
                        buffers.local_vertices[ids[2]].position,
                        buffers.local_vertices[ids[3]].position,
                    };
                    append_region_oriented_quad(buffers, pos, ids, {1.0, 0.0, 0.0});
                }
                if (iy0 == 0U || !lookup_neighbor(ix0 + u, iy0 - 1U, iz0 + v)) {
                    const std::array<std::size_t, 4> ids = {
                        region_vertex_index_for(runtime, buffers, ix0 + u, iy0, iz0 + v),
                        region_vertex_index_for(runtime, buffers, ix0 + u, iy0, iz0 + v + 1U),
                        region_vertex_index_for(runtime, buffers, ix0 + u + 1U, iy0, iz0 + v + 1U),
                        region_vertex_index_for(runtime, buffers, ix0 + u + 1U, iy0, iz0 + v),
                    };
                    const std::array<Vector3d, 4> pos = {
                        buffers.local_vertices[ids[0]].position,
                        buffers.local_vertices[ids[1]].position,
                        buffers.local_vertices[ids[2]].position,
                        buffers.local_vertices[ids[3]].position,
                    };
                    append_region_oriented_quad(buffers, pos, ids, {0.0, -1.0, 0.0});
                }
                if (iy0 + span >= runtime.fine_resolution ||
                    !lookup_neighbor(ix0 + u, iy0 + span, iz0 + v)) {
                    const std::array<std::size_t, 4> ids = {
                        region_vertex_index_for(runtime, buffers, ix0 + u, iy0 + span, iz0 + v),
                        region_vertex_index_for(runtime, buffers, ix0 + u + 1U, iy0 + span, iz0 + v),
                        region_vertex_index_for(runtime, buffers, ix0 + u + 1U, iy0 + span, iz0 + v + 1U),
                        region_vertex_index_for(runtime, buffers, ix0 + u, iy0 + span, iz0 + v + 1U),
                    };
                    const std::array<Vector3d, 4> pos = {
                        buffers.local_vertices[ids[0]].position,
                        buffers.local_vertices[ids[1]].position,
                        buffers.local_vertices[ids[2]].position,
                        buffers.local_vertices[ids[3]].position,
                    };
                    append_region_oriented_quad(buffers, pos, ids, {0.0, 1.0, 0.0});
                }
                if (iz0 == 0U || !lookup_neighbor(ix0 + u, iy0 + v, iz0 - 1U)) {
                    const std::array<std::size_t, 4> ids = {
                        region_vertex_index_for(runtime, buffers, ix0 + u, iy0 + v, iz0),
                        region_vertex_index_for(runtime, buffers, ix0 + u + 1U, iy0 + v, iz0),
                        region_vertex_index_for(runtime, buffers, ix0 + u + 1U, iy0 + v + 1U, iz0),
                        region_vertex_index_for(runtime, buffers, ix0 + u, iy0 + v + 1U, iz0),
                    };
                    const std::array<Vector3d, 4> pos = {
                        buffers.local_vertices[ids[0]].position,
                        buffers.local_vertices[ids[1]].position,
                        buffers.local_vertices[ids[2]].position,
                        buffers.local_vertices[ids[3]].position,
                    };
                    append_region_oriented_quad(buffers, pos, ids, {0.0, 0.0, -1.0});
                }
                if (iz0 + span >= runtime.fine_resolution ||
                    !lookup_neighbor(ix0 + u, iy0 + v, iz0 + span)) {
                    const std::array<std::size_t, 4> ids = {
                        region_vertex_index_for(runtime, buffers, ix0 + u, iy0 + v, iz0 + span),
                        region_vertex_index_for(runtime, buffers, ix0 + u, iy0 + v + 1U, iz0 + span),
                        region_vertex_index_for(runtime, buffers, ix0 + u + 1U, iy0 + v + 1U, iz0 + span),
                        region_vertex_index_for(runtime, buffers, ix0 + u + 1U, iy0 + v, iz0 + span),
                    };
                    const std::array<Vector3d, 4> pos = {
                        buffers.local_vertices[ids[0]].position,
                        buffers.local_vertices[ids[1]].position,
                        buffers.local_vertices[ids[2]].position,
                        buffers.local_vertices[ids[3]].position,
                    };
                    append_region_oriented_quad(buffers, pos, ids, {0.0, 0.0, 1.0});
                }
            }
        }
    }
}

inline void run_region_extract_surface_task(
    std::size_t region_id,
    const SurfaceExtractionRuntime &runtime,
    RegionSurfaceBuffers &out_buffers) {
    out_buffers.region_id = region_id;
    out_buffers.local_vertices.clear();
    out_buffers.local_triangles.clear();
    out_buffers.local_vertex_lookup.clear();
    emit_region_opened_surface(runtime.regions->at(region_id), runtime, out_buffers);
}

inline OpenedSurfaceMesh merge_region_surface_buffers(
    const std::vector<RegionSurfaceBuffers> &region_buffers) {
    OpenedSurfaceMesh mesh;
    std::unordered_map<std::uint64_t, std::size_t> vertex_lookup;
    std::vector<Vector3d> normal_accum;

    std::size_t vertex_capacity = 0U;
    std::size_t triangle_capacity = 0U;
    for (const RegionSurfaceBuffers &buffer : region_buffers) {
        vertex_capacity += buffer.local_vertices.size();
        triangle_capacity += buffer.local_triangles.size();
    }
    vertex_lookup.reserve(vertex_capacity);
    mesh.triangles.reserve(triangle_capacity);

    for (const RegionSurfaceBuffers &buffer : region_buffers) {
        std::vector<std::size_t> local_to_global(buffer.local_vertices.size(), 0U);
        for (std::size_t i = 0; i < buffer.local_vertices.size(); ++i) {
            const RegionSurfaceVertex &vertex = buffer.local_vertices[i];
            auto existing = vertex_lookup.find(vertex.key);
            if (existing == vertex_lookup.end()) {
                const std::size_t global_index = mesh.vertices.size();
                mesh.vertices.push_back({vertex.position, {0.0, 0.0, 0.0}});
                mesh.vertex_keys.push_back(vertex.key);
                normal_accum.push_back(vertex.normal_sum);
                vertex_lookup.emplace(vertex.key, global_index);
                local_to_global[i] = global_index;
            } else {
                local_to_global[i] = existing->second;
                normal_accum[existing->second].x += vertex.normal_sum.x;
                normal_accum[existing->second].y += vertex.normal_sum.y;
                normal_accum[existing->second].z += vertex.normal_sum.z;
            }
        }

        for (const RegionSurfaceTriangle &triangle : buffer.local_triangles) {
            mesh.triangles.push_back({
                local_to_global[triangle.local_vertex_index[0]],
                local_to_global[triangle.local_vertex_index[1]],
                local_to_global[triangle.local_vertex_index[2]],
            });
        }
    }

    std::vector<std::size_t> order(mesh.vertices.size(), 0U);
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return mesh.vertex_keys[a] < mesh.vertex_keys[b];
    });

    OpenedSurfaceMesh sorted_mesh;
    sorted_mesh.vertices.resize(mesh.vertices.size());
    sorted_mesh.vertex_keys.resize(mesh.vertex_keys.size());
    sorted_mesh.triangles = mesh.triangles;
    std::vector<Vector3d> sorted_normals(normal_accum.size(), {0.0, 0.0, 0.0});
    std::vector<std::size_t> old_to_new(order.size(), 0U);
    for (std::size_t new_index = 0; new_index < order.size(); ++new_index) {
        const std::size_t old_index = order[new_index];
        old_to_new[old_index] = new_index;
        sorted_mesh.vertices[new_index] = mesh.vertices[old_index];
        sorted_mesh.vertex_keys[new_index] = mesh.vertex_keys[old_index];
        sorted_normals[new_index] = normal_accum[old_index];
    }
    for (MeshTriangle &triangle : sorted_mesh.triangles) {
        triangle.vertex_indices[0] = old_to_new[triangle.vertex_indices[0]];
        triangle.vertex_indices[1] = old_to_new[triangle.vertex_indices[1]];
        triangle.vertex_indices[2] = old_to_new[triangle.vertex_indices[2]];
    }
    for (std::size_t i = 0; i < sorted_mesh.vertices.size(); ++i) {
        const double mag = std::sqrt(
            sorted_normals[i].x * sorted_normals[i].x +
            sorted_normals[i].y * sorted_normals[i].y +
            sorted_normals[i].z * sorted_normals[i].z);
        if (mag > 0.0) {
            sorted_mesh.vertices[i].normal = {
                sorted_normals[i].x / mag,
                sorted_normals[i].y / mag,
                sorted_normals[i].z / mag,
            };
        }
    }
    return sorted_mesh;
}

inline void run_merge_surface_buffers_task(
    const SurfaceTaskGraphRuntime &runtime,
    OpenedSurfaceMesh &out_mesh) {
    out_mesh = merge_region_surface_buffers(runtime.region_buffers);
}

inline OpenedSurfaceMesh run_opened_surface_task_graph(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &opened_inside,
    const std::vector<OctreeCell> &all_cells,
    const LeafSpatialIndex &spatial_index,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    std::uint32_t max_depth,
    std::uint32_t worker_count,
    double table_cadence_seconds) {
    if (worker_count == 0U) {
        worker_count = 1U;
    }

    SurfaceTaskGraphRuntime runtime;
    runtime.extraction.solid_leaves = &solid_leaves;
    runtime.extraction.opened_inside = &opened_inside;
    runtime.extraction.all_cells = &all_cells;
    runtime.extraction.spatial_index = &spatial_index;
    runtime.extraction.domain = &domain;
    runtime.extraction.cell_to_leaf_index = nullptr;
    runtime.extraction.base_resolution = base_resolution;
    runtime.extraction.max_depth = max_depth;
    runtime.extraction.fine_resolution = base_resolution * (1U << max_depth);
    runtime.extraction.fine_dx =
        (domain.max.x - domain.min.x) /
        static_cast<double>(runtime.extraction.fine_resolution);
    runtime.extraction.fine_dy =
        (domain.max.y - domain.min.y) /
        static_cast<double>(runtime.extraction.fine_resolution);
    runtime.extraction.fine_dz =
        (domain.max.z - domain.min.z) /
        static_cast<double>(runtime.extraction.fine_resolution);

    const std::vector<std::int64_t> cell_to_leaf_index =
        build_opened_cell_to_leaf_index(all_cells, solid_leaves);
    runtime.extraction.cell_to_leaf_index = &cell_to_leaf_index;

    const std::vector<SurfaceRegion> regions =
        build_surface_regions(solid_leaves, all_cells);
    runtime.extraction.regions = &regions;
    runtime.region_buffers.resize(regions.size());
    runtime.graph.pending_region_tasks.store(regions.size(), std::memory_order_relaxed);

    if (regions.empty()) {
        return {};
    }

    RefinementWorkQueue queue;
    queue.initialize(worker_count);
    for (std::size_t region_id = 0; region_id < regions.size(); ++region_id) {
        queue.push(
            {region_id, 0U, 0U, RefinementTaskKind::kRegionExtractSurface},
            static_cast<std::uint32_t>(region_id % worker_count));
    }
    queue.capture_initial_queue_size();

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::uint32_t worker_id = 0; worker_id < worker_count; ++worker_id) {
        workers.emplace_back([&, worker_id]() {
            RefinementTask task;
            while (queue.pop(worker_id, task)) {
                switch (task.kind) {
                    case RefinementTaskKind::kRegionExtractSurface: {
                        const std::size_t region_id = task.cell_index;
                        run_region_extract_surface_task(
                            region_id,
                            runtime.extraction,
                            runtime.region_buffers[region_id]);
                        if (runtime.graph.pending_region_tasks.fetch_sub(
                                1U, std::memory_order_acq_rel) == 1U) {
                            bool expected = false;
                            if (runtime.graph.merge_enqueued.compare_exchange_strong(
                                    expected,
                                    true,
                                    std::memory_order_acq_rel)) {
                                queue.push(
                                    {0U, 0U, 0U, RefinementTaskKind::kMergeSurfaceBuffers},
                                    0U);
                            }
                        }
                        break;
                    }
                    case RefinementTaskKind::kMergeSurfaceBuffers: {
                        OpenedSurfaceMesh merged;
                        run_merge_surface_buffers_task(runtime, merged);
                        {
                            std::lock_guard<std::mutex> lock(runtime.merged_mesh_mutex);
                            runtime.merged_mesh = std::move(merged);
                        }
                        runtime.graph.merge_completed.store(true, std::memory_order_release);
                        break;
                    }
                    default:
                        break;
                }
                queue.task_done();
                if (queue.try_claim_report(table_cadence_seconds)) {
                    const RefinementWorkQueueStats stats = queue.stats();
                    meshmerizer_log_detail::print_debug_status(
                        "Meshing",
                        "run_opened_surface_task_graph",
                        "scheduler: queue=%zu inflight=%zu pushed=%zu popped=%zu\n",
                        stats.queue_size,
                        stats.in_flight_count,
                        stats.push_count,
                        stats.pop_count);
                }
                queue.try_shutdown_if_idle();
            }
        });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }

    return runtime.merged_mesh;
}

struct CornerVote {
    std::uint32_t inside = 0U;
    std::uint32_t outside = 0U;
};

inline void unpack_surface_corner_coords(
    std::uint64_t key,
    std::uint32_t &ix,
    std::uint32_t &iy,
    std::uint32_t &iz) {
    ix = static_cast<std::uint32_t>((key >> 42U) & 0x1fffffU);
    iy = static_cast<std::uint32_t>((key >> 21U) & 0x1fffffU);
    iz = static_cast<std::uint32_t>(key & 0x1fffffU);
}

inline std::uint64_t pack_surface_corner_coords(
    std::uint32_t ix, std::uint32_t iy, std::uint32_t iz) {
    return (static_cast<std::uint64_t>(ix) << 42U) |
           (static_cast<std::uint64_t>(iy) << 21U) |
           static_cast<std::uint64_t>(iz);
}

inline bool resolve_opened_edge_ambiguities(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<OctreeCell> &all_cells,
    const LeafSpatialIndex &spatial_index,
    std::vector<std::uint8_t> &opened_inside,
    const OpenedSurfaceMesh &mesh) {
    struct EdgeKey {
        std::size_t a;
        std::size_t b;

        bool operator==(const EdgeKey &other) const {
            return a == other.a && b == other.b;
        }
    };
    struct EdgeKeyHash {
        std::size_t operator()(const EdgeKey &key) const {
            std::size_t h = key.a;
            h ^= key.b + 0x9e3779b9ULL + (h << 6U) + (h >> 2U);
            return h;
        }
    };

    if (mesh.triangles.empty() || mesh.vertex_keys.size() != mesh.vertices.size()) {
        return false;
    }

    const std::vector<std::int64_t> cell_to_leaf_index =
        build_opened_cell_to_leaf_index(all_cells, solid_leaves);

    auto lookup_leaf = [&](std::int64_t qx,
                           std::int64_t qy,
                           std::int64_t qz) -> std::int64_t {
        if (qx < 0 || qy < 0 || qz < 0 ||
            qx >= static_cast<std::int64_t>(spatial_index.fine_resolution) ||
            qy >= static_cast<std::int64_t>(spatial_index.fine_resolution) ||
            qz >= static_cast<std::int64_t>(spatial_index.fine_resolution)) {
            return -1;
        }
        const std::size_t cell_index = spatial_index.find_leaf_at(
            static_cast<std::uint32_t>(qx),
            static_cast<std::uint32_t>(qy),
            static_cast<std::uint32_t>(qz));
        if (cell_index == SIZE_MAX || cell_index >= cell_to_leaf_index.size()) {
            return -1;
        }
        return cell_to_leaf_index[cell_index];
    };

    auto opened_at_voxel = [&](std::int64_t qx,
                               std::int64_t qy,
                               std::int64_t qz) -> bool {
        const std::int64_t leaf_index = lookup_leaf(qx, qy, qz);
        return leaf_index >= 0 &&
               opened_inside[static_cast<std::size_t>(leaf_index)] != 0U;
    };

    std::unordered_map<EdgeKey, std::uint32_t, EdgeKeyHash> edge_counts;
    edge_counts.reserve(mesh.triangles.size() * 2U);
    ProgressCounter edge_counter(
        "Regularization", "resolve_opened_edge_ambiguities", "triangles", 1000);
    for (const MeshTriangle &triangle : mesh.triangles) {
        edge_counter.tick();
        for (std::size_t i = 0; i < 3U; ++i) {
            const std::size_t v0 = triangle.vertex_indices[i];
            const std::size_t v1 = triangle.vertex_indices[(i + 1U) % 3U];
            EdgeKey key{std::min(v0, v1), std::max(v0, v1)};
            ++edge_counts[key];
        }
    }
    edge_counter.finish();

    struct CandidateFill {
        std::size_t leaf_index;
        std::uint32_t score;
        double cell_size;
    };

    std::vector<CandidateFill> fills;
    fills.reserve(16U);
    ProgressCounter resolve_counter(
        "Regularization", "resolve_opened_edge_ambiguities", "edges", 100);

    for (const auto &entry : edge_counts) {
        if (entry.second <= 2U) {
            continue;
        }
        resolve_counter.tick();

        std::uint32_t ax, ay, az, bx, by, bz;
        unpack_surface_corner_coords(mesh.vertex_keys[entry.first.a], ax, ay, az);
        unpack_surface_corner_coords(mesh.vertex_keys[entry.first.b], bx, by, bz);

        int axis = -1;
        std::uint32_t gx = std::min(ax, bx);
        std::uint32_t gy = std::min(ay, by);
        std::uint32_t gz = std::min(az, bz);
        if (ax != bx && ay == by && az == bz) {
            axis = 0;
        } else if (ax == bx && ay != by && az == bz) {
            axis = 1;
        } else if (ax == bx && ay == by && az != bz) {
            axis = 2;
        } else {
            continue;
        }

        std::array<std::array<std::int64_t, 3>, 4> voxels;
        if (axis == 0) {
            voxels = {{{static_cast<std::int64_t>(gx), static_cast<std::int64_t>(gy) - 1, static_cast<std::int64_t>(gz) - 1},
                       {static_cast<std::int64_t>(gx), static_cast<std::int64_t>(gy), static_cast<std::int64_t>(gz) - 1},
                       {static_cast<std::int64_t>(gx), static_cast<std::int64_t>(gy) - 1, static_cast<std::int64_t>(gz)},
                       {static_cast<std::int64_t>(gx), static_cast<std::int64_t>(gy), static_cast<std::int64_t>(gz)}}};
        } else if (axis == 1) {
            voxels = {{{static_cast<std::int64_t>(gx) - 1, static_cast<std::int64_t>(gy), static_cast<std::int64_t>(gz) - 1},
                       {static_cast<std::int64_t>(gx), static_cast<std::int64_t>(gy), static_cast<std::int64_t>(gz) - 1},
                       {static_cast<std::int64_t>(gx) - 1, static_cast<std::int64_t>(gy), static_cast<std::int64_t>(gz)},
                       {static_cast<std::int64_t>(gx), static_cast<std::int64_t>(gy), static_cast<std::int64_t>(gz)}}};
        } else {
            voxels = {{{static_cast<std::int64_t>(gx) - 1, static_cast<std::int64_t>(gy) - 1, static_cast<std::int64_t>(gz)},
                       {static_cast<std::int64_t>(gx), static_cast<std::int64_t>(gy) - 1, static_cast<std::int64_t>(gz)},
                       {static_cast<std::int64_t>(gx) - 1, static_cast<std::int64_t>(gy), static_cast<std::int64_t>(gz)},
                       {static_cast<std::int64_t>(gx), static_cast<std::int64_t>(gy), static_cast<std::int64_t>(gz)}}};
        }

        std::array<bool, 4> inside = {false, false, false, false};
        std::array<std::int64_t, 4> leaf_indices = {-1, -1, -1, -1};
        std::uint32_t inside_count = 0U;
        for (std::size_t i = 0; i < 4U; ++i) {
            leaf_indices[i] = lookup_leaf(voxels[i][0], voxels[i][1], voxels[i][2]);
            inside[i] = leaf_indices[i] >= 0 &&
                        opened_inside[static_cast<std::size_t>(leaf_indices[i])] != 0U;
            inside_count += inside[i] ? 1U : 0U;
        }

        const bool diagonal_pattern =
            (inside[0] && inside[3] && !inside[1] && !inside[2]) ||
            (!inside[0] && !inside[3] && inside[1] && inside[2]);
        if (inside_count != 2U || !diagonal_pattern) {
            continue;
        }

        CandidateFill best{SIZE_MAX, 0U, std::numeric_limits<double>::infinity()};
        for (std::size_t i = 0; i < 4U; ++i) {
            if (inside[i] || leaf_indices[i] < 0) {
                continue;
            }

            std::uint32_t score = 0U;
            static const std::int32_t neighbor_offsets[6][3] = {
                {-1, 0, 0}, {1, 0, 0},
                {0, -1, 0}, {0, 1, 0},
                {0, 0, -1}, {0, 0, 1},
            };
            for (const auto &offset : neighbor_offsets) {
                if (opened_at_voxel(
                        voxels[i][0] + offset[0],
                        voxels[i][1] + offset[1],
                        voxels[i][2] + offset[2])) {
                    ++score;
                }
            }

            const std::size_t leaf_index = static_cast<std::size_t>(leaf_indices[i]);
            const double cell_size = solid_leaves[leaf_index].cell_size;
            if (score > best.score ||
                (score == best.score && cell_size < best.cell_size)) {
                best = {leaf_index, score, cell_size};
            }
        }

        if (best.leaf_index != SIZE_MAX) {
            fills.push_back(best);
        }
    }
    resolve_counter.finish();

    if (fills.empty()) {
        return false;
    }

    std::sort(
        fills.begin(),
        fills.end(),
        [](const CandidateFill &a, const CandidateFill &b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            return a.cell_size < b.cell_size;
        });

    std::size_t applied = 0U;
    for (const CandidateFill &fill : fills) {
        if (opened_inside[fill.leaf_index] != 0U) {
            continue;
        }
        opened_inside[fill.leaf_index] = 1U;
        ++applied;
    }

    if (applied > 0U) {
        meshmerizer_log_detail::print_debug_status(
            "Regularization",
            "resolve_opened_edge_ambiguities",
            "filled %zu leaves\n",
            applied);
    }
    return applied > 0U;
}

inline OpenedSurfaceMesh generate_opened_surface_mesh(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &opened_inside,
    const std::vector<OctreeCell> &all_cells,
    const LeafSpatialIndex &spatial_index,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    std::uint32_t max_depth,
    std::uint32_t worker_count = 1U,
    double table_cadence_seconds = 10.0) {
    return run_opened_surface_task_graph(
        solid_leaves,
        opened_inside,
        all_cells,
        spatial_index,
        domain,
        base_resolution,
        max_depth,
        worker_count,
        table_cadence_seconds);
}

inline void apply_qef_positions_to_opened_surface_mesh(
    OpenedSurfaceMesh &mesh,
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &opened_inside,
    const std::vector<OctreeCell> &all_cells,
    const std::vector<MeshVertex> &opened_qef_vertices,
    std::uint32_t max_depth) {
    struct PositionAccum {
        Vector3d sum = {0.0, 0.0, 0.0};
        std::uint32_t count = 0U;
    };

    std::unordered_map<std::uint64_t, PositionAccum> corner_accum;
    corner_accum.reserve(mesh.vertices.size());

    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        if (opened_inside[leaf_index] == 0U) {
            continue;
        }
        const OctreeCell &cell = all_cells[solid_leaves[leaf_index].cell_index];
        if (cell.representative_vertex_index < 0) {
            continue;
        }
        const MeshVertex &qef_vertex =
            opened_qef_vertices[static_cast<std::size_t>(cell.representative_vertex_index)];

        std::uint32_t cell_x = 0U;
        std::uint32_t cell_y = 0U;
        std::uint32_t cell_z = 0U;
        morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
        const std::uint32_t span = 1U << (max_depth - cell.depth);
        const std::uint32_t ix0 = cell_x << (max_depth - cell.depth);
        const std::uint32_t iy0 = cell_y << (max_depth - cell.depth);
        const std::uint32_t iz0 = cell_z << (max_depth - cell.depth);

        for (std::uint8_t corner = 0; corner < 8U; ++corner) {
            const std::uint32_t ix = ix0 + ((corner & 1U) ? span : 0U);
            const std::uint32_t iy = iy0 + ((corner & 2U) ? span : 0U);
            const std::uint32_t iz = iz0 + ((corner & 4U) ? span : 0U);
            PositionAccum &accum =
                corner_accum[pack_surface_corner_coords(ix, iy, iz)];
            accum.sum.x += qef_vertex.position.x;
            accum.sum.y += qef_vertex.position.y;
            accum.sum.z += qef_vertex.position.z;
            ++accum.count;
        }
    }

    for (std::size_t vertex_index = 0; vertex_index < mesh.vertices.size(); ++vertex_index) {
        auto it = corner_accum.find(mesh.vertex_keys[vertex_index]);
        if (it == corner_accum.end() || it->second.count == 0U) {
            continue;
        }
        mesh.vertices[vertex_index].position = {
            it->second.sum.x / static_cast<double>(it->second.count),
            it->second.sum.y / static_cast<double>(it->second.count),
            it->second.sum.z / static_cast<double>(it->second.count),
        };
    }
}

inline void apply_opened_corner_field(
    std::vector<OctreeCell> &all_cells,
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &opened_inside,
    std::uint32_t max_depth) {
    std::unordered_map<std::uint64_t, CornerVote> corner_votes;
    corner_votes.reserve(solid_leaves.size() * 4U);

    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        std::uint32_t cell_x = 0U;
        std::uint32_t cell_y = 0U;
        std::uint32_t cell_z = 0U;
        const OctreeCell &cell = all_cells[solid_leaves[leaf_index].cell_index];
        morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
        const std::uint32_t span = 1U << (max_depth - cell.depth);
        const std::uint32_t ix0 = cell_x << (max_depth - cell.depth);
        const std::uint32_t iy0 = cell_y << (max_depth - cell.depth);
        const std::uint32_t iz0 = cell_z << (max_depth - cell.depth);

        for (std::uint8_t corner = 0; corner < 8U; ++corner) {
            const std::uint32_t ix = ix0 + ((corner & 1U) ? span : 0U);
            const std::uint32_t iy = iy0 + ((corner & 2U) ? span : 0U);
            const std::uint32_t iz = iz0 + ((corner & 4U) ? span : 0U);
            CornerVote &vote = corner_votes[pack_surface_corner_coords(ix, iy, iz)];
            if (opened_inside[leaf_index] != 0U) {
                ++vote.inside;
            } else {
                ++vote.outside;
            }
        }
    }

    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        OctreeCell &cell = all_cells[solid_leaves[leaf_index].cell_index];
        std::uint32_t cell_x = 0U;
        std::uint32_t cell_y = 0U;
        std::uint32_t cell_z = 0U;
        morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
        const std::uint32_t span = 1U << (max_depth - cell.depth);
        const std::uint32_t ix0 = cell_x << (max_depth - cell.depth);
        const std::uint32_t iy0 = cell_y << (max_depth - cell.depth);
        const std::uint32_t iz0 = cell_z << (max_depth - cell.depth);

        for (std::uint8_t corner = 0; corner < 8U; ++corner) {
            const std::uint32_t ix = ix0 + ((corner & 1U) ? span : 0U);
            const std::uint32_t iy = iy0 + ((corner & 2U) ? span : 0U);
            const std::uint32_t iz = iz0 + ((corner & 4U) ? span : 0U);
            const CornerVote &vote = corner_votes[pack_surface_corner_coords(ix, iy, iz)];
            bool inside = vote.inside > vote.outside;
            if (vote.inside == vote.outside) {
                inside = opened_inside[leaf_index] != 0U;
            }
            cell.corner_values[corner] = inside ? 1.0 : -1.0;
        }

        cell.corner_sign_mask = compute_corner_sign_mask(cell.corner_values, 0.0);
        cell.has_surface = cell_may_contain_isosurface(cell.corner_values, 0.0);
        cell.is_topo_surface = false;
        cell.is_active = false;
        cell.representative_vertex_index = -1;
    }
}

inline bool opened_sign_at_fine_vertex(
    const LeafSpatialIndex &spatial_index,
    const std::vector<std::int64_t> &cell_to_leaf_index,
    const std::vector<std::uint8_t> &opened_inside,
    std::uint32_t fine_res,
    std::uint32_t ix,
    std::uint32_t iy,
    std::uint32_t iz) {
    std::uint32_t inside_votes = 0U;
    std::uint32_t outside_votes = 0U;

    for (std::uint32_t dx = 0U; dx < 2U; ++dx) {
        if (dx > ix) {
            continue;
        }
        const std::uint32_t qx = ix - dx;
        if (qx >= fine_res) {
            continue;
        }
        for (std::uint32_t dy = 0U; dy < 2U; ++dy) {
            if (dy > iy) {
                continue;
            }
            const std::uint32_t qy = iy - dy;
            if (qy >= fine_res) {
                continue;
            }
            for (std::uint32_t dz = 0U; dz < 2U; ++dz) {
                if (dz > iz) {
                    continue;
                }
                const std::uint32_t qz = iz - dz;
                if (qz >= fine_res) {
                    continue;
                }

                const std::size_t cell_index =
                    spatial_index.find_leaf_at(qx, qy, qz);
                if (cell_index == SIZE_MAX ||
                    cell_index >= cell_to_leaf_index.size()) {
                    continue;
                }

                const std::int64_t leaf_index = cell_to_leaf_index[cell_index];
                if (leaf_index < 0) {
                    continue;
                }

                if (opened_inside[static_cast<std::size_t>(leaf_index)] != 0U) {
                    ++inside_votes;
                } else {
                    ++outside_votes;
                }
            }
        }
    }

    return inside_votes > outside_votes;
}

inline bool opened_quad_has_four_distinct_vertices(
    const std::vector<OctreeCell> &all_cells,
    std::size_t c0,
    std::size_t c1,
    std::size_t c2,
    std::size_t c3) {
    if (c0 == SIZE_MAX || c1 == SIZE_MAX ||
        c2 == SIZE_MAX || c3 == SIZE_MAX) {
        return false;
    }

    const std::int64_t vi0 = all_cells[c0].representative_vertex_index;
    const std::int64_t vi1 = all_cells[c1].representative_vertex_index;
    const std::int64_t vi2 = all_cells[c2].representative_vertex_index;
    const std::int64_t vi3 = all_cells[c3].representative_vertex_index;
    if (vi0 < 0 || vi1 < 0 || vi2 < 0 || vi3 < 0) {
        return false;
    }

    return vi0 != vi1 && vi0 != vi2 && vi0 != vi3 &&
           vi1 != vi2 && vi1 != vi3 &&
           vi2 != vi3;
}

inline void mark_opened_topology_surface_cells(
    std::vector<OctreeCell> &all_cells,
    const LeafSpatialIndex &spatial_index,
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &opened_inside,
    std::uint32_t max_depth,
    std::uint32_t base_resolution) {
    if (max_depth >= 21U) {
        throw std::overflow_error(
            "max_depth >= 21 would overflow grid coordinates");
    }
    const std::uint64_t fine_res_wide =
        static_cast<std::uint64_t>(base_resolution) *
        (1ULL << max_depth);
    constexpr std::uint64_t MAX_COORD_FACE = (1ULL << 21U) - 1U;
    if (fine_res_wide > MAX_COORD_FACE) {
        throw std::overflow_error(
            "fine_res exceeds 21-bit grid coordinate capacity");
    }
    const std::uint32_t fine_res =
        static_cast<std::uint32_t>(fine_res_wide);

    for (const OccupiedSolidLeaf &leaf : solid_leaves) {
        OctreeCell &cell = all_cells[leaf.cell_index];
        cell.is_topo_surface = false;
        cell.is_active = false;
        cell.representative_vertex_index = -1;
    }

    const std::vector<std::int64_t> cell_to_leaf_index =
        build_opened_cell_to_leaf_index(all_cells, solid_leaves);

    auto mark_cell = [&](std::size_t cell_index) {
        if (cell_index == SIZE_MAX || cell_index >= all_cells.size()) {
            return;
        }
        OctreeCell &cell = all_cells[cell_index];
        cell.is_topo_surface = true;
        cell.is_active = true;
    };

    for (std::uint32_t ix = 0; ix < fine_res; ++ix) {
        for (std::uint32_t iy = 0; iy < fine_res; ++iy) {
            for (std::uint32_t iz = 0; iz < fine_res; ++iz) {
                const std::size_t cell_a_idx =
                    spatial_index.find_leaf_at(ix, iy, iz);
                if (cell_a_idx == SIZE_MAX) {
                    continue;
                }

                const bool sign_a = opened_sign_at_fine_vertex(
                    spatial_index, cell_to_leaf_index, opened_inside,
                    fine_res, ix, iy, iz);

                if (ix + 1U < fine_res) {
                    const std::size_t cell_b_idx =
                        spatial_index.find_leaf_at(ix + 1U, iy, iz);
                    if (cell_b_idx != SIZE_MAX) {
                        const bool sign_b = opened_sign_at_fine_vertex(
                            spatial_index, cell_to_leaf_index,
                            opened_inside, fine_res,
                            ix + 1U, iy, iz);
                        if (sign_a != sign_b && iy > 0U && iz > 0U) {
                            mark_cell(cell_a_idx);
                            mark_cell(spatial_index.find_leaf_at(ix, iy - 1U, iz));
                            mark_cell(spatial_index.find_leaf_at(ix, iy, iz - 1U));
                            mark_cell(spatial_index.find_leaf_at(ix, iy - 1U, iz - 1U));
                        }
                    }
                }

                if (iy + 1U < fine_res) {
                    const std::size_t cell_c_idx =
                        spatial_index.find_leaf_at(ix, iy + 1U, iz);
                    if (cell_c_idx != SIZE_MAX) {
                        const bool sign_c = opened_sign_at_fine_vertex(
                            spatial_index, cell_to_leaf_index,
                            opened_inside, fine_res,
                            ix, iy + 1U, iz);
                        if (sign_a != sign_c && ix > 0U && iz > 0U) {
                            mark_cell(cell_a_idx);
                            mark_cell(spatial_index.find_leaf_at(ix - 1U, iy, iz));
                            mark_cell(spatial_index.find_leaf_at(ix, iy, iz - 1U));
                            mark_cell(spatial_index.find_leaf_at(ix - 1U, iy, iz - 1U));
                        }
                    }
                }

                if (iz + 1U < fine_res) {
                    const std::size_t cell_d_idx =
                        spatial_index.find_leaf_at(ix, iy, iz + 1U);
                    if (cell_d_idx != SIZE_MAX) {
                        const bool sign_d = opened_sign_at_fine_vertex(
                            spatial_index, cell_to_leaf_index,
                            opened_inside, fine_res,
                            ix, iy, iz + 1U);
                        if (sign_a != sign_d && ix > 0U && iy > 0U) {
                            mark_cell(cell_a_idx);
                            mark_cell(spatial_index.find_leaf_at(ix - 1U, iy, iz));
                            mark_cell(spatial_index.find_leaf_at(ix, iy - 1U, iz));
                            mark_cell(spatial_index.find_leaf_at(ix - 1U, iy - 1U, iz));
                        }
                    }
                }
            }
        }
    }
}

inline std::vector<MeshTriangle> generate_opened_dual_contour_faces(
    const std::vector<OctreeCell> &all_cells,
    const std::vector<MeshVertex> &vertices,
    const LeafSpatialIndex &spatial_index,
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &opened_inside,
    std::uint32_t max_depth,
    std::uint32_t base_resolution) {
    if (max_depth >= 21U) {
        throw std::overflow_error(
            "max_depth >= 21 would overflow grid coordinates");
    }
    const std::uint64_t fine_res_wide =
        static_cast<std::uint64_t>(base_resolution) *
        (1ULL << max_depth);
    constexpr std::uint64_t MAX_COORD_FACE = (1ULL << 21U) - 1U;
    if (fine_res_wide > MAX_COORD_FACE) {
        throw std::overflow_error(
            "fine_res exceeds 21-bit grid coordinate capacity");
    }
    const std::uint32_t fine_res =
        static_cast<std::uint32_t>(fine_res_wide);

    const std::vector<std::int64_t> cell_to_leaf_index =
        build_opened_cell_to_leaf_index(all_cells, solid_leaves);

    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<MeshTriangle>> thread_triangles(
        static_cast<std::size_t>(n_threads));

    ProgressBar face_bar(
        "Meshing", "extract_opened_surface_mesh", static_cast<std::size_t>(fine_res));

#pragma omp parallel for schedule(dynamic)
    for (std::uint32_t ix = 0; ix < fine_res; ++ix) {
        std::vector<MeshTriangle> &local_triangles =
            thread_triangles[
                static_cast<std::size_t>(omp_get_thread_num())];

        for (std::uint32_t iy = 0; iy < fine_res; ++iy) {
            for (std::uint32_t iz = 0; iz < fine_res; ++iz) {
                const std::size_t cell_a_idx =
                    spatial_index.find_leaf_at(ix, iy, iz);
                if (cell_a_idx == SIZE_MAX) {
                    continue;
                }

                const bool sign_a = opened_sign_at_fine_vertex(
                    spatial_index, cell_to_leaf_index, opened_inside,
                    fine_res, ix, iy, iz);

                if (ix + 1U < fine_res) {
                    const std::size_t cell_b_idx =
                        spatial_index.find_leaf_at(ix + 1U, iy, iz);
                    if (cell_b_idx != SIZE_MAX) {
                        const bool sign_b = opened_sign_at_fine_vertex(
                            spatial_index, cell_to_leaf_index,
                            opened_inside, fine_res,
                            ix + 1U, iy, iz);
                        if (sign_a != sign_b && iy > 0U && iz > 0U) {
                            const std::size_t c1 =
                                spatial_index.find_leaf_at(ix, iy - 1U, iz);
                            const std::size_t c2 =
                                spatial_index.find_leaf_at(ix, iy, iz - 1U);
                            const std::size_t c3 =
                                spatial_index.find_leaf_at(
                                    ix, iy - 1U, iz - 1U);

                            if (!all_cells[cell_a_idx].is_topo_surface ||
                                c1 == SIZE_MAX || c2 == SIZE_MAX || c3 == SIZE_MAX ||
                                !all_cells[c1].is_topo_surface ||
                                !all_cells[c2].is_topo_surface ||
                                !all_cells[c3].is_topo_surface ||
                                !opened_quad_has_four_distinct_vertices(
                                    all_cells, cell_a_idx, c2, c3, c1)) {
                                continue;
                            }

                            emit_quad(
                                all_cells, vertices, local_triangles,
                                cell_a_idx, c2, c3, c1, sign_a);
                        }
                    }
                }

                if (iy + 1U < fine_res) {
                    const std::size_t cell_c_idx =
                        spatial_index.find_leaf_at(ix, iy + 1U, iz);
                    if (cell_c_idx != SIZE_MAX) {
                        const bool sign_c = opened_sign_at_fine_vertex(
                            spatial_index, cell_to_leaf_index,
                            opened_inside, fine_res,
                            ix, iy + 1U, iz);
                        if (sign_a != sign_c && ix > 0U && iz > 0U) {
                            const std::size_t c1 =
                                spatial_index.find_leaf_at(ix - 1U, iy, iz);
                            const std::size_t c2 =
                                spatial_index.find_leaf_at(ix, iy, iz - 1U);
                            const std::size_t c3 =
                                spatial_index.find_leaf_at(
                                    ix - 1U, iy, iz - 1U);

                            if (!all_cells[cell_a_idx].is_topo_surface ||
                                c1 == SIZE_MAX || c2 == SIZE_MAX || c3 == SIZE_MAX ||
                                !all_cells[c1].is_topo_surface ||
                                !all_cells[c2].is_topo_surface ||
                                !all_cells[c3].is_topo_surface ||
                                !opened_quad_has_four_distinct_vertices(
                                    all_cells, cell_a_idx, c1, c3, c2)) {
                                continue;
                            }

                            emit_quad(
                                all_cells, vertices, local_triangles,
                                cell_a_idx, c1, c3, c2, sign_a);
                        }
                    }
                }

                if (iz + 1U < fine_res) {
                    const std::size_t cell_d_idx =
                        spatial_index.find_leaf_at(ix, iy, iz + 1U);
                    if (cell_d_idx != SIZE_MAX) {
                        const bool sign_d = opened_sign_at_fine_vertex(
                            spatial_index, cell_to_leaf_index,
                            opened_inside, fine_res,
                            ix, iy, iz + 1U);
                        if (sign_a != sign_d && ix > 0U && iy > 0U) {
                            const std::size_t c1 =
                                spatial_index.find_leaf_at(ix - 1U, iy, iz);
                            const std::size_t c2 =
                                spatial_index.find_leaf_at(ix, iy - 1U, iz);
                            const std::size_t c3 =
                                spatial_index.find_leaf_at(
                                    ix - 1U, iy - 1U, iz);

                            if (!all_cells[cell_a_idx].is_topo_surface ||
                                c1 == SIZE_MAX || c2 == SIZE_MAX || c3 == SIZE_MAX ||
                                !all_cells[c1].is_topo_surface ||
                                !all_cells[c2].is_topo_surface ||
                                !all_cells[c3].is_topo_surface ||
                                !opened_quad_has_four_distinct_vertices(
                                    all_cells, cell_a_idx, c2, c3, c1)) {
                                continue;
                            }

                            emit_quad(
                                all_cells, vertices, local_triangles,
                                cell_a_idx, c2, c3, c1, sign_a);
                        }
                    }
                }
            }
        }
        face_bar.tick();
    }

    face_bar.finish();

    std::size_t total_triangles = 0;
    for (const auto &buf : thread_triangles) {
        total_triangles += buf.size();
    }
    std::vector<MeshTriangle> triangles;
    triangles.reserve(total_triangles);
    for (const auto &buf : thread_triangles) {
        triangles.insert(triangles.end(), buf.begin(), buf.end());
    }

    return triangles;
}

inline void apply_opened_topology_to_cells(
    std::vector<OctreeCell> &all_cells,
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &opened_inside,
    std::uint32_t max_depth,
    double isovalue) {
    std::unordered_map<std::uint64_t, CornerVote> corner_votes;
    corner_votes.reserve(solid_leaves.size() * 4U);

    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        std::uint32_t cell_x = 0U;
        std::uint32_t cell_y = 0U;
        std::uint32_t cell_z = 0U;
        const OctreeCell &cell = all_cells[solid_leaves[leaf_index].cell_index];
        morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
        const std::uint32_t span = 1U << (max_depth - cell.depth);
        const std::uint32_t ix0 = cell_x << (max_depth - cell.depth);
        const std::uint32_t iy0 = cell_y << (max_depth - cell.depth);
        const std::uint32_t iz0 = cell_z << (max_depth - cell.depth);

        for (std::uint8_t corner = 0; corner < 8U; ++corner) {
            const std::uint32_t ix = ix0 + ((corner & 1U) ? span : 0U);
            const std::uint32_t iy = iy0 + ((corner & 2U) ? span : 0U);
            const std::uint32_t iz = iz0 + ((corner & 4U) ? span : 0U);
            CornerVote &vote = corner_votes[pack_surface_corner_coords(ix, iy, iz)];
            if (opened_inside[leaf_index] != 0U) {
                ++vote.inside;
            } else {
                ++vote.outside;
            }
        }
    }

    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        OctreeCell &cell = all_cells[solid_leaves[leaf_index].cell_index];
        std::uint32_t cell_x = 0U;
        std::uint32_t cell_y = 0U;
        std::uint32_t cell_z = 0U;
        morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
        const std::uint32_t span = 1U << (max_depth - cell.depth);
        const std::uint32_t ix0 = cell_x << (max_depth - cell.depth);
        const std::uint32_t iy0 = cell_y << (max_depth - cell.depth);
        const std::uint32_t iz0 = cell_z << (max_depth - cell.depth);

        std::uint8_t opened_sign_mask = 0U;
        for (std::uint8_t corner = 0; corner < 8U; ++corner) {
            const std::uint32_t ix = ix0 + ((corner & 1U) ? span : 0U);
            const std::uint32_t iy = iy0 + ((corner & 2U) ? span : 0U);
            const std::uint32_t iz = iz0 + ((corner & 4U) ? span : 0U);
            const CornerVote &vote = corner_votes[pack_surface_corner_coords(ix, iy, iz)];
            bool inside = vote.inside > vote.outside;
            if (vote.inside == vote.outside) {
                inside = cell.corner_values[corner] >= isovalue;
            }
            if (inside) {
                opened_sign_mask |= static_cast<std::uint8_t>(1U << corner);
            }
        }

        cell.corner_sign_mask = opened_sign_mask;
        cell.has_surface = opened_sign_mask != 0U && opened_sign_mask != 0xFFU;
        cell.is_active = opened_inside[leaf_index] != 0U && cell.has_surface;
        if (!cell.is_active) {
            cell.representative_vertex_index = -1;
        }
    }
}

inline HermiteSample project_opened_boundary_sample(
    const BoundingBox &bounds,
    const Vector3d &face_center,
    const Vector3d &outward_normal,
    std::span<const std::size_t> contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue) {
    const double cell_size = bounds.max.x - bounds.min.x;
    const double search_distance = cell_size;
    Vector3d a = {
        face_center.x - outward_normal.x * search_distance,
        face_center.y - outward_normal.y * search_distance,
        face_center.z - outward_normal.z * search_distance,
    };
    Vector3d b = {
        face_center.x + outward_normal.x * search_distance,
        face_center.y + outward_normal.y * search_distance,
        face_center.z + outward_normal.z * search_distance,
    };

    double fa = evaluate_field_at_point(a, contributors, positions, smoothing_lengths) - isovalue;
    double fb = evaluate_field_at_point(b, contributors, positions, smoothing_lengths) - isovalue;
    Vector3d projected = face_center;

    if (fa == 0.0) {
        projected = a;
    } else if (fb == 0.0) {
        projected = b;
    } else if ((fa < 0.0 && fb > 0.0) || (fa > 0.0 && fb < 0.0)) {
        for (int iter = 0; iter < 12; ++iter) {
            const Vector3d mid = {
                0.5 * (a.x + b.x),
                0.5 * (a.y + b.y),
                0.5 * (a.z + b.z),
            };
            const double fm =
                evaluate_field_at_point(mid, contributors, positions, smoothing_lengths) - isovalue;
            projected = mid;
            if ((fa < 0.0 && fm > 0.0) || (fa > 0.0 && fm < 0.0)) {
                b = mid;
                fb = fm;
            } else {
                a = mid;
                fa = fm;
            }
        }
    }

    Vector3d normal = outward_normal;
    const Vector3d gradient = evaluate_field_gradient_at_point(
        projected, contributors, positions, smoothing_lengths);
    const double grad_mag = std::sqrt(
        gradient.x * gradient.x +
        gradient.y * gradient.y +
        gradient.z * gradient.z);
    if (grad_mag > 1e-12) {
        normal = {
            -gradient.x / grad_mag,
            -gradient.y / grad_mag,
            -gradient.z / grad_mag,
        };
    }

    return {projected, normal};
}

inline std::vector<MeshVertex> solve_opened_leaf_vertices(
    std::vector<OctreeCell> &all_cells,
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &opened_inside,
    const std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue) {
    std::vector<MeshVertex> vertices;
    vertices.reserve(solid_leaves.size());

    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        OctreeCell &cell = all_cells[solid_leaves[leaf_index].cell_index];
        cell.representative_vertex_index = -1;

        if (!cell.is_topo_surface) {
            continue;
        }

        const std::span<const std::size_t> contributors =
            cell_contributor_span(cell, all_contributors);
        std::vector<HermiteSample> samples;
        samples.reserve(6);
        const Vector3d center = cell.bounds.center();

        for (std::size_t face = 0; face < 6U; ++face) {
            const std::int64_t neighbor_leaf_index =
                solid_leaves[leaf_index].face_neighbor_leaf_indices[face];
            const bool self_inside = opened_inside[leaf_index] != 0U;
            const bool neighbor_inside =
                neighbor_leaf_index >= 0 &&
                opened_inside[static_cast<std::size_t>(neighbor_leaf_index)] != 0U;
            const bool touches_interface =
                neighbor_leaf_index < 0 || self_inside != neighbor_inside;
            if (!touches_interface) {
                continue;
            }

            Vector3d face_center = center;
            Vector3d outward_normal = {0.0, 0.0, 0.0};
            switch (face) {
                case 0:
                    face_center.x = cell.bounds.min.x;
                    outward_normal.x = -1.0;
                    break;
                case 1:
                    face_center.x = cell.bounds.max.x;
                    outward_normal.x = 1.0;
                    break;
                case 2:
                    face_center.y = cell.bounds.min.y;
                    outward_normal.y = -1.0;
                    break;
                case 3:
                    face_center.y = cell.bounds.max.y;
                    outward_normal.y = 1.0;
                    break;
                case 4:
                    face_center.z = cell.bounds.min.z;
                    outward_normal.z = -1.0;
                    break;
                case 5:
                    face_center.z = cell.bounds.max.z;
                    outward_normal.z = 1.0;
                    break;
                default:
                    break;
            }

            if (!self_inside) {
                outward_normal.x = -outward_normal.x;
                outward_normal.y = -outward_normal.y;
                outward_normal.z = -outward_normal.z;
            }

            samples.push_back(project_opened_boundary_sample(
                cell.bounds, face_center, outward_normal,
                contributors, positions, smoothing_lengths, isovalue));
        }

        if (samples.empty()) {
            Vector3d normal = {0.0, 0.0, 0.0};
            const Vector3d gradient = evaluate_field_gradient_at_point(
                center, contributors, positions, smoothing_lengths);
            const double grad_mag = std::sqrt(
                gradient.x * gradient.x +
                gradient.y * gradient.y +
                gradient.z * gradient.z);
            if (grad_mag > 1e-12) {
                normal = {
                    -gradient.x / grad_mag,
                    -gradient.y / grad_mag,
                    -gradient.z / grad_mag,
                };
            }

            cell.representative_vertex_index =
                static_cast<std::int64_t>(vertices.size());
            vertices.push_back({center, normal});
            continue;
        }

        const MeshVertex vertex = solve_qef_for_leaf(samples, cell.bounds);
        cell.representative_vertex_index =
            static_cast<std::int64_t>(vertices.size());
        vertices.push_back(vertex);
    }

    return vertices;
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_ADAPTIVE_SOLID_HPP_
