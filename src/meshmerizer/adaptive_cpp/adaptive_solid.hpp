#ifndef MESHMERIZER_ADAPTIVE_CPP_ADAPTIVE_SOLID_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_ADAPTIVE_SOLID_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cancellation.hpp"
#include "faces.hpp"
#include "octree_cell.hpp"

enum class OccupancyState : std::uint8_t {
    kOutside = 0U,
    kInside = 1U,
    kBoundaryInside = 2U,
    kBoundaryOutside = 3U,
};

struct OccupiedSolidLeaf {
    std::size_t cell_index;
    double center_value;
    double cell_size;
    std::uint32_t depth;
    OccupancyState occupancy;
    std::array<std::int64_t, 6> face_neighbor_leaf_indices;
};

struct OccupiedSolidClassificationCache {
    std::vector<std::uint8_t> inside_flags;
    std::vector<double> center_values;
    std::vector<std::uint8_t> occupancy_states;
    std::vector<std::array<std::size_t, 6>> face_neighbor_cell_indices;
};

struct OpenedBoundarySample {
    Vector3d position;
    Vector3d outward_normal;
    std::size_t leaf_index;
};

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

inline std::span<const std::size_t> cell_contributor_span(
    const OctreeCell &cell,
    const std::vector<std::size_t> &all_contributors) {
    const auto [begin_idx, end_idx] =
        cell_contributor_bounds(cell, all_contributors);
    return std::span<const std::size_t>(
        all_contributors.data() + begin_idx,
        end_idx - begin_idx);
}

inline double cell_edge_length(const OctreeCell &cell) {
    return cell.bounds.max.x - cell.bounds.min.x;
}

inline bool refine_surface_band_cells(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue,
    std::uint32_t max_depth,
    double max_surface_leaf_size,
    std::uint32_t minimum_usable_hermite_samples,
    double max_qef_rms_residual_ratio,
    double min_normal_alignment_threshold) {
    if (max_surface_leaf_size <= 0.0) {
        return false;
    }

    std::queue<std::size_t> cells_to_visit;
    for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
        const OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf || !cell.has_surface || cell.depth >= max_depth) {
            continue;
        }
        if (cell_edge_length(cell) > max_surface_leaf_size) {
            cells_to_visit.push(cell_idx);
        }
    }

    if (cells_to_visit.empty()) {
        meshmerizer_log_detail::print_status(
            "Regularization",
            "refine_surface_band_cells",
            "no surface-band splits needed (leaf_size_target=%.6g, total_cells=%zu)\n",
            max_surface_leaf_size,
            all_cells.size());
        return false;
    }

    meshmerizer_log_detail::print_status(
        "Regularization",
        "refine_surface_band_cells",
        "starting from %zu surface-band cells to reach leaf_size<=%.6g (total_cells_before=%zu)\n",
        cells_to_visit.size(),
        max_surface_leaf_size,
        all_cells.size());

    ProgressCounter refine_counter(
        "Regularization", "refine_surface_band_cells", "cells", 100);
    std::size_t split_count = 0U;
    std::size_t processed_count = 0U;

    while (!cells_to_visit.empty()) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(
            processed_count + split_count + 1U);
        const std::size_t cell_idx = cells_to_visit.front();
        cells_to_visit.pop();
        refine_counter.tick();
        ++processed_count;

        if (cell_idx >= all_cells.size()) {
            continue;
        }

        OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf || !cell.has_surface || cell.depth >= max_depth) {
            continue;
        }
        if (cell_edge_length(cell) <= max_surface_leaf_size) {
            continue;
        }

        split_octree_leaf(
            cell_idx, all_cells, all_contributors,
            positions, smoothing_lengths);
        ++split_count;

        const std::int64_t child_begin = all_cells[cell_idx].child_begin;
        if (child_begin < 0) {
            continue;
        }

        for (std::size_t child_offset = 0; child_offset < 8U; ++child_offset) {
            const std::size_t child_index =
                static_cast<std::size_t>(child_begin) + child_offset;
            if (child_index >= all_cells.size()) {
                continue;
            }

            OctreeCell &child = all_cells[child_index];
            const std::span<const std::size_t> contributors =
                cell_contributor_span(child, all_contributors);
            child.is_active = false;
            child.is_topo_surface = false;
            child.representative_vertex_index = -1;

            if (contributors.size() < 2U) {
                child.has_surface = false;
                child.corner_values = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                child.corner_sign_mask = 0U;
                continue;
            }

            child.corner_values = sample_cell_corners(
                child, contributors, positions, smoothing_lengths);
            child.corner_sign_mask = compute_corner_sign_mask(
                child.corner_values, isovalue);

            const bool corner_surface = cell_may_contain_isosurface(
                child.corner_values, isovalue);
            const bool inherited_surface_hint =
                all_cells[cell_idx].has_surface && !corner_surface;
            child.has_surface = corner_surface || inherited_surface_hint;
            if (!child.has_surface) {
                continue;
            }

            bool should_continue_refining =
                child.depth < max_depth &&
                cell_edge_length(child) > max_surface_leaf_size;

            if (!should_continue_refining &&
                child.depth < max_depth && corner_surface) {
                const std::vector<HermiteSample> samples =
                    compute_cell_hermite_samples(
                        child.bounds, child.corner_values,
                        child.corner_sign_mask, contributors,
                        positions, smoothing_lengths, isovalue);
                const QEFLeafDiagnostics qef_diagnostics =
                    analyze_qef_for_leaf(samples, child.bounds);
                const double dx = child.bounds.max.x - child.bounds.min.x;
                const double dy = child.bounds.max.y - child.bounds.min.y;
                const double dz = child.bounds.max.z - child.bounds.min.z;
                const double cell_radius =
                    0.5 * std::sqrt(dx * dx + dy * dy + dz * dz);
                const bool poor_qef_fit =
                    qef_diagnostics.usable_sample_count <
                        minimum_usable_hermite_samples ||
                    qef_diagnostics.used_fallback ||
                    minimum_hermite_normal_alignment(samples) <
                        min_normal_alignment_threshold ||
                    qef_diagnostics.rms_plane_residual >
                        max_qef_rms_residual_ratio * cell_radius;
                should_continue_refining = poor_qef_fit;
            }

            if (should_continue_refining) {
                cells_to_visit.push(child_index);
            } else {
                child.is_active = true;
            }
        }
    }

    refine_counter.finish();

    meshmerizer_log_detail::print_status(
        "Regularization",
        "refine_surface_band_cells",
        "processed=%zu split=%zu (total_cells_after_refine=%zu)\n",
        processed_count,
        split_count,
        all_cells.size());
    return split_count > 0U;
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

inline std::vector<OccupiedSolidLeaf> classify_occupied_solid_leaves(
    const std::vector<OctreeCell> &all_cells,
    const std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const LeafSpatialIndex &spatial_index,
    double isovalue,
    std::uint32_t max_depth,
    OccupiedSolidClassificationCache *cache = nullptr,
    const std::vector<std::uint8_t> *dirty_cells = nullptr,
    std::vector<std::uint8_t> *inside_mask = nullptr) {
    std::vector<OccupiedSolidLeaf> solid_leaves;
    solid_leaves.reserve(all_cells.size());
    std::vector<std::uint8_t> local_inside_flags;
    std::vector<double> local_center_values;
    if (cache == nullptr) {
        local_inside_flags.assign(all_cells.size(), 0U);
        local_center_values.assign(all_cells.size(), 0.0);
    } else {
        cache->inside_flags.resize(all_cells.size(), 0U);
        cache->center_values.resize(all_cells.size(), 0.0);
        cache->occupancy_states.resize(
            all_cells.size(),
            static_cast<std::uint8_t>(OccupancyState::kOutside));
        cache->face_neighbor_cell_indices.resize(
            all_cells.size(),
            {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX});
    }
    std::vector<std::uint8_t> &inside_flags =
        cache == nullptr ? local_inside_flags : cache->inside_flags;
    std::vector<double> &center_values =
        cache == nullptr ? local_center_values : cache->center_values;
    std::vector<std::int64_t> cell_to_leaf_index(all_cells.size(), -1);
    std::vector<std::uint8_t> leaf_is_included(all_cells.size(), 0U);
    std::vector<OccupiedSolidLeaf> leaf_buffer(all_cells.size());
    std::vector<std::uint8_t> local_occupancy_states(
        all_cells.size(),
        static_cast<std::uint8_t>(OccupancyState::kOutside));
    std::vector<std::array<std::size_t, 6>> local_face_neighbor_cell_indices(
        all_cells.size(),
        {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX});
    std::vector<std::uint8_t> &occupancy_states = cache == nullptr ?
        local_occupancy_states : cache->occupancy_states;
    std::vector<std::array<std::size_t, 6>> &face_neighbor_cell_indices =
        cache == nullptr ? local_face_neighbor_cell_indices :
                           cache->face_neighbor_cell_indices;
    std::vector<std::uint8_t> occupancy_dirty;

    if (dirty_cells != nullptr) {
        occupancy_dirty = *dirty_cells;
        if (occupancy_dirty.size() < all_cells.size()) {
            occupancy_dirty.resize(all_cells.size(), 0U);
        }
        for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
            meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
            if (cell_idx >= occupancy_dirty.size() || occupancy_dirty[cell_idx] == 0U) {
                continue;
            }
            const auto neighbors = face_neighbor_cells(
                all_cells[cell_idx], spatial_index, max_depth);
            for (std::size_t neighbor_idx : neighbors) {
                if (neighbor_idx == SIZE_MAX || neighbor_idx >= occupancy_dirty.size()) {
                    continue;
                }
                occupancy_dirty[neighbor_idx] = 1U;
            }
        }
    }

    if (inside_mask != nullptr) {
        inside_mask->clear();
        inside_mask->reserve(all_cells.size());
    }

    ProgressCounter classify_counter(
        "Regularization", "classify_occupied_solid_leaves", "cells", 1000);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                cell_idx)) {
            classify_counter.tick();
            continue;
        }
        classify_counter.tick();
        const OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf) {
            continue;
        }

        const bool reuse_cached_value =
            dirty_cells != nullptr && cell_idx < dirty_cells->size() &&
            (*dirty_cells)[cell_idx] == 0U;
        if (reuse_cached_value) {
            continue;
        }

        const std::span<const std::size_t> contributors =
            cell_contributor_span(cell, all_contributors);
        const double center_value = evaluate_field_at_point(
            cell.bounds.center(), contributors, positions, smoothing_lengths);
        center_values[cell_idx] = center_value;

        std::size_t inside_corner_count = 0;
        for (double corner_value : cell.corner_values) {
            if (corner_value >= isovalue) {
                ++inside_corner_count;
            }
        }

        const bool inside =
            center_value >= isovalue || inside_corner_count >= 4U;
        inside_flags[cell_idx] = inside ? 1U : 0U;
    }

    classify_counter.finish();

    ProgressCounter occupancy_counter(
        "Regularization", "classify_occupied_solid_leaves", "cells", 1000);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                cell_idx)) {
            occupancy_counter.tick();
            continue;
        }
        occupancy_counter.tick();
        const OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf) {
            continue;
        }

        const bool inside = inside_flags[cell_idx] != 0U;
        OccupancyState occupancy = OccupancyState::kOutside;
        if (dirty_cells != nullptr && cell_idx < occupancy_dirty.size() &&
            occupancy_dirty[cell_idx] == 0U) {
            occupancy = static_cast<OccupancyState>(occupancy_states[cell_idx]);
        } else {
            bool touches_opposite = false;
            const auto neighbors = face_neighbor_cells(
                cell, spatial_index, max_depth);
            face_neighbor_cell_indices[cell_idx] = neighbors;
            for (std::size_t neighbor_idx : neighbors) {
                if (neighbor_idx == SIZE_MAX || neighbor_idx >= all_cells.size()) {
                    continue;
                }
                if (inside_flags[neighbor_idx] != inside_flags[cell_idx]) {
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
            occupancy_states[cell_idx] =
                static_cast<std::uint8_t>(occupancy);
        }

        leaf_is_included[cell_idx] = 1U;
        leaf_buffer[cell_idx] = {
            cell_idx,
            center_values[cell_idx],
            cell_edge_length(cell),
            cell.depth,
            occupancy,
            {-1, -1, -1, -1, -1, -1},
        };
    }
    occupancy_counter.finish();

    for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
        if (leaf_is_included[cell_idx] == 0U) {
            continue;
        }
        solid_leaves.push_back(leaf_buffer[cell_idx]);
        cell_to_leaf_index[cell_idx] =
            static_cast<std::int64_t>(solid_leaves.size() - 1U);
        if (inside_mask != nullptr) {
            inside_mask->push_back(inside_flags[cell_idx] != 0U ? 1U : 0U);
        }
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
        const auto &neighbors = face_neighbor_cell_indices[leaf.cell_index];
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

    return solid_leaves;
}

inline std::vector<std::uint8_t> build_inside_mask(
    const std::vector<OccupiedSolidLeaf> &solid_leaves) {
    std::vector<std::uint8_t> inside_mask(solid_leaves.size(), 0U);
    ProgressCounter mask_counter(
        "Regularization", "build_inside_mask", "leaves", 1000);
    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(leaf_index);
        mask_counter.tick();
        inside_mask[leaf_index] =
            (solid_leaves[leaf_index].occupancy == OccupancyState::kInside ||
             solid_leaves[leaf_index].occupancy == OccupancyState::kBoundaryInside)
                ? 1U
                : 0U;
    }
    mask_counter.finish();
    return inside_mask;
}

inline std::vector<double> compute_outside_distance_from_inside_mask(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &inside_mask,
    double max_distance = std::numeric_limits<double>::infinity()) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> distance_from_inside(solid_leaves.size(), inf);
    using QueueEntry = std::pair<double, std::size_t>;
    std::priority_queue<
        QueueEntry,
        std::vector<QueueEntry>,
        std::greater<QueueEntry>> queue;

    ProgressCounter seed_counter(
        "Regularization", "compute_outside_distance_from_inside_mask",
        "leaves", 1000);
    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(leaf_index);
        seed_counter.tick();
        if (inside_mask[leaf_index] == 0U) {
            continue;
        }

        bool is_boundary_seed = false;
        for (std::int64_t neighbor_leaf_index :
             solid_leaves[leaf_index].face_neighbor_leaf_indices) {
            if (neighbor_leaf_index < 0) {
                is_boundary_seed = true;
                break;
            }

            const std::size_t neighbor =
                static_cast<std::size_t>(neighbor_leaf_index);
            if (inside_mask[neighbor] == 0U) {
                is_boundary_seed = true;
                break;
            }
        }

        if (is_boundary_seed) {
            distance_from_inside[leaf_index] = 0.0;
            queue.push({0.0, leaf_index});
        }
    }
    seed_counter.finish();

    ProgressCounter wavefront_counter(
        "Regularization", "compute_outside_distance_from_inside_mask",
        "queue pops", 10000);
    while (!queue.empty()) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(queue.size());
        wavefront_counter.tick();
        const auto [distance, leaf_index] = queue.top();
        queue.pop();
        if (distance > distance_from_inside[leaf_index]) {
            continue;
        }
        if (distance > max_distance) {
            continue;
        }

        for (std::int64_t neighbor_leaf_index :
             solid_leaves[leaf_index].face_neighbor_leaf_indices) {
            if (neighbor_leaf_index < 0) {
                continue;
            }

            const std::size_t neighbor =
                static_cast<std::size_t>(neighbor_leaf_index);
            if (inside_mask[neighbor] != 0U) {
                continue;
            }

            const double source_size = std::max(
                std::min(solid_leaves[leaf_index].cell_size, max_distance),
                0.0);
            const double neighbor_size = std::max(
                std::min(solid_leaves[neighbor].cell_size, max_distance),
                0.0);
            const double edge_cost = 0.5 * (
                source_size + neighbor_size);
            const double candidate = distance + edge_cost;
            if (candidate > max_distance) {
                continue;
            }
            if (candidate < distance_from_inside[neighbor]) {
                distance_from_inside[neighbor] = candidate;
                queue.push({candidate, neighbor});
            }
        }
    }
    wavefront_counter.finish();

    return distance_from_inside;
}

inline std::vector<std::uint8_t> dilate_inside_mask(
    const std::vector<std::uint8_t> &inside_mask,
    const std::vector<double> &distance_from_inside,
    double dilation_radius) {
    std::vector<std::uint8_t> dilated_inside(inside_mask.size(), 0U);

    ProgressCounter dilate_counter(
        "Regularization", "dilate_inside_mask", "leaves", 1000);
#pragma omp parallel for schedule(static)
    for (std::size_t leaf_index = 0; leaf_index < inside_mask.size(); ++leaf_index) {
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                leaf_index)) {
            dilate_counter.tick();
            continue;
        }
        dilate_counter.tick();
        dilated_inside[leaf_index] =
            (inside_mask[leaf_index] != 0U ||
             distance_from_inside[leaf_index] <= dilation_radius)
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
    double target_leaf_size,
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &inside_mask,
    const std::vector<double> &distance_from_inside,
    double thickening_radius,
    std::uint32_t minimum_usable_hermite_samples,
    double max_qef_rms_residual_ratio,
    double min_normal_alignment_threshold,
    std::vector<std::uint8_t> *dirty_cells = nullptr) {
    if (target_leaf_size <= 0.0 || thickening_radius <= 0.0) {
        return false;
    }

    const double refinement_halo_radius =
        thickening_radius + 2.0 * target_leaf_size;

    const auto target_depth_for_cell =
        [&](const OccupiedSolidLeaf &leaf) -> std::uint32_t {
            std::uint32_t target_depth = leaf.depth;
            double size = leaf.cell_size;
            while (target_depth < max_depth && size > target_leaf_size) {
                size *= 0.5;
                ++target_depth;
            }
            return target_depth;
        };

    std::queue<std::size_t> cells_to_visit;
    std::vector<std::uint32_t> queued_target_depths(all_cells.size(), 0U);
    std::vector<std::uint8_t> enqueued_cells(all_cells.size(), 0U);
    std::vector<std::size_t> band_seed_leaf_indices;

    auto enqueue_cell = [&](std::size_t cell_index, std::uint32_t target_depth) {
        if (cell_index >= all_cells.size()) {
            return;
        }
        if (cell_index >= queued_target_depths.size()) {
            queued_target_depths.resize(all_cells.size(), 0U);
        }
        if (cell_index >= enqueued_cells.size()) {
            enqueued_cells.resize(all_cells.size(), 0U);
        }
        if (enqueued_cells[cell_index] == 0U) {
            cells_to_visit.push(cell_index);
            enqueued_cells[cell_index] = 1U;
            queued_target_depths[cell_index] = target_depth;
            return;
        }
        queued_target_depths[cell_index] = std::max(
            queued_target_depths[cell_index], target_depth);
    };

    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(leaf_index);
        if (inside_mask[leaf_index] != 0U) {
            continue;
        }
        if (distance_from_inside[leaf_index] > refinement_halo_radius) {
            continue;
        }
        if (solid_leaves[leaf_index].depth >= max_depth ||
            solid_leaves[leaf_index].cell_size <= target_leaf_size) {
            continue;
        }
        const std::size_t cell_index = solid_leaves[leaf_index].cell_index;
        enqueue_cell(
            cell_index, target_depth_for_cell(solid_leaves[leaf_index]));
        band_seed_leaf_indices.push_back(leaf_index);
    }

    for (std::size_t leaf_index : band_seed_leaf_indices) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(leaf_index);
        for (std::int64_t neighbor_leaf_index :
             solid_leaves[leaf_index].face_neighbor_leaf_indices) {
            if (neighbor_leaf_index < 0) {
                continue;
            }
            const std::size_t neighbor =
                static_cast<std::size_t>(neighbor_leaf_index);
            if (inside_mask[neighbor] != 0U) {
                continue;
            }
            if (solid_leaves[neighbor].depth >= max_depth ||
                solid_leaves[neighbor].cell_size <= target_leaf_size) {
                continue;
            }
            enqueue_cell(
                solid_leaves[neighbor].cell_index,
                target_depth_for_cell(solid_leaves[neighbor]));
        }
    }

    if (cells_to_visit.empty()) {
        meshmerizer_log_detail::print_status(
            "Regularization",
            "refine_thickening_band_cells",
            "no growth-band splits needed (leaf_size_target=%.6g, radius=%.6g, halo_radius=%.6g, total_cells=%zu)\n",
            target_leaf_size,
            thickening_radius,
            refinement_halo_radius,
            all_cells.size());
        return false;
    }

    meshmerizer_log_detail::print_status(
        "Regularization",
        "refine_thickening_band_cells",
        "starting from %zu growth-band cells to reach leaf_size<=%.6g (radius=%.6g, halo_radius=%.6g, total_cells_before=%zu)\n",
        cells_to_visit.size(),
        target_leaf_size,
        thickening_radius,
        refinement_halo_radius,
        all_cells.size());

    ProgressCounter refine_counter(
        "Regularization", "refine_thickening_band_cells", "cells", 100);
    std::size_t split_count = 0U;
    std::size_t processed_count = 0U;

    auto mark_dirty = [&](std::size_t cell_index) {
        if (dirty_cells == nullptr) {
            return;
        }
        if (dirty_cells->size() <= cell_index) {
            dirty_cells->resize(cell_index + 1U, 0U);
        }
        (*dirty_cells)[cell_index] = 1U;
    };

    while (!cells_to_visit.empty()) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(
            processed_count + split_count + 1U);
        const std::size_t cell_idx = cells_to_visit.front();
        cells_to_visit.pop();
        if (cell_idx < enqueued_cells.size()) {
            enqueued_cells[cell_idx] = 0U;
        }
        refine_counter.tick();
        ++processed_count;

        if (cell_idx >= all_cells.size()) {
            continue;
        }

        OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf || cell.depth >= max_depth ||
            cell_edge_length(cell) <= target_leaf_size) {
            continue;
        }

        const std::uint32_t target_depth =
            cell_idx < queued_target_depths.size() ?
                queued_target_depths[cell_idx] :
                cell.depth;

        split_octree_leaf(
            cell_idx, all_cells, all_contributors, positions,
            smoothing_lengths);
        ++split_count;
        mark_dirty(cell_idx);

        const std::int64_t child_begin = all_cells[cell_idx].child_begin;
        if (child_begin < 0) {
            continue;
        }

        for (std::size_t child_offset = 0; child_offset < 8U; ++child_offset) {
            const std::size_t child_index =
                static_cast<std::size_t>(child_begin) + child_offset;
            if (child_index >= all_cells.size()) {
                continue;
            }
            mark_dirty(child_index);

            OctreeCell &child = all_cells[child_index];
            const std::span<const std::size_t> contributors =
                cell_contributor_span(child, all_contributors);
            child.is_active = false;
            child.is_topo_surface = false;
            child.representative_vertex_index = -1;

            if (contributors.size() < 2U) {
                child.has_surface = false;
                child.corner_values = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                child.corner_sign_mask = 0U;
                continue;
            }

            child.corner_values = sample_cell_corners(
                child, contributors, positions, smoothing_lengths);
            child.corner_sign_mask = compute_corner_sign_mask(
                child.corner_values, isovalue);

            const bool corner_surface = cell_may_contain_isosurface(
                child.corner_values, isovalue);
            const bool inherited_surface_hint =
                all_cells[cell_idx].has_surface && !corner_surface;
            child.has_surface = corner_surface || inherited_surface_hint;

            bool should_continue_refining =
                child.depth < max_depth &&
                (cell_edge_length(child) > target_leaf_size ||
                 child.depth < target_depth);

            if (!should_continue_refining && child.depth < max_depth &&
                corner_surface) {
                const std::vector<HermiteSample> samples =
                    compute_cell_hermite_samples(
                        child.bounds, child.corner_values,
                        child.corner_sign_mask, contributors,
                        positions, smoothing_lengths, isovalue);
                const QEFLeafDiagnostics qef_diagnostics =
                    analyze_qef_for_leaf(samples, child.bounds);
                const double dx = child.bounds.max.x - child.bounds.min.x;
                const double dy = child.bounds.max.y - child.bounds.min.y;
                const double dz = child.bounds.max.z - child.bounds.min.z;
                const double cell_radius =
                    0.5 * std::sqrt(dx * dx + dy * dy + dz * dz);
                const bool poor_qef_fit =
                    qef_diagnostics.usable_sample_count <
                        minimum_usable_hermite_samples ||
                    qef_diagnostics.used_fallback ||
                    minimum_hermite_normal_alignment(samples) <
                        min_normal_alignment_threshold ||
                    qef_diagnostics.rms_plane_residual >
                        max_qef_rms_residual_ratio * cell_radius;
                should_continue_refining = poor_qef_fit;
            }

            if (should_continue_refining) {
                enqueue_cell(child_index, target_depth);
            } else {
                child.is_active = true;
            }
        }
    }

    refine_counter.finish();

    meshmerizer_log_detail::print_status(
        "Regularization",
        "refine_thickening_band_cells",
        "processed=%zu split=%zu (total_cells_after_refine=%zu)\n",
        processed_count,
        split_count,
        all_cells.size());
    return split_count > 0U;
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

    meshmerizer_log_detail::print_status(
        "Regularization",
        "thickening_band_is_fully_refined",
        "unresolved strict-band leaves=%zu (target_leaf_size=%.6g, radius=%.6g)\n",
        unresolved_count,
        target_leaf_size,
        thickening_radius);

    return unresolved_count == 0U;
}

inline std::vector<double> compute_inside_clearance(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &inside_mask,
    double max_distance = std::numeric_limits<double>::infinity()) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> clearance(solid_leaves.size(), inf);
    using QueueEntry = std::pair<double, std::size_t>;
    std::priority_queue<
        QueueEntry,
        std::vector<QueueEntry>,
        std::greater<QueueEntry>> queue;

    ProgressCounter seed_counter(
        "Regularization", "compute_inside_clearance", "leaves", 1000);
    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(leaf_index);
        seed_counter.tick();
        if (inside_mask[leaf_index] == 0U) {
            continue;
        }

        bool is_boundary_seed = false;
        for (std::int64_t neighbor_leaf_index :
             solid_leaves[leaf_index].face_neighbor_leaf_indices) {
            if (neighbor_leaf_index < 0) {
                is_boundary_seed = true;
                break;
            }

            const std::size_t neighbor =
                static_cast<std::size_t>(neighbor_leaf_index);
            if (inside_mask[neighbor] == 0U) {
                is_boundary_seed = true;
                break;
            }
        }

        if (is_boundary_seed) {
            clearance[leaf_index] = 0.0;
            queue.push({0.0, leaf_index});
        }
    }
    seed_counter.finish();

    ProgressCounter wavefront_counter(
        "Regularization", "compute_inside_clearance", "queue pops", 10000);

    while (!queue.empty()) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(queue.size());
        wavefront_counter.tick();
        const auto [distance, leaf_index] = queue.top();
        queue.pop();
        if (distance > clearance[leaf_index]) {
            continue;
        }
        if (distance > max_distance) {
            continue;
        }

        for (std::int64_t neighbor_leaf_index :
             solid_leaves[leaf_index].face_neighbor_leaf_indices) {
            if (neighbor_leaf_index < 0) {
                continue;
            }
            const std::size_t neighbor =
                static_cast<std::size_t>(neighbor_leaf_index);
            if (inside_mask[neighbor] == 0U) {
                continue;
            }

            const double edge_cost = 0.5 * (
                solid_leaves[leaf_index].cell_size +
                solid_leaves[neighbor].cell_size);
            const double candidate = distance + edge_cost;
            if (candidate > max_distance) {
                continue;
            }
            if (candidate < clearance[neighbor]) {
                clearance[neighbor] = candidate;
                queue.push({candidate, neighbor});
            }
        }
    }

    wavefront_counter.finish();

    return clearance;
}

inline std::vector<std::uint8_t> erode_occupied_solid_leaves(
    const std::vector<std::uint8_t> &inside_mask,
    const std::vector<double> &clearance,
    double erosion_radius) {
    std::vector<std::uint8_t> kept_inside(inside_mask.size(), 0U);

    ProgressCounter erode_counter(
        "Regularization", "erode_occupied_solid_leaves", "leaves", 1000);
#pragma omp parallel for schedule(static)
    for (std::size_t leaf_index = 0; leaf_index < inside_mask.size(); ++leaf_index) {
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                leaf_index)) {
            erode_counter.tick();
            continue;
        }
        erode_counter.tick();
        if (inside_mask[leaf_index] == 0U) {
            continue;
        }
        kept_inside[leaf_index] = clearance[leaf_index] >= erosion_radius ? 1U : 0U;
    }
    erode_counter.finish();
    return kept_inside;
}

inline std::vector<double> compute_distance_to_eroded_solid(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &eroded_inside,
    double max_distance = std::numeric_limits<double>::infinity()) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> distance_to_eroded(solid_leaves.size(), inf);
    using QueueEntry = std::pair<double, std::size_t>;
    std::priority_queue<
        QueueEntry,
        std::vector<QueueEntry>,
        std::greater<QueueEntry>> queue;

    ProgressCounter seed_counter(
        "Regularization", "compute_distance_to_eroded_solid", "leaves", 1000);
    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(leaf_index);
        seed_counter.tick();
        if (eroded_inside[leaf_index] == 0U) {
            continue;
        }

        bool is_boundary_seed = false;
        for (std::int64_t neighbor_leaf_index :
             solid_leaves[leaf_index].face_neighbor_leaf_indices) {
            if (neighbor_leaf_index < 0) {
                is_boundary_seed = true;
                break;
            }
            if (eroded_inside[static_cast<std::size_t>(neighbor_leaf_index)] == 0U) {
                is_boundary_seed = true;
                break;
            }
        }

        if (is_boundary_seed) {
            distance_to_eroded[leaf_index] = 0.0;
            queue.push({0.0, leaf_index});
        }
    }
    seed_counter.finish();

    ProgressCounter wavefront_counter(
        "Regularization", "compute_distance_to_eroded_solid", "queue pops", 10000);

    while (!queue.empty()) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(queue.size());
        wavefront_counter.tick();
        const auto [distance, leaf_index] = queue.top();
        queue.pop();
        if (distance > distance_to_eroded[leaf_index]) {
            continue;
        }
        if (distance > max_distance) {
            continue;
        }

        for (std::int64_t neighbor_leaf_index :
             solid_leaves[leaf_index].face_neighbor_leaf_indices) {
            if (neighbor_leaf_index < 0) {
                continue;
            }
            const std::size_t neighbor =
                static_cast<std::size_t>(neighbor_leaf_index);
            if (eroded_inside[neighbor] != 0U) {
                continue;
            }

            const double edge_cost = 0.5 * (
                solid_leaves[leaf_index].cell_size +
                solid_leaves[neighbor].cell_size);
            const double candidate = distance + edge_cost;
            if (candidate > max_distance) {
                continue;
            }
            if (candidate < distance_to_eroded[neighbor]) {
                distance_to_eroded[neighbor] = candidate;
                queue.push({candidate, neighbor});
            }
        }
    }

    wavefront_counter.finish();

    return distance_to_eroded;
}

inline std::vector<std::uint8_t> dilate_eroded_solid_leaves(
    const std::vector<std::uint8_t> &eroded_inside,
    const std::vector<double> &distance_to_eroded,
    double dilation_radius) {
    std::vector<std::uint8_t> opened_inside(eroded_inside.size(), 0U);

    ProgressCounter dilate_counter(
        "Regularization", "dilate_eroded_solid_leaves", "leaves", 1000);
#pragma omp parallel for schedule(static)
    for (std::size_t leaf_index = 0; leaf_index < eroded_inside.size(); ++leaf_index) {
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                leaf_index)) {
            dilate_counter.tick();
            continue;
        }
        dilate_counter.tick();
        opened_inside[leaf_index] =
            (eroded_inside[leaf_index] != 0U ||
             distance_to_eroded[leaf_index] <= dilation_radius)
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
        meshmerizer_log_detail::print_status(
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
        meshmerizer_log_detail::print_status(
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

    meshmerizer_log_detail::print_status(
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
        meshmerizer_log_detail::print_status(
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
    std::uint32_t max_depth) {
    OpenedSurfaceMesh mesh;
    std::unordered_map<std::uint64_t, std::size_t> vertex_lookup;
    std::vector<Vector3d> normal_accum;
    vertex_lookup.reserve(solid_leaves.size() * 8U);
    ProgressCounter surface_counter(
        "Meshing", "generate_opened_surface_mesh", "leaves", 1000);

    const std::uint32_t fine_resolution =
        base_resolution * (1U << max_depth);
    const double fine_dx =
        (domain.max.x - domain.min.x) / static_cast<double>(fine_resolution);
    const double fine_dy =
        (domain.max.y - domain.min.y) / static_cast<double>(fine_resolution);
    const double fine_dz =
        (domain.max.z - domain.min.z) / static_cast<double>(fine_resolution);

    std::vector<std::int64_t> cell_to_leaf_index(all_cells.size(), -1);
    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        cell_to_leaf_index[solid_leaves[leaf_index].cell_index] =
            static_cast<std::int64_t>(leaf_index);
    }

    const auto vertex_index_for = [&](std::uint32_t ix, std::uint32_t iy,
                                      std::uint32_t iz) {
        const std::uint64_t key = pack_surface_corner_coords(ix, iy, iz);
        auto it = vertex_lookup.find(key);
        if (it != vertex_lookup.end()) {
            return it->second;
        }
        const Vector3d position = {
            domain.min.x + static_cast<double>(ix) * fine_dx,
            domain.min.y + static_cast<double>(iy) * fine_dy,
            domain.min.z + static_cast<double>(iz) * fine_dz,
        };
        const std::size_t index = mesh.vertices.size();
        mesh.vertices.push_back({position, {0.0, 0.0, 0.0}});
        mesh.vertex_keys.push_back(key);
        normal_accum.push_back({0.0, 0.0, 0.0});
        vertex_lookup[key] = index;
        return index;
    };

    const auto emit_oriented_quad = [&](const std::array<Vector3d, 4> &positions,
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

        mesh.triangles.push_back({
            static_cast<std::uint32_t>(oriented[0]),
            static_cast<std::uint32_t>(oriented[1]),
            static_cast<std::uint32_t>(oriented[2]),
        });
        mesh.triangles.push_back({
            static_cast<std::uint32_t>(oriented[0]),
            static_cast<std::uint32_t>(oriented[2]),
            static_cast<std::uint32_t>(oriented[3]),
        });

        for (std::size_t vertex_index : oriented) {
            normal_accum[vertex_index].x += outward.x;
            normal_accum[vertex_index].y += outward.y;
            normal_accum[vertex_index].z += outward.z;
        }
    };

    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        surface_counter.tick();
        if (opened_inside[leaf_index] == 0U) {
            continue;
        }

        std::uint32_t cell_x = 0U;
        std::uint32_t cell_y = 0U;
        std::uint32_t cell_z = 0U;
        const OctreeCell &cell = all_cells[solid_leaves[leaf_index].cell_index];
        morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
        const std::uint32_t span = 1U << (max_depth - cell.depth);
        const std::uint32_t ix0 = cell_x << (max_depth - cell.depth);
        const std::uint32_t iy0 = cell_y << (max_depth - cell.depth);
        const std::uint32_t iz0 = cell_z << (max_depth - cell.depth);

        for (std::uint32_t u = 0; u < span; ++u) {
            for (std::uint32_t v = 0; v < span; ++v) {
                const auto neighbor_opened = [&](std::int64_t neighbor_cell_index) {
                    if (neighbor_cell_index < 0) {
                        return false;
                    }
                    const std::size_t neighbor_leaf = static_cast<std::size_t>(neighbor_cell_index);
                    return opened_inside[neighbor_leaf] != 0U;
                };
                auto lookup_neighbor = [&](std::uint32_t qx, std::uint32_t qy,
                                           std::uint32_t qz) {
                    if (qx >= fine_resolution || qy >= fine_resolution || qz >= fine_resolution) {
                        return false;
                    }
                    const std::size_t neighbor_cell_index =
                        spatial_index.find_leaf_at(qx, qy, qz);
                    if (neighbor_cell_index == SIZE_MAX ||
                        neighbor_cell_index >= cell_to_leaf_index.size()) {
                        return false;
                    }
                    const std::int64_t neighbor_leaf_index =
                        cell_to_leaf_index[neighbor_cell_index];
                    return neighbor_opened(neighbor_leaf_index);
                };

                if (ix0 == 0U || !lookup_neighbor(ix0 - 1U, iy0 + u, iz0 + v)) {
                    const std::array<std::size_t, 4> ids = {
                        vertex_index_for(ix0, iy0 + u, iz0 + v),
                        vertex_index_for(ix0, iy0 + u + 1U, iz0 + v),
                        vertex_index_for(ix0, iy0 + u + 1U, iz0 + v + 1U),
                        vertex_index_for(ix0, iy0 + u, iz0 + v + 1U),
                    };
                    const std::array<Vector3d, 4> pos = {
                        mesh.vertices[ids[0]].position,
                        mesh.vertices[ids[1]].position,
                        mesh.vertices[ids[2]].position,
                        mesh.vertices[ids[3]].position,
                    };
                    emit_oriented_quad(pos, ids, {-1.0, 0.0, 0.0});
                }
                if (ix0 + span >= fine_resolution ||
                    !lookup_neighbor(ix0 + span, iy0 + u, iz0 + v)) {
                    const std::array<std::size_t, 4> ids = {
                        vertex_index_for(ix0 + span, iy0 + u, iz0 + v),
                        vertex_index_for(ix0 + span, iy0 + u, iz0 + v + 1U),
                        vertex_index_for(ix0 + span, iy0 + u + 1U, iz0 + v + 1U),
                        vertex_index_for(ix0 + span, iy0 + u + 1U, iz0 + v),
                    };
                    const std::array<Vector3d, 4> pos = {
                        mesh.vertices[ids[0]].position,
                        mesh.vertices[ids[1]].position,
                        mesh.vertices[ids[2]].position,
                        mesh.vertices[ids[3]].position,
                    };
                    emit_oriented_quad(pos, ids, {1.0, 0.0, 0.0});
                }
                if (iy0 == 0U || !lookup_neighbor(ix0 + u, iy0 - 1U, iz0 + v)) {
                    const std::array<std::size_t, 4> ids = {
                        vertex_index_for(ix0 + u, iy0, iz0 + v),
                        vertex_index_for(ix0 + u, iy0, iz0 + v + 1U),
                        vertex_index_for(ix0 + u + 1U, iy0, iz0 + v + 1U),
                        vertex_index_for(ix0 + u + 1U, iy0, iz0 + v),
                    };
                    const std::array<Vector3d, 4> pos = {
                        mesh.vertices[ids[0]].position,
                        mesh.vertices[ids[1]].position,
                        mesh.vertices[ids[2]].position,
                        mesh.vertices[ids[3]].position,
                    };
                    emit_oriented_quad(pos, ids, {0.0, -1.0, 0.0});
                }
                if (iy0 + span >= fine_resolution ||
                    !lookup_neighbor(ix0 + u, iy0 + span, iz0 + v)) {
                    const std::array<std::size_t, 4> ids = {
                        vertex_index_for(ix0 + u, iy0 + span, iz0 + v),
                        vertex_index_for(ix0 + u + 1U, iy0 + span, iz0 + v),
                        vertex_index_for(ix0 + u + 1U, iy0 + span, iz0 + v + 1U),
                        vertex_index_for(ix0 + u, iy0 + span, iz0 + v + 1U),
                    };
                    const std::array<Vector3d, 4> pos = {
                        mesh.vertices[ids[0]].position,
                        mesh.vertices[ids[1]].position,
                        mesh.vertices[ids[2]].position,
                        mesh.vertices[ids[3]].position,
                    };
                    emit_oriented_quad(pos, ids, {0.0, 1.0, 0.0});
                }
                if (iz0 == 0U || !lookup_neighbor(ix0 + u, iy0 + v, iz0 - 1U)) {
                    const std::array<std::size_t, 4> ids = {
                        vertex_index_for(ix0 + u, iy0 + v, iz0),
                        vertex_index_for(ix0 + u + 1U, iy0 + v, iz0),
                        vertex_index_for(ix0 + u + 1U, iy0 + v + 1U, iz0),
                        vertex_index_for(ix0 + u, iy0 + v + 1U, iz0),
                    };
                    const std::array<Vector3d, 4> pos = {
                        mesh.vertices[ids[0]].position,
                        mesh.vertices[ids[1]].position,
                        mesh.vertices[ids[2]].position,
                        mesh.vertices[ids[3]].position,
                    };
                    emit_oriented_quad(pos, ids, {0.0, 0.0, -1.0});
                }
                if (iz0 + span >= fine_resolution ||
                    !lookup_neighbor(ix0 + u, iy0 + v, iz0 + span)) {
                    const std::array<std::size_t, 4> ids = {
                        vertex_index_for(ix0 + u, iy0 + v, iz0 + span),
                        vertex_index_for(ix0 + u, iy0 + v + 1U, iz0 + span),
                        vertex_index_for(ix0 + u + 1U, iy0 + v + 1U, iz0 + span),
                        vertex_index_for(ix0 + u + 1U, iy0 + v, iz0 + span),
                    };
                    const std::array<Vector3d, 4> pos = {
                        mesh.vertices[ids[0]].position,
                        mesh.vertices[ids[1]].position,
                        mesh.vertices[ids[2]].position,
                        mesh.vertices[ids[3]].position,
                    };
                    emit_oriented_quad(pos, ids, {0.0, 0.0, 1.0});
                }
            }
        }
    }

    surface_counter.finish();

    ProgressCounter normal_counter(
        "Meshing", "generate_opened_surface_mesh", "vertices", 1000);
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        normal_counter.tick();
        const double mag = std::sqrt(
            normal_accum[i].x * normal_accum[i].x +
            normal_accum[i].y * normal_accum[i].y +
            normal_accum[i].z * normal_accum[i].z);
        if (mag > 0.0) {
            mesh.vertices[i].normal = {
                normal_accum[i].x / mag,
                normal_accum[i].y / mag,
                normal_accum[i].z / mag,
            };
        }
    }

    normal_counter.finish();

    return mesh;
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
