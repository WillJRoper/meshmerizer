#ifndef MESHMERIZER_ADAPTIVE_CPP_ADAPTIVE_SOLID_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_ADAPTIVE_SOLID_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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

struct OpenedBoundarySample {
    Vector3d position;
    Vector3d outward_normal;
    std::size_t leaf_index;
};

inline std::vector<std::size_t> gather_cell_contributors(
    const OctreeCell &cell,
    const std::vector<std::size_t> &all_contributors) {
    if (cell.contributor_begin < 0 ||
        cell.contributor_end <= cell.contributor_begin) {
        return {};
    }
    const auto begin_idx = static_cast<std::size_t>(cell.contributor_begin);
    const auto end_idx = static_cast<std::size_t>(cell.contributor_end);
    if (end_idx > all_contributors.size()) {
        return {};
    }
    return {
        all_contributors.begin() + static_cast<std::ptrdiff_t>(begin_idx),
        all_contributors.begin() + static_cast<std::ptrdiff_t>(end_idx),
    };
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
        const OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf || !cell.has_surface || cell.depth >= max_depth) {
            continue;
        }
        if (cell_edge_length(cell) > max_surface_leaf_size) {
            cells_to_visit.push(cell_idx);
        }
    }

    if (cells_to_visit.empty()) {
        std::fprintf(stdout,
                     "Regularization targeted refine: no surface-band splits needed "
                     "(leaf_size_target=%.6g, total_cells=%zu)\n",
                     max_surface_leaf_size,
                     all_cells.size());
        std::fflush(stdout);
        return false;
    }

    std::fprintf(stdout,
                 "Regularization targeted refine: starting from %zu surface-band "
                 "cells to reach leaf_size<=%.6g (total_cells_before=%zu)\n",
                 cells_to_visit.size(),
                 max_surface_leaf_size,
                 all_cells.size());
    std::fflush(stdout);

    ProgressCounter refine_counter(
        "Regularization targeted refine", "cells", 100);
    std::size_t split_count = 0U;
    std::size_t processed_count = 0U;

    while (!cells_to_visit.empty()) {
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
            const std::vector<std::size_t> contributors =
                gather_cell_contributors(child, all_contributors);
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

    std::fprintf(stdout,
                 "Regularization targeted refine: processed=%zu split=%zu "
                 "(total_cells_after_refine=%zu)\n",
                 processed_count,
                 split_count,
                 all_cells.size());
    std::fflush(stdout);
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
    std::uint32_t max_depth) {
    std::vector<OccupiedSolidLeaf> solid_leaves;
    solid_leaves.reserve(all_cells.size());
    std::vector<std::uint8_t> inside_flags(all_cells.size(), 0U);
    std::vector<double> center_values(all_cells.size(), 0.0);
    std::vector<std::int64_t> cell_to_leaf_index(all_cells.size(), -1);

    ProgressCounter classify_counter(
        "Classifying solid leaves", "cells", 1000);

#pragma omp parallel for schedule(dynamic)
    for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
        classify_counter.tick();
        const OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf) {
            continue;
        }

        const std::vector<std::size_t> contributors =
            gather_cell_contributors(cell, all_contributors);
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
        "Building occupancy", "cells", 1000);
    for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
        occupancy_counter.tick();
        const OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf) {
            continue;
        }

        const bool inside = inside_flags[cell_idx] != 0U;
        bool touches_opposite = false;
        const auto neighbors = face_neighbor_cells(cell, spatial_index, max_depth);
        for (std::size_t neighbor_idx : neighbors) {
            if (neighbor_idx == SIZE_MAX || neighbor_idx >= all_cells.size()) {
                continue;
            }
            if (inside_flags[neighbor_idx] != inside_flags[cell_idx]) {
                touches_opposite = true;
                break;
            }
        }

        OccupancyState occupancy = inside ? OccupancyState::kInside
                                          : OccupancyState::kOutside;
        if (touches_opposite) {
            occupancy = inside ? OccupancyState::kBoundaryInside
                               : OccupancyState::kBoundaryOutside;
        }

        solid_leaves.push_back({
            cell_idx,
            center_values[cell_idx],
            cell_edge_length(cell),
            cell.depth,
            occupancy,
            {-1, -1, -1, -1, -1, -1},
        });
        cell_to_leaf_index[cell_idx] =
            static_cast<std::int64_t>(solid_leaves.size() - 1U);
    }
    occupancy_counter.finish();

    ProgressCounter neighbor_counter(
        "Linking solid neighbors", "leaves", 1000);
    for (OccupiedSolidLeaf &leaf : solid_leaves) {
        neighbor_counter.tick();
        const auto neighbors = face_neighbor_cells(
            all_cells[leaf.cell_index], spatial_index, max_depth);
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

inline std::vector<double> compute_inside_clearance(
    const std::vector<OccupiedSolidLeaf> &solid_leaves) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> clearance(solid_leaves.size(), inf);
    using QueueEntry = std::pair<double, std::size_t>;
    std::priority_queue<
        QueueEntry,
        std::vector<QueueEntry>,
        std::greater<QueueEntry>> queue;

    ProgressCounter seed_counter(
        "Inside clearance seeds", "leaves", 1000);
    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        seed_counter.tick();
        if (solid_leaves[leaf_index].occupancy == OccupancyState::kBoundaryInside) {
            clearance[leaf_index] = 0.0;
            queue.push({0.0, leaf_index});
        }
    }
    seed_counter.finish();

    ProgressCounter wavefront_counter(
        "Inside clearance wavefront", "queue pops", 10000);

    while (!queue.empty()) {
        wavefront_counter.tick();
        const auto [distance, leaf_index] = queue.top();
        queue.pop();
        if (distance > clearance[leaf_index]) {
            continue;
        }

        for (std::int64_t neighbor_leaf_index :
             solid_leaves[leaf_index].face_neighbor_leaf_indices) {
            if (neighbor_leaf_index < 0) {
                continue;
            }
            const std::size_t neighbor =
                static_cast<std::size_t>(neighbor_leaf_index);
            if (solid_leaves[neighbor].occupancy == OccupancyState::kOutside ||
                solid_leaves[neighbor].occupancy == OccupancyState::kBoundaryOutside) {
                continue;
            }

            const double edge_cost = 0.5 * (
                solid_leaves[leaf_index].cell_size +
                solid_leaves[neighbor].cell_size);
            const double candidate = distance + edge_cost;
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
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<double> &clearance,
    double erosion_radius) {
    std::vector<std::uint8_t> kept_inside(solid_leaves.size(), 0U);

    ProgressCounter erode_counter("Eroding solid", "leaves", 1000);
#pragma omp parallel for schedule(static)
    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
        erode_counter.tick();
        if (solid_leaves[leaf_index].occupancy == OccupancyState::kOutside ||
            solid_leaves[leaf_index].occupancy == OccupancyState::kBoundaryOutside) {
            continue;
        }
        kept_inside[leaf_index] = clearance[leaf_index] >= erosion_radius ? 1U : 0U;
    }
    erode_counter.finish();
    return kept_inside;
}

inline std::vector<double> compute_distance_to_eroded_solid(
    const std::vector<OccupiedSolidLeaf> &solid_leaves,
    const std::vector<std::uint8_t> &eroded_inside) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> distance_to_eroded(solid_leaves.size(), inf);
    using QueueEntry = std::pair<double, std::size_t>;
    std::priority_queue<
        QueueEntry,
        std::vector<QueueEntry>,
        std::greater<QueueEntry>> queue;

    ProgressCounter seed_counter(
        "Dilation distance seeds", "leaves", 1000);
    for (std::size_t leaf_index = 0; leaf_index < solid_leaves.size(); ++leaf_index) {
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
        "Dilation distance wavefront", "queue pops", 10000);

    while (!queue.empty()) {
        wavefront_counter.tick();
        const auto [distance, leaf_index] = queue.top();
        queue.pop();
        if (distance > distance_to_eroded[leaf_index]) {
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

    ProgressCounter dilate_counter("Dilating solid", "leaves", 1000);
#pragma omp parallel for schedule(static)
    for (std::size_t leaf_index = 0; leaf_index < eroded_inside.size(); ++leaf_index) {
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
        "Opened boundary samples", "leaves", 1000);

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

inline std::uint64_t pack_surface_corner_coords(
    std::uint32_t ix, std::uint32_t iy, std::uint32_t iz) {
    return (static_cast<std::uint64_t>(ix) << 42U) |
           (static_cast<std::uint64_t>(iy) << 21U) |
           static_cast<std::uint64_t>(iz);
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
        "Opened surface extraction", "leaves", 1000);

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
        "Opened surface normals", "vertices", 1000);
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

    ProgressBar face_bar("Generating faces",
                         static_cast<std::size_t>(fine_res));

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
    const std::vector<std::size_t> &contributors,
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

        const std::vector<std::size_t> contributors =
            gather_cell_contributors(cell, all_contributors);
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
