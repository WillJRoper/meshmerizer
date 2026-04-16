/**
 * @file octree_cell.hpp
 * @brief Core adaptive octree cell type and refinement helpers.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_OCTREE_CELL_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_OCTREE_CELL_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "bounding_box.hpp"
#include "kernel_wendland_c2.hpp"
#include "morton.hpp"
#include "particle_grid.hpp"
#include "vector3d.hpp"

/**
 * @brief One adaptive octree cell in flat-array storage.
 */
struct OctreeCell {
    std::uint64_t morton_key;
    std::uint32_t depth;
    BoundingBox bounds;
    bool is_leaf;
    bool is_active;
    bool has_surface;
    std::int64_t child_begin;
    std::int64_t contributor_begin;
    std::int64_t contributor_end;
    std::int64_t representative_vertex_index;
    std::array<double, 8> corner_values;
    std::uint8_t corner_sign_mask;
};

/**
 * @brief Return whether sampled corner values can contain the isosurface.
 *
 * @param corner_values Scalar field samples at the eight cell corners.
 * @param isovalue Requested surface level.
 * @return True when the sampled value range straddles the isovalue.
 */
inline bool cell_may_contain_isosurface(
    const std::array<double, 8> &corner_values,
    double isovalue) {
    double minimum = corner_values[0];
    double maximum = corner_values[0];
    for (double value : corner_values) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    return minimum <= isovalue && maximum >= isovalue && minimum != maximum;
}

/**
 * @brief Return the sign mask of corner samples relative to an isovalue.
 *
 * @param corner_values Scalar field samples at the eight cell corners.
 * @param isovalue Requested surface level.
 * @return Bit mask with one bit per corner indicating `value >= isovalue`.
 */
inline std::uint8_t compute_corner_sign_mask(
    const std::array<double, 8> &corner_values,
    double isovalue) {
    std::uint8_t mask = 0U;
    for (std::size_t corner_index = 0; corner_index < corner_values.size();
         ++corner_index) {
        if (corner_values[corner_index] >= isovalue) {
            mask |= static_cast<std::uint8_t>(1U << corner_index);
        }
    }
    return mask;
}

/**
 * @brief Create the flat list of top-level octree cells.
 *
 * @param domain Working domain covered by the top-level cells.
 * @param base_resolution Number of top-level cells per axis.
 * @return Top-level cells in deterministic row-major order.
 */
inline std::vector<OctreeCell> create_top_level_cells(
    const BoundingBox &domain,
    std::uint32_t base_resolution) {
    if (base_resolution == 0U) {
        return {};
    }

    std::vector<OctreeCell> cells;
    cells.reserve(static_cast<std::size_t>(base_resolution) * base_resolution *
                  base_resolution);

    const Vector3d cell_size = {
        (domain.max.x - domain.min.x) / static_cast<double>(base_resolution),
        (domain.max.y - domain.min.y) / static_cast<double>(base_resolution),
        (domain.max.z - domain.min.z) / static_cast<double>(base_resolution),
    };

    for (std::uint32_t ix = 0; ix < base_resolution; ++ix) {
        for (std::uint32_t iy = 0; iy < base_resolution; ++iy) {
            for (std::uint32_t iz = 0; iz < base_resolution; ++iz) {
                const Vector3d minimum = {
                    domain.min.x + static_cast<double>(ix) * cell_size.x,
                    domain.min.y + static_cast<double>(iy) * cell_size.y,
                    domain.min.z + static_cast<double>(iz) * cell_size.z,
                };
                const Vector3d maximum = {
                    minimum.x + cell_size.x,
                    minimum.y + cell_size.y,
                    minimum.z + cell_size.z,
                };
                cells.push_back({
                    morton_encode_3d(ix, iy, iz),
                    0U,
                    {minimum, maximum},
                    true,
                    false,
                    false,
                    -1,
                    -1,
                    -1,
                    -1,
                    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                    0U,
                });
            }
        }
    }
    return cells;
}

/**
 * @brief Return one child bounding box from a parent cell.
 *
 * Child indices follow Morton-compatible local ordering:
 *
 * - bit 0 controls the x half
 * - bit 1 controls the y half
 * - bit 2 controls the z half
 *
 * @param parent_bounds Bounding box of the parent cell.
 * @param child_index Local child index in ``[0, 7]``.
 * @return Bounding box of the requested child.
 */
inline BoundingBox child_bounds_from_index(
    const BoundingBox &parent_bounds,
    std::uint8_t child_index) {
    const Vector3d midpoint = parent_bounds.center();
    const bool high_x = (child_index & 1U) != 0U;
    const bool high_y = (child_index & 2U) != 0U;
    const bool high_z = (child_index & 4U) != 0U;

    return {
        {
            high_x ? midpoint.x : parent_bounds.min.x,
            high_y ? midpoint.y : parent_bounds.min.y,
            high_z ? midpoint.z : parent_bounds.min.z,
        },
        {
            high_x ? parent_bounds.max.x : midpoint.x,
            high_y ? parent_bounds.max.y : midpoint.y,
            high_z ? parent_bounds.max.z : midpoint.z,
        },
    };
}

/**
 * @brief Create the eight children of one parent octree cell.
 *
 * @param parent Parent octree cell.
 * @return Child cells in deterministic Morton-compatible local order.
 */
inline std::vector<OctreeCell> create_child_cells(const OctreeCell &parent) {
    std::uint32_t parent_x = 0U;
    std::uint32_t parent_y = 0U;
    std::uint32_t parent_z = 0U;
    morton_decode_3d(parent.morton_key, parent_x, parent_y, parent_z);

    std::vector<OctreeCell> children;
    children.reserve(8U);
    for (std::uint8_t child_index = 0U; child_index < 8U; ++child_index) {
        const std::uint32_t child_x = parent_x * 2U + (child_index & 1U);
        const std::uint32_t child_y = parent_y * 2U + ((child_index >> 1U) & 1U);
        const std::uint32_t child_z = parent_z * 2U + ((child_index >> 2U) & 1U);
        children.push_back({
            morton_encode_3d(child_x, child_y, child_z),
            parent.depth + 1U,
            child_bounds_from_index(parent.bounds, child_index),
            true,
            false,
            false,
            -1,
            -1,
            -1,
            -1,
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            0U,
        });
    }
    return children;
}

/**
 * @brief Filter parent contributors into each child cell.
 *
 * @param parent_contributors Particle indices attached to the parent cell.
 * @param positions Particle positions in world space.
 * @param smoothing_lengths Per-particle support radii.
 * @param children Already-created child cells.
 * @return Per-child contributor lists in the same order as ``children``.
 */
inline std::vector<std::vector<std::size_t>> filter_child_contributors(
    const std::vector<std::size_t> &parent_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const std::vector<OctreeCell> &children) {
    std::vector<std::vector<std::size_t>> child_contributors(children.size());
    for (std::size_t particle_index : parent_contributors) {
        for (std::size_t child_index = 0; child_index < children.size();
             ++child_index) {
            if (particle_support_overlaps_box(
                    positions[particle_index],
                    smoothing_lengths[particle_index],
                    children[child_index].bounds)) {
                child_contributors[child_index].push_back(particle_index);
            }
        }
    }
    return child_contributors;
}

/**
 * @brief Evaluate the SPH field value at a given point.
 *
 * Sums the normalized Wendland C2 kernel contributions from all contributing
 * particles. This is the scalar field we use for isosurface extraction.
 *
 * @param query_point Position at which to evaluate the field.
 * @param contributor_indices Indices of particles that may contribute.
 * @param positions Particle positions in world space.
 * @param smoothing_lengths Per-particle support radii.
 * @return Scalar field value at the query point.
 */
inline double evaluate_field_at_point(
    const Vector3d &query_point,
    const std::vector<std::size_t> &contributor_indices,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths) {
    double field_value = 0.0;
    for (std::size_t particle_index : contributor_indices) {
        const Vector3d displacement = {
            query_point.x - positions[particle_index].x,
            query_point.y - positions[particle_index].y,
            query_point.z - positions[particle_index].z,
        };
        const double radius = std::sqrt(
            displacement.x * displacement.x + displacement.y * displacement.y +
            displacement.z * displacement.z);
        field_value +=
            evaluate_wendland_c2(radius, smoothing_lengths[particle_index], true);
    }
    return field_value;
}

/**
 * @brief Sample the eight corner values of a cell.
 *
 * @param cell The octree cell whose corners to sample.
 * @param contributor_indices Indices of particles that may contribute.
 * @param positions Particle positions in world space.
 * @param smoothing_lengths Per-particle support radii.
 * @return Array of eight field values, one per corner.
 */
inline std::array<double, 8> sample_cell_corners(
    const OctreeCell &cell,
    const std::vector<std::size_t> &contributor_indices,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths) {
    std::array<double, 8> corner_values;
    for (std::size_t corner_index = 0; corner_index < 8; ++corner_index) {
        const Vector3d corner = {
            (corner_index & 1U) ? cell.bounds.max.x : cell.bounds.min.x,
            (corner_index & 2U) ? cell.bounds.max.y : cell.bounds.min.y,
            (corner_index & 4U) ? cell.bounds.max.z : cell.bounds.min.z,
        };
        corner_values[corner_index] = evaluate_field_at_point(
            corner, contributor_indices, positions, smoothing_lengths);
    }
    return corner_values;
}

/**
 * @brief Breadth-first octree refinement based on corner sign changes.
 *
 * Starting from top-level cells, this function iteratively refines cells that
 * may contain the isosurface by evaluating corner field values. The algorithm
 * proceeds level by level (breadth-first) to maintain a balanced tree where
 * neighboring leaves differ by at most one refinement level.
 *
 * The refinement stops when:
 * - A cell does not contain the isosurface (corner values don't straddle
 *   the isovalue), or
 * - The maximum depth is reached, or
 * - A cell has fewer than two contributors (insufficient samples for
 *   meaningful refinement).
 *
 * @param initial_cells Top-level cells with attached contributors.
 * @param positions Particle positions in world space.
 * @param smoothing_lengths Per-particle support radii.
 * @param isovalue The target surface level.
 * @param max_depth Maximum refinement depth (prevents infinite refinement).
 * @return Tuple of (all_cells, all_contributors) where cells store indices
 *         into the contributors vector.
 */
inline std::pair<std::vector<OctreeCell>, std::vector<std::size_t>> refine_octree(
    std::vector<OctreeCell> initial_cells,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue,
    std::uint32_t max_depth) {
    if (initial_cells.empty()) {
        return {{}, {}};
    }

    std::vector<OctreeCell> all_cells;
    std::vector<std::size_t> all_contributors;
    std::queue<std::size_t> leaf_queue;

    for (std::size_t cell_index = 0; cell_index < initial_cells.size();
         ++cell_index) {
        OctreeCell cell = initial_cells[cell_index];
        const std::int64_t contrib_begin = cell.contributor_begin;
        const std::int64_t contrib_end = cell.contributor_end;

        if (contrib_begin >= 0 && contrib_end > contrib_begin) {
            cell.contributor_begin = static_cast<std::int64_t>(all_contributors.size());
            for (std::int64_t i = contrib_begin; i < contrib_end; ++i) {
                all_contributors.push_back(static_cast<std::size_t>(i));
            }
            cell.contributor_end = static_cast<std::int64_t>(all_contributors.size());
        } else {
            cell.contributor_begin = -1;
            cell.contributor_end = -1;
        }

        all_cells.push_back(cell);
        leaf_queue.push(all_cells.size() - 1);
    }

    while (!leaf_queue.empty()) {
        const std::size_t current_index = leaf_queue.front();
        leaf_queue.pop();
        OctreeCell &current_cell = all_cells[current_index];

        if (current_cell.depth >= max_depth) {
            current_cell.is_leaf = true;
            current_cell.is_active = false;
            current_cell.has_surface = false;
            continue;
        }

        const std::int64_t contrib_begin = current_cell.contributor_begin;
        const std::int64_t contrib_end = current_cell.contributor_end;
        if (contrib_begin < 0 || contrib_end < 0 || contrib_begin >= contrib_end) {
            current_cell.is_leaf = true;
            current_cell.is_active = false;
            current_cell.has_surface = false;
            continue;
        }

        std::vector<std::size_t> contributors;
        contributors.reserve(static_cast<std::size_t>(contrib_end - contrib_begin));
        for (std::int64_t i = contrib_begin; i < contrib_end; ++i) {
            if (static_cast<std::size_t>(i) < all_contributors.size()) {
                contributors.push_back(all_contributors[static_cast<std::size_t>(i)]);
            }
        }

        if (contributors.size() < 2) {
            current_cell.is_leaf = true;
            current_cell.is_active = false;
            current_cell.has_surface = false;
            continue;
        }

        std::array<double, 8> corner_values =
            sample_cell_corners(current_cell, contributors, positions,
                               smoothing_lengths);
        current_cell.corner_values = corner_values;
        current_cell.corner_sign_mask = compute_corner_sign_mask(corner_values, isovalue);

        if (!cell_may_contain_isosurface(corner_values, isovalue)) {
            current_cell.is_leaf = true;
            current_cell.is_active = false;
            current_cell.has_surface = false;
            continue;
        }

        current_cell.is_leaf = false;
        current_cell.is_active = true;
        current_cell.has_surface = true;

        std::vector<OctreeCell> children =
            create_child_cells(current_cell);
        std::vector<std::vector<std::size_t>> child_contributors =
            filter_child_contributors(contributors, positions, smoothing_lengths,
                                     children);

        const std::int64_t child_begin_offset =
            static_cast<std::int64_t>(all_contributors.size());
        current_cell.child_begin = child_begin_offset;

        for (std::size_t i = 0; i < children.size(); ++i) {
            const std::int64_t child_contrib_begin =
                static_cast<std::int64_t>(all_contributors.size());
            std::copy(child_contributors[i].begin(),
                      child_contributors[i].end(),
                      std::back_inserter(all_contributors));
            const std::int64_t child_contrib_end =
                static_cast<std::int64_t>(all_contributors.size());

            children[i].contributor_begin = child_contrib_begin;
            children[i].contributor_end = child_contrib_end;

            all_cells.push_back(children[i]);
            leaf_queue.push(all_cells.size() - 1);
        }
    }

    return {all_cells, all_contributors};
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_OCTREE_CELL_HPP_
