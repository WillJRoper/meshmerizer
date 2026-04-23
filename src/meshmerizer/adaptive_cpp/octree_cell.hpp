/**
 * @file octree_cell.hpp
 * @brief Core adaptive octree cell type and refinement helpers.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_OCTREE_CELL_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_OCTREE_CELL_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <span>
#include <queue>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bounding_box.hpp"
#include "hermite.hpp"
#include "kernel_wendland_c2.hpp"
#include "morton.hpp"
#include "particle_grid.hpp"
#include "progress_bar.hpp"
#include "qef.hpp"
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
    bool is_topo_surface;
    std::int64_t child_begin;
    std::int64_t contributor_begin;
    std::int64_t contributor_end;
    std::int64_t representative_vertex_index;
    std::array<double, 8> corner_values;
    std::uint8_t corner_sign_mask;
};

/**
 * @brief Print a concise summary of the final octree structure.
 *
 * The summary is intentionally compact: one total line plus one line per depth.
 * This preserves the existing progress counters while still making the final
 * tree shape visible to the user.
 *
 * @param all_cells Flat array of all octree cells.
 */
inline void print_octree_structure_summary(
    const std::vector<OctreeCell> &all_cells) {
    struct DepthSummary {
        std::size_t total = 0;
        std::size_t leaf = 0;
        std::size_t active = 0;
        std::size_t has_surface = 0;
    };

    if (all_cells.empty()) {
        meshmerizer_log_detail::print_status(
            "Tree", "print_octree_structure_summary",
            "total=0 leaf=0 internal=0 active=0 inactive=0 surface=0\n");
        return;
    }

    std::uint32_t max_cell_depth = 0;
    for (const OctreeCell &cell : all_cells) {
        max_cell_depth = std::max(max_cell_depth, cell.depth);
    }

    std::vector<DepthSummary> per_depth(max_cell_depth + 1U);
    std::size_t total_leaf = 0;
    std::size_t total_active = 0;
    std::size_t total_surface = 0;

    for (const OctreeCell &cell : all_cells) {
        DepthSummary &summary = per_depth[cell.depth];
        ++summary.total;
        if (cell.is_leaf) {
            ++summary.leaf;
            ++total_leaf;
        }
        if (cell.is_active) {
            ++summary.active;
            ++total_active;
        }
        if (cell.has_surface) {
            ++summary.has_surface;
            ++total_surface;
        }
    }

    meshmerizer_log_detail::print_status(
        "Tree",
        "print_octree_structure_summary",
        "total=%zu leaf=%zu internal=%zu active=%zu inactive=%zu surface=%zu\n",
        all_cells.size(),
        total_leaf,
        all_cells.size() - total_leaf,
        total_active,
        all_cells.size() - total_active,
        total_surface);
    for (std::uint32_t depth = 0; depth < per_depth.size(); ++depth) {
        const DepthSummary &summary = per_depth[depth];
        meshmerizer_log_detail::print_status(
            "Tree",
            "print_octree_structure_summary",
            "depth %u: total=%zu leaf=%zu internal=%zu active=%zu inactive=%zu surface=%zu\n",
            depth,
            summary.total,
            summary.leaf,
            summary.total - summary.leaf,
            summary.active,
            summary.total - summary.active,
            summary.has_surface);
    }
}

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
    std::span<const std::size_t> parent_contributors,
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
 * @brief Split one leaf cell and append its children to the flat arrays.
 */
inline void split_octree_leaf(
    std::size_t split_index,
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths) {
    if (split_index >= all_cells.size() || !all_cells[split_index].is_leaf) {
        return;
    }

    OctreeCell &parent = all_cells[split_index];
    std::vector<std::size_t> contributors;
    if (parent.contributor_begin >= 0 &&
        parent.contributor_end > parent.contributor_begin) {
        const auto begin_idx =
            static_cast<std::size_t>(parent.contributor_begin);
        const auto end_idx =
            static_cast<std::size_t>(parent.contributor_end);
        if (end_idx <= all_contributors.size()) {
            contributors.assign(
                all_contributors.begin() + static_cast<std::ptrdiff_t>(begin_idx),
                all_contributors.begin() + static_cast<std::ptrdiff_t>(end_idx));
        }
    }

    std::vector<OctreeCell> children = create_child_cells(parent);
    std::vector<std::vector<std::size_t>> child_contributors =
        filter_child_contributors(
            contributors, positions, smoothing_lengths, children);

    parent.is_leaf = false;
    parent.is_active = false;
    parent.has_surface = true;
    parent.is_topo_surface = false;
    parent.representative_vertex_index = -1;
    parent.child_begin = static_cast<std::int64_t>(all_cells.size());

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
    }
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
    std::span<const std::size_t> contributor_indices,
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
    std::span<const std::size_t> contributor_indices,
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

inline std::span<const std::size_t> contributor_span(
    const std::vector<std::size_t> &all_contributors,
    std::size_t begin,
    std::size_t end) {
    if (begin >= end || end > all_contributors.size()) {
        return std::span<const std::size_t>();
    }
    return std::span<const std::size_t>(
        all_contributors.data() + begin,
        end - begin);
}

/**
 * @brief Return whether a point lies inside a cell using closed bounds.
 *
 * Closed bounds are used intentionally here so that parent-derived crossing
 * positions or provisional representative vertices lying exactly on a child
 * boundary can seed all relevant children as refinement candidates.
 */
inline bool point_in_cell_closed(const BoundingBox &bounds,
                                 const Vector3d &point,
                                 double epsilon = 1e-12) {
    return point.x >= bounds.min.x - epsilon &&
           point.x <= bounds.max.x + epsilon &&
           point.y >= bounds.min.y - epsilon &&
           point.y <= bounds.max.y + epsilon &&
           point.z >= bounds.min.z - epsilon &&
           point.z <= bounds.max.z + epsilon;
}

/**
 * @brief Return whether a child should inherit surface consideration.
 *
 * When a known surface-bearing parent is split, a child may still contain the
 * surface even if its own corners are uniform.  To avoid discarding such
 * children prematurely, inherit surface consideration when either a parent
 * Hermite crossing or the parent's provisional QEF vertex lies inside the
 * child.
 */
inline bool child_inherits_surface_hint(
    const OctreeCell &child,
    const std::vector<HermiteSample> &parent_samples,
    const MeshVertex &parent_vertex) {
    if (point_in_cell_closed(child.bounds, parent_vertex.position)) {
        return true;
    }
    for (const HermiteSample &sample : parent_samples) {
        if (point_in_cell_closed(child.bounds, sample.position)) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Return the minimum alignment of usable Hermite normals to their mean.
 *
 * Values near 1 indicate locally coherent normals (flatter surface patch);
 * smaller values indicate stronger curvature or conflicting constraints, which
 * are signals that one representative vertex may be insufficient.
 */
inline double minimum_hermite_normal_alignment(
    const std::vector<HermiteSample> &samples) {
    Vector3d normal_sum = {0.0, 0.0, 0.0};
    std::size_t usable_count = 0;
    for (const HermiteSample &sample : samples) {
        const double nx = sample.normal.x;
        const double ny = sample.normal.y;
        const double nz = sample.normal.z;
        if (nx == 0.0 && ny == 0.0 && nz == 0.0) {
            continue;
        }
        normal_sum.x += nx;
        normal_sum.y += ny;
        normal_sum.z += nz;
        ++usable_count;
    }
    if (usable_count < 2U) {
        return 1.0;
    }

    const double mean_norm = std::sqrt(
        normal_sum.x * normal_sum.x +
        normal_sum.y * normal_sum.y +
        normal_sum.z * normal_sum.z);
    if (mean_norm < 1e-12) {
        return 0.0;
    }

    const Vector3d mean_normal = {
        normal_sum.x / mean_norm,
        normal_sum.y / mean_norm,
        normal_sum.z / mean_norm,
    };

    double min_alignment = 1.0;
    for (const HermiteSample &sample : samples) {
        const double nx = sample.normal.x;
        const double ny = sample.normal.y;
        const double nz = sample.normal.z;
        if (nx == 0.0 && ny == 0.0 && nz == 0.0) {
            continue;
        }
        const double alignment =
            nx * mean_normal.x + ny * mean_normal.y + nz * mean_normal.z;
        min_alignment = std::min(min_alignment, alignment);
    }
    return min_alignment;
}

/**
 * @brief Compute Morton key of a face-neighbor cell in a given direction.
 *
 * Given a cell Morton key and depth, return the neighbor key in the
 * positive or negative direction along one axis. Returns 0 if the
 * neighbor would be outside the domain covered by the root cells.
 *
 * Neighbor directions:
 * - 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z
 *
 * At depth d the coordinate grid spans ``[0, root_resolution * 2^d)`` on
 * each axis. A face neighbor differs by exactly one unit in the relevant
 * axis, so the only bound check needed is whether that one-unit step stays
 * inside the grid.
 *
 * @param key Current cell Morton key.
 * @param depth Cell depth.
 * @param direction Neighbor direction (0-5).
 * @param root_resolution Base resolution of the top-level grid.
 * @return Neighbor Morton key, or 0 if out of bounds.
 */
inline std::uint64_t neighbor_morton_key(std::uint64_t key,
                                         std::uint32_t depth,
                                         std::uint8_t direction,
                                         std::uint32_t root_resolution) {
    std::uint32_t x = 0U;
    std::uint32_t y = 0U;
    std::uint32_t z = 0U;
    morton_decode_3d(key, x, y, z);

    // Valid coordinates at depth d span [0, root_resolution * 2^d).
    // Guard: depth >= 32 would cause undefined behavior on the shift.
    if (depth >= 32U) {
        return 0ULL;  // No valid neighbor at extreme depths.
    }
    const std::uint64_t max_coord =
        static_cast<std::uint64_t>(root_resolution) << depth;

    switch (direction) {
        case 0:  // +X
            if (x + 1U >= max_coord) {
                return 0ULL;
            }
            ++x;
            break;
        case 1:  // -X
            if (x == 0U) {
                return 0ULL;
            }
            --x;
            break;
        case 2:  // +Y
            if (y + 1U >= max_coord) {
                return 0ULL;
            }
            ++y;
            break;
        case 3:  // -Y
            if (y == 0U) {
                return 0ULL;
            }
            --y;
            break;
        case 4:  // +Z
            if (z + 1U >= max_coord) {
                return 0ULL;
            }
            ++z;
            break;
        case 5:  // -Z
            if (z == 0U) {
                return 0ULL;
            }
            --z;
            break;
        default:
            return 0ULL;
    }

    return morton_encode_3d(x, y, z);
}

/**
 * @brief Return whether two leaf cells share a face with positive area.
 *
 * Two axis-aligned cells share a face when they touch along exactly one axis
 * (one box's max equals the other's min on that axis) and overlap with
 * positive length in each of the remaining two axes.
 *
 * This test works correctly for cells of different sizes, which is required
 * when detecting octree balance violations between leaves at different depths.
 *
 * @param a First cell.
 * @param b Second cell.
 * @return True when the cells share a face.
 */
inline bool leaf_cells_share_face(const OctreeCell &a, const OctreeCell &b) {
    // Floating-point equality is intentional here. All cell boundaries in this
    // codebase are derived by repeatedly halving the same initial domain
    // values, so the midpoint of any parent is computed identically for both
    // the parent's max and the child's min, giving exactly equal IEEE 754
    // results. If bounds are ever accumulated rather than computed fresh, this
    // assumption must be revisited.
    const bool touch_x = (a.bounds.max.x == b.bounds.min.x) ||
                         (b.bounds.max.x == a.bounds.min.x);
    const bool touch_y = (a.bounds.max.y == b.bounds.min.y) ||
                         (b.bounds.max.y == a.bounds.min.y);
    const bool touch_z = (a.bounds.max.z == b.bounds.min.z) ||
                         (b.bounds.max.z == a.bounds.min.z);
    const bool overlap_x = a.bounds.min.x < b.bounds.max.x &&
                           b.bounds.min.x < a.bounds.max.x;
    const bool overlap_y = a.bounds.min.y < b.bounds.max.y &&
                           b.bounds.min.y < a.bounds.max.y;
    const bool overlap_z = a.bounds.min.z < b.bounds.max.z &&
                           b.bounds.min.z < a.bounds.max.z;
    return (touch_x && overlap_y && overlap_z) ||
           (touch_y && overlap_x && overlap_z) ||
           (touch_z && overlap_x && overlap_y);
}

/**
 * @brief Pack three grid coordinates into a 64-bit key for balancing.
 *
 * Each axis gets 21 bits, matching the Morton key encoding capacity.
 * This is a local utility for the balance spatial hash; the same
 * packing scheme is used by LeafSpatialIndex in faces.hpp.
 *
 * @param ix X grid coordinate.
 * @param iy Y grid coordinate.
 * @param iz Z grid coordinate.
 * @return Packed 64-bit key.
 */
inline std::uint64_t balance_pack_coords(std::uint32_t ix,
                                          std::uint32_t iy,
                                          std::uint32_t iz) {
    return (static_cast<std::uint64_t>(ix) << 42U) |
           (static_cast<std::uint64_t>(iy) << 21U) |
           static_cast<std::uint64_t>(iz);
}

/**
 * @brief FNV-1a-style hash for 64-bit packed grid keys.
 */
struct BalanceKeyHash {
    std::size_t operator()(std::uint64_t key) const {
        std::uint64_t h = key;
        h ^= h >> 33U;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33U;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33U;
        return static_cast<std::size_t>(h);
    }
};

/**
 * @brief Spatial hash mapping leaf min-corner grid positions to cell indices.
 *
 * Used by the balance algorithm for O(1) neighbor lookups.  Each leaf
 * registers one entry keyed by its min-corner quantized to the finest
 * grid resolution.  Lookups use a hierarchical probe (same approach as
 * LeafSpatialIndex in faces.hpp) to find the leaf containing a given
 * fine-grid position.
 */
struct BalanceSpatialHash {
    std::unordered_map<std::uint64_t, std::size_t, BalanceKeyHash> map;
    std::uint32_t max_depth;
    double inv_cell_size_x;
    double inv_cell_size_y;
    double inv_cell_size_z;
    Vector3d domain_min;

    /**
     * @brief Build the hash from the current set of leaf cells.
     *
     * @param all_cells All octree cells.
     * @param domain Full domain bounding box.
     * @param md Maximum octree depth.
     * @param base_resolution Number of top-level cells per axis.
     */
    void build(const std::vector<OctreeCell> &all_cells,
               const BoundingBox &domain,
               std::uint32_t md,
               std::uint32_t base_resolution) {
        max_depth = md;
        domain_min = domain.min;

        const double fine_per_axis =
            static_cast<double>(base_resolution) *
            static_cast<double>(1U << max_depth);
        inv_cell_size_x = fine_per_axis / (domain.max.x - domain.min.x);
        inv_cell_size_y = fine_per_axis / (domain.max.y - domain.min.y);
        inv_cell_size_z = fine_per_axis / (domain.max.z - domain.min.z);

        map.clear();
        std::size_t n_leaves = 0;
        for (const auto &c : all_cells) {
            if (c.is_leaf) ++n_leaves;
        }
        map.reserve(n_leaves);

        for (std::size_t i = 0; i < all_cells.size(); ++i) {
            if (!all_cells[i].is_leaf) continue;
            std::uint32_t gx, gy, gz;
            quantize(all_cells[i].bounds.min, gx, gy, gz);
            map[balance_pack_coords(gx, gy, gz)] = i;
        }
    }

    /**
     * @brief Quantize a world-space position to fine-grid coordinates.
     */
    void quantize(const Vector3d &pos,
                  std::uint32_t &gx,
                  std::uint32_t &gy,
                  std::uint32_t &gz) const {
        gx = static_cast<std::uint32_t>(
            (pos.x - domain_min.x) * inv_cell_size_x + 0.5);
        gy = static_cast<std::uint32_t>(
            (pos.y - domain_min.y) * inv_cell_size_y + 0.5);
        gz = static_cast<std::uint32_t>(
            (pos.z - domain_min.z) * inv_cell_size_z + 0.5);
    }

    /**
     * @brief Find the leaf cell containing a given fine-grid position.
     *
     * Hierarchical probe from finest to coarsest alignment.
     *
     * @return Cell index, or SIZE_MAX if not found.
     */
    std::size_t find_leaf_at(std::uint32_t ix, std::uint32_t iy,
                             std::uint32_t iz) const {
        for (std::uint32_t k = 0; k <= max_depth; ++k) {
            const std::uint32_t mask = ~((1U << k) - 1U);
            const std::uint64_t key = balance_pack_coords(
                ix & mask, iy & mask, iz & mask);
            auto it = map.find(key);
            if (it != map.end()) {
                return it->second;
            }
        }
        return SIZE_MAX;
    }
};

/**
 * @brief Check if a leaf needs splitting using spatial hash neighbor lookup.
 *
 * For each of the 6 face directions, computes a probe point just inside
 * the neighboring region and looks up the leaf there via the spatial
 * hash.  If any neighbor leaf is more than 1 level deeper, the cell
 * needs splitting.
 *
 * Cost: O(6 * max_depth) hash probes per cell — much better than O(n).
 *
 * @param cell_index Index of the leaf to check.
 * @param all_cells All octree cells.
 * @param hash Prebuilt spatial hash of leaf cells.
 * @param max_depth Maximum octree depth.
 * @return True when this leaf needs to be split for balance.
 */
inline bool needs_balance_split(
    std::size_t cell_index,
    const std::vector<OctreeCell> &all_cells,
    const BalanceSpatialHash &hash,
    std::uint32_t max_depth) {
    const OctreeCell &cell = all_cells[cell_index];

    // The cell spans 'span' fine-grid cells per axis.
    const std::uint32_t span = 1U << (max_depth - cell.depth);

    // Quantize this cell's min corner.
    std::uint32_t gx, gy, gz;
    hash.quantize(cell.bounds.min, gx, gy, gz);

    // For each face, probe one fine-grid cell into the neighbor region.
    // The 6 face directions are: -X, +X, -Y, +Y, -Z, +Z.
    // For the positive direction, the probe is at min + span.
    // For the negative direction, the probe is at min - 1.
    // We probe at the center of the face (offset by span/2 in the
    // other two axes) to handle cases where the neighbor is larger.
    // However, for detecting deeper neighbors (which are smaller),
    // probing at the corner of the face is sufficient because any
    // deeper neighbor along this face will be found.

    // Face probe offsets: {dx, dy, dz} relative to cell min corner.
    // +X face: probe at (gx + span, gy, gz)
    // -X face: probe at (gx - 1, gy, gz)
    // +Y face: probe at (gx, gy + span, gz)
    // -Y face: probe at (gx, gy - 1, gz)
    // +Z face: probe at (gx, gy, gz + span)
    // -Z face: probe at (gx, gy, gz - 1)

    struct Probe {
        std::int64_t dx, dy, dz;
    };
    const Probe probes[6] = {
        {static_cast<std::int64_t>(span), 0, 0},    // +X
        {-1, 0, 0},                                   // -X
        {0, static_cast<std::int64_t>(span), 0},    // +Y
        {0, -1, 0},                                   // -Y
        {0, 0, static_cast<std::int64_t>(span)},    // +Z
        {0, 0, -1},                                   // -Z
    };

    for (const auto &p : probes) {
        const std::int64_t px =
            static_cast<std::int64_t>(gx) + p.dx;
        const std::int64_t py =
            static_cast<std::int64_t>(gy) + p.dy;
        const std::int64_t pz =
            static_cast<std::int64_t>(gz) + p.dz;

        // Skip probes outside the domain.
        if (px < 0 || py < 0 || pz < 0) continue;

        const std::size_t neighbor_idx = hash.find_leaf_at(
            static_cast<std::uint32_t>(px),
            static_cast<std::uint32_t>(py),
            static_cast<std::uint32_t>(pz));

        if (neighbor_idx == SIZE_MAX) continue;

        const OctreeCell &neighbor = all_cells[neighbor_idx];
        const std::int32_t depth_diff =
            static_cast<std::int32_t>(neighbor.depth) -
            static_cast<std::int32_t>(cell.depth);
        if (depth_diff > 1) {
            return true;
        }
    }
    return false;
}

inline void enqueue_balance_neighbors(
    std::size_t cell_index,
    const std::vector<OctreeCell> &all_cells,
    const BalanceSpatialHash &hash,
    std::uint32_t max_depth,
    std::vector<std::size_t> &queue,
    std::vector<std::uint8_t> &enqueued) {
    if (cell_index >= all_cells.size()) {
        return;
    }

    auto enqueue_index = [&](std::size_t candidate) {
        if (candidate >= all_cells.size()) {
            return;
        }
        if (!all_cells[candidate].is_leaf) {
            return;
        }
        if (candidate >= enqueued.size()) {
            enqueued.resize(all_cells.size(), 0U);
        }
        if (enqueued[candidate] != 0U) {
            return;
        }
        enqueued[candidate] = 1U;
        queue.push_back(candidate);
    };

    enqueue_index(cell_index);

    const OctreeCell &cell = all_cells[cell_index];
    if (!cell.is_leaf) {
        return;
    }

    const std::uint32_t span = 1U << (max_depth - cell.depth);
    std::uint32_t gx, gy, gz;
    hash.quantize(cell.bounds.min, gx, gy, gz);

    const std::uint32_t half = span > 1U ? span / 2U : 0U;
    struct Probe {
        std::int64_t dx, dy, dz;
    };
    const Probe probes[6] = {
        {static_cast<std::int64_t>(span), static_cast<std::int64_t>(half), static_cast<std::int64_t>(half)},
        {-1, static_cast<std::int64_t>(half), static_cast<std::int64_t>(half)},
        {static_cast<std::int64_t>(half), static_cast<std::int64_t>(span), static_cast<std::int64_t>(half)},
        {static_cast<std::int64_t>(half), -1, static_cast<std::int64_t>(half)},
        {static_cast<std::int64_t>(half), static_cast<std::int64_t>(half), static_cast<std::int64_t>(span)},
        {static_cast<std::int64_t>(half), static_cast<std::int64_t>(half), -1},
    };

    for (const Probe &probe : probes) {
        const std::int64_t px = static_cast<std::int64_t>(gx) + probe.dx;
        const std::int64_t py = static_cast<std::int64_t>(gy) + probe.dy;
        const std::int64_t pz = static_cast<std::int64_t>(gz) + probe.dz;
        if (px < 0 || py < 0 || pz < 0) {
            continue;
        }

        const std::size_t neighbor_idx = hash.find_leaf_at(
            static_cast<std::uint32_t>(px),
            static_cast<std::uint32_t>(py),
            static_cast<std::uint32_t>(pz));
        if (neighbor_idx == SIZE_MAX) {
            continue;
        }
        enqueue_index(neighbor_idx);
    }
}

/**
 * @brief Enforce the 2:1 octree balance rule on a refined octree.
 *
 * After the main breadth-first refinement, this function iteratively finds
 * pairs of adjacent leaf cells that violate the 2:1 balance rule (differing
 * by more than one refinement level) and splits the shallower leaf. The
 * process repeats until no violations remain.
 *
 * Each balance-forced split inherits contributors from the parent cell via
 * kernel-overlap filtering and samples corner values to support later surface
 * extraction.
 *
 * The algorithm uses a spatial hash of leaf min-corner positions for O(1)
 * neighbor lookups.  For each leaf, the 6 face-adjacent neighbors are
 * probed via the hash using a hierarchical search (finest to coarsest
 * alignment).  The hash is built once before the first round and updated
 * incrementally after each split (only new leaves are inserted).  Total
 * cost per iteration is O(n * max_depth) where n is the number of leaves.
 *
 * @param all_cells All octree cells (modified in place).
 * @param all_contributors Global flat contributor index array (modified in
 *     place).
 * @param positions Particle positions in world space.
 * @param smoothing_lengths Per-particle support radii.
 * @param isovalue The target surface level used to sample corner signs on
 *     newly created balance cells.
 * @param domain Full domain bounding box (for spatial hash construction).
 * @param base_resolution Number of top-level cells per axis.
 * @param max_depth Maximum depth; balance splits stop at this depth.
 */
inline void balance_octree(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    std::uint32_t max_depth) {
    ProgressCounter balance_counter(
        "Building", "balance_octree", "cells split", 10);

    // Build the spatial hash once; we will update it incrementally
    // as cells are split rather than rebuilding from scratch each
    // iteration.
    BalanceSpatialHash hash;
    hash.build(all_cells, domain, max_depth, base_resolution);

    std::vector<std::size_t> queue;
    queue.reserve(all_cells.size());
    std::vector<std::uint8_t> enqueued(all_cells.size(), 0U);
    for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
        if (!all_cells[cell_idx].is_leaf) {
            continue;
        }
        if (all_cells[cell_idx].depth >= max_depth) {
            continue;
        }
        queue.push_back(cell_idx);
        enqueued[cell_idx] = 1U;
    }

    std::size_t processed_count = 0U;
    std::size_t split_count = 0U;
    while (!queue.empty()) {
        const std::size_t split_index = queue.back();
        queue.pop_back();
        if (split_index < enqueued.size()) {
            enqueued[split_index] = 0U;
        }
        ++processed_count;

        if (split_index >= all_cells.size()) {
            continue;
        }
        if (!all_cells[split_index].is_leaf) {
            continue;
        }
        if (all_cells[split_index].depth >= max_depth) {
            continue;
        }
        if (!needs_balance_split(split_index, all_cells, hash, max_depth)) {
            continue;
        }

        const OctreeCell parent_snapshot = all_cells[split_index];
        const std::span<const std::size_t> parent_contributors =
            contributor_span(
                all_contributors,
                parent_snapshot.contributor_begin < 0 ? 0U :
                    static_cast<std::size_t>(parent_snapshot.contributor_begin),
                parent_snapshot.contributor_end <= parent_snapshot.contributor_begin ? 0U :
                    static_cast<std::size_t>(parent_snapshot.contributor_end));

        const std::vector<OctreeCell> children =
            create_child_cells(parent_snapshot);
        const std::vector<std::vector<std::size_t>> child_contributors =
            filter_child_contributors(
                parent_contributors, positions, smoothing_lengths, children);

        all_cells[split_index].is_leaf = false;
        all_cells[split_index].child_begin =
            static_cast<std::int64_t>(all_cells.size());

        std::uint32_t pgx, pgy, pgz;
        hash.quantize(parent_snapshot.bounds.min, pgx, pgy, pgz);
        hash.map.erase(balance_pack_coords(pgx, pgy, pgz));

        std::vector<std::size_t> affected_indices;
        affected_indices.reserve(1U + children.size());
        affected_indices.push_back(split_index);

        for (std::size_t ci = 0; ci < children.size(); ++ci) {
            OctreeCell child = children[ci];

            const std::int64_t contrib_begin =
                static_cast<std::int64_t>(all_contributors.size());
            for (std::size_t pidx : child_contributors[ci]) {
                all_contributors.push_back(pidx);
            }
            const std::int64_t contrib_end =
                static_cast<std::int64_t>(all_contributors.size());

            child.contributor_begin = contrib_begin;
            child.contributor_end = contrib_end;
            child.is_leaf = true;
            child.is_active = false;
            child.has_surface = false;
            child.is_topo_surface = false;

            if (!child_contributors[ci].empty()) {
                child.corner_values = sample_cell_corners(
                    child,
                    std::span<const std::size_t>(child_contributors[ci]),
                    positions,
                    smoothing_lengths);
                child.corner_sign_mask = compute_corner_sign_mask(
                    child.corner_values, isovalue);
                child.has_surface = cell_may_contain_isosurface(
                    child.corner_values, isovalue);
            }

            const std::size_t new_idx = all_cells.size();
            all_cells.push_back(child);
            affected_indices.push_back(new_idx);

            std::uint32_t cgx, cgy, cgz;
            hash.quantize(child.bounds.min, cgx, cgy, cgz);
            hash.map[balance_pack_coords(cgx, cgy, cgz)] = new_idx;
        }

        if (enqueued.size() < all_cells.size()) {
            enqueued.resize(all_cells.size(), 0U);
        }
        for (std::size_t affected_index : affected_indices) {
            enqueue_balance_neighbors(
                affected_index,
                all_cells,
                hash,
                max_depth,
                queue,
                enqueued);
        }

        ++split_count;
        balance_counter.tick();
    }

    meshmerizer_log_detail::print_status(
        "Building",
        "balance_octree",
        "local closure processed=%zu split=%zu (total_cells=%zu)\n",
        processed_count,
        split_count,
        all_cells.size());
    balance_counter.finish();
}

/**
 * @brief Breadth-first octree refinement followed by 2:1 balance enforcement.
 *
 * Starting from top-level cells, this function iteratively refines cells that
 * may contain the isosurface by evaluating corner field values. The algorithm
 * proceeds breadth-first (one depth level at a time), which simplifies later
 * parallelization and keeps the working set predictable.
 *
 * The surface-driven refinement stops for a cell when:
 * - The cell does not straddle the isovalue at any corner, or
 * - The maximum depth is reached, or
 * - The cell has fewer than two contributors.
 *
 * After the BFS completes, a separate balancing post-pass (``balance_octree``)
 * enforces the 2:1 rule: no two adjacent leaves may differ by more than one
 * refinement level. Balance-forced splits inherit contributors from their
 * parent and sample their own corner values.
 *
 * @param initial_cells Top-level cells with attached contributors.
 * @param positions Particle positions in world space.
 * @param smoothing_lengths Per-particle support radii.
 * @param isovalue The target surface level.
 * @param max_depth Maximum refinement depth (prevents infinite refinement).
 * @return Tuple of (all_cells, all_contributors) where cells store indices
 *         into the contributors vector.
 */
/**
 * @brief Overload that accepts pre-populated initial contributors.
 *
 * When initial cells are built in C++ (e.g., from the full pipeline),
 * the actual contributor particle indices are already known and stored
 * in @p initial_contributors.  Each cell's contributor_begin/end
 * index into this vector.  This avoids the legacy path that treats
 * contributor_begin..contributor_end as sequential particle indices.
 *
 * @param initial_cells Top-level cells with contributor ranges pointing
 *     into @p initial_contributors.
 * @param initial_contributors Pre-built flat contributor index array.
 * @param positions Particle positions.
 * @param smoothing_lengths Per-particle support radii.
 * @param isovalue Target surface level.
 * @param max_depth Maximum refinement depth.
 * @param domain Simulation domain bounding box.
 * @param base_resolution Top-level cells per axis.
 * @return Tuple of (all_cells, all_contributors).
 */
inline std::pair<std::vector<OctreeCell>, std::vector<std::size_t>>
refine_octree(
    std::vector<OctreeCell> initial_cells,
    std::vector<std::size_t> initial_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue,
    std::uint32_t max_depth,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    std::uint32_t minimum_usable_hermite_samples = 3U,
    double max_qef_rms_residual_ratio = 0.1,
    double min_normal_alignment_threshold = 0.97) {
    if (initial_cells.empty()) {
        return {{}, {}};
    }

    std::vector<OctreeCell> all_cells;
    // Start with the pre-built contributors; new contributors from
    // child splits will be appended after this initial segment.
    std::vector<std::size_t> all_contributors =
        std::move(initial_contributors);
    std::queue<std::size_t> leaf_queue;

    // Initial cells already have valid contributor_begin/end
    // pointing into all_contributors, so just copy them over.
    for (std::size_t cell_index = 0;
         cell_index < initial_cells.size(); ++cell_index) {
        all_cells.push_back(initial_cells[cell_index]);
        leaf_queue.push(all_cells.size() - 1);
    }

    ProgressCounter refine_counter(
        "Building", "refine_octree", "cells", 100);

    struct RefinementResult {
        bool has_surface = false;
        bool should_split = false;
        std::array<double, 8> corner_values = {
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        std::uint8_t corner_sign_mask = 0U;
        std::vector<OctreeCell> children;
        std::vector<std::vector<std::size_t>> child_contributors;
    };

    while (!leaf_queue.empty()) {
        const std::uint32_t batch_depth =
            all_cells[leaf_queue.front()].depth;
        std::vector<std::size_t> batch_indices;
        while (!leaf_queue.empty() &&
               all_cells[leaf_queue.front()].depth == batch_depth) {
            batch_indices.push_back(leaf_queue.front());
            leaf_queue.pop();
        }

        std::vector<RefinementResult> batch_results(batch_indices.size());

#pragma omp parallel for schedule(dynamic)
        for (std::size_t batch_i = 0; batch_i < batch_indices.size(); ++batch_i) {
            const std::size_t current_index = batch_indices[batch_i];
            const OctreeCell &current_cell = all_cells[current_index];
            RefinementResult &result = batch_results[batch_i];

            const std::int64_t contrib_begin = current_cell.contributor_begin;
            const std::int64_t contrib_end = current_cell.contributor_end;
            if (contrib_begin < 0 || contrib_end < 0 ||
                contrib_begin >= contrib_end) {
                continue;
            }

            const auto safe_begin = static_cast<std::size_t>(
                std::min(contrib_begin,
                         static_cast<std::int64_t>(all_contributors.size())));
            const auto safe_end = static_cast<std::size_t>(
                std::min(contrib_end,
                         static_cast<std::int64_t>(all_contributors.size())));
            const std::span<const std::size_t> contributors =
                contributor_span(all_contributors, safe_begin, safe_end);

            if (contributors.size() < 2) {
                continue;
            }

            result.corner_values = sample_cell_corners(
                current_cell, contributors, positions, smoothing_lengths);
            result.corner_sign_mask = compute_corner_sign_mask(
                result.corner_values, isovalue);

            const bool corner_surface = cell_may_contain_isosurface(
                result.corner_values, isovalue);
            const bool inherited_surface_hint =
                current_cell.has_surface && !corner_surface;
            result.has_surface = corner_surface || inherited_surface_hint;
            if (!result.has_surface) {
                continue;
            }

            if (current_cell.depth >= max_depth) {
                continue;
            }

            if (!corner_surface) {
                result.should_split = true;
                result.children = create_child_cells(current_cell);
                result.child_contributors = filter_child_contributors(
                    contributors, positions, smoothing_lengths,
                    result.children);
                continue;
            }

            const std::vector<HermiteSample> samples =
                compute_cell_hermite_samples(
                    current_cell.bounds, result.corner_values,
                    result.corner_sign_mask, contributors,
                    positions, smoothing_lengths, isovalue);
            const QEFLeafDiagnostics qef_diagnostics =
                analyze_qef_for_leaf(samples, current_cell.bounds);
            const double dx = current_cell.bounds.max.x - current_cell.bounds.min.x;
            const double dy = current_cell.bounds.max.y - current_cell.bounds.min.y;
            const double dz = current_cell.bounds.max.z - current_cell.bounds.min.z;
            const double cell_radius =
                0.5 * std::sqrt(dx * dx + dy * dy + dz * dz);
            const bool poor_qef_fit =
                qef_diagnostics.usable_sample_count < minimum_usable_hermite_samples ||
                qef_diagnostics.used_fallback ||
                minimum_hermite_normal_alignment(samples) <
                    min_normal_alignment_threshold ||
                qef_diagnostics.rms_plane_residual >
                    max_qef_rms_residual_ratio * cell_radius;
            if (!poor_qef_fit) {
                continue;
            }

            result.should_split = true;
            result.children = create_child_cells(current_cell);
            result.child_contributors = filter_child_contributors(
                contributors, positions, smoothing_lengths, result.children);
            for (OctreeCell &child : result.children) {
                child.has_surface = child_inherits_surface_hint(
                    child, samples, qef_diagnostics.vertex);
            }
        }

        for (std::size_t batch_i = 0; batch_i < batch_indices.size(); ++batch_i) {
            refine_counter.tick();
            OctreeCell &current_cell = all_cells[batch_indices[batch_i]];
            const RefinementResult &result = batch_results[batch_i];

            current_cell.corner_values = result.corner_values;
            current_cell.corner_sign_mask = result.corner_sign_mask;

            if (!result.has_surface) {
                current_cell.is_leaf = true;
                current_cell.is_active = false;
                current_cell.has_surface = false;
                current_cell.is_topo_surface = false;
                current_cell.child_begin = -1;
                continue;
            }

            if (!result.should_split) {
                current_cell.is_leaf = true;
                current_cell.is_active = true;
                current_cell.has_surface = true;
                current_cell.is_topo_surface = false;
                current_cell.child_begin = -1;
                continue;
            }

            current_cell.is_leaf = false;
            current_cell.is_active = true;
            current_cell.has_surface = true;
            current_cell.is_topo_surface = false;

            const std::int64_t child_begin_offset =
                static_cast<std::int64_t>(all_cells.size());
            current_cell.child_begin = child_begin_offset;

            for (std::size_t i = 0; i < result.children.size(); ++i) {
                const std::int64_t child_contrib_begin =
                    static_cast<std::int64_t>(all_contributors.size());
                std::copy(result.child_contributors[i].begin(),
                          result.child_contributors[i].end(),
                          std::back_inserter(all_contributors));
                const std::int64_t child_contrib_end =
                    static_cast<std::int64_t>(all_contributors.size());

                OctreeCell child = result.children[i];
                child.contributor_begin = child_contrib_begin;
                child.contributor_end = child_contrib_end;

                all_cells.push_back(child);
                leaf_queue.push(all_cells.size() - 1);
            }
        }
    }

    refine_counter.finish();

    // Enforce the 2:1 balance rule.
    balance_octree(all_cells, all_contributors, positions,
                   smoothing_lengths, isovalue, domain,
                   base_resolution, max_depth);

    return {all_cells, all_contributors};
}

inline std::pair<std::vector<OctreeCell>, std::vector<std::size_t>> refine_octree(
    std::vector<OctreeCell> initial_cells,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue,
    std::uint32_t max_depth,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    std::uint32_t minimum_usable_hermite_samples = 3U,
    double max_qef_rms_residual_ratio = 0.1,
    double min_normal_alignment_threshold = 0.97) {
    if (initial_cells.empty()) {
        return {{}, {}};
    }

    std::vector<std::size_t> initial_contributors;
    initial_contributors.reserve(positions.size());
    for (std::size_t cell_index = 0; cell_index < initial_cells.size();
         ++cell_index) {
        OctreeCell &cell = initial_cells[cell_index];
        const std::int64_t contrib_begin = cell.contributor_begin;
        const std::int64_t contrib_end = cell.contributor_end;

        if (contrib_begin >= 0 && contrib_end > contrib_begin) {
            const std::int64_t original_size =
                static_cast<std::int64_t>(initial_contributors.size());
            cell.contributor_begin = original_size;
            for (std::int64_t i = contrib_begin; i < contrib_end; ++i) {
                initial_contributors.push_back(static_cast<std::size_t>(i));
            }
            cell.contributor_end =
                static_cast<std::int64_t>(initial_contributors.size());
        } else {
            cell.contributor_begin = -1;
            cell.contributor_end = -1;
        }
    }

    return refine_octree(
        std::move(initial_cells), std::move(initial_contributors),
        positions, smoothing_lengths, isovalue, max_depth,
        domain, base_resolution,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold);
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_OCTREE_CELL_HPP_
