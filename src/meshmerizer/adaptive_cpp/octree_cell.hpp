/**
 * @file octree_cell.hpp
 * @brief Core adaptive octree cell type and refinement helpers.
 *
 * This header defines the flat-array octree representation used throughout the
 * adaptive meshing pipeline. The design goals are deterministic child ordering,
 * cache-friendly storage of cells and contributor ranges, and reusable helper
 * routines for refinement, closure propagation support, and diagnostics.
 *
 * The flat representation is important: instead of storing child pointers and
 * nested ownership, cells are appended to one vector and reference child ranges
 * and contributor slices by integer offsets. That representation is reused by
 * the Python binding layer, serialization code, and topology regularization
 * stages.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_OCTREE_CELL_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_OCTREE_CELL_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <span>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bounding_box.hpp"
#include "cancellation.hpp"
#include "hermite.hpp"
#include "kernel_wendland_c2.hpp"
#include "morton.hpp"
#include "particle_grid.hpp"
#include "qef.hpp"
#include "refinement_closure.hpp"
#include "vector3d.hpp"

/**
 * @brief One adaptive octree cell in flat-array storage.
 *
 * Invariants:
 * - ``morton_key`` and ``depth`` identify the logical cell location.
 * - ``bounds`` stores the geometric extent in domain coordinates.
 * - leaf cells have ``child_begin < 0`` and may carry contributor ranges.
 * - internal cells have ``child_begin`` pointing to the first of eight
 *   consecutively stored children.
 * - ``contributor_begin`` / ``contributor_end`` index a half-open slice in the
 *   shared contributor vector when contributor data is available.
 * - ``corner_values`` and ``corner_sign_mask`` describe the sampled field state
 *   used for refinement and meshing decisions.
 */
struct OctreeCell {
    /** Morton-compatible key encoding the cell's logical grid position. */
    std::uint64_t morton_key;
    /** Refinement depth measured from the top-level grid. */
    std::uint32_t depth;
    /** Geometric extent of the cell in world/domain coordinates. */
    BoundingBox bounds;
    /** Whether the cell currently has no children in the flat array. */
    bool is_leaf;
    /** Whether the cell is considered active for direct contour extraction. */
    bool is_active;
    /** Whether the sampled field suggests that a surface crosses the cell. */
    bool has_surface;
    /** Whether topology regularization marks the cell as a topology surface. */
    bool is_topo_surface;
    /** Index of the first child in ``all_cells``, or ``-1`` for leaves. */
    std::int64_t child_begin;
    /** Begin offset into the flat contributor array. */
    std::int64_t contributor_begin;
    /** End offset into the flat contributor array. */
    std::int64_t contributor_end;
    /** Representative mesh vertex index assigned during vertex solving. */
    std::int64_t representative_vertex_index;
    /** Scalar field samples at the eight cell corners. */
    std::array<double, 8> corner_values;
    /** Bit mask encoding ``corner_values >= isovalue``. */
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
        meshmerizer_log_detail::print_debug_status(
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

    meshmerizer_log_detail::print_debug_status(
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
        meshmerizer_log_detail::print_debug_status(
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
 *
 * This helper is the core structural mutation primitive used during adaptive
 * refinement. It preserves the flat-array storage model by decoding the parent
 * contributor slice, filtering those contributors into each child, appending
 * child contributor ranges to the shared contributor array, and then appending
 * the child cells contiguously to ``all_cells``.
 *
 * @param split_index Index of the leaf to split.
 * @param all_cells Flat cell array updated in place.
 * @param all_contributors Flat contributor array updated in place.
 * @param positions Particle positions in world space.
 * @param smoothing_lengths Per-particle support radii.
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
 *
 * @param bounds Candidate cell bounds.
 * @param point Query point.
 * @param epsilon Tolerance used for inclusive face checks.
 * @return ``true`` when the point lies inside or on the cell boundary.
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
 *
 * @param child Candidate child cell.
 * @param parent_samples Hermite samples computed on the parent.
 * @param parent_vertex Parent provisional QEF vertex.
 * @return ``true`` when the child should continue surface-focused refinement.
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
 *
 * @param samples Hermite samples used for one cell.
 * @return Minimum dot-product alignment against the mean usable normal.
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
    std::uint32_t worker_count = 1U,
    double table_cadence_seconds = 20.0,
    std::uint32_t minimum_usable_hermite_samples = 3U,
    double max_qef_rms_residual_ratio = 0.1,
    double min_normal_alignment_threshold = 0.97) {
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
        "Building",
        "refine_octree",
        "refine_octree",
    };

    return refine_with_closure(
        std::move(initial_cells),
        std::move(initial_contributors),
        positions,
        smoothing_lengths,
        closure_config);
}

inline std::pair<std::vector<OctreeCell>, std::vector<std::size_t>> refine_octree(
    std::vector<OctreeCell> initial_cells,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue,
    std::uint32_t max_depth,
    const BoundingBox &domain,
    std::uint32_t base_resolution,
    std::uint32_t worker_count = 1U,
    double table_cadence_seconds = 20.0,
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
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_index);
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
        domain, base_resolution, worker_count, table_cadence_seconds,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold);
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_OCTREE_CELL_HPP_
