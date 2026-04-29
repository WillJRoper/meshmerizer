/**
 * @file faces.hpp
 * @brief Dual contouring face generation from octree connectivity.
 *
 * Phase 9 of the adaptive meshing pipeline.  Given an octree whose active
 * leaf cells each carry a representative mesh vertex (from the QEF solve in
 * Phase 8), this module constructs the triangle mesh by finding every
 * sign-changing primal edge and connecting the representative vertices of
 * the (up to 4) leaf cells incident on that edge.
 *
 * The algorithm iterates every primal edge at the **finest grid resolution**
 * (``base_resolution * 2^max_depth`` cells per axis).  At each fine-grid
 * vertex the field sign is determined by trilinear interpolation of the
 * containing leaf cell's corner values.  For a sign-changing edge, the four
 * incident leaf cells are located via a hash-map spatial index and a quad
 * is emitted connecting their QEF vertices.  Quads are split into two
 * triangles with consistent winding determined by the sign-change direction.
 *
 * In an adaptive (2:1-balanced) octree, some of the four incident cells
 * may be the *same* cell (when a coarser cell covers multiple fine-grid
 * positions).  Degenerate quads where fewer than 3 distinct vertices
 * remain are discarded.
 *
 * Boundary edges (edges on the domain boundary where fewer than 4 cells
 * exist) are silently skipped.  The isosurface must be fully contained
 * within the domain interior for the resulting mesh to be watertight.
 *
 * Design for parallelism:
 * - The outer loop over fine-grid vertices can be partitioned across
 *   threads with thread-local triangle buffers.
 * - Vertex solving is per-leaf and independent, suitable for parallel
 *   for-each.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_FACES_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_FACES_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "hermite.hpp"
#include "mesh.hpp"
#include "octree_cell.hpp"
#include "cancellation.hpp"
#include "omp_config.hpp"
#include "progress_bar.hpp"
#include "qef.hpp"
#include "refinement_closure.hpp"
#include "vector3d.hpp"

/**
 * @brief Spatial index mapping fine-grid positions to leaf cell indices.
 *
 * Uses a hash map keyed by a quantized grid position to provide O(1)
 * lookup of the leaf cell that contains a given world-space point.
 * The quantization grid matches the finest octree level so that each
 * fine-grid position maps to at most one leaf cell.
 *
 * Memory efficiency: each leaf cell registers exactly **one** entry in
 * the hash map, keyed by its min-corner fine-grid coordinate.  Lookups
 * use a hierarchical probe: starting at the queried fine-grid position,
 * progressively round down to coarser alignments (stride 1, 2, 4, ...)
 * until the containing leaf is found.  This costs at most
 * ``max_depth + 1`` hash probes per lookup but reduces memory from
 * O(fine_res^3) to O(num_leaves).
 */
struct LeafSpatialIndex {
    /**
     * @brief Hash key encoding a quantized (ix, iy, iz) triple.
     *
     * Uses bit packing: 21 bits per axis, which matches the Morton
     * key encoding capacity of the codebase.
     */
    struct GridKey {
        std::uint64_t packed;

        bool operator==(const GridKey &other) const {
            return packed == other.packed;
        }
    };

    /** @brief Hash function for GridKey using a simple bit mixer. */
    struct GridKeyHash {
        std::size_t operator()(const GridKey &key) const {
            // FNV-1a-style mixer.
            std::uint64_t h = key.packed;
            h ^= h >> 33U;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33U;
            h *= 0xc4ceb9fe1a85ec53ULL;
            h ^= h >> 33U;
            return static_cast<std::size_t>(h);
        }
    };

    /** @brief Map from leaf min-corner grid position to cell index. */
    std::unordered_map<GridKey, std::size_t, GridKeyHash> lookup;

    /** @brief Inverse of the finest cell edge length. */
    double inv_cell_size_x;
    double inv_cell_size_y;
    double inv_cell_size_z;

    /** @brief Domain minimum corner used for quantization. */
    Vector3d domain_min;

    /** @brief Maximum octree depth, cached for hierarchical probe. */
    std::uint32_t cached_max_depth;

    /** @brief Finest-grid resolution per axis. */
    std::uint32_t fine_resolution;

    /**
     * @brief Build the spatial index from a set of octree cells.
     *
     * Only leaf cells are indexed.  Each leaf registers exactly one
     * entry keyed by its min-corner fine-grid coordinate.
     *
     * @param all_cells All octree cells (leaves and internal nodes).
     * @param domain The full simulation domain bounding box.
     * @param max_depth Maximum octree depth (determines fine-grid resolution).
     * @param base_resolution Number of top-level cells per axis.
     */
    void build(const std::vector<OctreeCell> &all_cells,
               const BoundingBox &domain,
               std::uint32_t max_depth,
               std::uint32_t base_resolution) {
        // Guard: max_depth must be small enough that
        // base_resolution * 2^max_depth fits in 21 bits (the
        // pack_grid_coords capacity) and does not overflow uint32.
        if (max_depth >= 21U) {
            throw std::overflow_error(
                "max_depth >= 21 would overflow 21-bit grid "
                "coordinate packing");
        }
        // Compute in uint64_t to detect overflow before narrowing.
        const std::uint64_t fine_res_wide =
            static_cast<std::uint64_t>(base_resolution) *
            (1ULL << max_depth);
        constexpr std::uint64_t MAX_COORD = (1ULL << 21U) - 1U;
        if (fine_res_wide > MAX_COORD) {
            throw std::overflow_error(
                "fine_res exceeds 21-bit grid coordinate capacity");
        }

        cached_max_depth = max_depth;
        fine_resolution = static_cast<std::uint32_t>(fine_res_wide);

        // The finest grid has base_resolution * 2^max_depth cells per axis.
        const double fine_cells_per_axis_x =
            static_cast<double>(base_resolution) *
            static_cast<double>(1U << max_depth);
        const double fine_cells_per_axis_y = fine_cells_per_axis_x;
        const double fine_cells_per_axis_z = fine_cells_per_axis_x;

        const double domain_size_x = domain.max.x - domain.min.x;
        const double domain_size_y = domain.max.y - domain.min.y;
        const double domain_size_z = domain.max.z - domain.min.z;

        const double fine_cell_size_x =
            domain_size_x / fine_cells_per_axis_x;
        const double fine_cell_size_y =
            domain_size_y / fine_cells_per_axis_y;
        const double fine_cell_size_z =
            domain_size_z / fine_cells_per_axis_z;

        inv_cell_size_x = 1.0 / fine_cell_size_x;
        inv_cell_size_y = 1.0 / fine_cell_size_y;
        inv_cell_size_z = 1.0 / fine_cell_size_z;
        domain_min = domain.min;

        lookup.clear();

        // Count leaves first so we can reserve the hash map capacity
        // and avoid rehashing during insertion.
        std::size_t n_leaves = 0;
        for (const auto &cell : all_cells) {
            if (cell.is_leaf) {
                ++n_leaves;
            }
        }
        lookup.reserve(n_leaves);

        ProgressCounter build_counter(
            "Regularization",
            "LeafSpatialIndex::build",
            "leaves",
            10000);
        for (std::size_t cell_idx = 0; cell_idx < all_cells.size();
             ++cell_idx) {
            const OctreeCell &cell = all_cells[cell_idx];
            if (!cell.is_leaf) {
                continue;
            }
            build_counter.tick();

            // Quantize the cell's min corner to fine-grid coordinates.
            // This gives the unique key for this leaf.
            const std::uint32_t ix_min = static_cast<std::uint32_t>(
                (cell.bounds.min.x - domain_min.x) * inv_cell_size_x +
                0.5);
            const std::uint32_t iy_min = static_cast<std::uint32_t>(
                (cell.bounds.min.y - domain_min.y) * inv_cell_size_y +
                0.5);
            const std::uint32_t iz_min = static_cast<std::uint32_t>(
                (cell.bounds.min.z - domain_min.z) * inv_cell_size_z +
                0.5);

            const GridKey key = {pack_grid_coords(
                ix_min, iy_min, iz_min)};
            lookup[key] = cell_idx;
        }
        build_counter.finish();
    }

    /**
     * @brief Pack three grid coordinates into a single 64-bit key.
     *
     * @param ix X grid coordinate.
     * @param iy Y grid coordinate.
     * @param iz Z grid coordinate.
     * @return Packed 64-bit key.
     */
    static std::uint64_t pack_grid_coords(std::uint32_t ix,
                                           std::uint32_t iy,
                                           std::uint32_t iz) {
        // Each axis gets 21 bits.  Values >= 2^21 would collide with
        // adjacent fields, producing silent hash collisions.
        constexpr std::uint32_t MAX_COORD = (1U << 21U) - 1U;
        if (ix > MAX_COORD || iy > MAX_COORD || iz > MAX_COORD) {
            throw std::overflow_error(
                "Grid coordinate exceeds 21-bit capacity in "
                "pack_grid_coords");
        }
        return (static_cast<std::uint64_t>(ix) << 42U) |
               (static_cast<std::uint64_t>(iy) << 21U) |
               static_cast<std::uint64_t>(iz);
    }

    /**
     * @brief Look up the leaf cell index at a given fine-grid position.
     *
     * Uses a hierarchical probe: starting at stride 1 (finest level),
     * rounds the query coordinates down to progressively coarser
     * alignments until a matching leaf is found.  A cell at depth d
     * has its min corner aligned to stride ``2^(max_depth - d)``, so
     * rounding down to that stride recovers the cell's key.
     *
     * Worst case: ``max_depth + 1`` hash probes.
     *
     * @param ix X fine-grid coordinate.
     * @param iy Y fine-grid coordinate.
     * @param iz Z fine-grid coordinate.
     * @return Index into the all_cells array, or SIZE_MAX if not found.
     */
    std::size_t find_leaf_at(std::uint32_t ix, std::uint32_t iy,
                             std::uint32_t iz) const {
        // Probe from finest to coarsest.  At each level, round the
        // query coordinates down to the alignment of that level.
        // Level 0 corresponds to max_depth (stride 1, finest cells).
        // Level k corresponds to depth (max_depth - k) with stride 2^k.
        for (std::uint32_t k = 0; k <= cached_max_depth; ++k) {
            const std::uint32_t mask = ~((1U << k) - 1U);
            const std::uint32_t ax = ix & mask;
            const std::uint32_t ay = iy & mask;
            const std::uint32_t az = iz & mask;
            const GridKey key = {pack_grid_coords(ax, ay, az)};
            auto it = lookup.find(key);
            if (it != lookup.end()) {
                return it->second;
            }
        }
        return SIZE_MAX;
    }

    /**
     * @brief Quantize a cell's min corner to fine-grid coordinates.
     *
     * @param cell_min Minimum corner of the cell in world space.
     * @param ix Output X fine-grid coordinate.
     * @param iy Output Y fine-grid coordinate.
     * @param iz Output Z fine-grid coordinate.
     */
    void quantize_min_corner(const Vector3d &cell_min,
                             std::uint32_t &ix,
                             std::uint32_t &iy,
                             std::uint32_t &iz) const {
        ix = static_cast<std::uint32_t>(
            (cell_min.x - domain_min.x) * inv_cell_size_x + 0.5);
        iy = static_cast<std::uint32_t>(
            (cell_min.y - domain_min.y) * inv_cell_size_y + 0.5);
        iz = static_cast<std::uint32_t>(
            (cell_min.z - domain_min.z) * inv_cell_size_z + 0.5);
    }
};

/**
 * @brief Solve QEF vertices for all active leaf cells and assign indices.
 *
 * This function iterates over all cells, identifies active leaf cells
 * (leaves with a sign-changing surface), solves the QEF for each, and
 * stores the resulting mesh vertex.  Each active leaf's
 * ``representative_vertex_index`` is set to the index into the returned
 * vertex array.
 *
 * A leaf is considered active if it is a leaf with ``has_surface`` set
 * (meaning its corner values straddle the isovalue).
 *
 * @param all_cells All octree cells (modified: ``representative_vertex_index``
 *     is set for active leaves).
 * @param all_contributors Flat contributor index array.
 * @param positions Particle positions.
 * @param smoothing_lengths Per-particle support radii.
 * @param isovalue Target surface level.
 * @return Array of mesh vertices, one per active leaf.
 */
inline std::vector<MeshVertex> solve_all_leaf_vertices(
    std::vector<OctreeCell> &all_cells,
    const std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue) {
    const std::size_t n_cells = all_cells.size();

    // Per-cell results computed in parallel.  Each entry is either a
    // valid MeshVertex (when the cell is an active surface leaf) or
    // an empty placeholder.  A separate flag marks which cells are
    // active so the serial index-assignment pass can skip non-active
    // cells cheaply.
    std::vector<MeshVertex> per_cell_vertex(n_cells);
    // Use char instead of bool to avoid the std::vector<bool> bit-packing
    // specialisation, which causes data races when adjacent elements are
    // written by different threads (they share the same byte).
    std::vector<char> is_active_leaf(n_cells, 0);

    ProgressBar vertex_bar("Meshing", "solve_vertices", n_cells);

    // Pass 1 (parallel): For each leaf cell, lazily sample corners if
    // needed, compute Hermite samples, and solve the QEF.  Each
    // iteration touches only its own cell and reads shared immutable
    // data (positions, smoothing_lengths, all_contributors), so there
    // are no data races.
#pragma omp parallel for schedule(dynamic)
    for (std::size_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                cell_idx)) {
            vertex_bar.tick();
            continue;
        }
        OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf) {
            vertex_bar.tick();
            continue;
        }

        // Gather contributor indices for this cell.  Construct the
        // vector directly from the iterator range into
        // all_contributors, avoiding per-element push_back overhead.
        // Validate the range to prevent out-of-bounds access from
        // corrupt cell data.
        std::span<const std::size_t> contributors;
        if (cell.contributor_begin >= 0 &&
            cell.contributor_end > cell.contributor_begin) {
            const auto begin_idx =
                static_cast<std::size_t>(cell.contributor_begin);
            const auto end_idx =
                static_cast<std::size_t>(cell.contributor_end);
            if (end_idx > all_contributors.size()) {
                // Cannot throw from an OpenMP region on all
                // compilers; mark the cell inactive and continue.
                vertex_bar.tick();
                continue;
            }
            contributors = contributor_span(
                all_contributors, begin_idx, end_idx);
        }

        // If corner values were not sampled during refinement (e.g.,
        // the cell was at max depth or had too few contributors at
        // refinement time), sample them now so we can determine
        // whether this leaf actually straddles the isosurface.
        if (!cell.has_surface && !contributors.empty()) {
            cell.corner_values = sample_cell_corners(
                cell, contributors, positions, smoothing_lengths);
            cell.corner_sign_mask = compute_corner_sign_mask(
                cell.corner_values, isovalue);
            cell.has_surface = cell_may_contain_isosurface(
                cell.corner_values, isovalue);
        }

        if (!cell.has_surface) {
            vertex_bar.tick();
            continue;
        }

        // If no sign change, skip (safety check).  The corner_sign_mask
        // was already set during refinement (octree_cell.hpp) or in the
        // lazy sampling block above, so no recomputation is needed here.
        if (cell.corner_sign_mask == 0U ||
            cell.corner_sign_mask == 0xFFU) {
            cell.has_surface = false;
            vertex_bar.tick();
            continue;
        }

        // Compute Hermite samples for this leaf.
        const std::vector<HermiteSample> samples =
            compute_cell_hermite_samples(
                cell.bounds, cell.corner_values,
                cell.corner_sign_mask, contributors,
                positions, smoothing_lengths, isovalue);

        // Solve the QEF.
        per_cell_vertex[cell_idx] =
            solve_qef_for_leaf(samples, cell.bounds);
        is_active_leaf[cell_idx] = 1;
        vertex_bar.tick();
    }

    vertex_bar.finish();

    // Pass 2 (serial): Assign contiguous vertex indices and build
    // the compact output vertex array.  This must be serial because
    // vertex indices must be dense and deterministic.
    std::vector<MeshVertex> vertices;
    for (std::size_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
        OctreeCell &cell = all_cells[cell_idx];
        if (is_active_leaf[cell_idx]) {
            cell.representative_vertex_index =
                static_cast<std::int64_t>(vertices.size());
            cell.is_active = true;
            vertices.push_back(per_cell_vertex[cell_idx]);
        } else {
            cell.representative_vertex_index = -1;
            cell.is_active = false;
        }
    }

    return vertices;
}

/**
 * @brief Emit a quad (as two triangles) from four incident cell indices.
 *
 * Validates that all four cells are found, have representative vertices,
 * and that at least 3 distinct vertices exist.  Degenerate quads with
 * fewer than 3 distinct vertices are silently discarded.
 *
 * Winding order: when ``sign_a_above`` is true (the starting corner is
 * above the isovalue, i.e., inside the fluid), the quad normal should
 * point away from the fluid.  The winding is chosen so that the outward
 * normal (by the right-hand rule) points toward the "below" side.
 *
 * The quad is formed by vertices (v0, v1, v2, v3) and split into
 * triangles (v0, v1, v2) and (v0, v2, v3).  If sign_a_above is true,
 * the winding is reversed to (v0, v2, v1) and (v0, v3, v2).
 *
 * @param all_cells All octree cells.
 * @param vertices Mesh vertices.
 * @param triangles Output triangle array (appended to).
 * @param c0 Index of first incident cell.
 * @param c1 Index of second incident cell.
 * @param c2 Index of third incident cell.
 * @param c3 Index of fourth incident cell.
 * @param sign_a_above Whether the starting corner is above the isovalue.
 */
inline void emit_quad(
    const std::vector<OctreeCell> &all_cells,
    const std::vector<MeshVertex> &vertices,
    std::vector<MeshTriangle> &triangles,
    std::size_t c0, std::size_t c1,
    std::size_t c2, std::size_t c3,
    bool sign_a_above) {
    // Validate all four cell lookups succeeded.
    if (c0 == SIZE_MAX || c1 == SIZE_MAX ||
        c2 == SIZE_MAX || c3 == SIZE_MAX) {
        return;
    }

    // Get representative vertex indices for all four cells.
    const std::int64_t vi0 =
        all_cells[c0].representative_vertex_index;
    const std::int64_t vi1 =
        all_cells[c1].representative_vertex_index;
    const std::int64_t vi2 =
        all_cells[c2].representative_vertex_index;
    const std::int64_t vi3 =
        all_cells[c3].representative_vertex_index;

    // All four cells must have representative vertices (be active).
    if (vi0 < 0 || vi1 < 0 || vi2 < 0 || vi3 < 0) {
        return;
    }

    const std::size_t v0 = static_cast<std::size_t>(vi0);
    const std::size_t v1 = static_cast<std::size_t>(vi1);
    const std::size_t v2 = static_cast<std::size_t>(vi2);
    const std::size_t v3 = static_cast<std::size_t>(vi3);

    // Count distinct vertices and collect the first 3 in cyclic order.
    // With adaptive octrees, some incident cells may be the same coarse
    // cell, giving duplicate vertex indices.  We walk the cyclic
    // sequence (v0, v1, v2, v3) and keep vertices that differ from
    // all previously kept ones.  This preserves the cyclic winding
    // order, which is essential for correct triangle orientation.
    const std::size_t cyclic[4] = {v0, v1, v2, v3};
    std::size_t unique[4];
    std::size_t num_unique = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        bool is_dup = false;
        for (std::size_t j = 0; j < num_unique; ++j) {
            if (cyclic[i] == unique[j]) {
                is_dup = true;
                break;
            }
        }
        if (!is_dup) {
            unique[num_unique++] = cyclic[i];
        }
    }

    if (num_unique < 3) {
        // Degenerate: fewer than 3 distinct vertices, no valid triangle.
        return;
    }

    // Emit two triangles for the quad.  The winding direction is
    // determined by whether the starting corner is above or below the
    // isovalue.  When above (inside fluid), we reverse the winding so
    // the outward normal points away from the fluid.
    if (sign_a_above) {
        // Reversed winding for outward normal away from fluid.
        if (num_unique == 4) {
            triangles.push_back({{v0, v2, v1}});
            triangles.push_back({{v0, v3, v2}});
        } else {
            // 3 distinct vertices: emit one triangle.
            triangles.push_back({{unique[0], unique[2], unique[1]}});
        }
    } else {
        // Standard winding.
        if (num_unique == 4) {
            triangles.push_back({{v0, v1, v2}});
            triangles.push_back({{v0, v2, v3}});
        } else {
            triangles.push_back({{unique[0], unique[1], unique[2]}});
        }
    }
}

/**
 * @brief Trilinearly interpolate the sign at a fine-grid vertex.
 *
 * Given a fine-grid vertex position ``(ix, iy, iz)`` and the leaf cell
 * that contains it, computes the field value at that vertex by trilinear
 * interpolation of the cell's 8 corner values.  Returns true if the
 * interpolated value is above the isovalue (i.e., "inside" the fluid).
 *
 * For cells at the finest depth (span = 1), the fine-grid vertex
 * coincides with corner 0 and no interpolation is needed.
 *
 * @param cell        Leaf cell containing the fine-grid vertex.
 * @param ix          Fine-grid X coordinate of the vertex.
 * @param iy          Fine-grid Y coordinate of the vertex.
 * @param iz          Fine-grid Z coordinate of the vertex.
 * @param spatial_index Spatial index (used for quantization parameters).
 * @param max_depth   Maximum octree depth (to compute cell span).
 * @param isovalue    Target isovalue.
 * @return True if the interpolated field value >= isovalue.
 */
inline bool sign_at_fine_vertex(
    const OctreeCell &cell,
    std::uint32_t ix, std::uint32_t iy, std::uint32_t iz,
    const LeafSpatialIndex &spatial_index,
    std::uint32_t max_depth,
    double isovalue) {
    // The cell spans 'span' fine-grid cells per axis.  A cell at depth
    // d covers 2^(max_depth - d) fine cells along each axis.
    // Guard: cell.depth must not exceed max_depth.
    if (cell.depth > max_depth) {
        return false;  // Defensive: should never happen.
    }
    const std::uint32_t span = 1U << (max_depth - cell.depth);

    // For finest-level cells (span=1), the fine-grid vertex IS the min
    // corner — just read corner 0 directly, no interpolation needed.
    if (span == 1U) {
        return cell.corner_values[0] >= isovalue;
    }

    // Use the octree lattice coordinates encoded by the Morton key rather than
    // re-quantizing world-space bounds. This matches the previously working
    // implementation more closely and avoids rounding mismatches between the
    // octree lattice and world-space quantization.
    std::uint32_t cell_x = 0U, cell_y = 0U, cell_z = 0U;
    morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
    const std::uint32_t cx = cell_x << (max_depth - cell.depth);
    const std::uint32_t cy = cell_y << (max_depth - cell.depth);
    const std::uint32_t cz = cell_z << (max_depth - cell.depth);

    // corner_values are at the 8 corners of the cell bounding box:
    //   corner 0 = (cx, cy, cz),  corner 7 = (cx+span, cy+span, cz+span).
    // Compute the fractional position of (ix,iy,iz) within the cell
    // in [0, 1] per axis, then trilinearly interpolate.
    const double fx =
        static_cast<double>(ix - cx) / static_cast<double>(span);
    const double fy =
        static_cast<double>(iy - cy) / static_cast<double>(span);
    const double fz =
        static_cast<double>(iz - cz) / static_cast<double>(span);

    // Trilinear interpolation of corner values.
    // Corner indexing: index = 4*dz + 2*dy + dx, where dx/dy/dz are
    // 0 (low) or 1 (high) along each axis.  Variable names below use
    // the pattern v_<dx><dy><dz> to match this convention.
    const double v_000 = cell.corner_values[0];  // dx=0, dy=0, dz=0
    const double v_100 = cell.corner_values[1];  // dx=1, dy=0, dz=0
    const double v_010 = cell.corner_values[2];  // dx=0, dy=1, dz=0
    const double v_110 = cell.corner_values[3];  // dx=1, dy=1, dz=0
    const double v_001 = cell.corner_values[4];  // dx=0, dy=0, dz=1
    const double v_101 = cell.corner_values[5];  // dx=1, dy=0, dz=1
    const double v_011 = cell.corner_values[6];  // dx=0, dy=1, dz=1
    const double v_111 = cell.corner_values[7];  // dx=1, dy=1, dz=1

    // Interpolate along X first (dx direction).
    const double vx_00 = v_000 + fx * (v_100 - v_000);
    const double vx_01 = v_010 + fx * (v_110 - v_010);
    const double vx_10 = v_001 + fx * (v_101 - v_001);
    const double vx_11 = v_011 + fx * (v_111 - v_011);

    // Then along Y (dy direction).
    const double vxy_0 = vx_00 + fy * (vx_01 - vx_00);
    const double vxy_1 = vx_10 + fy * (vx_11 - vx_10);

    // Then along Z (dz direction).
    const double value = vxy_0 + fz * (vxy_1 - vxy_0);

    return value >= isovalue;
}

/**
 * @brief Gather contributor indices for one cell.
 */
inline std::span<const std::size_t> gather_face_cell_contributors(
    const OctreeCell &cell,
    const std::vector<std::size_t> &all_contributors) {
    if (cell.contributor_begin < 0 ||
        cell.contributor_end <= cell.contributor_begin) {
        return std::span<const std::size_t>();
    }
    const auto begin_idx = static_cast<std::size_t>(cell.contributor_begin);
    const auto end_idx = static_cast<std::size_t>(cell.contributor_end);
    if (end_idx > all_contributors.size()) {
        return std::span<const std::size_t>();
    }
    return contributor_span(all_contributors, begin_idx, end_idx);
}

/**
 * @brief Collect leaves incident to sign-changing fine-grid edges that still
 *        lack representative vertices.
 */
inline std::vector<std::size_t> collect_missing_incident_cells(
    const std::vector<OctreeCell> &all_cells,
    const LeafSpatialIndex &spatial_index,
    std::uint32_t max_depth,
    std::uint32_t base_resolution,
    double isovalue) {
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
    const std::uint32_t fine_res = static_cast<std::uint32_t>(fine_res_wide);

    std::vector<char> needs_vertex(all_cells.size(), 0);
    ProgressBar activation_bar(
        "Meshing", "activate_incident_cells", static_cast<std::size_t>(fine_res));

#pragma omp parallel
    {
        std::vector<char> local_needs_vertex(all_cells.size(), 0);

#pragma omp for schedule(dynamic)
        for (std::uint32_t ix = 0; ix < fine_res; ++ix) {
            if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(ix)) {
                activation_bar.tick();
                continue;
            }
            for (std::uint32_t iy = 0; iy < fine_res; ++iy) {
                for (std::uint32_t iz = 0; iz < fine_res; ++iz) {
                    const std::size_t cell_a_idx =
                        spatial_index.find_leaf_at(ix, iy, iz);
                    if (cell_a_idx == SIZE_MAX) {
                        continue;
                    }
                    const OctreeCell &cell_a = all_cells[cell_a_idx];
                    const bool sign_a = sign_at_fine_vertex(
                        cell_a, ix, iy, iz, spatial_index, max_depth,
                        isovalue);

                    if (ix + 1U < fine_res) {
                        const std::size_t cell_b_idx =
                            spatial_index.find_leaf_at(ix + 1U, iy, iz);
                        if (cell_b_idx != SIZE_MAX) {
                            const bool sign_b = sign_at_fine_vertex(
                                all_cells[cell_b_idx], ix + 1U, iy, iz,
                                spatial_index, max_depth, isovalue);
                            if (sign_a != sign_b && iy > 0U && iz > 0U) {
                                const std::size_t incident[4] = {
                                    cell_a_idx,
                                    spatial_index.find_leaf_at(ix, iy, iz - 1U),
                                    spatial_index.find_leaf_at(
                                        ix, iy - 1U, iz - 1U),
                                    spatial_index.find_leaf_at(ix, iy - 1U, iz),
                                };
                                for (std::size_t ci : incident) {
                                    if (ci != SIZE_MAX && all_cells[ci].is_leaf &&
                                        all_cells[ci].representative_vertex_index < 0) {
                                        local_needs_vertex[ci] = 1;
                                    }
                                }
                            }
                        }
                    }

                    if (iy + 1U < fine_res) {
                        const std::size_t cell_c_idx =
                            spatial_index.find_leaf_at(ix, iy + 1U, iz);
                        if (cell_c_idx != SIZE_MAX) {
                            const bool sign_c = sign_at_fine_vertex(
                                all_cells[cell_c_idx], ix, iy + 1U, iz,
                                spatial_index, max_depth, isovalue);
                            if (sign_a != sign_c && ix > 0U && iz > 0U) {
                                const std::size_t incident[4] = {
                                    cell_a_idx,
                                    spatial_index.find_leaf_at(ix - 1U, iy, iz),
                                    spatial_index.find_leaf_at(
                                        ix - 1U, iy, iz - 1U),
                                    spatial_index.find_leaf_at(ix, iy, iz - 1U),
                                };
                                for (std::size_t ci : incident) {
                                    if (ci != SIZE_MAX && all_cells[ci].is_leaf &&
                                        all_cells[ci].representative_vertex_index < 0) {
                                        local_needs_vertex[ci] = 1;
                                    }
                                }
                            }
                        }
                    }

                    if (iz + 1U < fine_res) {
                        const std::size_t cell_d_idx =
                            spatial_index.find_leaf_at(ix, iy, iz + 1U);
                        if (cell_d_idx != SIZE_MAX) {
                            const bool sign_d = sign_at_fine_vertex(
                                all_cells[cell_d_idx], ix, iy, iz + 1U,
                                spatial_index, max_depth, isovalue);
                            if (sign_a != sign_d && ix > 0U && iy > 0U) {
                                const std::size_t incident[4] = {
                                    cell_a_idx,
                                    spatial_index.find_leaf_at(ix, iy - 1U, iz),
                                    spatial_index.find_leaf_at(
                                        ix - 1U, iy - 1U, iz),
                                    spatial_index.find_leaf_at(ix - 1U, iy, iz),
                                };
                                for (std::size_t ci : incident) {
                                    if (ci != SIZE_MAX && all_cells[ci].is_leaf &&
                                        all_cells[ci].representative_vertex_index < 0) {
                                        local_needs_vertex[ci] = 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            activation_bar.tick();
        }

#pragma omp critical
        {
            for (std::size_t i = 0; i < all_cells.size(); ++i) {
                if (local_needs_vertex[i]) {
                    needs_vertex[i] = 1;
                }
            }
        }
    }

    activation_bar.finish();

    std::vector<std::size_t> missing_cells;
    for (std::size_t i = 0; i < all_cells.size(); ++i) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(i);
        if (needs_vertex[i]) {
            missing_cells.push_back(i);
        }
    }

    // Expand over a bounded local 26-neighbourhood. A true fixed-point
    // closure can percolate through huge connected regions of the octree and
    // effectively never finish on large datasets, so keep the expansion local.
    std::vector<char> expanded_needs_vertex = needs_vertex;
    constexpr std::size_t MAX_CLOSURE_ITERATIONS = 2;
    for (std::size_t closure_iteration = 1;
         closure_iteration <= MAX_CLOSURE_ITERATIONS;
         ++closure_iteration) {
        std::vector<char> next_needs_vertex = expanded_needs_vertex;
        ProgressCounter closure_counter(
            "Meshing", "close_incident_neighbourhood", "cells", 100);
        std::size_t newly_marked = 0;
        for (std::size_t cell_idx = 0; cell_idx < all_cells.size(); ++cell_idx) {
            meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
            closure_counter.tick();
            if (!expanded_needs_vertex[cell_idx]) {
                continue;
            }
            const OctreeCell &cell = all_cells[cell_idx];
            std::uint32_t ix = 0U, iy = 0U, iz = 0U;
            spatial_index.quantize_min_corner(cell.bounds.min, ix, iy, iz);
            const std::uint32_t span = 1U << (max_depth - cell.depth);

            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (dx == 0 && dy == 0 && dz == 0) {
                            continue;
                        }

                        const std::int64_t px =
                            dx < 0 ? static_cast<std::int64_t>(ix) - 1LL
                                   : dx > 0 ? static_cast<std::int64_t>(ix + span)
                                            : static_cast<std::int64_t>(ix);
                        const std::int64_t py =
                            dy < 0 ? static_cast<std::int64_t>(iy) - 1LL
                                   : dy > 0 ? static_cast<std::int64_t>(iy + span)
                                            : static_cast<std::int64_t>(iy);
                        const std::int64_t pz =
                            dz < 0 ? static_cast<std::int64_t>(iz) - 1LL
                                   : dz > 0 ? static_cast<std::int64_t>(iz + span)
                                            : static_cast<std::int64_t>(iz);
                        if (px < 0 || py < 0 || pz < 0 ||
                            px >= static_cast<std::int64_t>(fine_res) ||
                            py >= static_cast<std::int64_t>(fine_res) ||
                            pz >= static_cast<std::int64_t>(fine_res)) {
                            continue;
                        }

                        const std::size_t neighbor_idx = spatial_index.find_leaf_at(
                            static_cast<std::uint32_t>(px),
                            static_cast<std::uint32_t>(py),
                            static_cast<std::uint32_t>(pz));
                        if (neighbor_idx != SIZE_MAX && all_cells[neighbor_idx].is_leaf &&
                            all_cells[neighbor_idx].representative_vertex_index < 0 &&
                            !next_needs_vertex[neighbor_idx]) {
                            next_needs_vertex[neighbor_idx] = 1;
                            ++newly_marked;
                        }
                    }
                }
            }
        }
        closure_counter.finish();
        meshmerizer_log_detail::print_debug_status(
            "Meshing",
            "close_incident_neighbourhood",
            "iteration %zu complete (+%zu cells)\n",
            closure_iteration,
            newly_marked);
        expanded_needs_vertex.swap(next_needs_vertex);
        if (newly_marked == 0) {
            break;
        }
    }

    std::vector<std::size_t> expanded_missing_cells;
    for (std::size_t i = 0; i < all_cells.size(); ++i) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(i);
        if (expanded_needs_vertex[i]) {
            expanded_missing_cells.push_back(i);
        }
    }
    return expanded_missing_cells;
}

/**
 * @brief Refine topology-required incident cells that still have no local
 *        Hermite support.
 *
 * Returns true when at least one missing incident cell was split, so the
 * caller can rebuild the spatial index and re-solve leaf vertices.
 */
inline bool refine_zero_sample_incident_cells(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const LeafSpatialIndex &spatial_index,
    std::uint32_t max_depth,
    std::uint32_t base_resolution,
    double isovalue,
    const BoundingBox &domain,
    double table_cadence_seconds = 0.0) {
    const std::size_t initial_cell_count = all_cells.size();
    const std::vector<std::size_t> missing_cells =
        collect_missing_incident_cells(
            all_cells, spatial_index, max_depth, base_resolution, isovalue);

    std::vector<std::size_t> zero_sample_cells;
    zero_sample_cells.reserve(missing_cells.size());
    std::size_t zero_sample_count = 0;
    for (std::size_t cell_idx : missing_cells) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
        if (cell_idx >= all_cells.size()) {
            continue;
        }
        OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf || cell.representative_vertex_index >= 0) {
            continue;
        }

        const std::span<const std::size_t> contributors =
            gather_face_cell_contributors(cell, all_contributors);
        if (cell.corner_sign_mask == 0U && !contributors.empty()) {
            cell.corner_values = sample_cell_corners(
                cell, contributors, positions, smoothing_lengths);
            cell.corner_sign_mask = compute_corner_sign_mask(
                cell.corner_values, isovalue);
        }
        const std::vector<HermiteSample> samples =
            compute_cell_hermite_samples(
                cell.bounds, cell.corner_values, cell.corner_sign_mask,
                contributors, positions, smoothing_lengths, isovalue);
        if (!samples.empty()) {
            continue;
        }

        ++zero_sample_count;
        if (cell.depth < max_depth) {
            zero_sample_cells.push_back(cell_idx);
        }
    }

    const RefinementClosureConfig closure_config = {
        isovalue,
        max_depth,
        domain,
        base_resolution,
        1U,
        3U,
        0.1,
        0.97,
        table_cadence_seconds,
        "Meshing",
        "refine_zero_sample_incident_cells",
        "incident_zero_sample",
    };

    const bool split_any = refine_cells_to_next_depth_with_closure(
        all_cells,
        all_contributors,
        positions,
        smoothing_lengths,
        closure_config,
        zero_sample_cells);

    meshmerizer_log_detail::print_debug_status(
        "Meshing",
        "refine_zero_sample_incident_cells",
        "missing=%zu zero_sample=%zu split=%zu\n",
        missing_cells.size(),
        zero_sample_count,
        split_any ? all_cells.size() - initial_cell_count : 0U);

    if (!split_any) {
        return false;
    }
    return true;
}

/**
 * @brief Solve representative vertices for incident cells missed by the main
 *        per-leaf activation pass.
 */
inline void activate_missing_incident_cells(
    std::vector<OctreeCell> &all_cells,
    std::vector<MeshVertex> &vertices,
    const std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const LeafSpatialIndex &spatial_index,
    std::uint32_t max_depth,
    std::uint32_t base_resolution,
    double isovalue) {
    const std::vector<std::size_t> missing_cells =
        collect_missing_incident_cells(
            all_cells, spatial_index, max_depth, base_resolution, isovalue);

    std::size_t already_present_count = 0;
    std::size_t zero_sample_count = 0;
    std::size_t activated_count = 0;
    for (std::size_t cell_idx : missing_cells) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_idx);
        OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf) {
            continue;
        }
        if (cell.representative_vertex_index >= 0) {
            ++already_present_count;
            continue;
        }

        const std::span<const std::size_t> contributors =
            gather_face_cell_contributors(cell, all_contributors);
        if (cell.corner_sign_mask == 0U && !contributors.empty()) {
            cell.corner_values = sample_cell_corners(
                cell, contributors, positions, smoothing_lengths);
            cell.corner_sign_mask = compute_corner_sign_mask(
                cell.corner_values, isovalue);
        }

        const std::vector<HermiteSample> samples =
            compute_cell_hermite_samples(
                cell.bounds, cell.corner_values, cell.corner_sign_mask,
                contributors, positions, smoothing_lengths, isovalue);
        if (samples.empty()) {
            ++zero_sample_count;
            continue;
        }

        cell.representative_vertex_index =
            static_cast<std::int64_t>(vertices.size());
        cell.is_active = true;
        vertices.push_back(solve_qef_for_leaf(samples, cell.bounds));
        ++activated_count;
    }

    meshmerizer_log_detail::print_debug_status(
        "Meshing",
        "activate_missing_incident_cells",
        "missing=%zu already_present=%zu zero_sample=%zu added=%zu\n",
        missing_cells.size(),
        already_present_count,
        zero_sample_count,
        activated_count);
}

/**
 * @brief Generate mesh triangles by iterating fine-grid primal edges.
 *
 * This is the correct approach for adaptive (2:1-balanced) octrees.
 * Instead of iterating per-leaf-cell with cell-resolution edges (which
 * causes duplicates and gaps at T-junctions between cells of different
 * depths), we iterate **every primal edge at the finest grid resolution**.
 *
 * For each fine-grid vertex ``(ix, iy, iz)``, we process 3 edges:
 *   - X-edge to ``(ix+1, iy, iz)``
 *   - Y-edge to ``(ix, iy+1, iz)``
 *   - Z-edge to ``(ix, iy, iz+1)``
 *
 * For each edge:
 *   1. Look up the leaf cells containing both endpoints via the spatial
 *      index.
 *   2. Determine the field sign at each endpoint by trilinear
 *      interpolation of the containing cell's corner values.
 *   3. If signs differ (sign-changing edge), find the 4 incident leaf
 *      cells and emit a quad.
 *
 * Each fine-grid edge is visited exactly once by this iteration, so
 * there are no duplicates.  Boundary edges (where fewer than 4 incident
 * cells exist) are skipped to produce a closed mesh.
 *
 * The cost is O(fine_resolution^3 * 3) edge checks, which is acceptable
 * because we only perform cheap hash-map lookups per edge.
 *
 * Design for parallelism:
 * - The outer loop over fine-grid vertices can be partitioned across
 *   threads with thread-local triangle buffers.
 *
 * @param all_cells All octree cells (with representative_vertex_index set).
 * @param vertices Mesh vertices produced by solve_all_leaf_vertices.
 * @param spatial_index Prebuilt spatial index for leaf lookup.
 * @param max_depth Maximum octree depth (fine-grid resolution parameter).
 * @param base_resolution Top-level cells per axis.
 * @param isovalue Target isovalue for sign interpolation.
 * @return Triangle array forming the output mesh.
 */
inline std::vector<MeshTriangle> generate_dual_contour_faces(
    const std::vector<OctreeCell> &all_cells,
    const std::vector<MeshVertex> &vertices,
    const LeafSpatialIndex &spatial_index,
    std::uint32_t max_depth,
    std::uint32_t base_resolution,
    double isovalue) {

    // Guard: ensure fine_res does not overflow uint32 or exceed
    // the 21-bit pack_grid_coords capacity.
    if (max_depth >= 21U) {
        throw std::overflow_error(
            "max_depth >= 21 would overflow grid coordinates");
    }
    // Compute in uint64_t to detect overflow before narrowing.
    const std::uint64_t fine_res_wide =
        static_cast<std::uint64_t>(base_resolution) *
        (1ULL << max_depth);
    constexpr std::uint64_t MAX_COORD_FACE = (1ULL << 21U) - 1U;
    if (fine_res_wide > MAX_COORD_FACE) {
        throw std::overflow_error(
            "fine_res exceeds 21-bit grid coordinate capacity");
    }
    // The fine-grid has this many cells per axis.
    const std::uint32_t fine_res =
        static_cast<std::uint32_t>(fine_res_wide);

    std::vector<std::uint64_t> candidate_origins;
    candidate_origins.reserve(vertices.size() * 8U);
    for (const OctreeCell &cell : all_cells) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(
            candidate_origins.size() + 1U);
        if (!cell.is_leaf || cell.representative_vertex_index < 0) {
            continue;
        }

        std::uint32_t cell_x = 0U;
        std::uint32_t cell_y = 0U;
        std::uint32_t cell_z = 0U;
        morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
        const std::uint32_t level_shift = max_depth - cell.depth;
        const std::uint32_t span = 1U << level_shift;
        const std::uint32_t ix0 = cell_x << level_shift;
        const std::uint32_t iy0 = cell_y << level_shift;
        const std::uint32_t iz0 = cell_z << level_shift;

        const std::uint32_t ix_begin = ix0 > 0U ? ix0 - 1U : 0U;
        const std::uint32_t iy_begin = iy0 > 0U ? iy0 - 1U : 0U;
        const std::uint32_t iz_begin = iz0 > 0U ? iz0 - 1U : 0U;
        const std::uint32_t ix_end = std::min(fine_res, ix0 + span);
        const std::uint32_t iy_end = std::min(fine_res, iy0 + span);
        const std::uint32_t iz_end = std::min(fine_res, iz0 + span);

        for (std::uint32_t ix = ix_begin; ix < ix_end; ++ix) {
            for (std::uint32_t iy = iy_begin; iy < iy_end; ++iy) {
                for (std::uint32_t iz = iz_begin; iz < iz_end; ++iz) {
                    candidate_origins.push_back(
                        LeafSpatialIndex::pack_grid_coords(ix, iy, iz));
                }
            }
        }
    }

    std::sort(candidate_origins.begin(), candidate_origins.end());
    candidate_origins.erase(
        std::unique(candidate_origins.begin(), candidate_origins.end()),
        candidate_origins.end());

    // Determine the number of threads.  When OpenMP is disabled the
    // stub returns 1 and the code degrades to a single-threaded path
    // with no overhead beyond the extra indirection through the
    // per-thread vector.
    const int n_threads = omp_get_max_threads();

    // Per-thread triangle buffers.  Each thread appends only to its
    // own buffer, avoiding synchronization inside the hot loop.
    std::vector<std::vector<MeshTriangle>> thread_triangles(
        static_cast<std::size_t>(n_threads));

    ProgressBar face_bar(
        "Meshing", "generate_faces", candidate_origins.size());

    // Iterate over all fine-grid vertex positions.  Each vertex
    // position (ix, iy, iz) is the min corner of a fine-grid cell.
    // We process the 3 edges emanating from it in the +X, +Y, +Z
    // directions.
    //
    // The outer loop over ix is parallelised.  Each ix slice is
    // independent because edges only connect adjacent vertices and
    // emit_quad reads (but never writes) shared cell/vertex data.
    // The only writes go into the thread-local triangle buffer.
#pragma omp parallel for schedule(dynamic)
    for (std::int64_t candidate_index = 0;
         candidate_index < static_cast<std::int64_t>(candidate_origins.size());
         ++candidate_index) {
        if (meshmerizer_cancel_detail::poll_for_cancellation_in_parallel(
                static_cast<std::size_t>(candidate_index))) {
            face_bar.tick();
            continue;
        }
        // Select this thread's triangle buffer.
        std::vector<MeshTriangle> &local_triangles =
            thread_triangles[
                static_cast<std::size_t>(omp_get_thread_num())];

        const std::uint64_t packed =
            candidate_origins[static_cast<std::size_t>(candidate_index)];
        const std::uint32_t ix = static_cast<std::uint32_t>(packed >> 42U);
        const std::uint32_t iy =
            static_cast<std::uint32_t>((packed >> 21U) & ((1U << 21U) - 1U));
        const std::uint32_t iz =
            static_cast<std::uint32_t>(packed & ((1U << 21U) - 1U));

        // Look up the leaf cell containing this vertex.
        const std::size_t cell_a_idx =
            spatial_index.find_leaf_at(ix, iy, iz);
        if (cell_a_idx == SIZE_MAX) {
            face_bar.tick();
            continue;
        }
        const OctreeCell &cell_a = all_cells[cell_a_idx];

        // Determine the sign at vertex (ix, iy, iz).
        const bool sign_a = sign_at_fine_vertex(
            cell_a, ix, iy, iz, spatial_index,
            max_depth, isovalue);

        // --- X-edge: (ix,iy,iz) -> (ix+1,iy,iz) ---
        if (ix + 1U < fine_res) {
            const std::size_t cell_b_idx =
                spatial_index.find_leaf_at(ix + 1U, iy, iz);
            if (cell_b_idx != SIZE_MAX) {
                const bool sign_b = sign_at_fine_vertex(
                    all_cells[cell_b_idx],
                    ix + 1U, iy, iz,
                    spatial_index, max_depth, isovalue);

                if (sign_a != sign_b) {
                    if (iy > 0U && iz > 0U) {
                        const std::size_t c1 =
                            spatial_index.find_leaf_at(ix, iy - 1U, iz);
                        const std::size_t c2 =
                            spatial_index.find_leaf_at(ix, iy, iz - 1U);
                        const std::size_t c3 =
                            spatial_index.find_leaf_at(ix, iy - 1U, iz - 1U);

                        emit_quad(
                            all_cells, vertices,
                            local_triangles,
                            cell_a_idx, c2, c3, c1,
                            sign_a);
                    }
                }
            }
        }

        // --- Y-edge: (ix,iy,iz) -> (ix,iy+1,iz) ---
        if (iy + 1U < fine_res) {
            const std::size_t cell_c_idx =
                spatial_index.find_leaf_at(ix, iy + 1U, iz);
            if (cell_c_idx != SIZE_MAX) {
                const bool sign_c = sign_at_fine_vertex(
                    all_cells[cell_c_idx],
                    ix, iy + 1U, iz,
                    spatial_index, max_depth, isovalue);

                if (sign_a != sign_c) {
                    if (ix > 0U && iz > 0U) {
                        const std::size_t c1 =
                            spatial_index.find_leaf_at(ix - 1U, iy, iz);
                        const std::size_t c2 =
                            spatial_index.find_leaf_at(ix, iy, iz - 1U);
                        const std::size_t c3 =
                            spatial_index.find_leaf_at(ix - 1U, iy, iz - 1U);

                        emit_quad(
                            all_cells, vertices,
                            local_triangles,
                            cell_a_idx, c1, c3, c2,
                            sign_a);
                    }
                }
            }
        }

        // --- Z-edge: (ix,iy,iz) -> (ix,iy,iz+1) ---
        if (iz + 1U < fine_res) {
            const std::size_t cell_d_idx =
                spatial_index.find_leaf_at(ix, iy, iz + 1U);
            if (cell_d_idx != SIZE_MAX) {
                const bool sign_d = sign_at_fine_vertex(
                    all_cells[cell_d_idx],
                    ix, iy, iz + 1U,
                    spatial_index, max_depth, isovalue);

                if (sign_a != sign_d) {
                    if (ix > 0U && iy > 0U) {
                        const std::size_t c1 =
                            spatial_index.find_leaf_at(ix - 1U, iy, iz);
                        const std::size_t c2 =
                            spatial_index.find_leaf_at(ix, iy - 1U, iz);
                        const std::size_t c3 =
                            spatial_index.find_leaf_at(ix - 1U, iy - 1U, iz);

                        emit_quad(
                            all_cells, vertices,
                            local_triangles,
                            cell_a_idx, c2, c3, c1,
                            sign_a);
                    }
                }
            }
        }
        face_bar.tick();
    }

    face_bar.finish();

    // Merge thread-local triangle buffers into a single output.
    // Count total triangles first for a single allocation.
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

/**
 * @brief Run the complete face generation pipeline.
 *
 * This is the top-level entry point for Phase 9.  Given the fully refined
 * octree (with contributor indices and corner values from refinement), it:
 *
 * 1. Solves QEF vertices for all active leaf cells.
 * 2. Builds a spatial index for leaf cell lookup.
 * 3. Generates mesh triangles by iterating over sign-changing primal edges.
 *
 * @param all_cells All octree cells (modified: representative_vertex_index
 *     and is_active are set for active leaves).
 * @param all_contributors Flat contributor index array.
 * @param positions Particle positions.
 * @param smoothing_lengths Per-particle support radii.
 * @param isovalue Target surface level.
 * @param domain Full simulation domain bounding box.
 * @param max_depth Maximum octree depth.
 * @param base_resolution Top-level cells per axis.
 * @return Pair of (vertices, triangles) forming the output mesh.
 */
inline std::pair<std::vector<MeshVertex>, std::vector<MeshTriangle>>
generate_mesh(
    std::vector<OctreeCell> &all_cells,
    const std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue,
    const BoundingBox &domain,
    std::uint32_t max_depth,
    std::uint32_t base_resolution) {
    // Step 1: Solve QEF vertices for every active leaf.
    std::vector<MeshVertex> vertices = solve_all_leaf_vertices(
        all_cells, all_contributors, positions, smoothing_lengths,
        isovalue);

    if (vertices.empty()) {
        return {vertices, {}};
    }

    // Step 2: Build spatial index for neighbor lookups.
    LeafSpatialIndex spatial_index;
    spatial_index.build(all_cells, domain, max_depth, base_resolution);

    // Step 3: Generate triangles from dual contouring connectivity.
    std::vector<MeshTriangle> triangles =
        generate_dual_contour_faces(
            all_cells, vertices, spatial_index,
            max_depth, base_resolution, isovalue);

    return {vertices, triangles};
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_FACES_HPP_
