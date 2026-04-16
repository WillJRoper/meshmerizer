/**
 * @file faces.hpp
 * @brief Dual contouring face generation from octree connectivity.
 *
 * Phase 9 of the adaptive meshing pipeline.  Given an octree whose active
 * leaf cells each carry a representative mesh vertex (from the QEF solve in
 * Phase 8), this module constructs the triangle mesh by finding every
 * sign-changing primal edge of the octree and connecting the representative
 * vertices of the (up to 4) leaf cells incident on that edge.
 *
 * The algorithm iterates over leaf cells and, for each leaf, considers the
 * three primal edges emanating from its minimum corner (one per axis).
 * This "min-corner ownership" convention ensures each interior primal edge
 * is processed exactly once.  For each sign-changing edge, the four
 * incident leaf cells are located by spatial lookup, and a quad connecting
 * their QEF vertices is emitted.  Quads are split into two triangles with
 * consistent winding determined by the sign-change direction.
 *
 * In an adaptive (2:1-balanced) octree, some of the four incident cells
 * may be the *same* cell (when a coarser cell covers multiple fine-grid
 * positions).  Degenerate quads where fewer than 3 distinct vertices
 * remain are discarded.
 *
 * Boundary edges (edges on the domain boundary where fewer than 4 cells
 * exist) are silently skipped to produce a closed mesh.
 *
 * Design for parallelism:
 * - Edge ownership is per-leaf and deterministic, so the iteration can be
 *   parallelized by partitioning leaves across workers.
 * - Face emission uses thread-local buffers that are concatenated at the
 *   end. (The initial implementation is serial but follows this pattern.)
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_FACES_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_FACES_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "hermite.hpp"
#include "mesh.hpp"
#include "octree_cell.hpp"
#include "qef.hpp"
#include "vector3d.hpp"

/**
 * @brief Spatial index mapping positions to leaf cell indices.
 *
 * Uses a hash map keyed by a quantized grid position to provide O(1)
 * lookup of the leaf cell that contains a given world-space point.
 * The quantization grid matches the finest octree level so that each
 * fine-grid position maps to at most one leaf cell.
 *
 * For coarse cells that span multiple fine-grid positions, every
 * fine-grid position within the cell is registered at construction
 * time.  The 2:1 balance constraint ensures a coarse cell spans at
 * most 2x the fine-grid spacing, so the registration cost per coarse
 * cell is at most 8 entries (one octant of fine-grid positions).
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

    /** @brief Map from quantized grid position to leaf cell index. */
    std::unordered_map<GridKey, std::size_t, GridKeyHash> lookup;

    /** @brief Inverse of the finest cell edge length. */
    double inv_cell_size_x;
    double inv_cell_size_y;
    double inv_cell_size_z;

    /** @brief Domain minimum corner used for quantization. */
    Vector3d domain_min;

    /**
     * @brief Build the spatial index from a set of octree cells.
     *
     * Only leaf cells are indexed.  For each leaf, every fine-grid
     * position that falls within the cell is registered.  The fine
     * grid resolution is determined by the maximum depth present in
     * the octree.
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

        for (std::size_t cell_idx = 0; cell_idx < all_cells.size();
             ++cell_idx) {
            const OctreeCell &cell = all_cells[cell_idx];
            if (!cell.is_leaf) {
                continue;
            }

            // Quantize the cell's min corner to fine-grid coordinates.
            const std::uint32_t ix_min = static_cast<std::uint32_t>(
                (cell.bounds.min.x - domain_min.x) * inv_cell_size_x +
                0.5);
            const std::uint32_t iy_min = static_cast<std::uint32_t>(
                (cell.bounds.min.y - domain_min.y) * inv_cell_size_y +
                0.5);
            const std::uint32_t iz_min = static_cast<std::uint32_t>(
                (cell.bounds.min.z - domain_min.z) * inv_cell_size_z +
                0.5);

            // Determine how many fine-grid cells this leaf spans per axis.
            // A leaf at depth d spans 2^(max_depth - d) fine cells.
            const std::uint32_t span =
                1U << (max_depth - cell.depth);

            // Register every fine-grid position within this leaf.
            for (std::uint32_t dx = 0; dx < span; ++dx) {
                for (std::uint32_t dy = 0; dy < span; ++dy) {
                    for (std::uint32_t dz = 0; dz < span; ++dz) {
                        const GridKey key = {pack_grid_coords(
                            ix_min + dx, iy_min + dy, iz_min + dz)};
                        lookup[key] = cell_idx;
                    }
                }
            }
        }
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
        return (static_cast<std::uint64_t>(ix) << 42U) |
               (static_cast<std::uint64_t>(iy) << 21U) |
               static_cast<std::uint64_t>(iz);
    }

    /**
     * @brief Look up the leaf cell index at a given fine-grid position.
     *
     * @param ix X fine-grid coordinate.
     * @param iy Y fine-grid coordinate.
     * @param iz Z fine-grid coordinate.
     * @return Index into the all_cells array, or SIZE_MAX if not found.
     */
    std::size_t find_leaf_at(std::uint32_t ix, std::uint32_t iy,
                             std::uint32_t iz) const {
        const GridKey key = {pack_grid_coords(ix, iy, iz)};
        auto it = lookup.find(key);
        if (it != lookup.end()) {
            return it->second;
        }
        return SIZE_MAX;
    }

    /**
     * @brief Quantize a cell's min corner to fine-grid coordinates.
     *
     * @param cell_min Minimum corner of the cell in world space.
     * @return Tuple of (ix, iy, iz) fine-grid coordinates.
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
    std::vector<MeshVertex> vertices;

    for (std::size_t cell_idx = 0; cell_idx < all_cells.size();
         ++cell_idx) {
        OctreeCell &cell = all_cells[cell_idx];
        if (!cell.is_leaf) {
            cell.representative_vertex_index = -1;
            continue;
        }

        // Gather contributor indices for this cell.
        std::vector<std::size_t> contributors;
        if (cell.contributor_begin >= 0 &&
            cell.contributor_end > cell.contributor_begin) {
            contributors.reserve(static_cast<std::size_t>(
                cell.contributor_end - cell.contributor_begin));
            for (std::int64_t k = cell.contributor_begin;
                 k < cell.contributor_end; ++k) {
                contributors.push_back(
                    all_contributors[static_cast<std::size_t>(k)]);
            }
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
            cell.representative_vertex_index = -1;
            continue;
        }

        // Recompute the sign mask to ensure consistency.
        cell.corner_sign_mask =
            compute_corner_sign_mask(cell.corner_values, isovalue);

        // If no sign change, skip (safety check).
        if (cell.corner_sign_mask == 0U ||
            cell.corner_sign_mask == 0xFFU) {
            cell.representative_vertex_index = -1;
            cell.has_surface = false;
            continue;
        }

        // Compute Hermite samples for this leaf.
        const std::vector<HermiteSample> samples =
            compute_cell_hermite_samples(
                cell.bounds, cell.corner_values,
                cell.corner_sign_mask, contributors,
                positions, smoothing_lengths, isovalue);

        // Solve the QEF.
        const MeshVertex vertex =
            solve_qef_for_leaf(samples, cell.bounds);

        cell.representative_vertex_index =
            static_cast<std::int64_t>(vertices.size());
        cell.is_active = true;
        vertices.push_back(vertex);
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

    // Count distinct vertices.  With adaptive octrees, some incident
    // cells may be the same coarse cell, giving duplicate vertex indices.
    std::size_t unique[4] = {v0, v1, v2, v3};
    std::size_t num_unique = 4;
    // Simple O(n^2) deduplicate for n=4.
    for (std::size_t i = 0; i < num_unique; ++i) {
        for (std::size_t j = i + 1; j < num_unique;) {
            if (unique[j] == unique[i]) {
                unique[j] = unique[num_unique - 1];
                --num_unique;
            } else {
                ++j;
            }
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
    const std::uint32_t span = 1U << (max_depth - cell.depth);

    // For finest-level cells (span=1), the fine-grid vertex IS the min
    // corner — just read corner 0 directly, no interpolation needed.
    if (span == 1U) {
        return cell.corner_values[0] >= isovalue;
    }

    // Quantize the cell's own min corner to fine-grid coordinates.
    std::uint32_t cx = 0U, cy = 0U, cz = 0U;
    spatial_index.quantize_min_corner(cell.bounds.min, cx, cy, cz);

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
    // Corner indexing: bit 0 = +X, bit 1 = +Y, bit 2 = +Z.
    const double c000 = cell.corner_values[0];
    const double c001 = cell.corner_values[1];
    const double c010 = cell.corner_values[2];
    const double c011 = cell.corner_values[3];
    const double c100 = cell.corner_values[4];
    const double c101 = cell.corner_values[5];
    const double c110 = cell.corner_values[6];
    const double c111 = cell.corner_values[7];

    // Interpolate along X first.
    const double c00 = c000 + fx * (c001 - c000);
    const double c01 = c010 + fx * (c011 - c010);
    const double c10 = c100 + fx * (c101 - c100);
    const double c11 = c110 + fx * (c111 - c110);

    // Then along Y.
    const double c0 = c00 + fy * (c01 - c00);
    const double c1 = c10 + fy * (c11 - c10);

    // Then along Z.
    const double value = c0 + fz * (c1 - c0);

    return value >= isovalue;
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
    std::vector<MeshTriangle> triangles;

    // The fine-grid has this many cells per axis.
    const std::uint32_t fine_res =
        base_resolution * (1U << max_depth);

    // Iterate over all fine-grid vertex positions.  Each vertex
    // position (ix, iy, iz) is the min corner of a fine-grid cell.
    // We process the 3 edges emanating from it in the +X, +Y, +Z
    // directions.
    for (std::uint32_t ix = 0; ix < fine_res; ++ix) {
        for (std::uint32_t iy = 0; iy < fine_res; ++iy) {
            for (std::uint32_t iz = 0; iz < fine_res; ++iz) {
                // Look up the leaf cell containing this vertex.
                const std::size_t cell_a_idx =
                    spatial_index.find_leaf_at(ix, iy, iz);
                if (cell_a_idx == SIZE_MAX) {
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
                        spatial_index.find_leaf_at(
                            ix + 1U, iy, iz);
                    if (cell_b_idx != SIZE_MAX) {
                        const bool sign_b = sign_at_fine_vertex(
                            all_cells[cell_b_idx],
                            ix + 1U, iy, iz,
                            spatial_index, max_depth, isovalue);

                        if (sign_a != sign_b) {
                            // 4 incident cells for an X-edge at
                            // (ix, iy, iz).  In the Y-Z plane:
                            //   c0=(iy,iz)     c2=(iy,iz-1)
                            //   c1=(iy-1,iz)   c3=(iy-1,iz-1)
                            // Cyclic CCW around +X: c0, c2, c3, c1.
                            if (iy > 0U && iz > 0U) {
                                const std::size_t c0 =
                                    spatial_index.find_leaf_at(
                                        ix, iy, iz);
                                const std::size_t c1 =
                                    spatial_index.find_leaf_at(
                                        ix, iy - 1U, iz);
                                const std::size_t c2 =
                                    spatial_index.find_leaf_at(
                                        ix, iy, iz - 1U);
                                const std::size_t c3 =
                                    spatial_index.find_leaf_at(
                                        ix, iy - 1U, iz - 1U);

                                emit_quad(
                                    all_cells, vertices, triangles,
                                    c0, c2, c3, c1, sign_a);
                            }
                        }
                    }
                }

                // --- Y-edge: (ix,iy,iz) -> (ix,iy+1,iz) ---
                if (iy + 1U < fine_res) {
                    const std::size_t cell_c_idx =
                        spatial_index.find_leaf_at(
                            ix, iy + 1U, iz);
                    if (cell_c_idx != SIZE_MAX) {
                        const bool sign_c = sign_at_fine_vertex(
                            all_cells[cell_c_idx],
                            ix, iy + 1U, iz,
                            spatial_index, max_depth, isovalue);

                        if (sign_a != sign_c) {
                            // 4 incident cells for a Y-edge at
                            // (ix, iy, iz).  In the X-Z plane:
                            //   c0=(ix,iz)     c2=(ix,iz-1)
                            //   c1=(ix-1,iz)   c3=(ix-1,iz-1)
                            // Cyclic CCW around +Y: c0, c1, c3, c2.
                            if (ix > 0U && iz > 0U) {
                                const std::size_t c0 =
                                    spatial_index.find_leaf_at(
                                        ix, iy, iz);
                                const std::size_t c1 =
                                    spatial_index.find_leaf_at(
                                        ix - 1U, iy, iz);
                                const std::size_t c2 =
                                    spatial_index.find_leaf_at(
                                        ix, iy, iz - 1U);
                                const std::size_t c3 =
                                    spatial_index.find_leaf_at(
                                        ix - 1U, iy, iz - 1U);

                                emit_quad(
                                    all_cells, vertices, triangles,
                                    c0, c1, c3, c2, sign_a);
                            }
                        }
                    }
                }

                // --- Z-edge: (ix,iy,iz) -> (ix,iy,iz+1) ---
                if (iz + 1U < fine_res) {
                    const std::size_t cell_d_idx =
                        spatial_index.find_leaf_at(
                            ix, iy, iz + 1U);
                    if (cell_d_idx != SIZE_MAX) {
                        const bool sign_d = sign_at_fine_vertex(
                            all_cells[cell_d_idx],
                            ix, iy, iz + 1U,
                            spatial_index, max_depth, isovalue);

                        if (sign_a != sign_d) {
                            // 4 incident cells for a Z-edge at
                            // (ix, iy, iz).  In the X-Y plane:
                            //   c0=(ix,iy)     c2=(ix,iy-1)
                            //   c1=(ix-1,iy)   c3=(ix-1,iy-1)
                            // Cyclic CCW around +Z: c0, c2, c3, c1.
                            if (ix > 0U && iy > 0U) {
                                const std::size_t c0 =
                                    spatial_index.find_leaf_at(
                                        ix, iy, iz);
                                const std::size_t c1 =
                                    spatial_index.find_leaf_at(
                                        ix - 1U, iy, iz);
                                const std::size_t c2 =
                                    spatial_index.find_leaf_at(
                                        ix, iy - 1U, iz);
                                const std::size_t c3 =
                                    spatial_index.find_leaf_at(
                                        ix - 1U, iy - 1U, iz);

                                emit_quad(
                                    all_cells, vertices, triangles,
                                    c0, c2, c3, c1, sign_a);
                            }
                        }
                    }
                }
            }
        }
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
