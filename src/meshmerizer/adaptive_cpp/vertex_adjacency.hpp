#ifndef MESHMERIZER_ADAPTIVE_CPP_VERTEX_ADJACENCY_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_VERTEX_ADJACENCY_HPP_

/**
 * @file vertex_adjacency.hpp
 * @brief Build vertex adjacency from octree leaf connectivity.
 *
 * Given an adaptive octree with QEF vertices solved on active leaf
 * cells, this module constructs a CSR (Compressed Sparse Row)
 * adjacency structure that records which active leaf vertices are
 * face-neighbors.  Two active leaves are face-neighbors if they
 * share a face of the octree grid (i.e., they are adjacent along
 * one of the 6 axis-aligned directions).
 *
 * The adjacency is stored in CSR format:
 *   - adj_offsets[i] .. adj_offsets[i+1] gives the range of
 *     neighbor indices for vertex i.
 *   - adj_neighbors[adj_offsets[i] .. adj_offsets[i+1]] are the
 *     vertex indices of i's neighbors.
 *
 * This structure is used by:
 *   - Phase 23b: Laplacian vertex smoothing.
 *   - Phase 24: Gap filling (edge subdivision) to identify long
 *     edges between adjacent vertices.
 *
 * The approach iterates each active leaf cell and probes the 6
 * face-adjacent directions at the fine-grid level using the
 * LeafSpatialIndex.  For coarse cells (depth < max_depth), a
 * single face may border multiple finer cells, so we probe at
 * every fine-grid position along that face to discover all
 * neighbors.  Duplicate neighbor indices are suppressed.
 *
 * Complexity: O(n_active_leaves * max_face_area_in_fine_cells)
 * where max_face_area_in_fine_cells = span^2 and span =
 * 2^(max_depth - depth).  For typical max_depth differences of
 * 1-3 levels, this is very fast.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "faces.hpp"
#include "octree_cell.hpp"
#include "omp_config.hpp"

/**
 * @brief CSR-format vertex adjacency structure.
 *
 * For vertex i, its neighbors are:
 *   neighbors[offsets[i]], neighbors[offsets[i]+1], ...,
 *   neighbors[offsets[i+1]-1]
 *
 * offsets has size n_vertices + 1.
 * neighbors has size equal to the total number of directed edges
 * (each undirected edge is stored twice, once per endpoint).
 */
struct VertexAdjacency {
    /// CSR row offsets.  offsets[i] is the start index in
    /// `neighbors` for vertex i.  offsets[n_vertices] is the
    /// total number of entries in `neighbors`.
    std::vector<std::size_t> offsets;

    /// Flat array of neighbor vertex indices.
    std::vector<std::size_t> neighbors;
};

/**
 * @brief Build vertex adjacency from octree leaf connectivity.
 *
 * For each active leaf cell (one that has a QEF vertex), this
 * function probes the 6 face-adjacent directions using the
 * spatial index to discover neighboring active leaves.  The
 * result is a symmetric CSR adjacency structure over vertex
 * indices.
 *
 * On a 2:1 balanced octree, each leaf has at most 6 face
 * directions, and each face of a coarse cell can border at most
 * 4 finer cells (since the depth difference is at most 1).  So
 * the maximum degree of any vertex is 6 * 4 = 24, though in
 * practice most vertices have 6 neighbors.
 *
 * @param all_cells       All octree cells (with
 *     representative_vertex_index set for active leaves).
 * @param spatial_index   Prebuilt spatial index for leaf lookup.
 * @param max_depth       Maximum octree refinement depth.
 * @param n_vertices      Number of active leaf vertices (the
 *     size of the vertex array from solve_all_leaf_vertices).
 * @return VertexAdjacency in CSR format.
 */
inline VertexAdjacency build_vertex_adjacency(
    const std::vector<OctreeCell> &all_cells,
    const LeafSpatialIndex &spatial_index,
    std::uint32_t max_depth,
    std::size_t n_vertices) {

    // ----------------------------------------------------------
    // Pass 1: For each active leaf, collect its neighbor vertex
    // indices into a per-vertex temporary list.
    //
    // We iterate all cells, skip non-active-leaves, and for each
    // active leaf probe 6 directions.  For each direction, we
    // probe every fine-grid position along the adjacent face to
    // find all neighboring leaves (which may be at a finer depth
    // than the current cell).
    // ----------------------------------------------------------

    // Per-vertex neighbor lists (temporary, unsorted, may have
    // duplicates within each list).
    std::vector<std::vector<std::size_t>> per_vertex_neighbors(
        n_vertices);

    // The 6 face-adjacent direction offsets in fine-grid coords.
    // For a cell with min-corner (cx, cy, cz) and span s:
    //   +X face: probe at (cx + s, cy + j, cz + k)
    //   -X face: probe at (cx - 1, cy + j, cz + k)
    //   +Y face: probe at (cx + j, cy + s, cz + k)
    //   -Y face: probe at (cx + j, cy - 1, cz + k)
    //   +Z face: probe at (cx + j, cy + k, cz + s)
    //   -Z face: probe at (cx + j, cy + k, cz - 1)
    // where j, k iterate over [0, span) to cover the full face.

#pragma omp parallel for schedule(dynamic)
    for (std::int64_t ci = 0;
         ci < static_cast<std::int64_t>(all_cells.size()); ++ci) {
        const OctreeCell &cell = all_cells[static_cast<std::size_t>(ci)];

        // Skip non-leaf, non-active cells.
        if (!cell.is_leaf || !cell.is_active) {
            continue;
        }
        if (cell.representative_vertex_index < 0) {
            continue;
        }

        const std::size_t vi = static_cast<std::size_t>(
            cell.representative_vertex_index);

        // Quantize cell min corner to fine-grid coordinates.
        std::uint32_t cx = 0, cy = 0, cz = 0;
        spatial_index.quantize_min_corner(
            cell.bounds.min, cx, cy, cz);

        // Cell span in fine-grid cells per axis.
        const std::uint32_t span =
            1U << (max_depth - cell.depth);

        // Fine-grid resolution (for boundary checks).
        // The fine-grid has base_resolution * 2^max_depth cells
        // per axis, but we don't have base_resolution here.
        // Instead, we can infer the grid extent from the spatial
        // index: any position outside the domain will return
        // SIZE_MAX from find_leaf_at, which we handle gracefully.
        // We still need an upper bound for the boundary check to
        // avoid underflow on unsigned subtraction.  Use a
        // sentinel: if cx == 0, skip -X; if cx + span >= some
        // large value, skip +X.  Since find_leaf_at returns
        // SIZE_MAX for out-of-bounds, we can simply always probe
        // and let the lookup handle it — except for the -1
        // underflow on uint32.  So we guard the -X/-Y/-Z cases
        // with cx > 0, cy > 0, cz > 0.

        // Helper: add a neighbor vertex index to vi's list,
        // avoiding self-loops.
        auto add_neighbor = [&](std::size_t neighbor_cell_idx) {
            if (neighbor_cell_idx == SIZE_MAX) {
                return;
            }
            const OctreeCell &nbr = all_cells[neighbor_cell_idx];
            if (!nbr.is_leaf || !nbr.is_active) {
                return;
            }
            if (nbr.representative_vertex_index < 0) {
                return;
            }
            const std::size_t vj = static_cast<std::size_t>(
                nbr.representative_vertex_index);
            if (vj == vi) {
                return;  // Self-loop (shouldn't happen, but guard).
            }
            per_vertex_neighbors[vi].push_back(vj);
        };

        // Probe each face.  For a cell of span s, the face has
        // s * s fine-grid positions.  On a 2:1 balanced tree with
        // max depth difference of 1, span is at most 2, so each
        // face has at most 4 probe positions.  For deeper cells
        // (span = 1), each face is a single probe.
        //
        // However, neighboring cells on the same face may be
        // coarser (span > our span), in which case multiple probe
        // points on our face will map to the same neighbor cell.
        // We deduplicate below.

        // +X face: positions (cx + span, cy + j, cz + k)
        for (std::uint32_t j = 0; j < span; ++j) {
            for (std::uint32_t k = 0; k < span; ++k) {
                add_neighbor(spatial_index.find_leaf_at(
                    cx + span, cy + j, cz + k));
            }
        }

        // -X face: positions (cx - 1, cy + j, cz + k)
        if (cx > 0) {
            for (std::uint32_t j = 0; j < span; ++j) {
                for (std::uint32_t k = 0; k < span; ++k) {
                    add_neighbor(spatial_index.find_leaf_at(
                        cx - 1U, cy + j, cz + k));
                }
            }
        }

        // +Y face: positions (cx + j, cy + span, cz + k)
        for (std::uint32_t j = 0; j < span; ++j) {
            for (std::uint32_t k = 0; k < span; ++k) {
                add_neighbor(spatial_index.find_leaf_at(
                    cx + j, cy + span, cz + k));
            }
        }

        // -Y face: positions (cx + j, cy - 1, cz + k)
        if (cy > 0) {
            for (std::uint32_t j = 0; j < span; ++j) {
                for (std::uint32_t k = 0; k < span; ++k) {
                    add_neighbor(spatial_index.find_leaf_at(
                        cx + j, cy - 1U, cz + k));
                }
            }
        }

        // +Z face: positions (cx + j, cy + k, cz + span)
        for (std::uint32_t j = 0; j < span; ++j) {
            for (std::uint32_t k = 0; k < span; ++k) {
                add_neighbor(spatial_index.find_leaf_at(
                    cx + j, cy + k, cz + span));
            }
        }

        // -Z face: positions (cx + j, cy + k, cz - 1)
        if (cz > 0) {
            for (std::uint32_t j = 0; j < span; ++j) {
                for (std::uint32_t k = 0; k < span; ++k) {
                    add_neighbor(spatial_index.find_leaf_at(
                        cx + j, cy + k, cz - 1U));
                }
            }
        }
    }

    // ----------------------------------------------------------
    // Pass 2: Deduplicate each vertex's neighbor list and build
    // the CSR structure.
    // ----------------------------------------------------------

    VertexAdjacency adj;
    adj.offsets.resize(n_vertices + 1, 0);

    // Sort and deduplicate each per-vertex neighbor list.
#pragma omp parallel for schedule(dynamic)
    for (std::int64_t vi = 0;
         vi < static_cast<std::int64_t>(n_vertices); ++vi) {
        auto &nbrs = per_vertex_neighbors[static_cast<std::size_t>(vi)];
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(
            std::unique(nbrs.begin(), nbrs.end()),
            nbrs.end());
    }

    // Compute CSR offsets.
    for (std::size_t vi = 0; vi < n_vertices; ++vi) {
        adj.offsets[vi + 1] =
            adj.offsets[vi] + per_vertex_neighbors[vi].size();
    }

    // Fill CSR neighbors array.
    adj.neighbors.resize(adj.offsets[n_vertices]);
    for (std::size_t vi = 0; vi < n_vertices; ++vi) {
        const std::size_t start = adj.offsets[vi];
        const auto &nbrs = per_vertex_neighbors[vi];
        for (std::size_t k = 0; k < nbrs.size(); ++k) {
            adj.neighbors[start + k] = nbrs[k];
        }
    }

    return adj;
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_VERTEX_ADJACENCY_HPP_
