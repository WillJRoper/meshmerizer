#ifndef MESHMERIZER_ADAPTIVE_CPP_EDGE_SUBDIV_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_EDGE_SUBDIV_HPP_

/**
 * @file edge_subdiv.hpp
 * @brief Edge subdivision for gap filling in Dual Contouring meshes.
 *
 * After DC face generation, some triangle edges may be much longer
 * than the local cell size — especially at octree depth transitions
 * or where QEF vertices are poorly placed.  These long edges produce
 * large triangles with poor surface resolution and contribute to
 * holes in the mesh.
 *
 * This module identifies long edges, inserts intermediate vertices
 * by linear interpolation, and subdivides affected triangles into
 * smaller triangles.  The subdivision is consistent: each edge is
 * keyed by its ordered vertex pair, so both triangles sharing an
 * edge see the same intermediate vertices.
 *
 * The threshold for subdivision is:
 *   max_edge_ratio * min(cell_size_i, cell_size_j)
 * where cell_size_i and cell_size_j are the octree cell edge
 * lengths of the two endpoint vertices.  Default max_edge_ratio
 * is 1.5.
 *
 * Intermediate vertex positions and normals are linearly
 * interpolated from the two endpoint vertices (normals are
 * renormalized).  No isosurface evaluation is performed — this
 * is a purely geometric operation.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "mesh.hpp"
#include "octree_cell.hpp"
#include "vector3d.hpp"

/**
 * @brief Key for identifying a directed edge by its two vertex
 *        indices, canonicalized so that the smaller index comes
 *        first.
 *
 * Using a canonical ordering ensures that both triangles sharing
 * an edge produce the same key and therefore share the same set
 * of intermediate vertices.
 */
struct EdgeKey {
    std::size_t v_lo;  ///< Smaller vertex index.
    std::size_t v_hi;  ///< Larger vertex index.

    EdgeKey(std::size_t a, std::size_t b)
        : v_lo(std::min(a, b)), v_hi(std::max(a, b)) {}

    bool operator==(const EdgeKey &other) const {
        return v_lo == other.v_lo && v_hi == other.v_hi;
    }
};

/**
 * @brief Hash function for EdgeKey.
 *
 * Uses a simple bit-mixing approach combining both vertex indices.
 */
struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey &key) const {
        // Cantor pairing + bit mix.
        std::size_t h = key.v_lo;
        h ^= key.v_hi + 0x9e3779b9ULL + (h << 6U) + (h >> 2U);
        return h;
    }
};

/**
 * @brief Information about a subdivided edge.
 *
 * Stores the indices of intermediate vertices inserted along
 * the edge, ordered from v_lo to v_hi.  For an edge split into
 * k segments, there are k-1 intermediate vertices.
 */
struct SubdividedEdge {
    /// Indices into the (extended) vertex array, ordered from
    /// v_lo toward v_hi.
    std::vector<std::size_t> intermediate_indices;
};

/**
 * @brief Per-vertex cell size, computed from the octree.
 *
 * For each QEF vertex (one per active leaf), this is the edge
 * length of the leaf cell that produced the vertex.  Used to
 * compute the local subdivision threshold.
 *
 * @param all_cells   All octree cells (representative_vertex_index
 *     set for active leaves).
 * @param n_vertices  Number of QEF vertices.
 * @return Vector of cell edge lengths indexed by vertex index.
 */
inline std::vector<double> compute_vertex_cell_sizes(
    const std::vector<OctreeCell> &all_cells,
    std::size_t n_vertices) {

    std::vector<double> cell_sizes(n_vertices, 0.0);

    for (const auto &cell : all_cells) {
        if (!cell.is_leaf || !cell.is_active) {
            continue;
        }
        if (cell.representative_vertex_index < 0) {
            continue;
        }
        const std::size_t vi = static_cast<std::size_t>(
            cell.representative_vertex_index);
        if (vi >= n_vertices) {
            continue;
        }
        // Cell edge length along X axis (cells are cubic on a
        // uniform-base octree, so all axes are equal).
        cell_sizes[vi] =
            cell.bounds.max.x - cell.bounds.min.x;
    }

    return cell_sizes;
}

/**
 * @brief Identify long edges and subdivide them, inserting new
 *        vertices and re-triangulating affected faces.
 *
 * This is the main entry point for the gap-filling pass.  It:
 *
 * 1. Scans all triangle edges, computing edge lengths and
 *    comparing against the local threshold.
 * 2. For long edges, computes the number of subdivisions and
 *    inserts intermediate vertices into the vertex array.
 * 3. Walks each triangle and, if any of its edges were
 *    subdivided, replaces the triangle with smaller triangles.
 *
 * The vertex array is extended in-place with new intermediate
 * vertices.  The triangle array is replaced entirely (both
 * subdivided and non-subdivided triangles are written to the
 * output).
 *
 * @param vertices       Vertex array (extended in-place with
 *     new intermediate vertices).
 * @param triangles      Triangle array (replaced with the
 *     subdivided result).
 * @param cell_sizes     Per-vertex cell edge lengths from
 *     compute_vertex_cell_sizes.
 * @param max_edge_ratio Maximum edge length as a multiple of
 *     local cell size.  Default 1.5.
 */
inline void subdivide_long_edges(
    std::vector<MeshVertex> &vertices,
    std::vector<MeshTriangle> &triangles,
    const std::vector<double> &cell_sizes,
    double max_edge_ratio) {

    if (triangles.empty() || max_edge_ratio <= 0.0) {
        return;
    }

    // ============================================================
    // Pass 1: Identify all edges that need subdivision and insert
    // intermediate vertices.
    // ============================================================

    // Map from canonical edge key to subdivision info.
    std::unordered_map<EdgeKey, SubdividedEdge, EdgeKeyHash>
        edge_subdivisions;

    // Scan all triangle edges.  Each edge is visited potentially
    // twice (once per adjacent triangle), but the canonical key
    // ensures we only subdivide once.
    for (const auto &tri : triangles) {
        for (int e = 0; e < 3; ++e) {
            const std::size_t va = tri.vertex_indices[e];
            const std::size_t vb =
                tri.vertex_indices[(e + 1) % 3];

            EdgeKey key(va, vb);

            // Skip if already processed.
            if (edge_subdivisions.count(key) > 0) {
                continue;
            }

            // Compute edge length.
            const Vector3d &pa = vertices[va].position;
            const Vector3d &pb = vertices[vb].position;
            const double dx = pb.x - pa.x;
            const double dy = pb.y - pa.y;
            const double dz = pb.z - pa.z;
            const double edge_len =
                std::sqrt(dx * dx + dy * dy + dz * dz);

            // Local threshold: ratio * min(cell_size_a, cell_size_b).
            // Use cell_sizes for original vertices.  For vertices
            // added by earlier subdivision passes (if any), use a
            // fallback of 0 which would make the threshold 0 and
            // always trigger subdivision — but since we only run
            // one pass, all vertices here are original QEF vertices.
            const double cs_a =
                (va < cell_sizes.size()) ? cell_sizes[va] : 0.0;
            const double cs_b =
                (vb < cell_sizes.size()) ? cell_sizes[vb] : 0.0;
            const double local_cell_size =
                std::min(cs_a, cs_b);

            // Guard: if either cell size is zero (shouldn't happen
            // for valid QEF vertices), skip subdivision for this
            // edge to avoid division by zero.
            if (local_cell_size <= 0.0) {
                continue;
            }

            const double threshold =
                max_edge_ratio * local_cell_size;

            if (edge_len <= threshold) {
                continue;  // Edge is short enough, no subdivision.
            }

            // Number of segments to split the edge into.
            // ceil(edge_len / threshold) ensures each segment
            // is at most `threshold` long.
            const std::size_t n_segments =
                static_cast<std::size_t>(
                    std::ceil(edge_len / threshold));
            if (n_segments <= 1) {
                continue;  // Rounding: edge is just barely over.
            }

            // Insert intermediate vertices.  Positions and normals
            // are linearly interpolated from the endpoints.  We
            // interpolate from v_lo to v_hi (canonical order).
            const Vector3d &p_lo =
                vertices[key.v_lo].position;
            const Vector3d &p_hi =
                vertices[key.v_hi].position;
            const Vector3d &n_lo =
                vertices[key.v_lo].normal;
            const Vector3d &n_hi =
                vertices[key.v_hi].normal;

            SubdividedEdge sub;
            sub.intermediate_indices.reserve(n_segments - 1);

            for (std::size_t m = 1; m < n_segments; ++m) {
                const double t =
                    static_cast<double>(m) /
                    static_cast<double>(n_segments);

                // Linearly interpolated position.
                Vector3d pos = {
                    p_lo.x + t * (p_hi.x - p_lo.x),
                    p_lo.y + t * (p_hi.y - p_lo.y),
                    p_lo.z + t * (p_hi.z - p_lo.z)};

                // Linearly interpolated normal (renormalized).
                Vector3d nrm = {
                    n_lo.x + t * (n_hi.x - n_lo.x),
                    n_lo.y + t * (n_hi.y - n_lo.y),
                    n_lo.z + t * (n_hi.z - n_lo.z)};
                const double nrm_len = std::sqrt(
                    nrm.x * nrm.x + nrm.y * nrm.y +
                    nrm.z * nrm.z);
                if (nrm_len > 1e-15) {
                    nrm.x /= nrm_len;
                    nrm.y /= nrm_len;
                    nrm.z /= nrm_len;
                }

                const std::size_t new_idx = vertices.size();
                vertices.push_back({pos, nrm});
                sub.intermediate_indices.push_back(new_idx);
            }

            edge_subdivisions[key] = std::move(sub);
        }
    }

    // If no edges were subdivided, nothing to do.
    if (edge_subdivisions.empty()) {
        return;
    }

    // ============================================================
    // Pass 2: Re-triangulate.  Walk each original triangle and,
    // for each edge that was subdivided, split the triangle
    // accordingly.
    // ============================================================
    //
    // Strategy: For each triangle, collect the full vertex chain
    // along each of its 3 edges (including any intermediate
    // vertices).  Then triangulate the resulting polygon using a
    // fan from the "apex" vertex (the vertex opposite the most-
    // subdivided edge), or a more general approach for multiple
    // subdivided edges.
    //
    // The simplest correct approach for arbitrary combinations:
    // - Build the full ordered vertex list around the triangle
    //   perimeter (inserting intermediates along each edge).
    // - Fan-triangulate from the first vertex in the list.
    //
    // This produces well-shaped triangles when only one edge is
    // subdivided (fan from the opposite vertex) and acceptable
    // triangles for multi-edge subdivision.

    std::vector<MeshTriangle> new_triangles;
    new_triangles.reserve(triangles.size() * 2);

    for (const auto &tri : triangles) {
        // For each of the 3 edges, build the chain of vertices
        // from tri.vertex_indices[e] to tri.vertex_indices[(e+1)%3],
        // including intermediate vertices if the edge was subdivided.
        //
        // The perimeter vertex list concatenates these 3 chains
        // (excluding the last vertex of each chain, since it equals
        // the first vertex of the next chain).

        // Collect full perimeter.
        std::vector<std::size_t> perimeter;
        perimeter.reserve(12);  // Typical: 3 + a few intermediates.

        bool has_subdivision = false;

        for (int e = 0; e < 3; ++e) {
            const std::size_t va = tri.vertex_indices[e];
            const std::size_t vb =
                tri.vertex_indices[(e + 1) % 3];

            // Start with the first vertex of this edge.
            perimeter.push_back(va);

            // Check if this edge was subdivided.
            EdgeKey key(va, vb);
            auto it = edge_subdivisions.find(key);
            if (it != edge_subdivisions.end()) {
                has_subdivision = true;
                const auto &intermediates =
                    it->second.intermediate_indices;

                // The intermediates are stored in order from
                // v_lo to v_hi.  If va == v_lo, append in
                // forward order; if va == v_hi, append in
                // reverse order.
                if (va == key.v_lo) {
                    for (std::size_t idx : intermediates) {
                        perimeter.push_back(idx);
                    }
                } else {
                    // va == v_hi: reverse order.
                    for (std::size_t k = intermediates.size();
                         k > 0; --k) {
                        perimeter.push_back(
                            intermediates[k - 1]);
                    }
                }
            }
            // Note: vb is NOT appended here — it will be
            // appended as va of the next edge.
        }

        if (!has_subdivision) {
            // No edges subdivided: keep original triangle.
            new_triangles.push_back(tri);
            continue;
        }

        // Fan-triangulate from perimeter[0].  This produces
        // (perimeter.size() - 2) triangles.
        const std::size_t v0 = perimeter[0];
        for (std::size_t k = 1; k + 1 < perimeter.size(); ++k) {
            MeshTriangle t;
            t.vertex_indices[0] = v0;
            t.vertex_indices[1] = perimeter[k];
            t.vertex_indices[2] = perimeter[k + 1];
            new_triangles.push_back(t);
        }
    }

    // Replace original triangles with subdivided result.
    triangles = std::move(new_triangles);
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_EDGE_SUBDIV_HPP_
