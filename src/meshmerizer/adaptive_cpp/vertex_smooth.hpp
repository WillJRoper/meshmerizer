#ifndef MESHMERIZER_ADAPTIVE_CPP_VERTEX_SMOOTH_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_VERTEX_SMOOTH_HPP_

/**
 * @file vertex_smooth.hpp
 * @brief Laplacian vertex smoothing for QEF vertices.
 *
 * After the QEF solve positions each active leaf's representative
 * vertex independently, adjacent vertices can have inconsistent
 * spacing — especially in regions with noisy particle distributions
 * or poor Hermite data.  Laplacian smoothing regularizes vertex
 * positions by iteratively moving each vertex toward the centroid
 * of its neighbors.
 *
 * The algorithm uses Jacobi-style iteration: each iteration reads
 * all positions from the previous step and writes updated positions
 * to a separate buffer, then swaps.  This avoids order-dependent
 * results and is trivially parallelizable with OpenMP.
 *
 * No cell-boundary clamping is applied.  The DC face topology is
 * fixed before smoothing runs, so a vertex drifting outside its
 * original cell has no topological consequence.  In complex regions
 * we explicitly want the smoothing to deviate from the noisy SPH
 * isosurface to produce a cleaner mesh.
 *
 * Reference:
 *   - Taubin, "A Signal Processing Approach To Fair Surface
 *     Design", SIGGRAPH 1995 (for the general Laplacian smoothing
 *     framework; we use the simpler one-parameter variant here).
 *
 * @see vertex_adjacency.hpp for the CSR adjacency structure.
 */

#include <cstddef>
#include <vector>

#include "mesh.hpp"
#include "omp_config.hpp"
#include "vector3d.hpp"
#include "vertex_adjacency.hpp"

/**
 * @brief Apply Laplacian smoothing to QEF vertex positions.
 *
 * For each iteration, every vertex is moved toward the centroid
 * of its neighbors:
 *
 *   v_i' = v_i + lambda * (centroid(neighbors(i)) - v_i)
 *
 * where lambda (strength) controls how far each vertex moves
 * toward the centroid per iteration.  lambda = 0 means no
 * movement; lambda = 1 snaps to the centroid.
 *
 * Vertices with no neighbors (degree 0) are left unchanged.
 *
 * Only the position field of MeshVertex is updated.  Normals
 * are left as-is (they remain the QEF-derived normals, which
 * are used for visual orientation only and do not affect
 * topology).
 *
 * @param vertices        Vertex array to smooth (modified
 *     in-place).  Only the position field is updated.
 * @param adjacency       CSR adjacency structure from
 *     build_vertex_adjacency.
 * @param iterations      Number of smoothing iterations.  0
 *     means no smoothing (early return).
 * @param strength        Smoothing strength lambda in (0, 1].
 *     Values outside this range are clamped.
 */
inline void laplacian_smooth_vertices(
    std::vector<MeshVertex> &vertices,
    const VertexAdjacency &adjacency,
    std::uint32_t iterations,
    double strength) {

    const std::size_t n = vertices.size();
    if (n == 0 || iterations == 0) {
        return;
    }

    // Clamp strength to [0, 1] for safety.
    if (strength < 0.0) {
        strength = 0.0;
    }
    if (strength > 1.0) {
        strength = 1.0;
    }
    if (strength == 0.0) {
        return;  // No movement requested.
    }

    // Temporary buffer for new positions (Jacobi iteration:
    // read from current, write to new, then swap).
    std::vector<Vector3d> new_positions(n);

    for (std::uint32_t iter = 0; iter < iterations; ++iter) {
        // Parallel pass: compute new positions for all vertices.
#pragma omp parallel for schedule(static)
        for (std::size_t vi = 0; vi < n; ++vi) {
            const std::size_t nbr_begin = adjacency.offsets[vi];
            const std::size_t nbr_end =
                adjacency.offsets[vi + 1];
            const std::size_t degree = nbr_end - nbr_begin;

            if (degree == 0) {
                // No neighbors: keep position unchanged.
                new_positions[vi] = vertices[vi].position;
                continue;
            }

            // Compute centroid of neighbor positions.
            Vector3d centroid = {0.0, 0.0, 0.0};
            for (std::size_t k = nbr_begin; k < nbr_end; ++k) {
                const std::size_t vj = adjacency.neighbors[k];
                centroid.x += vertices[vj].position.x;
                centroid.y += vertices[vj].position.y;
                centroid.z += vertices[vj].position.z;
            }
            const double inv_deg =
                1.0 / static_cast<double>(degree);
            centroid.x *= inv_deg;
            centroid.y *= inv_deg;
            centroid.z *= inv_deg;

            // Move vertex toward centroid by strength lambda.
            // v_new = v_old + lambda * (centroid - v_old)
            //       = (1 - lambda) * v_old + lambda * centroid
            const Vector3d &pos = vertices[vi].position;
            const double one_minus_lambda = 1.0 - strength;
            new_positions[vi] = {
                one_minus_lambda * pos.x +
                    strength * centroid.x,
                one_minus_lambda * pos.y +
                    strength * centroid.y,
                one_minus_lambda * pos.z +
                    strength * centroid.z};
        }

        // Copy new positions back into vertex array.
        for (std::size_t vi = 0; vi < n; ++vi) {
            vertices[vi].position = new_positions[vi];
        }
    }
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_VERTEX_SMOOTH_HPP_
