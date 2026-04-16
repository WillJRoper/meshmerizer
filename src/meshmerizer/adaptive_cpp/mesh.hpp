/**
 * @file mesh.hpp
 * @brief Fundamental mesh output types for the adaptive meshing pipeline.
 *
 * These types carry the final mesh geometry produced by dual contouring.
 * They are intentionally minimal: the adaptive pipeline fills them during
 * the vertex solve (Phase 8) and face construction (Phase 9) stages.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_MESH_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_MESH_HPP_

#include <cstddef>

#include "vector3d.hpp"

/**
 * @brief One vertex of the output mesh.
 *
 * Each active leaf cell produces exactly one representative MeshVertex via the
 * QEF solve. The position is the point inside the leaf that best satisfies the
 * tangent-plane constraints from the leaf's Hermite samples. The normal is the
 * normalized mean of those sample normals.
 */
struct MeshVertex {
    Vector3d position;
    Vector3d normal;
};

/**
 * @brief One triangle of the output mesh.
 *
 * Indices reference entries in the flat vertex array produced during face
 * construction. Winding order is consistent and outward-facing as defined
 * during Phase 9.
 */
struct MeshTriangle {
    std::size_t vertex_indices[3];
};

#endif  // MESHMERIZER_ADAPTIVE_CPP_MESH_HPP_
