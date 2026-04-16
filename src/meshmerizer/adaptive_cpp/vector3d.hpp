/**
 * @file vector3d.hpp
 * @brief Minimal three-component vector for adaptive geometry utilities.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_VECTOR3D_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_VECTOR3D_HPP_

/**
 * @brief Small world-space vector with explicit `double` storage.
 *
 * The adaptive rewrite uses this type for geometric bookkeeping where the
 * semantics of each coordinate should stay obvious in the code and generated
 * documentation.
 */
struct Vector3d {
    double x;
    double y;
    double z;
};

#endif  // MESHMERIZER_ADAPTIVE_CPP_VECTOR3D_HPP_
