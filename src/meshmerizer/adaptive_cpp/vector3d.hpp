/**
 * @file vector3d.hpp
 * @brief Minimal three-component vector for adaptive geometry utilities.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_VECTOR3D_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_VECTOR3D_HPP_

#include <cmath>

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

    /* ---- arithmetic operators ----------------------------------------- */

    /**
     * @brief Component-wise addition.
     */
    Vector3d operator+(const Vector3d &rhs) const {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }

    /**
     * @brief Component-wise subtraction.
     */
    Vector3d operator-(const Vector3d &rhs) const {
        return {x - rhs.x, y - rhs.y, z - rhs.z};
    }

    /**
     * @brief Scalar multiplication (vector * scalar).
     */
    Vector3d operator*(double s) const {
        return {x * s, y * s, z * s};
    }

    /**
     * @brief In-place component-wise addition.
     */
    Vector3d &operator+=(const Vector3d &rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    /**
     * @brief In-place component-wise subtraction.
     */
    Vector3d &operator-=(const Vector3d &rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    /**
     * @brief In-place scalar multiplication.
     */
    Vector3d &operator*=(double s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    /**
     * @brief Dot product of two vectors.
     */
    double dot(const Vector3d &rhs) const {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }

    /**
     * @brief Euclidean length of the vector.
     */
    double length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    /**
     * @brief Squared Euclidean length of the vector.
     */
    double length_squared() const {
        return x * x + y * y + z * z;
    }
};

/**
 * @brief Scalar multiplication (scalar * vector).
 */
inline Vector3d operator*(double s, const Vector3d &v) {
    return {s * v.x, s * v.y, s * v.z};
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_VECTOR3D_HPP_
