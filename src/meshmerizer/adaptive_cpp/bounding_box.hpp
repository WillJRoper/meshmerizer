/**
 * @file bounding_box.hpp
 * @brief Axis-aligned bounding-box helpers for adaptive meshing.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_BOUNDING_BOX_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_BOUNDING_BOX_HPP_

#include "vector3d.hpp"

/**
 * @brief Axis-aligned bounding box in world space.
 *
 * The box stores its inclusive lower corner and exclusive upper corner so the
 * same type can be used consistently for cell ownership and overlap tests.
 */
struct BoundingBox {
    Vector3d min;
    Vector3d max;

    /**
     * @brief Return whether a point lies inside the box.
     *
     * @param point World-space point to test.
     * @return True when the point lies within the half-open box bounds.
     */
    bool contains(const Vector3d &point) const {
        return point.x >= min.x && point.x < max.x && point.y >= min.y &&
               point.y < max.y && point.z >= min.z && point.z < max.z;
    }

    /**
     * @brief Return whether two bounding boxes overlap with positive volume.
     *
     * @param other Bounding box to test against.
     * @return True when the boxes overlap on all three axes.
     */
    bool overlaps(const BoundingBox &other) const {
        return min.x < other.max.x && max.x > other.min.x &&
               min.y < other.max.y && max.y > other.min.y &&
               min.z < other.max.z && max.z > other.min.z;
    }

    /**
     * @brief Return the geometric center of the box.
     *
     * @return Center point halfway between the minimum and maximum corners.
     */
    Vector3d center() const {
        return {
            0.5 * (min.x + max.x),
            0.5 * (min.y + max.y),
            0.5 * (min.z + max.z),
        };
    }

    /**
     * @brief Return the extent (width, height, depth) of the box.
     *
     * @return Vector indicating the box dimensions along each axis.
     */
    Vector3d extent() const {
        return {
            max.x - min.x,
            max.y - min.y,
            max.z - min.z,
        };
    }
};

#endif  // MESHMERIZER_ADAPTIVE_CPP_BOUNDING_BOX_HPP_
