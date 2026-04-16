/**
 * @file hermite.hpp
 * @brief Hermite sample generation for dual contouring on the adaptive octree.
 *
 * A Hermite sample is a pair (position, normal) that represents one local
 * geometric constraint on the isosurface. For each sign-changing edge of a
 * leaf cell we linearly interpolate a crossing point, then evaluate the SPH
 * gradient at that point to obtain the outward surface normal.
 *
 * These samples are the input to the QEF vertex solve in Phase 8.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_HERMITE_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_HERMITE_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "bounding_box.hpp"
#include "kernel_wendland_c2.hpp"
#include "vector3d.hpp"

/**
 * @brief One Hermite sample: a surface crossing point and its outward normal.
 *
 * The position is the linearly interpolated point along a sign-changing edge
 * where the scalar field equals the isovalue.
 *
 * The normal is the outward surface normal at that point, computed from the
 * normalized negation of the SPH gradient:
 *
 *   normal = -normalize( sum_i nabla W(x - x_i, h_i) )
 *
 * The SPH gradient points into the fluid (toward higher field values), so
 * negating it gives the outward direction.
 *
 * A zero-length normal indicates a degenerate sample where all contributing
 * kernels evaluated to zero gradient at this position. The QEF solver in
 * Phase 8 should treat such samples as low-confidence constraints.
 */
struct HermiteSample {
    Vector3d position;
    Vector3d normal;
};

/**
 * @brief The 12 edges of a unit cube, as pairs of corner indices.
 *
 * Corner indices encode position within the cell bounding box:
 * - bit 0: high x half (1 = max.x, 0 = min.x)
 * - bit 1: high y half (1 = max.y, 0 = min.y)
 * - bit 2: high z half (1 = max.z, 0 = min.z)
 *
 * Edges are ordered by axis group: 4 X-edges, then 4 Y-edges, then 4 Z-edges.
 * Within each group the edges are enumerated in the order that the remaining
 * two bits (the perpendicular axes) count from 00 to 11.
 *
 * X-edges (bit 0 varies, bits 1-2 fixed):
 *   0: (0,1)  1: (2,3)  2: (4,5)  3: (6,7)
 * Y-edges (bit 1 varies, bits 0,2 fixed):
 *   4: (0,2)  5: (1,3)  6: (4,6)  7: (5,7)
 * Z-edges (bit 2 varies, bits 0,1 fixed):
 *   8: (0,4)  9: (1,5) 10: (2,6) 11: (3,7)
 */
constexpr std::array<std::array<std::uint8_t, 2>, 12> CELL_EDGE_PAIRS = {{
    // X-edges
    {0, 1},
    {2, 3},
    {4, 5},
    {6, 7},
    // Y-edges
    {0, 2},
    {1, 3},
    {4, 6},
    {5, 7},
    // Z-edges
    {0, 4},
    {1, 5},
    {2, 6},
    {3, 7},
}};

/**
 * @brief Return the world-space position of one corner of a bounding box.
 *
 * Corner index encoding matches the rest of the octree codebase:
 * bit 0 = high x, bit 1 = high y, bit 2 = high z.
 *
 * @param bounds Cell bounding box.
 * @param corner_index Corner index in ``[0, 7]``.
 * @return World-space position of the corner.
 */
inline Vector3d corner_position(const BoundingBox &bounds,
                                std::uint8_t corner_index) {
    return {
        (corner_index & 1U) ? bounds.max.x : bounds.min.x,
        (corner_index & 2U) ? bounds.max.y : bounds.min.y,
        (corner_index & 4U) ? bounds.max.z : bounds.min.z,
    };
}

/**
 * @brief Linearly interpolate a crossing point along a sign-changing edge.
 *
 * Given two endpoint field values that straddle the isovalue, returns the
 * parameter ``t`` in ``[0, 1]`` such that:
 *
 *   field_value_at(pos_a + t * (pos_b - pos_a)) == isovalue
 *
 * If the difference ``value_b - value_a`` is below the minimum representable
 * step (i.e. the field is effectively flat across this edge), ``t`` is clamped
 * to 0.5 so the crossing falls at the midpoint.
 *
 * @param value_a Scalar field value at the first endpoint.
 * @param value_b Scalar field value at the second endpoint.
 * @param isovalue Target surface level.
 * @return Interpolation parameter in ``[0, 1]``.
 */
inline double edge_crossing_parameter(double value_a, double value_b,
                                      double isovalue) {
    const double denominator = value_b - value_a;
    if (std::abs(denominator) < 1e-12) {
        return 0.5;
    }
    const double t = (isovalue - value_a) / denominator;
    // Clamp to the edge in case of floating-point overshoot.
    return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
}

/**
 * @brief Evaluate the SPH gradient of the scalar field at a point.
 *
 * The gradient is the sum of the normalized Wendland C2 kernel gradients from
 * all contributing particles:
 *
 *   nabla_rho(x) = sum_i nabla W(x - x_i, h_i)
 *
 * This vector points into the fluid (toward higher density). The caller is
 * responsible for negating it to obtain the outward normal.
 *
 * @param query_point Position at which to evaluate the gradient.
 * @param contributor_indices Indices of particles that may contribute.
 * @param positions Particle positions in world space.
 * @param smoothing_lengths Per-particle support radii.
 * @return Unnormalized SPH gradient vector at the query point.
 */
inline Vector3d evaluate_field_gradient_at_point(
    const Vector3d &query_point,
    const std::vector<std::size_t> &contributor_indices,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths) {
    Vector3d gradient = {0.0, 0.0, 0.0};
    for (std::size_t particle_index : contributor_indices) {
        const Vector3d displacement = {
            query_point.x - positions[particle_index].x,
            query_point.y - positions[particle_index].y,
            query_point.z - positions[particle_index].z,
        };
        const Vector3d kernel_gradient = evaluate_wendland_c2_gradient(
            displacement, smoothing_lengths[particle_index],
            /*normalize=*/true);
        gradient.x += kernel_gradient.x;
        gradient.y += kernel_gradient.y;
        gradient.z += kernel_gradient.z;
    }
    return gradient;
}

/**
 * @brief Compute all Hermite samples for one leaf cell.
 *
 * Inspects each of the 12 cell edges. For every edge whose two endpoints
 * differ in sign relative to the isovalue, this function:
 *
 * 1. Interpolates the crossing point along the edge using the corner values.
 * 2. Evaluates the SPH gradient at the crossing point.
 * 3. Normalizes the negated gradient to obtain the outward surface normal.
 * 4. Stores the crossing position and outward normal as a ``HermiteSample``.
 *
 * A cell with no sign-changing edges produces an empty result. A cell with
 * all corners on the same side of the isovalue never reaches this function in
 * practice because ``cell_may_contain_isosurface`` would have returned false.
 *
 * @param bounds Cell bounding box in world space.
 * @param corner_values Scalar field samples at the eight cell corners,
 *     indexed by the same corner-index convention used elsewhere in the
 *     codebase (bit 0 = high x, bit 1 = high y, bit 2 = high z).
 * @param corner_sign_mask Precomputed sign mask: bit i is set when
 *     ``corner_values[i] >= isovalue``.
 * @param contributor_indices Indices of particles that contribute to this
 *     cell's field.
 * @param positions Particle positions in world space.
 * @param smoothing_lengths Per-particle support radii.
 * @param isovalue The target surface level.
 * @return Hermite samples, one per sign-changing edge.
 */
inline std::vector<HermiteSample> compute_cell_hermite_samples(
    const BoundingBox &bounds,
    const std::array<double, 8> &corner_values,
    std::uint8_t corner_sign_mask,
    const std::vector<std::size_t> &contributor_indices,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    double isovalue) {
    std::vector<HermiteSample> samples;
    samples.reserve(4);  // Most cells have at most a handful of crossings.

    for (const auto &edge : CELL_EDGE_PAIRS) {
        const std::uint8_t corner_a = edge[0];
        const std::uint8_t corner_b = edge[1];

        // Determine whether the two endpoints are on opposite sides of the
        // isovalue. A sign change exists when exactly one of the two bits is
        // set in the precomputed corner sign mask.
        const bool above_a = ((corner_sign_mask >> corner_a) & 1U) != 0U;
        const bool above_b = ((corner_sign_mask >> corner_b) & 1U) != 0U;
        if (above_a == above_b) {
            continue;
        }

        // Linearly interpolate the crossing position along the edge.
        const Vector3d pos_a = corner_position(bounds, corner_a);
        const Vector3d pos_b = corner_position(bounds, corner_b);
        const double t = edge_crossing_parameter(
            corner_values[corner_a], corner_values[corner_b], isovalue);
        const Vector3d crossing = {
            pos_a.x + t * (pos_b.x - pos_a.x),
            pos_a.y + t * (pos_b.y - pos_a.y),
            pos_a.z + t * (pos_b.z - pos_a.z),
        };

        // Evaluate the SPH gradient at the crossing point. The gradient
        // points into the fluid (toward higher field values). Negate it to
        // obtain the outward normal.
        const Vector3d gradient = evaluate_field_gradient_at_point(
            crossing, contributor_indices, positions, smoothing_lengths);
        const double magnitude = std::sqrt(gradient.x * gradient.x +
                                           gradient.y * gradient.y +
                                           gradient.z * gradient.z);

        // Normalize the outward normal. A near-zero gradient indicates a
        // degenerate sample; store a zero normal so the QEF solver can
        // recognize and down-weight it.
        Vector3d outward_normal = {0.0, 0.0, 0.0};
        if (magnitude > 1e-12) {
            outward_normal = {
                -gradient.x / magnitude,
                -gradient.y / magnitude,
                -gradient.z / magnitude,
            };
        }

        samples.push_back({crossing, outward_normal});
    }

    return samples;
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_HERMITE_HPP_
