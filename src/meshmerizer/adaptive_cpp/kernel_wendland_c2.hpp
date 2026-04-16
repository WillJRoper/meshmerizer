/**
 * @file kernel_wendland_c2.hpp
 * @brief Wendland C2 SPH kernel helpers for the adaptive meshing core.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_KERNEL_WENDLAND_C2_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_KERNEL_WENDLAND_C2_HPP_

#include <cmath>

#include "vector3d.hpp"

/**
 * @brief Return the unnormalized Wendland C2 kernel profile.
 *
 * The profile is scaled so the value at the particle center is 1 and the value
 * at the support radius is 0.
 *
 * @param q Radius divided by smoothing length.
 * @return Unnormalized kernel weight.
 */
inline double wendland_c2_profile(double q) {
    if (q < 0.0 || q >= 1.0) {
        return 0.0;
    }
    const double one_minus_q = 1.0 - q;
    return std::pow(one_minus_q, 4) * (1.0 + 4.0 * q);
}

/**
 * @brief Return the 3-D Wendland C2 normalization constant.
 *
 * @param smoothing_length Support radius of the particle kernel.
 * @return Multiplicative constant that makes the kernel integrate to unity.
 */
inline double wendland_c2_normalization(double smoothing_length) {
    if (smoothing_length <= 0.0) {
        return 0.0;
    }
    return 21.0 / (2.0 * M_PI * std::pow(smoothing_length, 3));
}

/**
 * @brief Return the Wendland C2 kernel value at a given radius.
 *
 * @param radius Distance from the particle center.
 * @param smoothing_length Support radius of the particle kernel.
 * @param normalize Whether to apply the 3-D normalization constant.
 * @return Kernel value at the requested radius.
 */
inline double evaluate_wendland_c2(double radius, double smoothing_length,
                                   bool normalize) {
    if (smoothing_length <= 0.0) {
        return 0.0;
    }
    double weight = wendland_c2_profile(radius / smoothing_length);
    if (normalize) {
        weight *= wendland_c2_normalization(smoothing_length);
    }
    return weight;
}

/**
 * @brief Return the derivative of the unnormalized profile with respect to q.
 *
 * @param q Radius divided by smoothing length.
 * @return Derivative of the unnormalized profile.
 */
inline double wendland_c2_profile_derivative(double q) {
    if (q < 0.0 || q >= 1.0) {
        return 0.0;
    }
    return -20.0 * q * std::pow(1.0 - q, 3);
}

/**
 * @brief Return the gradient of the Wendland C2 kernel at a displacement.
 *
 * @param displacement Vector from particle center to query position.
 * @param smoothing_length Support radius of the particle kernel.
 * @param normalize Whether to apply the 3-D normalization constant.
 * @return Kernel gradient vector.
 */
inline Vector3d evaluate_wendland_c2_gradient(const Vector3d &displacement,
                                              double smoothing_length,
                                              bool normalize) {
    const double radius =
        std::sqrt(displacement.x * displacement.x +
                  displacement.y * displacement.y +
                  displacement.z * displacement.z);
    if (smoothing_length <= 0.0 || radius <= 0.0 || radius >= smoothing_length) {
        return {0.0, 0.0, 0.0};
    }

    double scale = wendland_c2_profile_derivative(radius / smoothing_length);
    scale /= (smoothing_length * radius);
    if (normalize) {
        scale *= wendland_c2_normalization(smoothing_length);
    }

    return {
        displacement.x * scale,
        displacement.y * scale,
        displacement.z * scale,
    };
}

#endif  // MESHMERIZER_ADAPTIVE_CPP_KERNEL_WENDLAND_C2_HPP_
