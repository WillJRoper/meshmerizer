/**
 * @file particle.hpp
 * @brief Lightweight particle payload for the adaptive SPH field.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_PARTICLE_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_PARTICLE_HPP_

#include <cstdint>

#include "vector3d.hpp"

/**
 * @brief Minimal particle payload used by the adaptive meshing core.
 *
 * Only the data required for scalar and gradient evaluation is stored in the
 * particle core type.
 */
struct Particle {
    Vector3d position;
    float value;
    float smoothing_length;
    std::uint64_t id;
};

#endif  // MESHMERIZER_ADAPTIVE_CPP_PARTICLE_HPP_
