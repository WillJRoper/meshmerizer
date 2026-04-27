#ifndef MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_CLOSURE_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_CLOSURE_HPP_

#include <utility>
#include <vector>

#include "bounding_box.hpp"
#include "refinement_context.hpp"
#include "vector3d.hpp"

struct RefinementClosureConfig {
    double isovalue = 0.0;
    std::uint32_t max_depth = 0U;
    BoundingBox domain;
    std::uint32_t base_resolution = 0U;
    std::uint32_t worker_count = 1U;
    std::uint32_t minimum_usable_hermite_samples = 3U;
    double max_qef_rms_residual_ratio = 0.1;
    double min_normal_alignment_threshold = 0.97;
};

/**
 * @brief Placeholder serial closure driver.
 *
 * The initial implementation only prepares the shared abstraction boundary.
 * Later commits will move the current refinement logic behind this interface.
 */
std::pair<std::vector<OctreeCell>, std::vector<std::size_t>>
refine_with_closure(
    std::vector<OctreeCell> initial_cells,
    std::vector<std::size_t> initial_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config);

#endif  // MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_CLOSURE_HPP_
