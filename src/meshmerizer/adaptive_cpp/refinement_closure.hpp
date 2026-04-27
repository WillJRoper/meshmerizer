#ifndef MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_CLOSURE_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_CLOSURE_HPP_

#include <string>
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
    double table_cadence_seconds = 20.0;
    std::string status_operation = "Building";
    std::string status_function = "refine_octree";
    std::string phase_name = "refine_octree";
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

bool refine_surface_band_with_closure(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config,
    double max_surface_leaf_size);

bool refine_thickening_band_with_closure(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config,
    double target_leaf_size,
    const std::vector<std::uint32_t> &seed_target_depths,
    std::vector<std::uint8_t> *dirty_cells = nullptr);

bool refine_cells_to_next_depth_with_closure(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config,
    const std::vector<std::size_t> &seed_cell_indices);

#endif  // MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_CLOSURE_HPP_
