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
    double table_cadence_seconds = 10.0;
    std::string status_operation = "Building";
    std::string status_function = "refine_octree";
    std::string phase_name = "refine_octree";
    // When > 0.0, ``apply_balance_split`` / ``apply_surface_split`` will
    // self-enqueue any newly published child whose edge length exceeds
    // this target AND has ``has_surface`` set AND has depth < max_depth.
    // This fuses the surface-band outer loop into the closure: a single
    // closure run reaches convergence instead of N passes that each
    // re-discover newly-exposed surface children. Set by the surface-band
    // entry point; left at 0.0 elsewhere so unrelated phases ignore it.
    // Placed at the end of the struct so existing positional aggregate
    // initializers in callers (adaptive_solid.hpp, faces.hpp, etc.) remain
    // valid without requiring designated initializers.
    double surface_band_target_leaf_size = 0.0;

    // When > 0.0, the incremental thickening-band task chain is active.
    // kClassify tasks evaluate the SPH field at each cell centre and write
    // the inside/outside label to the per-cell classification side-car.
    // kDistanceUpdate tasks walk face-neighbours to compute the approximate
    // outside distance and enqueue kRefine when the cell is in the band.
    // kRefine children are re-enqueued as kClassify to continue the chain.
    // Set by ``refine_thickening_band_with_closure`` when B2 task-based
    // convergence is requested; left at 0.0 to disable.
    double thickening_band_target_leaf_size = 0.0;

    // The outward thickening radius used with the B2 task chain.  Only
    // consulted when ``thickening_band_target_leaf_size > 0.0``.
    double thickening_radius = 0.0;
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
    const std::vector<std::size_t> &seed_cell_indices,
    const std::vector<std::uint32_t> &seed_target_depths,
    const std::vector<std::uint8_t> *initial_inside_flags = nullptr,
    const std::vector<double> *initial_center_values = nullptr,
    const std::vector<std::uint8_t> *initial_occupancy_states = nullptr,
    std::vector<std::uint8_t> *dirty_cells = nullptr,
    std::vector<std::uint8_t> *classified_inside_flags = nullptr,
    std::vector<double> *classified_center_values = nullptr,
    std::vector<std::uint8_t> *classified_occupancy_states = nullptr,
    std::vector<std::array<std::size_t, 6>> *classified_face_neighbors =
        nullptr);

bool refine_cells_to_next_depth_with_closure(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config,
    const std::vector<std::size_t> &seed_cell_indices);

#endif  // MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_CLOSURE_HPP_
