#ifndef MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_CONTEXT_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_CONTEXT_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

struct OctreeCell;

/**
 * @file refinement_context.hpp
 * @brief Side-car scheduling state for refinement closure.
 *
 * This context owns transient, non-serialized scheduler metadata that should
 * not live directly on `OctreeCell`.
 */

enum class RefinementTaskState : std::uint8_t {
    kIdle = 0U,
    kQueued = 1U,
    kProcessing = 2U,
    kRetired = 3U,
};

struct RefinementCellState {
    /** Monotone demanded minimum depth for this cell. */
    std::uint32_t required_depth = 0U;

    /** Current scheduling state for this cell. */
    RefinementTaskState task_state = RefinementTaskState::kIdle;

    /** Optional generation counter for stale-task diagnostics. */
    std::uint32_t generation = 0U;
};

class RefinementContext {
public:
    RefinementContext(
        std::vector<OctreeCell> &all_cells,
        std::vector<std::size_t> &all_contributors);

    /** Ensure the side-car state covers all currently allocated cells. */
    void sync_cell_state_size();

    /** Return mutable state for a valid cell index. */
    RefinementCellState &state(std::size_t cell_index);

    /** Return const state for a valid cell index. */
    const RefinementCellState &state(std::size_t cell_index) const;

    /** Return the number of tracked cells. */
    std::size_t size() const;

    /** Reset non-persistent scheduler state for all tracked cells. */
    void reset_task_states();

    /**
     * @brief Raise a cell's required depth monotonically.
     *
     * Returns true only when the stored requirement increased.
     */
    bool raise_required_depth_to(
        std::size_t cell_index,
        std::uint32_t new_required_depth);

    /** Return the current required depth for a cell. */
    std::uint32_t get_required_depth(std::size_t cell_index) const;

    /** Attempt to move a cell from idle to queued. */
    bool mark_queued(std::size_t cell_index);

    /** Mark a queued cell as processing. */
    bool mark_processing(std::size_t cell_index);

    /** Mark a processing cell as idle. */
    void mark_idle(std::size_t cell_index);

    /** Mark a cell as retired. */
    void mark_retired(std::size_t cell_index);

    /** Expose the octree cell storage used by the closure driver. */
    std::vector<OctreeCell> &cells();

    /** Expose the flat contributor storage used by the closure driver. */
    std::vector<std::size_t> &contributors();

private:
    std::vector<OctreeCell> &all_cells_;
    std::vector<std::size_t> &all_contributors_;
    std::vector<RefinementCellState> cell_state_;
};

#endif  // MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_CONTEXT_HPP_
