#ifndef MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_CONTEXT_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_CONTEXT_HPP_

#include <cstddef>
#include <cstdint>
#include <array>
#include <atomic>
#include <span>
#include <vector>

#include "refinement_arena.hpp"

struct OctreeCell;

/**
 * @file refinement_context.hpp
 * @brief Side-car scheduling state for refinement closure.
 *
 * The context owns transient, non-serialized scheduler metadata that should
 * not live directly on `OctreeCell`. All storage (cells, contributors, and
 * the four scheduler side-cars) is backed by ``ChunkedArena<T>`` so that
 * appends are wait-free in the common case and reads never take a mutex.
 * Every side-car arena is grown in lockstep with the cell arena via
 * @ref reserve_cell_block, which guarantees that side-car index ``i`` is
 * valid for any ``i`` reachable through ``cells()``.
 */

enum class RefinementTaskState : std::uint8_t {
    kIdle = 0U,
    kQueued = 1U,
    kProcessing = 2U,
    kRetired = 3U,
};

struct RefinementChildBlock {
    /** Number of stable child indices owned by this parent state. */
    std::uint8_t child_count = 0U;

    /** Stable scheduler-visible child indices for this parent. */
    std::array<std::size_t, 8> child_indices = {};
};

struct RefinementContributorRange {
    std::int64_t begin = -1;
    std::int64_t end = -1;
};

class RefinementContext {
public:
    /**
     * @brief Adopt the caller's cell/contributor vectors into arenas.
     *
     * The vectors passed in are read once to seed the arenas; subsequent
     * mutations during the closure run flow through the arenas. The
     * caller-visible vectors are not kept in sync until @ref
     * materialize_into is invoked at run end.
     */
    RefinementContext(
        std::vector<OctreeCell> &all_cells,
        std::vector<std::size_t> &all_contributors);

    /**
     * @brief Ensure side-car arenas cover all currently allocated cells.
     *
     * Called once by the constructor and (idempotently) at the start of
     * each closure entry point so that any cells appended outside of the
     * worker loop have their scheduler state initialized before parallel
     * work begins.
     */
    void sync_cell_state_size();

    /** Return the current number of tracked cells (cell-arena size). */
    std::size_t size() const;

    /** Return the parent index for a cell, or SIZE_MAX for roots. */
    std::size_t parent_index(std::size_t cell_index) const;

    /** Reset non-persistent scheduler state for all tracked cells. */
    void reset_task_states();

    /** Record a stable child block for one parent cell. */
    void set_child_block(
        std::size_t cell_index,
        const std::vector<std::size_t> &child_indices);

    /** Append stable child indices for one parent cell. */
    void append_child_indices(
        std::size_t cell_index,
        std::vector<std::size_t> &out_indices) const;

    /** Return one stable child index for a parent/local-child slot. */
    std::size_t child_index_at(
        std::size_t cell_index,
        std::uint8_t child_slot) const;

    /** Record the contributor slice owned by one cell. */
    void set_contributor_range(
        std::size_t cell_index,
        std::int64_t begin,
        std::int64_t end);

    /** Return the context-owned contributor range for one cell. */
    RefinementContributorRange contributor_range(
        std::size_t cell_index) const;

    /** Return a cell's contributor slice as a span over arena storage. */
    std::span<const std::size_t> contributor_span(
        std::size_t cell_index) const;

    /** Copy a cell's contributor slice into a local vector. */
    void copy_contributors_for_cell(
        std::size_t cell_index,
        std::vector<std::size_t> &out_indices) const;

    /**
     * @brief Raise a cell's required depth monotonically.
     *
     * Returns true only when the stored requirement increased. This is a
     * single CAS on the cell's own slot; no ancestor traversal is
     * performed because the closure pipeline reads ``required_depth`` only
     * for the cell itself, not as a subtree aggregate.
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
    ChunkedArena<OctreeCell> &cells();

    /** Expose the octree cell storage used by the closure driver. */
    const ChunkedArena<OctreeCell> &cells() const;

    /** Expose the flat contributor storage used by the closure driver. */
    ChunkedArena<std::size_t> &contributors();

    /** Expose the flat contributor storage used by the closure driver. */
    const ChunkedArena<std::size_t> &contributors() const;

    /**
     * @brief Reserve a contiguous block of ``count`` cell slots in lockstep
     *        with all four side-car arenas.
     *
     * The cell arena's reservation may include chunk-alignment rounding so
     * that the returned block fits inside a single chunk (required for
     * ``child_begin + offset`` walks). Side-car arenas are then advanced
     * to ``begin + count`` via ``ChunkedArena::reserve_to`` so that every
     * index in ``[begin, begin + count)`` is valid for state queries the
     * moment this call returns.
     *
     * @param count Number of contiguous cell slots to reserve. Must be
     *              ``> 0`` and ``<= kArenaChunkSize``.
     * @return Flat starting index of the reserved block.
     */
    std::size_t reserve_cell_block(std::size_t count);

    /**
     * @brief Lock-free append of one cell. Returns the new flat index.
     *
     * Thin wrapper around @ref reserve_cell_block with ``count == 1``.
     */
    std::size_t append_cell(OctreeCell cell);

    /**
     * @brief Reserve a contiguous slice of contributor entries.
     *
     * @return Flat starting index of the reserved slice. Callers write into
     *         ``contributors()[base + i]`` for ``i`` in ``[0, count)``.
     */
    std::size_t reserve_contributor_slice(std::size_t count);

    /**
     * @brief Materialize arena contents back into the caller-visible vectors.
     *
     * Called at the end of a closure run to hand storage back to legacy
     * downstream pipelines that consume ``std::vector<OctreeCell>`` and
     * ``std::vector<std::size_t>``. Also fixes up the ``is_leaf`` boolean
     * on each cell to match the publication state encoded in
     * ``child_begin`` (``child_begin < 0`` means leaf).
     */
    void materialize_into(
        std::vector<OctreeCell> &out_cells,
        std::vector<std::size_t> &out_contributors);

    /**
     * @brief Materialize thickening side-cars back into flat vectors.
     *
     * Any null output pointer is ignored.
     */
    void materialize_thickening_state(
        std::vector<std::uint8_t> *out_cell_classification,
        std::vector<double> *out_center_values,
        std::vector<std::uint8_t> *out_occupancy_states,
        std::vector<std::array<std::size_t, 6>> *out_face_neighbors) const;

    void initialize_thickening_state(
        const std::vector<std::uint8_t> *initial_cell_classification,
        const std::vector<double> *initial_center_values,
        const std::vector<std::uint8_t> *initial_occupancy_states);

    // -----------------------------------------------------------------------
    // Incremental thickening side-cars
    //
    // These arenas are grown in lockstep with the cell arena via
    // reserve_cell_block.  They allow the kClassify / kDistanceUpdate task
    // kinds to record per-cell state without touching OctreeCell.
    //
    // cell_classification_:  0 = outside, 1 = inside, 255 = unset.
    // outside_distance_:     distance from the cell centre to the nearest
    //                        outside point; +inf while unset.
    // -----------------------------------------------------------------------

    /** Expose the per-cell inside/outside classification side-car. */
    ChunkedArena<std::atomic<std::uint8_t>> &cell_classification();

    /** Expose the per-cell outside-distance side-car. */
    ChunkedArena<std::atomic<std::uint32_t>> &outside_distance_bits();

    /** Expose the per-cell centre field-value side-car. */
    ChunkedArena<std::atomic<std::uint64_t>> &center_value_bits();

    /** Expose the per-cell occupancy-state side-car. */
    ChunkedArena<std::atomic<std::uint8_t>> &occupancy_state_bits();

    /** Expose the per-cell face-neighbor indices side-car. */
    ChunkedArena<std::array<std::size_t, 6>> &face_neighbor_indices();

private:
    /** External vectors retained only so that materialize_into can target them. */
    std::vector<OctreeCell> &out_cells_;
    std::vector<std::size_t> &out_contributors_;
    ChunkedArena<OctreeCell> cell_arena_;
    ChunkedArena<std::size_t> contrib_arena_;
    /**
     * @brief Side-car scheduler state, all backed by chunked arenas.
     *
     * Each arena is grown in lockstep with ``cell_arena_`` via
     * @ref reserve_cell_block. The element type for the atomic side-cars
     * is value-initialized on chunk allocation, which gives a
     * zero-initialized atomic (RefinementTaskState::kIdle == 0,
     * required_depth == 0, generation == 0). Reads and writes go through
     * the atomic interface and never take a mutex; the only lock in the
     * arena is the chunk-growth mutex, which fires once per ~128K appends.
     */
    ChunkedArena<std::atomic<std::uint32_t>> required_depth_;
    ChunkedArena<std::atomic<std::uint8_t>> task_state_;
    ChunkedArena<std::atomic<std::uint32_t>> generation_;
    ChunkedArena<RefinementChildBlock> child_blocks_;
    ChunkedArena<RefinementContributorRange> contributor_ranges_;
    // Incremental thickening side-cars (grown in lockstep with cell_arena_).
    ChunkedArena<std::atomic<std::uint8_t>> cell_classification_;
    // Outside distance stored as raw bits of a float (bit_cast equivalent).
    // 0x7F800000 == +inf when interpreted as IEEE 754 float.
    ChunkedArena<std::atomic<std::uint32_t>> outside_distance_bits_;
    // Cell-centre field value stored as raw bits of a double.
    ChunkedArena<std::atomic<std::uint64_t>> center_value_bits_;
    // Occupancy state aligned with adaptive_solid::OccupancyState values.
    ChunkedArena<std::atomic<std::uint8_t>> occupancy_state_bits_;
    // Face-neighbor cell indices aligned with OccupiedSolidClassificationCache.
    ChunkedArena<std::array<std::size_t, 6>> face_neighbor_indices_;

    /**
     * @brief Propagate required_depth upward from a cell that was just raised.
     *
     * Walks up the ancestor chain starting from the parent of `cell_index`,
     * and raises each ancestor's required_depth to max(current, new_required_depth).
     * Stops when an ancestor already has required_depth >= new_required_depth
     * or when the root is reached.
     */
    void propagate_required_depth_upward(
        std::size_t cell_index,
        std::uint32_t new_required_depth);
};

#endif  // MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_CONTEXT_HPP_
