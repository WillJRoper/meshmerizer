#include "refinement_context.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

#include "octree_cell.hpp"

namespace {

inline std::uint8_t task_state_value(RefinementTaskState state) {
    return static_cast<std::uint8_t>(state);
}

inline double bits_to_double(std::uint64_t bits) {
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace

RefinementContext::RefinementContext(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors)
    : out_cells_(all_cells),
      out_contributors_(all_contributors) {
    cell_arena_.adopt(all_cells);
    contrib_arena_.adopt(all_contributors);
    sync_cell_state_size();
    for (std::size_t i = 0; i < cell_arena_.size(); ++i) {
        contributor_ranges_[i] = {
            cell_arena_[i].contributor_begin,
            cell_arena_[i].contributor_end,
        };
    }
}

void RefinementContext::sync_cell_state_size() {
    // Mirror the cell arena's current size into every side-car arena so
    // that index ``i`` is valid for any ``i < cell_arena_.size()``. Each
    // side-car's ``reserve_to`` is idempotent and concurrency-safe; if the
    // arena is already large enough this is a single acquire-load.
    //
    // Newly allocated chunks are value-initialized, which gives
    // zero-initialized atomics (RefinementTaskState::kIdle == 0,
    // required_depth == 0, generation == 0) and default-initialized
    // RefinementChildBlock (child_count = 0).
    // cell_classification_ starts at 0 (== outside / unset default).
    // outside_distance_bits_ starts at 0 (== 0.0f, overwritten on first use).
    const std::size_t target = cell_arena_.size();
    required_depth_.reserve_to(target);
    task_state_.reserve_to(target);
    generation_.reserve_to(target);
    child_blocks_.reserve_to(target);
    contributor_ranges_.reserve_to(target);
    cell_classification_.reserve_to(target);
    outside_distance_bits_.reserve_to(target);
    center_value_bits_.reserve_to(target);
    occupancy_state_bits_.reserve_to(target);
}

std::size_t RefinementContext::size() const {
    // The cell arena is the canonical source of truth; side-cars track it
    // in lockstep via reserve_cell_block.
    return cell_arena_.size();
}

std::size_t RefinementContext::parent_index(std::size_t cell_index) const {
    if (cell_index >= cell_arena_.size()) {
        throw std::out_of_range("refinement cell index out of range");
    }
    const OctreeCell &cell = cell_arena_[cell_index];
    // Field-based parent lookup: O(1), no map probe. parent_index is set
    // during split publication in the closure pipeline; root cells leave
    // it at -1.
    if (cell.parent_index < 0) {
        return std::numeric_limits<std::size_t>::max();
    }
    return static_cast<std::size_t>(cell.parent_index);
}

void RefinementContext::reset_task_states() {
    // Walk every tracked task state slot and set it to kIdle. This is
    // serial by design; the closure driver only calls reset_task_states
    // between phases, never under contention.
    const std::size_t total = task_state_.size();
    for (std::size_t i = 0; i < total; ++i) {
        task_state_[i].store(task_state_value(RefinementTaskState::kIdle),
                             std::memory_order_release);
    }
}

void RefinementContext::set_child_block(
    std::size_t cell_index,
    const std::vector<std::size_t> &child_indices) {
    if (cell_index >= cell_arena_.size()) {
        throw std::out_of_range("refinement child block index out of range");
    }
    RefinementChildBlock &child_block = child_blocks_[cell_index];
    if (child_indices.size() > child_block.child_indices.size()) {
        throw std::out_of_range("refinement child block exceeded fixed capacity");
    }
    child_block.child_count = static_cast<std::uint8_t>(child_indices.size());
    for (std::size_t index = 0; index < child_indices.size(); ++index) {
        child_block.child_indices[index] = child_indices[index];
    }
}

void RefinementContext::append_child_indices(
    std::size_t cell_index,
    std::vector<std::size_t> &out_indices) const {
    if (cell_index >= cell_arena_.size()) {
        throw std::out_of_range("refinement child block index out of range");
    }
    const RefinementChildBlock &child_block = child_blocks_[cell_index];
    for (std::size_t index = 0; index < child_block.child_count; ++index) {
        out_indices.push_back(child_block.child_indices[index]);
    }
}

std::size_t RefinementContext::child_index_at(
    std::size_t cell_index,
    std::uint8_t child_slot) const {
    if (cell_index >= cell_arena_.size()) {
        throw std::out_of_range("refinement child block index out of range");
    }
    const RefinementChildBlock &child_block = child_blocks_[cell_index];
    if (child_slot >= child_block.child_count) {
        return std::numeric_limits<std::size_t>::max();
    }
    return child_block.child_indices[child_slot];
}

void RefinementContext::set_contributor_range(
    std::size_t cell_index,
    std::int64_t begin,
    std::int64_t end) {
    if (cell_index >= cell_arena_.size()) {
        throw std::out_of_range("refinement contributor range index out of range");
    }
    contributor_ranges_[cell_index] = {begin, end};
}

RefinementContributorRange RefinementContext::contributor_range(
    std::size_t cell_index) const {
    if (cell_index >= cell_arena_.size()) {
        throw std::out_of_range("refinement contributor range index out of range");
    }
    return contributor_ranges_[cell_index];
}

void RefinementContext::copy_contributors_for_cell(
    std::size_t cell_index,
    std::vector<std::size_t> &out_indices) const {
    out_indices.clear();
    const RefinementContributorRange range = contributor_range(cell_index);
    if (range.begin < 0 || range.end <= range.begin) {
        return;
    }
    const std::size_t safe_begin =
        static_cast<std::size_t>(std::max<std::int64_t>(0, range.begin));
    const std::size_t safe_end = static_cast<std::size_t>(std::min(
        range.end,
        static_cast<std::int64_t>(contrib_arena_.size())));
    out_indices.reserve(safe_end - safe_begin);
    for (std::size_t i = safe_begin; i < safe_end; ++i) {
        out_indices.push_back(contrib_arena_[i]);
    }
}

bool RefinementContext::raise_required_depth_to(
    std::size_t cell_index,
    std::uint32_t new_required_depth) {
    if (cell_index >= cell_arena_.size()) {
        return false;
    }
    std::atomic<std::uint32_t> &required_depth = required_depth_[cell_index];
    std::uint32_t observed = required_depth.load(std::memory_order_acquire);
    while (observed < new_required_depth) {
        if (required_depth.compare_exchange_weak(
                observed,
                new_required_depth,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            // Successfully raised this cell. Propagate upward so that
            // parent required_depth matches the max of its descendants.
            propagate_required_depth_upward(cell_index, new_required_depth);
            return true;
        }
    }
    return false;
}

std::uint32_t RefinementContext::get_required_depth(std::size_t cell_index) const {
    if (cell_index >= cell_arena_.size()) {
        throw std::out_of_range("refinement required depth index out of range");
    }
    return required_depth_[cell_index].load(std::memory_order_acquire);
}

bool RefinementContext::mark_queued(std::size_t cell_index) {
    if (cell_index >= cell_arena_.size()) {
        throw std::out_of_range("refinement task state index out of range");
    }

    std::atomic<std::uint8_t> &task_state = task_state_[cell_index];
    std::uint8_t observed = task_state.load(std::memory_order_acquire);
    while (observed != task_state_value(RefinementTaskState::kQueued) &&
           observed != task_state_value(RefinementTaskState::kProcessing)) {
        if (task_state.compare_exchange_weak(
                observed,
                task_state_value(RefinementTaskState::kQueued),
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            generation_[cell_index].fetch_add(1U, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

bool RefinementContext::mark_processing(std::size_t cell_index) {
    if (cell_index >= cell_arena_.size()) {
        throw std::out_of_range("refinement task state index out of range");
    }

    std::uint8_t expected = task_state_value(RefinementTaskState::kQueued);
    return task_state_[cell_index].compare_exchange_strong(
        expected,
        task_state_value(RefinementTaskState::kProcessing),
        std::memory_order_acq_rel,
        std::memory_order_acquire);
}

void RefinementContext::mark_idle(std::size_t cell_index) {
    if (cell_index >= cell_arena_.size()) {
        throw std::out_of_range("refinement task state index out of range");
    }
    task_state_[cell_index].store(
        task_state_value(RefinementTaskState::kIdle),
        std::memory_order_release);
}

void RefinementContext::mark_retired(std::size_t cell_index) {
    if (cell_index >= cell_arena_.size()) {
        throw std::out_of_range("refinement task state index out of range");
    }
    task_state_[cell_index].store(
        task_state_value(RefinementTaskState::kRetired),
        std::memory_order_release);
}

ChunkedArena<OctreeCell> &RefinementContext::cells() {
    return cell_arena_;
}

const ChunkedArena<OctreeCell> &RefinementContext::cells() const {
    return cell_arena_;
}

ChunkedArena<std::size_t> &RefinementContext::contributors() {
    return contrib_arena_;
}

const ChunkedArena<std::size_t> &RefinementContext::contributors() const {
    return contrib_arena_;
}

std::size_t RefinementContext::reserve_cell_block(std::size_t count) {
    // Reserve in the cell arena first; this performs any chunk-alignment
    // rounding so that the returned block fits inside one chunk and the
    // eight children of a split are physically contiguous.
    const std::size_t begin = cell_arena_.reserve_block(count);
    const std::size_t target = begin + count;
    // Advance every side-car arena to ``target``. ``reserve_to`` is
    // idempotent across racing callers, so concurrent reservations from
    // multiple workers each contribute their own (begin, count) pair and
    // the arenas converge to the largest target. Fresh slots are
    // value-initialized to zero atomics / default-constructed child
    // blocks, which is the correct initial scheduler state.
    required_depth_.reserve_to(target);
    task_state_.reserve_to(target);
    generation_.reserve_to(target);
    child_blocks_.reserve_to(target);
    contributor_ranges_.reserve_to(target);
    cell_classification_.reserve_to(target);
    outside_distance_bits_.reserve_to(target);
    center_value_bits_.reserve_to(target);
    occupancy_state_bits_.reserve_to(target);
    // Explicitly initialize the scheduler side-cars for the newly reserved
    // slots. Do not rely on default construction semantics of
    // ``std::atomic<T>`` arrays here: newly appended children must always
    // start from a known kIdle / zero-depth / zero-generation state before
    // any worker can attempt to queue them.
    for (std::size_t i = begin; i < target; ++i) {
        required_depth_[i].store(0U, std::memory_order_relaxed);
        task_state_[i].store(
            task_state_value(RefinementTaskState::kIdle),
            std::memory_order_relaxed);
        generation_[i].store(0U, std::memory_order_relaxed);
        child_blocks_[i] = RefinementChildBlock{};
        contributor_ranges_[i] = RefinementContributorRange{};
        cell_classification_[i].store(0U, std::memory_order_relaxed);
        outside_distance_bits_[i].store(0U, std::memory_order_relaxed);
        center_value_bits_[i].store(0U, std::memory_order_relaxed);
        occupancy_state_bits_[i].store(0U, std::memory_order_relaxed);
    }
    return begin;
}

std::size_t RefinementContext::append_cell(OctreeCell cell) {
    const std::size_t new_index = reserve_cell_block(1U);
    cell_arena_[new_index] = std::move(cell);
    return new_index;
}

std::size_t RefinementContext::reserve_contributor_slice(std::size_t count) {
    if (count == 0U) {
        return contrib_arena_.size();
    }
    return contrib_arena_.reserve_block(count);
}

ChunkedArena<std::atomic<std::uint8_t>> &RefinementContext::cell_classification() {
    return cell_classification_;
}

ChunkedArena<std::atomic<std::uint32_t>> &RefinementContext::outside_distance_bits() {
    return outside_distance_bits_;
}

ChunkedArena<std::atomic<std::uint64_t>> &RefinementContext::center_value_bits() {
    return center_value_bits_;
}

ChunkedArena<std::atomic<std::uint8_t>> &RefinementContext::occupancy_state_bits() {
    return occupancy_state_bits_;
}

void RefinementContext::materialize_into(
    std::vector<OctreeCell> &out_cells,
    std::vector<std::size_t> &out_contributors) {
    const std::size_t cell_count = cell_arena_.size();
    out_cells.clear();
    out_cells.reserve(cell_count);
    for (std::size_t i = 0; i < cell_count; ++i) {
        OctreeCell cell = cell_arena_[i];
        // Reconcile the legacy ``is_leaf`` boolean with the publication state
        // encoded by ``child_begin``. During a parallel closure run workers
        // never mutate ``is_leaf`` on a parent (that would race with neighbor
        // walks); instead the publication is the release-store on
        // ``child_begin``. Downstream pipelines consume ``is_leaf`` as a
        // plain bool, so we set it here once at run end where there is no
        // concurrency.
        cell.is_leaf = (cell.child_begin < 0);
        const RefinementContributorRange range = contributor_ranges_[i];
        cell.contributor_begin = range.begin;
        cell.contributor_end = range.end;
        out_cells.push_back(std::move(cell));
    }
    const std::size_t contrib_count = contrib_arena_.size();
    out_contributors.clear();
    out_contributors.reserve(contrib_count);
    for (std::size_t i = 0; i < contrib_count; ++i) {
        out_contributors.push_back(contrib_arena_[i]);
    }
}

void RefinementContext::materialize_thickening_state(
    std::vector<std::uint8_t> *out_cell_classification,
    std::vector<double> *out_center_values,
    std::vector<std::uint8_t> *out_occupancy_states) const {
    const std::size_t cell_count = cell_arena_.size();
    if (out_cell_classification != nullptr) {
        out_cell_classification->assign(cell_count, 0U);
        for (std::size_t i = 0; i < cell_count; ++i) {
            (*out_cell_classification)[i] =
                cell_classification_[i].load(std::memory_order_acquire);
        }
    }
    if (out_center_values != nullptr) {
        out_center_values->assign(cell_count, 0.0);
        for (std::size_t i = 0; i < cell_count; ++i) {
            (*out_center_values)[i] = bits_to_double(
                center_value_bits_[i].load(std::memory_order_acquire));
        }
    }
    if (out_occupancy_states != nullptr) {
        out_occupancy_states->assign(cell_count, 0U);
        for (std::size_t i = 0; i < cell_count; ++i) {
            (*out_occupancy_states)[i] =
                occupancy_state_bits_[i].load(std::memory_order_acquire);
        }
    }
}

void RefinementContext::initialize_thickening_state(
    const std::vector<std::uint8_t> *initial_cell_classification,
    const std::vector<double> *initial_center_values,
    const std::vector<std::uint8_t> *initial_occupancy_states) {
    const std::size_t cell_count = cell_arena_.size();
    for (std::size_t i = 0; i < cell_count; ++i) {
        if (initial_cell_classification != nullptr &&
            i < initial_cell_classification->size()) {
            cell_classification_[i].store(
                (*initial_cell_classification)[i], std::memory_order_relaxed);
        }
        if (initial_center_values != nullptr && i < initial_center_values->size()) {
            std::uint64_t bits = 0U;
            const double value = (*initial_center_values)[i];
            std::memcpy(&bits, &value, sizeof(bits));
            center_value_bits_[i].store(bits, std::memory_order_relaxed);
        }
        if (initial_occupancy_states != nullptr &&
            i < initial_occupancy_states->size()) {
            occupancy_state_bits_[i].store(
                (*initial_occupancy_states)[i], std::memory_order_relaxed);
        }
    }
}

void RefinementContext::propagate_required_depth_upward(
    std::size_t cell_index,
    std::uint32_t new_required_depth) {
    // Walk up the ancestor chain starting from the parent of cell_index.
    if (cell_index >= cell_arena_.size()) {
        return;
    }
    std::size_t current = cell_index;
    while (true) {
        const OctreeCell &cell = cell_arena_[current];
        std::int64_t parent_idx = cell.parent_index;
        if (parent_idx < 0) {
            // Reached the root.
            break;
        }
        std::size_t parent_index = static_cast<std::size_t>(parent_idx);

        // Try to raise the parent's required_depth.
        std::atomic<std::uint32_t> &parent_required =
            required_depth_[parent_index];
        std::uint32_t parent_observed =
            parent_required.load(std::memory_order_acquire);
        bool raised = false;
        while (parent_observed < new_required_depth) {
            if (parent_required.compare_exchange_weak(
                    parent_observed,
                    new_required_depth,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                raised = true;
                break;
            }
        }

        if (!raised) {
            // Parent already has required_depth >= new_required_depth.
            break;
        }

        // Continue walking up from the parent.
        current = parent_index;
    }
}
