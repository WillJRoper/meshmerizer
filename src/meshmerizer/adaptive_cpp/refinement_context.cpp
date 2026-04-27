#include "refinement_context.hpp"

#include <stdexcept>

#include "octree_cell.hpp"

RefinementContext::RefinementContext(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors)
    : all_cells_(all_cells),
      all_contributors_(all_contributors),
      cell_state_(all_cells.size()) {}

void RefinementContext::sync_cell_state_size() {
    if (cell_state_.size() < all_cells_.size()) {
        cell_state_.resize(all_cells_.size());
    }
}

RefinementCellState &RefinementContext::state(std::size_t cell_index) {
    if (cell_index >= cell_state_.size()) {
        throw std::out_of_range("refinement cell state index out of range");
    }
    return cell_state_[cell_index];
}

const RefinementCellState &RefinementContext::state(
    std::size_t cell_index) const {
    if (cell_index >= cell_state_.size()) {
        throw std::out_of_range("refinement cell state index out of range");
    }
    return cell_state_[cell_index];
}

std::size_t RefinementContext::size() const {
    return cell_state_.size();
}

void RefinementContext::reset_task_states() {
    for (RefinementCellState &cell_state : cell_state_) {
        cell_state.task_state = RefinementTaskState::kIdle;
    }
}

bool RefinementContext::raise_required_depth_to(
    std::size_t cell_index,
    std::uint32_t new_required_depth) {
    RefinementCellState &cell_state = state(cell_index);
    if (new_required_depth <= cell_state.required_depth) {
        return false;
    }
    cell_state.required_depth = new_required_depth;
    return true;
}

std::uint32_t RefinementContext::get_required_depth(
    std::size_t cell_index) const {
    return state(cell_index).required_depth;
}

bool RefinementContext::mark_queued(std::size_t cell_index) {
    RefinementCellState &cell_state = state(cell_index);
    if (cell_state.task_state == RefinementTaskState::kQueued ||
        cell_state.task_state == RefinementTaskState::kProcessing ||
        cell_state.task_state == RefinementTaskState::kRetired) {
        return false;
    }
    cell_state.task_state = RefinementTaskState::kQueued;
    ++cell_state.generation;
    return true;
}

bool RefinementContext::mark_processing(std::size_t cell_index) {
    RefinementCellState &cell_state = state(cell_index);
    if (cell_state.task_state != RefinementTaskState::kQueued) {
        return false;
    }
    cell_state.task_state = RefinementTaskState::kProcessing;
    return true;
}

void RefinementContext::mark_idle(std::size_t cell_index) {
    RefinementCellState &cell_state = state(cell_index);
    cell_state.task_state = RefinementTaskState::kIdle;
}

void RefinementContext::mark_retired(std::size_t cell_index) {
    RefinementCellState &cell_state = state(cell_index);
    cell_state.task_state = RefinementTaskState::kRetired;
}

std::vector<OctreeCell> &RefinementContext::cells() {
    return all_cells_;
}

std::vector<std::size_t> &RefinementContext::contributors() {
    return all_contributors_;
}
