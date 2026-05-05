#include "refinement_closure.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <span>
#include <thread>

#include "morton.hpp"
#include "octree_cell.hpp"
#include "progress_bar.hpp"
#include "refinement_context.hpp"
#include "refinement_work_queue.hpp"

namespace {

// Atomic publication helper. Writers store ``child_begin`` with release order
// after children are fully constructed; readers in neighbor-walk paths load
// it with acquire order to decide whether a cell is currently a leaf. The
// legacy ``is_leaf`` boolean is intentionally NOT flipped during a split
// because doing so would race with concurrent neighbor walks; it is fixed up
// post-run in ``RefinementContext::materialize_into``.
inline bool closure_is_leaf(const OctreeCell &cell) {
    // ``atomic_ref<const T>`` is not load-capable on this libc++; cast the
    // const away on the underlying storage. ``atomic_ref`` itself is the
    // synchronization mechanism, so this is sound: we are not mutating the
    // cell, only acquire-loading the publication slot.
    const std::int64_t observed =
        std::atomic_ref<std::int64_t>(
            const_cast<std::int64_t &>(cell.child_begin))
            .load(std::memory_order_acquire);
    return observed < 0;
}

inline std::int64_t closure_child_begin_acquire(const OctreeCell &cell) {
    return std::atomic_ref<std::int64_t>(
               const_cast<std::int64_t &>(cell.child_begin))
        .load(std::memory_order_acquire);
}

inline void closure_publish_child_begin(
    OctreeCell &cell, std::int64_t value) {
    std::atomic_ref<std::int64_t>(cell.child_begin)
        .store(value, std::memory_order_release);
}

struct RefinementResult {
    bool has_surface = false;
    bool should_split = false;
    std::array<double, 8> corner_values = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::uint8_t corner_sign_mask = 0U;
    std::vector<OctreeCell> children;
    std::vector<std::vector<std::size_t>> child_contributors;
};

struct PublishedChildren {
    std::vector<std::size_t> child_indices;
    std::vector<std::size_t> affected_indices;
};

struct ContributorReservation {
    std::int64_t begin = 0;
    std::size_t count = 0U;
};

struct ChildSlotReservation {
    std::size_t begin = 0U;
    std::size_t count = 0U;
};

struct PreparedChildPublication {
    OctreeCell cell;
    std::uint32_t required_depth = 0U;
    std::vector<std::size_t> contributors;
};

class ClosureStorage {
public:
    explicit ClosureStorage(RefinementContext &context) : context_(context) {}

    ChildSlotReservation reserve_children(std::size_t child_count) {
        if (child_count == 0U) {
            return {context_.cells().size(), 0U};
        }
        // One CAS reserves a contiguous eight-child block in the arena;
        // chunk-boundary alignment in ``reserve_block`` guarantees that all
        // eight children live in the same chunk so child_begin + offset is
        // Reserve a contiguous arena slice for the eight (or fewer)
        // children. ``reserve_cell_block`` advances the cell arena and all
        // four side-car arenas in lockstep, so every child index in
        // ``[begin, begin + child_count)`` has its scheduler state
        // initialized (kIdle, depth=0, generation=0) before this call
        // returns. The caller writes the OctreeCell payloads into the
        // reserved slots and publishes the parent's ``child_begin`` to
        // give downstream walks a stable contiguous index.
        const std::size_t begin =
            context_.reserve_cell_block(child_count);
        return {begin, child_count};
    }

    std::size_t publish_cell(
        const ChildSlotReservation &reservation,
        std::size_t offset,
        OctreeCell cell) {
        if (offset >= reservation.count) {
            throw std::runtime_error("child publication offset out of range");
        }
        const std::size_t new_index = reservation.begin + offset;
        // Direct write into the pre-reserved slot. No push_back, no growth
        // races; the arena slot is uninitialized memory that this writer
        // exclusively owns until it publishes the parent's child_begin.
        // Side-car scheduler state for ``new_index`` was initialized to
        // defaults (kIdle, depth=0, generation=0) by reserve_cell_block.
        context_.cells()[new_index] = std::move(cell);
        return new_index;
    }

    std::vector<std::size_t> publish_child_batch(
        const ChildSlotReservation &reservation,
        std::vector<PreparedChildPublication> prepared_children) {
        if (prepared_children.size() > reservation.count) {
            throw std::runtime_error(
                "child batch publication exceeded reserved count");
        }

        std::vector<std::size_t> published_indices;
        published_indices.reserve(prepared_children.size());

        std::size_t contributor_count = 0U;
        for (const PreparedChildPublication &prepared : prepared_children) {
            contributor_count += prepared.contributors.size();
        }

        // One arena reservation for the whole child batch's contributor
        // slice. Lock-free; chunk-aligned so the slice is contiguous.
        // We still publish into one shared flat contributor arena because
        // downstream code consumes contributor_begin/contributor_end as flat
        // offsets; replacing that with cell-owned storage is a later step.
        const std::size_t contributor_begin =
            context_.reserve_contributor_slice(contributor_count);
        std::size_t next_contributor_offset = contributor_begin;

        for (std::size_t offset = 0; offset < prepared_children.size(); ++offset) {
            PreparedChildPublication &prepared = prepared_children[offset];
            prepared.cell.contributor_begin =
                static_cast<std::int64_t>(next_contributor_offset);
            for (std::size_t contributor_index : prepared.contributors) {
                context_.contributors()[next_contributor_offset++] =
                    contributor_index;
            }
            prepared.cell.contributor_end =
                static_cast<std::int64_t>(next_contributor_offset);
            const std::int64_t published_contributor_begin =
                prepared.cell.contributor_begin;
            const std::int64_t published_contributor_end =
                prepared.cell.contributor_end;
            const std::size_t new_index = publish_cell(
                reservation,
                offset,
                std::move(prepared.cell));
            context_.set_contributor_range(
                new_index,
                published_contributor_begin,
                published_contributor_end);
            context_.raise_required_depth_to(new_index, prepared.required_depth);
            published_indices.push_back(new_index);
        }

        return published_indices;
    }

    RefinementContext &context() { return context_; }

private:
    struct EmittedSnapshot {
        std::size_t queue_size = 0U;
        std::size_t push_count = 0U;
        std::size_t pop_count = 0U;
        std::size_t stale_task_count = 0U;
        std::size_t processed_leaf_count = 0U;
        std::size_t surface_leaf_count = 0U;
        std::size_t split_count = 0U;
        std::size_t total_cells = 0U;
        std::size_t required_depth_raise_count = 0U;
        std::size_t internal_wake_count = 0U;
        std::size_t peak_queue = 0U;

        bool operator==(const EmittedSnapshot &other) const {
            return queue_size == other.queue_size &&
                   push_count == other.push_count &&
                   pop_count == other.pop_count &&
                   stale_task_count == other.stale_task_count &&
                   processed_leaf_count == other.processed_leaf_count &&
                   surface_leaf_count == other.surface_leaf_count &&
                   split_count == other.split_count &&
                   total_cells == other.total_cells &&
                   required_depth_raise_count ==
                       other.required_depth_raise_count &&
                   internal_wake_count == other.internal_wake_count &&
                   peak_queue == other.peak_queue;
        }

        bool has_any_work() const {
            return push_count > 0U || pop_count > 0U ||
                   processed_leaf_count > 0U || split_count > 0U ||
                   required_depth_raise_count > 0U ||
                   internal_wake_count > 0U || stale_task_count > 0U;
        }
    };

    RefinementContext &context_;
};

// (ClosureStorage EmittedSnapshot above is the storage-private one;
//  ClosureStatusReporter has its own below.)

class ClosurePublisher {
public:
    ClosurePublisher(
        ClosureStorage &storage,
        const std::vector<Vector3d> &positions,
        const std::vector<double> &smoothing_lengths,
        double isovalue)
        : storage_(storage),
          positions_(positions),
          smoothing_lengths_(smoothing_lengths),
          isovalue_(isovalue) {}

    PublishedChildren publish_children(
        std::size_t parent_index,
        std::uint32_t parent_required_depth,
        const std::vector<OctreeCell> &source_children,
        const std::vector<std::vector<std::size_t>> &source_child_contributors,
        bool compute_balance_samples,
        std::uint32_t preferred_worker) {
        PublishedChildren published;
        published.affected_indices.reserve(1U + source_children.size());
        published.affected_indices.push_back(parent_index);
        published.child_indices.reserve(source_children.size());
        const ChildSlotReservation child_slots =
            storage_.reserve_children(source_children.size());
        std::vector<PreparedChildPublication> prepared_children;
        prepared_children.reserve(source_children.size());

        for (std::size_t child_index = 0;
             child_index < source_children.size();
             ++child_index) {
            OctreeCell child = source_children[child_index];
            if (compute_balance_samples &&
                !source_child_contributors[child_index].empty()) {
                child.is_leaf = true;
                child.is_active = false;
                child.has_surface = false;
                child.is_topo_surface = false;
                child.corner_values = sample_cell_corners(
                    child,
                    std::span<const std::size_t>(
                        source_child_contributors[child_index]),
                    positions_,
                    smoothing_lengths_);
                child.corner_sign_mask = compute_corner_sign_mask(
                    child.corner_values, isovalue_);
                child.has_surface = cell_may_contain_isosurface(
                    child.corner_values, isovalue_);
            }

            PreparedChildPublication prepared_child;
            prepared_child.cell = std::move(child);
            prepared_child.required_depth = std::max(
                source_children[child_index].depth,
                parent_required_depth);
            prepared_child.contributors = source_child_contributors[child_index];
            prepared_children.push_back(std::move(prepared_child));
        }

        const std::vector<std::size_t> published_indices =
            storage_.publish_child_batch(
                child_slots,
                std::move(prepared_children));

        for (std::size_t new_index : published_indices) {
            published.child_indices.push_back(new_index);
            published.affected_indices.push_back(new_index);
        }

        storage_.context().set_child_block(parent_index, published.child_indices);

        return published;
    }

private:
    ClosureStorage &storage_;
    const std::vector<Vector3d> &positions_;
    const std::vector<double> &smoothing_lengths_;
    double isovalue_;
};

struct ClosureRunStats {
    std::atomic<std::size_t> stale_task_count{0U};
    std::atomic<std::size_t> processed_leaf_count{0U};
    std::atomic<std::size_t> surface_leaf_count{0U};
    std::atomic<std::size_t> split_count{0U};
    std::atomic<std::size_t> required_depth_raise_count{0U};
    std::atomic<std::size_t> internal_wake_count{0U};
    // Per-kind pop counters.  Incremented in the worker dispatch switch so
    // the status table can show classify / dist_upd / refine breakdown.
    std::atomic<std::size_t> classify_pop_count{0U};
    std::atomic<std::size_t> distance_update_pop_count{0U};
    std::atomic<std::size_t> refine_pop_count{0U};
};

// Plain-old-data snapshot of ClosureRunStats, suitable for copying out
// for status reporting / final summary.
struct ClosureRunStatsSnapshot {
    std::size_t stale_task_count = 0U;
    std::size_t processed_leaf_count = 0U;
    std::size_t surface_leaf_count = 0U;
    std::size_t split_count = 0U;
    std::size_t required_depth_raise_count = 0U;
    std::size_t internal_wake_count = 0U;
    // Per-kind breakdown.
    std::size_t classify_pop_count = 0U;
    std::size_t distance_update_pop_count = 0U;
    std::size_t refine_pop_count = 0U;
};

inline ClosureRunStatsSnapshot snapshot_run_stats(
    const ClosureRunStats &stats) {
    ClosureRunStatsSnapshot out;
    out.stale_task_count = stats.stale_task_count.load(
        std::memory_order_relaxed);
    out.processed_leaf_count = stats.processed_leaf_count.load(
        std::memory_order_relaxed);
    out.surface_leaf_count = stats.surface_leaf_count.load(
        std::memory_order_relaxed);
    out.split_count = stats.split_count.load(std::memory_order_relaxed);
    out.required_depth_raise_count = stats.required_depth_raise_count.load(
        std::memory_order_relaxed);
    out.internal_wake_count = stats.internal_wake_count.load(
        std::memory_order_relaxed);
    out.classify_pop_count = stats.classify_pop_count.load(
        std::memory_order_relaxed);
    out.distance_update_pop_count = stats.distance_update_pop_count.load(
        std::memory_order_relaxed);
    out.refine_pop_count = stats.refine_pop_count.load(
        std::memory_order_relaxed);
    return out;
}

// =====================================================================
// Lightweight closure profiler.
//
// All members are atomic so workers can update them without taking
// any extra mutex. Time accumulators store nanoseconds. Counts are
// raw event counts. A single summary is emitted at the end of a run.
// =====================================================================
struct ClosureProfiler {
    // Time accumulators (nanoseconds).
    std::atomic<std::uint64_t> ns_pop_wait{0U};
    std::atomic<std::uint64_t> ns_leaf_eval{0U};
    std::atomic<std::uint64_t> ns_apply_split{0U};
    std::atomic<std::uint64_t> ns_storage_lock_wait{0U};
    std::atomic<std::uint64_t> ns_storage_lock_held{0U};
    std::atomic<std::uint64_t> ns_neighbor_wake{0U};
    std::atomic<std::uint64_t> ns_required_depth_raise{0U};
    std::atomic<std::uint64_t> ns_internal_descent{0U};
    std::atomic<std::uint64_t> ns_local_dfs_total{0U};

    // Event counts.
    std::atomic<std::uint64_t> tasks_popped{0U};
    std::atomic<std::uint64_t> tasks_retired_no_split{0U};
    std::atomic<std::uint64_t> tasks_stale_or_skipped{0U};
    std::atomic<std::uint64_t> leaves_evaluated{0U};
    std::atomic<std::uint64_t> splits{0U};
    std::atomic<std::uint64_t> internal_wakes{0U};
    std::atomic<std::uint64_t> neighbor_wakes_pushed{0U};
    std::atomic<std::uint64_t> required_depth_raises{0U};
    std::atomic<std::uint64_t> local_stack_pushes{0U};
    std::atomic<std::uint64_t> spill_pushes{0U};
    std::atomic<std::uint64_t> steal_attempts{0U};
    std::atomic<std::uint64_t> steal_successes{0U};
    std::atomic<std::uint64_t> pop_idle_sleeps{0U};
    std::atomic<std::uint64_t> internal_cells_visited{0U};
    std::atomic<std::uint64_t> non_leaf_retired_immediately{0U};
};

class ScopedNsTimer {
public:
    explicit ScopedNsTimer(std::atomic<std::uint64_t> &accumulator)
        : accumulator_(accumulator),
          start_(std::chrono::steady_clock::now()) {}
    ~ScopedNsTimer() {
        const auto end = std::chrono::steady_clock::now();
        accumulator_.fetch_add(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - start_).count()),
            std::memory_order_relaxed);
    }
private:
    std::atomic<std::uint64_t> &accumulator_;
    std::chrono::steady_clock::time_point start_;
};

struct ClosureWorkerState {
    RefinementContext &context;
    const std::vector<Vector3d> &positions;
    const std::vector<double> &smoothing_lengths;
    const RefinementClosureConfig &config;
    RefinementWorkQueue &queue;
    const std::vector<std::size_t> *startup_seed_cells;
    std::atomic<std::size_t> *startup_seed_cursor;
    std::uint32_t worker_id;
    ClosureRunStats &run_stats;
    ClosureProfiler &profiler;
};

constexpr int kTableElapsedWidth = 9;
constexpr int kTableCountWidth = 12;

inline void poll_closure_cancellation(
    const ClosureWorkerState &worker,
    std::size_t counter,
    std::size_t interval) {
    // Workers must NEVER touch the Python GIL. Cancellation is delivered via
    // a pure C++ atomic flag set by the Python boundary (e.g. when the
    // calling thread observes a pending KeyboardInterrupt). Each worker
    // simply polls that atomic; no syscalls, no signal handling, no GIL.
    (void)worker;
    (void)counter;
    (void)interval;
    if (meshmerizer_cancel_detail::is_cancel_requested()) {
        throw meshmerizer_cancel_detail::OperationCancelled();
    }
}

class ClosureStatusReporter {
public:
    ClosureStatusReporter(
        const RefinementClosureConfig &config,
        const RefinementContext &context,
        const RefinementWorkQueue &queue,
        const ClosureRunStats &run_stats)
        : config_(config),
          context_(context),
          queue_(queue),
          run_stats_(run_stats),
          start_time_(std::chrono::steady_clock::now()),
          last_progress_bucket_(0U),
          header_printed_(false) {}

private:
    struct EmittedSnapshot {
        std::size_t queue_size = 0U;
        std::size_t push_count = 0U;
        std::size_t pop_count = 0U;
        std::size_t stale_task_count = 0U;
        std::size_t processed_leaf_count = 0U;
        std::size_t surface_leaf_count = 0U;
        std::size_t split_count = 0U;
        std::size_t total_cells = 0U;
        std::size_t required_depth_raise_count = 0U;
        std::size_t internal_wake_count = 0U;
        std::size_t peak_queue = 0U;
        // Per-task-kind pop counters for table breakdown.
        std::size_t classify_pop_count = 0U;
        std::size_t distance_update_pop_count = 0U;
        std::size_t refine_pop_count = 0U;

        bool operator==(const EmittedSnapshot &other) const {
            return queue_size == other.queue_size &&
                   push_count == other.push_count &&
                   pop_count == other.pop_count &&
                   stale_task_count == other.stale_task_count &&
                   processed_leaf_count == other.processed_leaf_count &&
                   surface_leaf_count == other.surface_leaf_count &&
                   split_count == other.split_count &&
                   total_cells == other.total_cells &&
                   required_depth_raise_count ==
                       other.required_depth_raise_count &&
                   internal_wake_count == other.internal_wake_count &&
                   peak_queue == other.peak_queue &&
                   classify_pop_count == other.classify_pop_count &&
                   distance_update_pop_count ==
                       other.distance_update_pop_count &&
                   refine_pop_count == other.refine_pop_count;
        }

        bool has_any_work() const {
            return push_count > 0U || pop_count > 0U ||
                   processed_leaf_count > 0U || split_count > 0U ||
                   required_depth_raise_count > 0U ||
                   internal_wake_count > 0U || stale_task_count > 0U;
        }
    };

public:

    void maybe_emit_periodic() {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(emit_mutex_);
        const EmittedSnapshot snapshot = build_snapshot();
        if (!snapshot.has_any_work()) {
            return;
        }
        if (has_last_snapshot_ && snapshot == last_snapshot_) {
            return;
        }
        last_snapshot_ = emit_row(now, snapshot);
        has_last_snapshot_ = true;
        last_progress_bucket_ = progress_bucket_for(snapshot);
    }

    void finish() {
        if (!queue_.try_claim_final_report()) {
            return;
        }
        std::lock_guard<std::mutex> lock(emit_mutex_);
        const auto now = std::chrono::steady_clock::now();
        const EmittedSnapshot snapshot = build_snapshot();
        if (!snapshot.has_any_work()) {
            return;
        }
        if (has_last_snapshot_ && snapshot == last_snapshot_) {
            return;
        }
        last_snapshot_ = emit_row(now, snapshot);
        has_last_snapshot_ = true;
        last_progress_bucket_ = std::max(last_progress_bucket_, progress_bucket_for(snapshot));
    }

    void emit_header() {
        meshmerizer_log_detail::print_indented_status(
            "scheduler status table follows (progress=1%%, task graph=%s)\n",
            config_.phase_name.c_str());
        meshmerizer_log_detail::print_indented_status(
            "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s"
            " %*s %*s %*s %*s\n",
            kTableElapsedWidth,
            "elapsed_s",
            kTableCountWidth,
            "queue",
            kTableCountWidth,
            "inflight",
            kTableCountWidth,
            "pushed",
            kTableCountWidth,
            "popped",
            kTableCountWidth,
            "stale",
            kTableCountWidth,
            "refine",
            kTableCountWidth,
            "classify",
            kTableCountWidth,
            "dist_upd",
            kTableCountWidth,
            "split",
            kTableCountWidth,
            "balance",
            kTableCountWidth,
            "processed",
            kTableCountWidth,
            "surface",
            kTableCountWidth,
            "cells");
    }

    EmittedSnapshot build_snapshot() const {
        const RefinementWorkQueueStats queue_stats = queue_.stats();
        const ClosureRunStatsSnapshot run_stats_snapshot =
            snapshot_run_stats(run_stats_);

        return {
            queue_stats.queue_size,
            queue_stats.push_count,
            queue_stats.pop_count,
            run_stats_snapshot.stale_task_count,
            run_stats_snapshot.processed_leaf_count,
            run_stats_snapshot.surface_leaf_count,
            run_stats_snapshot.split_count,
            context_.cells().size(),
            run_stats_snapshot.required_depth_raise_count,
            run_stats_snapshot.internal_wake_count,
            queue_stats.high_watermark,
            run_stats_snapshot.classify_pop_count,
            run_stats_snapshot.distance_update_pop_count,
            run_stats_snapshot.refine_pop_count,
        };
    }

    EmittedSnapshot emit_row(const std::chrono::steady_clock::time_point &now) {
        return emit_row(now, build_snapshot());
    }

    EmittedSnapshot emit_row(
        const std::chrono::steady_clock::time_point &now,
        const EmittedSnapshot &snapshot) {

        if (!header_printed_) {
            emit_header();
            header_printed_ = true;
        }

        const double elapsed_seconds =
            std::chrono::duration<double>(now - start_time_).count();
        meshmerizer_log_detail::print_indented_status(
            "%*.1f %*zu %*zu %*zu %*zu %*zu %*zu %*zu %*zu %*zu"
            " %*zu %*zu %*zu %*zu\n",
            kTableElapsedWidth,
            elapsed_seconds,
            kTableCountWidth,
            snapshot.queue_size,
            kTableCountWidth,
            queue_.stats().in_flight_count,
            kTableCountWidth,
            snapshot.push_count,
            kTableCountWidth,
            snapshot.pop_count,
            kTableCountWidth,
            snapshot.stale_task_count,
            kTableCountWidth,
            snapshot.refine_pop_count,
            kTableCountWidth,
            snapshot.classify_pop_count,
            kTableCountWidth,
            snapshot.distance_update_pop_count,
            kTableCountWidth,
            snapshot.split_count,
            kTableCountWidth,
            snapshot.required_depth_raise_count,
            kTableCountWidth,
            snapshot.processed_leaf_count,
            kTableCountWidth,
            snapshot.surface_leaf_count,
            kTableCountWidth,
            snapshot.total_cells);
        return snapshot;
    }

    std::size_t progress_bucket_for(const EmittedSnapshot &snapshot) const {
        const std::size_t work_scale = std::max(
            snapshot.pop_count,
            snapshot.peak_queue);
        if (work_scale == 0U) {
            return 0U;
        }
        const std::size_t threshold = std::max<std::size_t>(1U, work_scale / 100U);
        return snapshot.pop_count / threshold;
    }

    const RefinementClosureConfig &config_;
    const RefinementContext &context_;
    const RefinementWorkQueue &queue_;
    const ClosureRunStats &run_stats_;
    std::chrono::steady_clock::time_point start_time_;
    std::mutex emit_mutex_;
    std::size_t last_progress_bucket_;
    EmittedSnapshot last_snapshot_{};
    bool has_last_snapshot_ = false;
    bool header_printed_;
};

inline RefinementResult evaluate_refinement_for_leaf(
    const OctreeCell &current_cell,
    const std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config) {
    RefinementResult result;

    const std::int64_t contrib_begin = current_cell.contributor_begin;
    const std::int64_t contrib_end = current_cell.contributor_end;
    if (contrib_begin < 0 || contrib_end < 0 || contrib_begin >= contrib_end) {
        return result;
    }

    const auto safe_begin = static_cast<std::size_t>(
        std::min(contrib_begin,
                 static_cast<std::int64_t>(all_contributors.size())));
    const auto safe_end = static_cast<std::size_t>(
        std::min(contrib_end,
                 static_cast<std::int64_t>(all_contributors.size())));
    const std::span<const std::size_t> contributors =
        contributor_span(all_contributors, safe_begin, safe_end);

    if (contributors.size() < 2) {
        return result;
    }

    result.corner_values = sample_cell_corners(
        current_cell, contributors, positions, smoothing_lengths);
    result.corner_sign_mask = compute_corner_sign_mask(
        result.corner_values, config.isovalue);

    const bool corner_surface = cell_may_contain_isosurface(
        result.corner_values, config.isovalue);
    const bool inherited_surface_hint =
        current_cell.has_surface && !corner_surface;
    result.has_surface = corner_surface || inherited_surface_hint;
    if (!result.has_surface) {
        return result;
    }

    if (current_cell.depth >= config.max_depth) {
        return result;
    }

    if (!corner_surface) {
        result.should_split = true;
        result.children = create_child_cells(current_cell);
        result.child_contributors = filter_child_contributors(
            contributors, positions, smoothing_lengths, result.children);
        return result;
    }

    const std::vector<HermiteSample> samples = compute_cell_hermite_samples(
        current_cell.bounds, result.corner_values,
        result.corner_sign_mask, contributors,
        positions, smoothing_lengths, config.isovalue);
    const QEFLeafDiagnostics qef_diagnostics =
        analyze_qef_for_leaf(samples, current_cell.bounds);
    const double dx = current_cell.bounds.max.x - current_cell.bounds.min.x;
    const double dy = current_cell.bounds.max.y - current_cell.bounds.min.y;
    const double dz = current_cell.bounds.max.z - current_cell.bounds.min.z;
    const double cell_radius =
        0.5 * std::sqrt(dx * dx + dy * dy + dz * dz);
    const bool poor_qef_fit =
        qef_diagnostics.usable_sample_count <
            config.minimum_usable_hermite_samples ||
        qef_diagnostics.used_fallback ||
        minimum_hermite_normal_alignment(samples) <
            config.min_normal_alignment_threshold ||
        qef_diagnostics.rms_plane_residual >
            config.max_qef_rms_residual_ratio * cell_radius;
    if (!poor_qef_fit) {
        return result;
    }

    result.should_split = true;
    result.children = create_child_cells(current_cell);
    result.child_contributors = filter_child_contributors(
        contributors, positions, smoothing_lengths, result.children);
    for (OctreeCell &child : result.children) {
        child.has_surface = child_inherits_surface_hint(
            child, samples, qef_diagnostics.vertex);
    }

    return result;
}

// =====================================================================
// Lock-free neighbor walk via parent_index morton-ascend.
//
// For each of the six face directions of ``cell_index``, we sample the
// face at fine-grid resolution. Each sample point identifies a target
// fine-grid cell ``(tx, ty, tz)`` at depth ``max_depth``. We resolve that
// target to whatever leaf currently covers it by:
//
// 1. Starting from the recently-split cell, walking UP via parent_index
//    until the ancestor's spatial extent at its depth contains the
//    target. Walk-up cost is bounded by max_depth and uses one O(1)
//    field load (cell.parent_index) per step.
// 2. From that common ancestor, walking DOWN by reading
//    ``ancestor.child_begin`` (atomic-acquire) and indexing
//    ``cells[child_begin + slot]`` where slot is the local octant
//    derived from the target morton bits at that depth. Walk-down stops
//    when we hit a leaf (closure_is_leaf) or reach max_depth.
//
// The walk performs ZERO map probes, ZERO ``std::set`` allocations, and
// touches no shared state other than atomic loads on ``child_begin``.
// =====================================================================

inline std::size_t closure_morton_descend_to_leaf(
    std::size_t ancestor_index,
    const ChunkedArena<OctreeCell> &all_cells,
    std::uint64_t target_morton_at_max_depth,
    std::uint32_t max_depth) {
    std::size_t current = ancestor_index;
    while (true) {
        const OctreeCell &cell = all_cells[current];
        if (cell.depth >= max_depth) {
            return current;
        }
        const std::int64_t child_begin = closure_child_begin_acquire(cell);
        if (child_begin < 0) {
            return current;
        }
        // Slot within the parent's eight-child block: extract the three
        // morton bits at the child's depth from the target key.
        const std::uint32_t shift = (max_depth - (cell.depth + 1U)) * 3U;
        const std::uint64_t shifted = target_morton_at_max_depth >> shift;
        const std::uint8_t slot = static_cast<std::uint8_t>(shifted & 0x7U);
        current = static_cast<std::size_t>(child_begin) +
                  static_cast<std::size_t>(slot);
    }
}

inline std::size_t closure_morton_ascend_to_ancestor(
    std::size_t start_index,
    const ChunkedArena<OctreeCell> &all_cells,
    std::uint64_t target_morton_at_max_depth,
    std::uint32_t max_depth) {
    // Walk up via parent_index until the ancestor's morton prefix at its
    // depth matches the target's morton prefix at that depth. That is the
    // unique tree-ancestor whose volume contains the target.
    std::size_t current = start_index;
    while (true) {
        const OctreeCell &cell = all_cells[current];
        const std::uint32_t shift = (max_depth - cell.depth) * 3U;
        const std::uint64_t target_prefix =
            target_morton_at_max_depth >> shift;
        if (target_prefix == cell.morton_key) {
            return current;
        }
        if (cell.parent_index < 0) {
            // Walked off the root: target lies outside the tree.
            return std::numeric_limits<std::size_t>::max();
        }
        current = static_cast<std::size_t>(cell.parent_index);
    }
}

inline std::size_t closure_locate_cell_for_target(
    std::size_t origin_index,
    const ChunkedArena<OctreeCell> &all_cells,
    std::uint64_t target_morton_at_max_depth,
    std::uint32_t max_depth,
    std::uint32_t base_resolution) {
    const std::size_t ancestor = closure_morton_ascend_to_ancestor(
        origin_index, all_cells, target_morton_at_max_depth, max_depth);
    if (ancestor == std::numeric_limits<std::size_t>::max()) {
        std::uint32_t tx = 0U;
        std::uint32_t ty = 0U;
        std::uint32_t tz = 0U;
        morton_decode_3d(target_morton_at_max_depth, tx, ty, tz);
        const std::uint32_t root_x = tx >> max_depth;
        const std::uint32_t root_y = ty >> max_depth;
        const std::uint32_t root_z = tz >> max_depth;
        if (root_x >= base_resolution || root_y >= base_resolution ||
            root_z >= base_resolution) {
            return std::numeric_limits<std::size_t>::max();
        }
        const std::size_t root_index =
            (static_cast<std::size_t>(root_x) * base_resolution + root_y) *
                base_resolution +
            root_z;
        if (root_index >= all_cells.size()) {
            return std::numeric_limits<std::size_t>::max();
        }
        return closure_morton_descend_to_leaf(
            root_index, all_cells, target_morton_at_max_depth, max_depth);
    }
    return closure_morton_descend_to_leaf(
        ancestor, all_cells, target_morton_at_max_depth, max_depth);
}

inline void schedule_balance_neighbors_for_cell(
    std::size_t cell_index,
    RefinementContext &context,
    std::uint32_t max_depth,
    std::uint32_t base_resolution,
    std::uint32_t preferred_worker,
    std::vector<RefinementTask> &wake_tasks,
    ClosureRunStats &run_stats) {
    (void)preferred_worker;
    const ChunkedArena<OctreeCell> &all_cells = context.cells();
    if (cell_index >= all_cells.size()) {
        return;
    }

    auto raise_and_enqueue = [&](std::size_t candidate,
                                 std::uint32_t demanded_depth) {
        if (candidate >= all_cells.size()) {
            return;
        }
        if (context.get_required_depth(candidate) >= demanded_depth) {
            return;
        }
        if (!context.raise_required_depth_to(candidate, demanded_depth)) {
            return;
        }
        run_stats.required_depth_raise_count.fetch_add(
            1U, std::memory_order_relaxed);
        const bool internal = !closure_is_leaf(all_cells[candidate]);
        if (internal) {
            run_stats.internal_wake_count.fetch_add(
                1U, std::memory_order_relaxed);
        }
        if (context.mark_queued(candidate)) {
            wake_tasks.push_back({
                candidate,
                context.get_required_depth(candidate),
                0U,
            });
        }
    };

    const OctreeCell &cell = all_cells[cell_index];
    const std::uint32_t demanded_depth =
        cell.depth > 0U ? cell.depth - 1U : 0U;
    raise_and_enqueue(cell_index, demanded_depth);
    if (!closure_is_leaf(cell)) {
        return;
    }

    // Decode the cell's fine-grid origin (coords at max_depth).
    std::uint32_t cell_x = 0U;
    std::uint32_t cell_y = 0U;
    std::uint32_t cell_z = 0U;
    morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
    const std::uint32_t shift = max_depth - cell.depth;
    const std::uint32_t fine_min_x = cell_x << shift;
    const std::uint32_t fine_min_y = cell_y << shift;
    const std::uint32_t fine_min_z = cell_z << shift;
    const std::uint32_t span = 1U << shift;

    auto sample_face_interior = [&](int axis, bool positive) {
        // Face coord just outside the cell along ``axis``.
        const std::int64_t base[3] = {
            static_cast<std::int64_t>(fine_min_x),
            static_cast<std::int64_t>(fine_min_y),
            static_cast<std::int64_t>(fine_min_z),
        };
        const std::int64_t face_coord =
            positive ? base[axis] + static_cast<std::int64_t>(span)
                     : base[axis] - 1;
        if (face_coord < 0) {
            return;
        }
        const std::int64_t fine_max = static_cast<std::int64_t>(
            base_resolution << max_depth);
        if (face_coord >= fine_max) {
            return;
        }
        const int u_axis = (axis == 0) ? 1 : 0;
        const int v_axis = (axis == 2) ? 1 : 2;
        // For face-only 2:1 balance, any violating coarser neighbor spans the
        // entire open interior of the face in an aligned dyadic octree. One
        // strict interior sample is therefore sufficient to identify the
        // unique neighbor leaf covering that face.
        std::int64_t target[3] = {base[0], base[1], base[2]};
        target[axis] = face_coord;
        target[u_axis] += static_cast<std::int64_t>(span / 2U);
        target[v_axis] += static_cast<std::int64_t>(span / 2U);
        if (target[u_axis] < 0 || target[v_axis] < 0) {
            return;
        }
        if (target[u_axis] >= fine_max || target[v_axis] >= fine_max) {
            return;
        }
        const std::uint64_t target_morton = morton_encode_3d(
            static_cast<std::uint32_t>(target[0]),
            static_cast<std::uint32_t>(target[1]),
            static_cast<std::uint32_t>(target[2]));
        const std::size_t neighbor_idx = closure_locate_cell_for_target(
            cell_index, all_cells, target_morton, max_depth, base_resolution);
        if (neighbor_idx == std::numeric_limits<std::size_t>::max()) {
            return;
        }
        if (all_cells[neighbor_idx].depth < demanded_depth) {
            raise_and_enqueue(neighbor_idx, demanded_depth);
        }
    };

    sample_face_interior(0, true);
    sample_face_interior(0, false);
    sample_face_interior(1, true);
    sample_face_interior(1, false);
    sample_face_interior(2, true);
    sample_face_interior(2, false);
}

inline void enqueue_local_children(
    const std::vector<RefinementTask> &claimed_tasks,
    ClosureWorkerState &worker,
    std::vector<RefinementTask> &local_stack);

inline void collect_internal_refinement_indices(
    const PublishedChildren &published,
    ClosureWorkerState &worker,
    std::vector<std::size_t> &demanded_internal_indices) {
    if (published.child_indices.empty()) {
        return;
    }

    demanded_internal_indices.reserve(published.child_indices.size());
    for (std::size_t child_index : published.child_indices) {
        const std::size_t parent_index = worker.context.parent_index(child_index);
        if (parent_index == SIZE_MAX) {
            continue;
        }
        const OctreeCell &parent = worker.context.cells()[parent_index];
        if (closure_is_leaf(parent)) {
            continue;
        }
        if (worker.context.get_required_depth(parent_index) <= parent.depth) {
            continue;
        }
        demanded_internal_indices.push_back(parent_index);
    }

    std::sort(
        demanded_internal_indices.begin(),
        demanded_internal_indices.end());
    demanded_internal_indices.erase(
        std::unique(
            demanded_internal_indices.begin(),
            demanded_internal_indices.end()),
        demanded_internal_indices.end());
}

// Set parent_index on each newly created child so that the lock-free
// neighbor walk in ``schedule_balance_neighbors_for_cell`` can ascend
// from any descendant up to its tree ancestors via O(1) field loads.
inline void closure_assign_parent_index(
    std::vector<OctreeCell> &children, std::size_t parent_index) {
    for (OctreeCell &child : children) {
        child.parent_index = static_cast<std::int64_t>(parent_index);
    }
}

// Phase-A: surface-band self-enqueue.
//
// After a split publishes ``child_count`` children at indices
// [child_begin, child_begin + child_count), inspect each child and
// schedule it for further refinement *if* it still meets the surface-
// band criterion (``has_surface && depth < max_depth && edge > target``).
//
// We do NOT push directly into the global queue; instead we append to
// ``wake_tasks`` so the caller's existing ``merge_wake_tasks_into_claimed
// + queue_requirement_wakes`` path handles the kQueued transition exactly
// like balance-driven wakes. This keeps the kIdle/kQueued/kProcessing
// state machine identical to the original outer-loop approach.
//
// Cost: one cell read + one branch per published child. Skipped entirely
// when ``surface_band_target_leaf_size <= 0.0`` (every non-surface-band
// phase).
inline double closure_cell_edge_length(const OctreeCell &cell);
inline void enqueue_surface_band_children(
    const PublishedChildren &published,
    ClosureWorkerState &worker,
    std::vector<RefinementTask> &wake_tasks) {
    const double target = worker.config.surface_band_target_leaf_size;
    if (target <= 0.0) {
        return;
    }
    const std::uint32_t max_depth = worker.config.max_depth;
    RefinementContext &context = worker.context;
    for (std::size_t child_index : published.child_indices) {
        if (child_index >= context.size()) {
            continue;
        }
        const OctreeCell &child = context.cells()[child_index];
        if (!child.has_surface || child.depth >= max_depth) {
            continue;
        }
        if (closure_cell_edge_length(child) <= target) {
            continue;
        }
        // Raise the required depth so the child is treated as needing
        // further refinement, mirroring the seed pass in
        // ``refine_surface_band_with_closure``.
        context.raise_required_depth_to(child_index, child.depth);
        if (!context.mark_queued(child_index)) {
            continue;
        }
        wake_tasks.push_back({
            child_index,
            child.depth,
            0U,
        });
    }
}

// Phase-B2: thickening-band child classify enqueue.
//
// After any split publishes children, if the thickening task chain is active
// (``thickening_band_target_leaf_size > 0``), push a kClassify task for each
// published child.  This is how the incremental chain
//   kRefine → (new children) → kClassify → kDistanceUpdate → kRefine ...
// propagates without needing an outer re-classification pass.
//
// Unlike the surface-band path we push directly to the queue (not via
// wake_tasks / mark_queued) because kClassify tasks do not interact with
// the kProcessing/kIdle state machine — they are one-shot evaluations
// with no concurrency hazard on the cell state.
inline void enqueue_classify_children(
    const PublishedChildren &published,
    ClosureWorkerState &worker) {
    if (worker.config.thickening_band_target_leaf_size <= 0.0) {
        return;
    }
    for (std::size_t child_index : published.child_indices) {
        if (child_index >= worker.context.size()) {
            continue;
        }
        worker.queue.push(
            {child_index, 0U, 0U, RefinementTaskKind::kClassify},
            worker.worker_id);
    }
}

inline void enqueue_distance_update_neighbors(
    const PublishedChildren &published,
    ClosureWorkerState &worker) {
    if (worker.config.thickening_band_target_leaf_size <= 0.0 ||
        worker.config.thickening_radius <= 0.0) {
        return;
    }
    const double target = worker.config.thickening_band_target_leaf_size;
    for (std::size_t affected_index : published.affected_indices) {
        if (affected_index >= worker.context.size()) {
            continue;
        }
        const OctreeCell &cell = worker.context.cells()[affected_index];
        if (!closure_is_leaf(cell) ||
            cell.depth >= worker.config.max_depth ||
            closure_cell_edge_length(cell) <= target) {
            continue;
        }
        const std::uint8_t cls =
            worker.context.cell_classification()[affected_index].load(
                std::memory_order_acquire);
        if (cls != 1U) {
            continue;
        }
        worker.queue.push(
            {affected_index, 0U, 0U, RefinementTaskKind::kDistanceUpdate},
            worker.worker_id);
    }
}

inline PublishedChildren apply_balance_split(
    std::size_t split_index,
    ClosureWorkerState &worker,
    std::vector<RefinementTask> &wake_tasks) {
    ScopedNsTimer apply_timer(worker.profiler.ns_apply_split);
    RefinementContext &context = worker.context;
    OctreeCell &current_cell = context.cells()[split_index];
    // Snapshot is safe: this worker holds kProcessing on split_index, and
    // the cell's content fields (depth, morton_key, contributor_*, bounds)
    // are not concurrently mutated outside of split publication, which is
    // also serialized by kProcessing.
    const OctreeCell parent_snapshot = current_cell;
    if (!closure_is_leaf(parent_snapshot)) {
        return PublishedChildren{};
    }
    const std::uint32_t parent_required_depth =
        context.get_required_depth(split_index);

    std::vector<std::size_t> parent_contributors;
    context.copy_contributors_for_cell(split_index, parent_contributors);

    std::vector<OctreeCell> children = create_child_cells(parent_snapshot);
    closure_assign_parent_index(children, split_index);
    const std::vector<std::vector<std::size_t>> child_contributors =
        filter_child_contributors(
            std::span<const std::size_t>(
                parent_contributors.data(), parent_contributors.size()),
            worker.positions,
            worker.smoothing_lengths,
            children);

    ClosureStorage storage(context);
    ClosurePublisher publisher(
        storage,
        worker.positions,
        worker.smoothing_lengths,
        worker.config.isovalue);

    const PublishedChildren published =
        publisher.publish_children(
            split_index,
            parent_required_depth,
            children,
            child_contributors,
            true,
            worker.worker_id);

    // Atomic publication: release-store the child_begin so concurrent
    // neighbor walks observing this value are guaranteed to see the
    // children's writes that completed inside publish_children. We do
    // NOT mutate ``is_leaf`` here; that boolean is reconciled post-run
    // in materialize_into to avoid racing with neighbor reads.
    if (!published.child_indices.empty()) {
        closure_publish_child_begin(
            current_cell,
            static_cast<std::int64_t>(published.child_indices.front()));
    }

    for (std::size_t affected_index : published.affected_indices) {
        ScopedNsTimer wake_timer(worker.profiler.ns_neighbor_wake);
        const std::size_t wake_count_before = wake_tasks.size();
        schedule_balance_neighbors_for_cell(
            affected_index,
            context,
            worker.config.max_depth,
            worker.config.base_resolution,
            worker.worker_id,
            wake_tasks,
            worker.run_stats);
        if (wake_tasks.size() > wake_count_before) {
            worker.profiler.neighbor_wakes_pushed.fetch_add(
                wake_tasks.size() - wake_count_before,
                std::memory_order_relaxed);
        }
    }

    enqueue_surface_band_children(published, worker, wake_tasks);
    enqueue_classify_children(published, worker);
    enqueue_distance_update_neighbors(published, worker);

    return published;
}

inline PublishedChildren apply_surface_split(
    std::size_t split_index,
    const RefinementResult &result,
    ClosureWorkerState &worker,
    std::vector<RefinementTask> &wake_tasks) {
    ScopedNsTimer apply_timer(worker.profiler.ns_apply_split);
    RefinementContext &context = worker.context;
    OctreeCell &current_cell = context.cells()[split_index];
    const std::uint32_t parent_required_depth =
        context.get_required_depth(split_index);

    // Mutations on the parent cell other than ``child_begin`` are safe
    // because this worker holds kProcessing; neighbor walks read only
    // ``child_begin`` (atomic) and ``depth`` (immutable).
    current_cell.is_active = true;
    current_cell.has_surface = true;
    current_cell.is_topo_surface = false;

    // Copy children so we can stamp parent_index without mutating the
    // RefinementResult passed in by const ref.
    std::vector<OctreeCell> children = result.children;
    closure_assign_parent_index(children, split_index);

    ClosureStorage storage(context);
    ClosurePublisher publisher(
        storage,
        worker.positions,
        worker.smoothing_lengths,
        worker.config.isovalue);

    const PublishedChildren published =
        publisher.publish_children(
            split_index,
            parent_required_depth,
            children,
            result.child_contributors,
            false,
            worker.worker_id);

    if (!published.child_indices.empty()) {
        closure_publish_child_begin(
            current_cell,
            static_cast<std::int64_t>(published.child_indices.front()));
    }

    for (std::size_t affected_index : published.affected_indices) {
        ScopedNsTimer wake_timer(worker.profiler.ns_neighbor_wake);
        const std::size_t wake_count_before = wake_tasks.size();
        schedule_balance_neighbors_for_cell(
            affected_index,
            context,
            worker.config.max_depth,
            worker.config.base_resolution,
            worker.worker_id,
            wake_tasks,
            worker.run_stats);
        if (wake_tasks.size() > wake_count_before) {
            worker.profiler.neighbor_wakes_pushed.fetch_add(
                wake_tasks.size() - wake_count_before,
                std::memory_order_relaxed);
        }
    }

    enqueue_surface_band_children(published, worker, wake_tasks);
    enqueue_classify_children(published, worker);
    enqueue_distance_update_neighbors(published, worker);

    return published;
}

inline void maybe_shutdown_queue(ClosureWorkerState &worker) {
    worker.queue.try_shutdown_if_idle();
}

// Cap on per-worker local DFS stack size before spilling to the global
// queue. Higher values reduce queue traffic and let workers keep more
// related work cache-hot, at the cost of slightly worse load balancing
// when one worker gets a deep subtree. 4096 was chosen so each worker can
// hold a full octree subtree branch (max_depth ~12-16) entirely in its
// local stack without spilling on common workloads.
constexpr std::size_t kLocalDepthFirstBudget = 4096U;

inline void claim_refinement_tasks(
    const std::vector<std::size_t> &cell_indices,
    ClosureWorkerState &worker,
    std::vector<RefinementTask> &claimed_tasks) {
    claimed_tasks.reserve(claimed_tasks.size() + cell_indices.size());
    for (std::size_t cell_index : cell_indices) {
        if (!worker.context.mark_queued(cell_index)) {
            continue;
        }
        claimed_tasks.push_back({
            cell_index,
            worker.context.get_required_depth(cell_index),
            0U,
        });
    }
}

inline void merge_wake_tasks_into_claimed(
    std::vector<RefinementTask> &claimed_tasks,
    std::vector<RefinementTask> &wake_tasks) {
    if (wake_tasks.empty()) {
        return;
    }
    claimed_tasks.reserve(claimed_tasks.size() + wake_tasks.size());
    for (RefinementTask &task : wake_tasks) {
        claimed_tasks.push_back(task);
    }
}

inline void queue_requirement_wakes(
    std::vector<RefinementTask> &claimed_tasks,
    ClosureWorkerState &worker) {
    if (claimed_tasks.empty()) {
        return;
    }

    std::vector<RefinementTask> local_wakes;
    local_wakes.reserve(claimed_tasks.size());
    for (const RefinementTask &task : claimed_tasks) {
        if (!worker.context.mark_queued(task.cell_index)) {
            continue;
        }
        local_wakes.push_back({
            task.cell_index,
            worker.context.get_required_depth(task.cell_index),
            task.reason_flags,
        });
    }
    claimed_tasks.swap(local_wakes);
}

inline bool seed_queue_from_cells(
    RefinementContext &context,
    RefinementWorkQueue &queue,
    std::uint32_t worker_count,
    const std::vector<std::size_t> &seed_cells) {
    bool queued_any = false;
    for (std::size_t ordinal = 0; ordinal < seed_cells.size(); ++ordinal) {
        const std::size_t cell_index = seed_cells[ordinal];
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_index);
        if (cell_index >= context.size()) {
            continue;
        }
        if (!context.mark_queued(cell_index)) {
            continue;
        }
        queue.push(
            {
                cell_index,
                context.get_required_depth(cell_index),
                0U,
            },
            static_cast<std::uint32_t>(ordinal % std::max(1U, worker_count)));
        queued_any = true;
    }
    return queued_any;
}

inline std::vector<std::size_t> build_coarse_startup_seed_order(
    const std::vector<std::size_t> &seed_cells,
    std::uint32_t coarse_seed_count) {
    if (seed_cells.empty()) {
        return {};
    }
    const std::size_t coarse_count = std::min<std::size_t>(
        std::max<std::uint32_t>(1U, coarse_seed_count), seed_cells.size());
    const std::size_t stride =
        (seed_cells.size() + coarse_count - 1U) / coarse_count;

    std::vector<std::size_t> ordered;
    ordered.reserve(seed_cells.size());
    for (std::size_t offset = 0; offset < stride; ++offset) {
        for (std::size_t index = offset; index < seed_cells.size();
             index += stride) {
            ordered.push_back(seed_cells[index]);
        }
    }
    return ordered;
}

inline std::size_t count_remaining_surface_band_eligible(
    RefinementContext &context,
    double max_surface_leaf_size,
    std::uint32_t max_depth) {
    std::size_t count = 0U;
    for (std::size_t cell_index = 0; cell_index < context.size(); ++cell_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_index);
        const OctreeCell &cell = context.cells()[cell_index];
        if (!closure_is_leaf(cell) || !cell.has_surface || cell.depth >= max_depth) {
            continue;
        }
        if (closure_cell_edge_length(cell) <= max_surface_leaf_size) {
            continue;
        }
        ++count;
    }
    return count;
}

inline std::size_t count_remaining_thickening_band_eligible(
    RefinementContext &context,
    double target_leaf_size,
    std::uint32_t max_depth) {
    std::size_t count = 0U;
    for (std::size_t cell_index = 0; cell_index < context.size(); ++cell_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_index);
        const OctreeCell &cell = context.cells()[cell_index];
        if (!closure_is_leaf(cell) || cell.depth >= max_depth ||
            closure_cell_edge_length(cell) <= target_leaf_size) {
            continue;
        }
        const std::uint8_t cls =
            context.cell_classification()[cell_index].load(std::memory_order_acquire);
        if (cls != 1U) {
            continue;
        }
        ++count;
    }
    return count;
}

inline double closure_cell_edge_length(const OctreeCell &cell) {
    return cell.bounds.max.x - cell.bounds.min.x;
}

inline std::vector<std::size_t> select_surface_band_seed_cells(
    const RefinementContext &context,
    std::uint32_t worker_count,
    double max_surface_leaf_size,
    std::vector<std::size_t> *all_matching_cells = nullptr) {
    std::vector<std::size_t> matching_cells;
    std::vector<std::size_t> seeds;
    seeds.reserve(std::max<std::uint32_t>(worker_count, 1U) * 64U);

    std::uint32_t minimum_depth = std::numeric_limits<std::uint32_t>::max();
    for (std::size_t cell_index = 0; cell_index < context.size(); ++cell_index) {
        const OctreeCell &cell = context.cells()[cell_index];
        if (!cell.is_leaf || !cell.has_surface) {
            continue;
        }
        if (closure_cell_edge_length(cell) <= max_surface_leaf_size) {
            continue;
        }
        matching_cells.push_back(cell_index);
        minimum_depth = std::min(minimum_depth, cell.depth);
    }

    if (all_matching_cells != nullptr) {
        *all_matching_cells = matching_cells;
    }
    if (matching_cells.empty()) {
        return seeds;
    }

    seeds = matching_cells;
    return seeds;
}

inline std::vector<std::size_t> select_targeted_seed_cells(
    const RefinementContext &context,
    std::uint32_t worker_count,
    const std::vector<std::size_t> &candidate_cells,
    std::vector<std::size_t> *all_matching_cells = nullptr) {
    std::vector<std::size_t> matching_cells;
    std::vector<std::size_t> seeds;
    seeds.reserve(std::max<std::uint32_t>(worker_count, 1U) * 64U);

    std::uint32_t minimum_depth = std::numeric_limits<std::uint32_t>::max();
    for (std::size_t cell_index : candidate_cells) {
        if (cell_index >= context.size()) {
            continue;
        }
        const OctreeCell &cell = context.cells()[cell_index];
        if (!cell.is_leaf) {
            continue;
        }
        matching_cells.push_back(cell_index);
        minimum_depth = std::min(minimum_depth, cell.depth);
    }

    if (all_matching_cells != nullptr) {
        *all_matching_cells = matching_cells;
    }
    if (matching_cells.empty()) {
        return seeds;
    }

    seeds = matching_cells;
    return seeds;
}

inline void enqueue_local_children(
    const std::vector<RefinementTask> &claimed_tasks,
    ClosureWorkerState &worker,
    std::vector<RefinementTask> &local_stack) {
    std::vector<RefinementTask> spill_tasks;
    spill_tasks.reserve(claimed_tasks.size());
    std::size_t local_pushed = 0U;
    for (const RefinementTask &task : claimed_tasks) {
        if (local_stack.size() < kLocalDepthFirstBudget) {
            local_stack.push_back(task);
            ++local_pushed;
        } else {
            spill_tasks.push_back(task);
        }
    }
    if (local_pushed > 0U) {
        worker.profiler.local_stack_pushes.fetch_add(
            local_pushed, std::memory_order_relaxed);
    }
    if (!spill_tasks.empty()) {
        worker.profiler.spill_pushes.fetch_add(
            spill_tasks.size(), std::memory_order_relaxed);
        worker.queue.push_batch(spill_tasks, worker.worker_id);
    }
}

inline void enqueue_internal_children_if_needed(
    std::size_t parent_index,
    const OctreeCell &parent_snapshot,
    ClosureWorkerState &worker,
    std::vector<std::size_t> &child_indices) {
    // Use the publication-based leaf check rather than the legacy
    // ``is_leaf`` boolean. After R1, splits publish via ``child_begin``
    // and never mutate ``is_leaf``; the boolean is reconciled only in
    // ``materialize_into`` at run end.
    if (closure_is_leaf(parent_snapshot)) {
        return;
    }

    worker.context.append_child_indices(parent_index, child_indices);

    std::vector<std::size_t> demanded_child_indices;
    demanded_child_indices.reserve(child_indices.size());
    for (std::size_t child_index : child_indices) {
        // ``depth`` is immutable for the lifetime of a cell once published,
        // so we can read it without any synchronization.
        const std::uint32_t child_depth =
            worker.context.cells()[child_index].depth;
        const std::uint32_t child_required_depth =
            worker.context.get_required_depth(child_index);
        if (child_required_depth <= child_depth) {
            continue;
        }
        demanded_child_indices.push_back(child_index);
    }
    child_indices.swap(demanded_child_indices);
}

inline void process_closure_task(
    const RefinementTask &root_task,
    ClosureWorkerState &worker) {
    ScopedNsTimer dfs_timer(worker.profiler.ns_local_dfs_total);
    std::vector<RefinementTask> local_stack;
    local_stack.push_back(root_task);
    worker.profiler.local_stack_pushes.fetch_add(1U, std::memory_order_relaxed);

    while (!local_stack.empty()) {
        const RefinementTask task = local_stack.back();
        local_stack.pop_back();
        poll_closure_cancellation(worker, task.cell_index, 4096U);
        RefinementContext &context = worker.context;

        OctreeCell cell_snapshot;
        std::vector<std::size_t> contributor_snapshot;
        std::uint32_t required_depth_snapshot = 0U;
        std::vector<RefinementTask> claimed_tasks;
        std::vector<std::size_t> claimed_indices;
        bool queued_followup_work = false;
        bool ready_for_leaf_evaluation = false;

        if (task.cell_index >= context.size() ||
            task.cell_index >= context.cells().size()) {
            worker.run_stats.stale_task_count.fetch_add(
                1U, std::memory_order_relaxed);
            worker.profiler.tasks_stale_or_skipped.fetch_add(
                1U, std::memory_order_relaxed);
            continue;
        }

        if (!context.mark_processing(task.cell_index)) {
            worker.run_stats.stale_task_count.fetch_add(
                1U, std::memory_order_relaxed);
            worker.profiler.tasks_stale_or_skipped.fetch_add(
                1U, std::memory_order_relaxed);
            continue;
        }

        const OctreeCell current_cell_snapshot =
            context.cells()[task.cell_index];
        cell_snapshot = current_cell_snapshot;
        required_depth_snapshot = context.get_required_depth(task.cell_index);

        if (!closure_is_leaf(current_cell_snapshot)) {
            ScopedNsTimer internal_timer(worker.profiler.ns_internal_descent);
            worker.profiler.internal_cells_visited.fetch_add(
                1U, std::memory_order_relaxed);
            if (required_depth_snapshot <= current_cell_snapshot.depth) {
                context.mark_retired(task.cell_index);
                worker.profiler.non_leaf_retired_immediately.fetch_add(
                    1U, std::memory_order_relaxed);
                continue;
            }
            context.mark_idle(task.cell_index);
            claimed_indices.clear();
            enqueue_internal_children_if_needed(
                task.cell_index,
                current_cell_snapshot,
                worker,
                claimed_indices);
            claim_refinement_tasks(claimed_indices, worker, claimed_tasks);
            queued_followup_work = !claimed_tasks.empty();
        } else {
            context.copy_contributors_for_cell(
                task.cell_index, contributor_snapshot);

            cell_snapshot.contributor_begin = 0;
            cell_snapshot.contributor_end =
                static_cast<std::int64_t>(contributor_snapshot.size());
            ready_for_leaf_evaluation = true;
        }

        if (queued_followup_work) {
            enqueue_local_children(claimed_tasks, worker, local_stack);
            continue;
        }
        if (!ready_for_leaf_evaluation) {
            continue;
        }

        {
            worker.run_stats.processed_leaf_count.fetch_add(
                1U, std::memory_order_relaxed);
            if (cell_snapshot.has_surface) {
                worker.run_stats.surface_leaf_count.fetch_add(
                    1U, std::memory_order_relaxed);
            }
        }

        // Balance-driven split for the surface band (with 2:1 enforcement).
        // The thickening band skips this check to match the original loop
        // which had no 2:1 balancing.
        if (cell_snapshot.depth < required_depth_snapshot &&
            worker.config.thickening_band_target_leaf_size <= 0.0) {
            // Balance-driven split: directly apply the split without any
            // cross-worker mutex. The kProcessing claim on this cell
            // already serializes split publication for this parent, and
            // ``apply_balance_split`` publishes children via release-store
            // on ``child_begin``. We do still detect a "retired_early"
            // case where another worker observed our cell as a non-leaf
            // before we got here (impossible with kProcessing, but kept
            // as a defensive check on ``closure_is_leaf``).
            bool did_split = false;
            std::vector<RefinementTask> wake_tasks;
            const PublishedChildren published =
                apply_balance_split(task.cell_index, worker, wake_tasks);
            const bool retired_early = published.child_indices.empty();
            if (retired_early) {
                context.mark_retired(task.cell_index);
                continue;
            }
            claimed_indices = published.child_indices;
            collect_internal_refinement_indices(
                published,
                worker,
                claimed_indices);
            did_split = true;
            context.mark_retired(task.cell_index);
            claim_refinement_tasks(claimed_indices, worker, claimed_tasks);
            merge_wake_tasks_into_claimed(claimed_tasks, wake_tasks);
            if (did_split) {
                worker.run_stats.split_count.fetch_add(
                    1U, std::memory_order_relaxed);
                worker.profiler.splits.fetch_add(1U, std::memory_order_relaxed);
                enqueue_local_children(
                    claimed_tasks,
                    worker,
                    local_stack);
            }
            continue;
        }

        // For the thickening band, force split if above target_leaf_size.
        // This replicates the original loop which split regardless of QEF.
        // We must check this BEFORE calling evaluate_refinement_for_leaf
        // so the result can be non-const and modified.
        const bool force_split_thickening =
            worker.config.thickening_band_target_leaf_size > 0.0 &&
            closure_cell_edge_length(cell_snapshot) >
                worker.config.thickening_band_target_leaf_size;

        RefinementResult result = [&]() {
            ScopedNsTimer leaf_timer(worker.profiler.ns_leaf_eval);
            worker.profiler.leaves_evaluated.fetch_add(
                1U, std::memory_order_relaxed);
            return evaluate_refinement_for_leaf(
                cell_snapshot,
                contributor_snapshot,
                worker.positions,
                worker.smoothing_lengths,
                worker.config);
        }();

        if (force_split_thickening) {
            result.should_split = true;
            if (result.children.empty()) {
                result.children = create_child_cells(cell_snapshot);
                result.child_contributors = filter_child_contributors(
                    std::span<const std::size_t>(
                        contributor_snapshot.data(),
                        contributor_snapshot.size()),
                    worker.positions,
                    worker.smoothing_lengths,
                    result.children);
            }
        }

        bool did_split = false;
        bool should_continue = false;
        // No-split / no-surface paths: mutate cell content fields under
        // our kProcessing claim. ``is_leaf`` was already true for this
        // path (we entered via ``closure_is_leaf(current_cell_snapshot)``)
        // and we never flip it during the run, so leave it as-is.
        {
            if (task.cell_index >= context.cells().size()) {
                context.mark_retired(task.cell_index);
                continue;
            }
            OctreeCell &current_cell = context.cells()[task.cell_index];
            if (!closure_is_leaf(current_cell)) {
                context.mark_retired(task.cell_index);
                continue;
            }

            current_cell.corner_values = result.corner_values;
            current_cell.corner_sign_mask = result.corner_sign_mask;

            if (!result.has_surface) {
                current_cell.is_active = false;
                current_cell.has_surface = false;
                current_cell.is_topo_surface = false;
                current_cell.child_begin = -1;
                context.mark_idle(task.cell_index);
                should_continue = true;
                worker.profiler.tasks_retired_no_split.fetch_add(
                    1U, std::memory_order_relaxed);
            }

            if (!should_continue && !result.should_split) {
                current_cell.is_active = true;
                current_cell.has_surface = true;
                current_cell.is_topo_surface = false;
                current_cell.child_begin = -1;
                context.mark_idle(task.cell_index);
                should_continue = true;
                worker.profiler.tasks_retired_no_split.fetch_add(
                    1U, std::memory_order_relaxed);
            }
        }
        if (should_continue) {
            continue;
        }
        {
            std::vector<RefinementTask> wake_tasks;
            const PublishedChildren published =
                apply_surface_split(task.cell_index, result, worker, wake_tasks);
            if (published.child_indices.empty()) {
                context.mark_retired(task.cell_index);
                continue;
            }
            claimed_tasks.clear();
            claimed_indices = published.child_indices;
            collect_internal_refinement_indices(
                published,
                worker,
                claimed_indices);
            did_split = true;
            merge_wake_tasks_into_claimed(claimed_tasks, wake_tasks);
        }
        context.mark_retired(task.cell_index);
        claim_refinement_tasks(claimed_indices, worker, claimed_tasks);
        if (did_split) {
            worker.run_stats.split_count.fetch_add(
                1U, std::memory_order_relaxed);
            worker.profiler.splits.fetch_add(1U, std::memory_order_relaxed);
            enqueue_local_children(
                claimed_tasks,
                worker,
                local_stack);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers for IEEE-754 float bit manipulation used by the thickening side-cars.
// ---------------------------------------------------------------------------

// Raw bits of +infinity as a 32-bit float.
static constexpr std::uint32_t kPlusInfBits = 0x7F800000U;

inline float bits_to_float(std::uint32_t bits) {
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

inline std::uint32_t float_to_bits(float f) {
    std::uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(bits));
    return bits;
}

inline std::uint64_t double_to_bits(double value) {
    std::uint64_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline void refresh_occupancy_state_for_cell(
    std::size_t idx,
    ClosureWorkerState &worker) {

    RefinementContext &context = worker.context;
    if (idx >= context.size()) {
        return;
    }
    if (!closure_is_leaf(context.cells()[idx])) {
        return;
    }

    const OctreeCell &cell = context.cells()[idx];
    constexpr std::uint8_t kInsideOccupancy = 1U;
    constexpr std::uint8_t kBoundaryInsideOccupancy = 2U;
    constexpr std::uint8_t kBoundaryOutsideOccupancy = 3U;
    const std::uint8_t cls =
        context.cell_classification()[idx].load(std::memory_order_acquire);
    const bool inside = cls != 0U;
    std::uint8_t occupancy = inside ? kInsideOccupancy : 0U;

    std::uint32_t cell_x = 0U;
    std::uint32_t cell_y = 0U;
    std::uint32_t cell_z = 0U;
    morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
    const std::uint32_t cell_shift = worker.config.max_depth - cell.depth;
    const std::uint32_t fine_min_x = cell_x << cell_shift;
    const std::uint32_t fine_min_y = cell_y << cell_shift;
    const std::uint32_t fine_min_z = cell_z << cell_shift;
    const std::uint32_t span = 1U << cell_shift;
    const std::uint32_t half_span = span / 2U > 0U ? span / 2U : 1U;
    const std::int64_t fine_max = static_cast<std::int64_t>(
        worker.config.base_resolution << worker.config.max_depth);

    struct FaceInfo { int axis; bool positive; };
    const FaceInfo faces[6] = {
        {0, false}, {0, true}, {1, false},
        {1, true},  {2, false}, {2, true},
    };
    bool touches_opposite = false;
    for (const FaceInfo &face : faces) {
        const std::int64_t base_x = static_cast<std::int64_t>(fine_min_x);
        const std::int64_t base_y = static_cast<std::int64_t>(fine_min_y);
        const std::int64_t base_z = static_cast<std::int64_t>(fine_min_z);
        std::int64_t probe[3] = {
            base_x + static_cast<std::int64_t>(half_span),
            base_y + static_cast<std::int64_t>(half_span),
            base_z + static_cast<std::int64_t>(half_span),
        };
        if (face.positive) {
            probe[face.axis] = face.axis == 0 ?
                base_x + static_cast<std::int64_t>(span) :
                face.axis == 1 ?
                    base_y + static_cast<std::int64_t>(span) :
                    base_z + static_cast<std::int64_t>(span);
        } else {
            probe[face.axis] = face.axis == 0 ? base_x - 1
                             : face.axis == 1 ? base_y - 1
                                              : base_z - 1;
        }
        if (probe[0] < 0 || probe[0] >= fine_max ||
            probe[1] < 0 || probe[1] >= fine_max ||
            probe[2] < 0 || probe[2] >= fine_max) {
            touches_opposite = true;
            continue;
        }
        const std::uint64_t target_morton = morton_encode_3d(
            static_cast<std::uint32_t>(probe[0]),
            static_cast<std::uint32_t>(probe[1]),
            static_cast<std::uint32_t>(probe[2]));
        const std::size_t neighbor_idx = closure_locate_cell_for_target(
            idx,
            context.cells(),
            target_morton,
            worker.config.max_depth,
            worker.config.base_resolution);
        if (neighbor_idx == std::numeric_limits<std::size_t>::max() ||
            neighbor_idx >= context.size()) {
            touches_opposite = true;
            continue;
        }
        const bool neighbor_inside =
            context.cell_classification()[neighbor_idx].load(
                std::memory_order_acquire) != 0U;
        if (neighbor_inside != inside) {
            touches_opposite = true;
        }
    }

    if (touches_opposite) {
        occupancy = inside ? kBoundaryInsideOccupancy
                           : kBoundaryOutsideOccupancy;
    }
    context.occupancy_state_bits()[idx].store(
        occupancy,
        std::memory_order_release);
}

inline void refresh_occupancy_state_for_cell_and_neighbors(
    std::size_t idx,
    ClosureWorkerState &worker) {

    RefinementContext &context = worker.context;
    if (idx >= context.size()) {
        return;
    }
    refresh_occupancy_state_for_cell(idx, worker);
    if (!closure_is_leaf(context.cells()[idx])) {
        return;
    }

    const OctreeCell &cell = context.cells()[idx];
    std::uint32_t cell_x = 0U;
    std::uint32_t cell_y = 0U;
    std::uint32_t cell_z = 0U;
    morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
    const std::uint32_t cell_shift = worker.config.max_depth - cell.depth;
    const std::uint32_t fine_min_x = cell_x << cell_shift;
    const std::uint32_t fine_min_y = cell_y << cell_shift;
    const std::uint32_t fine_min_z = cell_z << cell_shift;
    const std::uint32_t span = 1U << cell_shift;
    const std::uint32_t half_span = span / 2U > 0U ? span / 2U : 1U;
    const std::int64_t fine_max = static_cast<std::int64_t>(
        worker.config.base_resolution << worker.config.max_depth);

    struct FaceInfo { int axis; bool positive; };
    const FaceInfo faces[6] = {
        {0, false}, {0, true}, {1, false},
        {1, true},  {2, false}, {2, true},
    };
    for (const FaceInfo &face : faces) {
        const std::int64_t base_x = static_cast<std::int64_t>(fine_min_x);
        const std::int64_t base_y = static_cast<std::int64_t>(fine_min_y);
        const std::int64_t base_z = static_cast<std::int64_t>(fine_min_z);
        std::int64_t probe[3] = {
            base_x + static_cast<std::int64_t>(half_span),
            base_y + static_cast<std::int64_t>(half_span),
            base_z + static_cast<std::int64_t>(half_span),
        };
        if (face.positive) {
            probe[face.axis] = face.axis == 0 ?
                base_x + static_cast<std::int64_t>(span) :
                face.axis == 1 ?
                    base_y + static_cast<std::int64_t>(span) :
                    base_z + static_cast<std::int64_t>(span);
        } else {
            probe[face.axis] = face.axis == 0 ? base_x - 1
                             : face.axis == 1 ? base_y - 1
                                              : base_z - 1;
        }
        if (probe[0] < 0 || probe[0] >= fine_max ||
            probe[1] < 0 || probe[1] >= fine_max ||
            probe[2] < 0 || probe[2] >= fine_max) {
            continue;
        }
        const std::uint64_t target_morton = morton_encode_3d(
            static_cast<std::uint32_t>(probe[0]),
            static_cast<std::uint32_t>(probe[1]),
            static_cast<std::uint32_t>(probe[2]));
        const std::size_t neighbor_idx = closure_locate_cell_for_target(
            idx,
            context.cells(),
            target_morton,
            worker.config.max_depth,
            worker.config.base_resolution);
        if (neighbor_idx == std::numeric_limits<std::size_t>::max() ||
            neighbor_idx >= context.size() ||
            neighbor_idx == idx) {
            continue;
        }
        refresh_occupancy_state_for_cell(neighbor_idx, worker);
    }
}

// Atomically store a float distance into the outside_distance_bits side-car
// only if the new value is strictly smaller (i.e., maintain a min-register).
inline void atomic_update_distance_min(
    std::atomic<std::uint32_t> &slot,
    float new_distance) {
    std::uint32_t new_bits = float_to_bits(new_distance);
    std::uint32_t observed = slot.load(std::memory_order_acquire);
    while (bits_to_float(observed) > new_distance) {
        if (slot.compare_exchange_weak(
                observed, new_bits,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Step 4: kClassify task handler.
//
// Evaluates whether the cell centre is inside the particle field and records
// the result in the cell_classification_ side-car.  If inside, initialises
// outside_distance_bits_ to +inf and enqueues a kDistanceUpdate task.
// ---------------------------------------------------------------------------

inline void process_classify_task(
    const RefinementTask &task,
    ClosureWorkerState &worker) {

    const std::size_t idx = task.cell_index;
    RefinementContext &context = worker.context;

    if (idx >= context.size()) {
        return;
    }

    // Only classify leaf cells; internal cells are not relevant.
    if (!closure_is_leaf(context.cells()[idx])) {
        return;
    }

    const OctreeCell &cell = context.cells()[idx];

    std::vector<std::size_t> contributors;
    context.copy_contributors_for_cell(idx, contributors);

    // Evaluate the SPH density at the cell centre.
    const Vector3d centre = cell.bounds.center();
    double field_value = 0.0;
    for (std::size_t pi : contributors) {
        const Vector3d d = {
            centre.x - worker.positions[pi].x,
            centre.y - worker.positions[pi].y,
            centre.z - worker.positions[pi].z,
        };
        const double r = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        field_value += evaluate_wendland_c2(
            r, worker.smoothing_lengths[pi], true);
    }

    // Classification: inside when field_value >= isovalue (matches the
    // classification logic used by classify_occupied_solid_leaves).
    const bool inside = (field_value >= worker.config.isovalue);
    context.center_value_bits()[idx].store(
        double_to_bits(field_value), std::memory_order_release);
    context.cell_classification()[idx].store(
        inside ? std::uint8_t{1U} : std::uint8_t{0U},
        std::memory_order_release);
    refresh_occupancy_state_for_cell_and_neighbors(idx, worker);

    if (!inside) {
        // Outside cells: distance is trivially 0 (they ARE the outside);
        // leave outside_distance_bits_ unchanged (it will be read as-is
        // by any kDistanceUpdate that reaches a neighbour).
        return;
    }

    // Inside cell: initialise outside distance to +inf so a kDistanceUpdate
    // can overwrite it with a finite estimate.
    context.outside_distance_bits()[idx].store(
        kPlusInfBits, std::memory_order_release);

    // Only enqueue a distance update for cells that could still be refined.
    const double target = worker.config.thickening_band_target_leaf_size;
    if (target <= 0.0) {
        return;
    }
    const double edge = closure_cell_edge_length(cell);
    if (cell.depth >= worker.config.max_depth || edge <= target) {
        return;
    }

    // Push a kDistanceUpdate for this inside cell.
    worker.queue.push(
        {idx, 0U, 0U, RefinementTaskKind::kDistanceUpdate},
        worker.worker_id);
}

// ---------------------------------------------------------------------------
// Step 5: kDistanceUpdate task handler.
//
// Walks the six face-neighbour cells via the morton-based closure walk.
// If any neighbour is classified outside, computes the Euclidean distance
// from this cell centre to that neighbour's centre.  If the minimum outside
// distance is within [thickening_radius + halo] and the cell is still above
// target leaf size, enqueues a kRefine task.
// ---------------------------------------------------------------------------

inline void process_distance_update_task(
    const RefinementTask &task,
    ClosureWorkerState &worker) {

    const std::size_t idx = task.cell_index;
    RefinementContext &context = worker.context;

    if (idx >= context.size()) {
        return;
    }

    // Only process leaf cells.
    if (!closure_is_leaf(context.cells()[idx])) {
        return;
    }

    const double target = worker.config.thickening_band_target_leaf_size;
    const double radius = worker.config.thickening_radius;
    if (target <= 0.0 || radius <= 0.0) {
        return;
    }

    // If classification says outside, nothing to do.
    const std::uint8_t cls =
        context.cell_classification()[idx].load(std::memory_order_acquire);
    if (cls != 1U) {
        return;
    }

    const OctreeCell &cell = context.cells()[idx];

    // Walk six face-neighbours.  For each, if it's classified as outside,
    // compute the Euclidean distance between cell centres.
    const std::uint32_t max_depth = worker.config.max_depth;
    const ChunkedArena<OctreeCell> &all_cells = context.cells();

    // Decode the cell's fine-grid origin.
    std::uint32_t cell_x = 0U;
    std::uint32_t cell_y = 0U;
    std::uint32_t cell_z = 0U;
    morton_decode_3d(cell.morton_key, cell_x, cell_y, cell_z);
    const std::uint32_t cell_shift = max_depth - cell.depth;
    const std::uint32_t fine_min_x = cell_x << cell_shift;
    const std::uint32_t fine_min_y = cell_y << cell_shift;
    const std::uint32_t fine_min_z = cell_z << cell_shift;
    const std::uint32_t span = 1U << cell_shift;
    const std::uint32_t half_span = span / 2U > 0U ? span / 2U : 1U;

    // Face sample: centre of each face.
    const std::int64_t fine_max = static_cast<std::int64_t>(
        worker.config.base_resolution << max_depth);
    const Vector3d centre = cell.bounds.center();

    float min_outside_distance = std::numeric_limits<float>::infinity();

    struct FaceInfo {
        int axis;
        bool positive;
    };
    const FaceInfo faces[6] = {
        {0, false}, {0, true},
        {1, false}, {1, true},
        {2, false}, {2, true},
    };
    for (const FaceInfo &face : faces) {
        const std::int64_t base_x =
            static_cast<std::int64_t>(fine_min_x);
        const std::int64_t base_y =
            static_cast<std::int64_t>(fine_min_y);
        const std::int64_t base_z =
            static_cast<std::int64_t>(fine_min_z);
        std::int64_t probe[3] = {
            base_x + static_cast<std::int64_t>(half_span),
            base_y + static_cast<std::int64_t>(half_span),
            base_z + static_cast<std::int64_t>(half_span),
        };
        // Move probe just outside the face.
        if (face.positive) {
            probe[face.axis] = base_x + static_cast<std::int64_t>(span);
            if (face.axis == 1) {
                probe[face.axis] =
                    base_y + static_cast<std::int64_t>(span);
            }
            if (face.axis == 2) {
                probe[face.axis] =
                    base_z + static_cast<std::int64_t>(span);
            }
        } else {
            probe[face.axis] = (face.axis == 0) ? base_x - 1
                             : (face.axis == 1) ? base_y - 1
                                                : base_z - 1;
        }
        if (probe[0] < 0 || probe[0] >= fine_max ||
            probe[1] < 0 || probe[1] >= fine_max ||
            probe[2] < 0 || probe[2] >= fine_max) {
            continue;
        }
        const std::uint64_t target_morton = morton_encode_3d(
            static_cast<std::uint32_t>(probe[0]),
            static_cast<std::uint32_t>(probe[1]),
            static_cast<std::uint32_t>(probe[2]));
        const std::size_t neighbour_idx =
            closure_locate_cell_for_target(
                idx, all_cells, target_morton, max_depth,
                worker.config.base_resolution);
        if (neighbour_idx == std::numeric_limits<std::size_t>::max()) {
            continue;
        }
        if (neighbour_idx >= context.size()) {
            continue;
        }
        // Check if this neighbour is outside (classification == 0).
        // Unset (255) is treated as outside for distance purposes.
        const std::uint8_t n_cls =
            context.cell_classification()[neighbour_idx].load(
                std::memory_order_acquire);
        if (n_cls == 1U) {
            // Inside neighbour: not useful for outside distance.
            continue;
        }
        // Compute distance between cell centres.
        const Vector3d n_centre = all_cells[neighbour_idx].bounds.center();
        const double dx = centre.x - n_centre.x;
        const double dy = centre.y - n_centre.y;
        const double dz = centre.z - n_centre.z;
        const float dist = static_cast<float>(
            std::sqrt(dx * dx + dy * dy + dz * dz));
        if (dist < min_outside_distance) {
            min_outside_distance = dist;
        }
    }

    if (!std::isfinite(min_outside_distance)) {
        // No outside neighbour found within one face step — use a fallback
        // that still permits refinement: cell_size is the coarsest sensible
        // estimate (distance to next neighbour is at most 1 cell).
        const double edge = closure_cell_edge_length(cell);
        min_outside_distance = static_cast<float>(edge);
    }

    // Write the minimum outside distance back to the side-car.
    atomic_update_distance_min(
        context.outside_distance_bits()[idx], min_outside_distance);

    // Decide whether to enqueue a kRefine task.
    // Criterion: cell is inside, outside_distance <= thickening_radius + halo,
    // and cell is still above the target leaf size.
    const double edge = closure_cell_edge_length(cell);
    if (cell.depth >= max_depth || edge <= target) {
        return;
    }

    const double refinement_halo = radius + 2.0 * target;
    if (static_cast<double>(min_outside_distance) > refinement_halo) {
        return;
    }

    // Raise required depth to trigger the balance-split path in the kRefine
    // handler (process_closure_task).
    context.raise_required_depth_to(idx, cell.depth + 1U);

    if (context.mark_queued(idx)) {
        worker.queue.push(
            {idx, context.get_required_depth(idx), 0U,
             RefinementTaskKind::kRefine},
            worker.worker_id);
    }
}

inline void run_closure_worker_loop(
    ClosureWorkerState &worker,
    ClosureStatusReporter &reporter) {
    RefinementTask task;
    RefinementTask pending_startup_task;
    std::size_t processed_count = 0U;
    bool have_pending_startup_task = false;
    bool task_counts_as_in_flight = false;
    while (true) {
        poll_closure_cancellation(worker, processed_count, 4096U);
        if (worker.queue.try_claim_report(
                worker.config.table_cadence_seconds)) {
            reporter.maybe_emit_periodic();
        }
        bool got_task = false;
        if (have_pending_startup_task) {
            task = pending_startup_task;
            got_task = true;
            have_pending_startup_task = false;
            task_counts_as_in_flight = true;
        } else {
            ScopedNsTimer pop_timer(worker.profiler.ns_pop_wait);
            got_task = worker.queue.pop(worker.worker_id, task);
            task_counts_as_in_flight = got_task;
        }
        if (!got_task) {
            break;
        }
        worker.profiler.tasks_popped.fetch_add(1U, std::memory_order_relaxed);
        // Dispatch by task kind and count per-kind pops for the status table.
        switch (task.kind) {
            case RefinementTaskKind::kClassify:
                worker.run_stats.classify_pop_count.fetch_add(
                    1U, std::memory_order_relaxed);
                process_classify_task(task, worker);
                break;
            case RefinementTaskKind::kDistanceUpdate:
                worker.run_stats.distance_update_pop_count.fetch_add(
                    1U, std::memory_order_relaxed);
                process_distance_update_task(task, worker);
                break;
            case RefinementTaskKind::kRefine:
            default:
                worker.run_stats.refine_pop_count.fetch_add(
                    1U, std::memory_order_relaxed);
                process_closure_task(task, worker);
                break;
        }
        if (worker.startup_seed_cells != nullptr &&
            worker.startup_seed_cursor != nullptr) {
            while (true) {
                const std::size_t ordinal = worker.startup_seed_cursor->fetch_add(
                    1U, std::memory_order_acq_rel);
                if (ordinal >= worker.startup_seed_cells->size()) {
                    break;
                }
                const std::size_t cell_index =
                    (*worker.startup_seed_cells)[ordinal];
                if (cell_index >= worker.context.size()) {
                    continue;
                }
                if (!worker.context.mark_queued(cell_index)) {
                    continue;
                }
                worker.queue.begin_external_task();
                pending_startup_task = {
                    cell_index,
                    worker.context.get_required_depth(cell_index),
                    0U,
                    RefinementTaskKind::kRefine,
                };
                have_pending_startup_task = true;
                break;
            }
        }
        if (task_counts_as_in_flight) {
            worker.queue.task_done();
        }
        maybe_shutdown_queue(worker);
        ++processed_count;
    }
}

inline void run_closure_queue(
    RefinementContext &context,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config,
    RefinementWorkQueue &queue,
    const std::vector<std::size_t> *startup_seed_cells = nullptr,
    std::size_t startup_seed_cursor_begin = 0U) {
    ClosureRunStats run_stats;
    ClosureProfiler profiler;
    const auto run_wallclock_start = std::chrono::steady_clock::now();
    ClosureStatusReporter reporter(
        config,
        context,
        queue,
        run_stats);

    const std::uint32_t worker_count = std::max(1U, config.worker_count);
    std::atomic<std::size_t> startup_seed_cursor{startup_seed_cursor_begin};
    // Queue must be initialized by the caller before seeding tasks so we do
    // not discard already-enqueued work here.

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::uint32_t worker_index = 0; worker_index < worker_count;
         ++worker_index) {
        workers.emplace_back([
                                  &context,
                                  &positions,
                                  &smoothing_lengths,
                                  &config,
                                  &queue,
                                  &run_stats,
                                  &reporter,
                                  &profiler,
                                  startup_seed_cells,
                                  &startup_seed_cursor,
                                  worker_index
                              ]() {
            ClosureWorkerState worker = {
                context,
                positions,
                smoothing_lengths,
                config,
                queue,
                startup_seed_cells,
                startup_seed_cells != nullptr ? &startup_seed_cursor : nullptr,
                worker_index,
                run_stats,
                profiler,
            };
            run_closure_worker_loop(worker, reporter);
        });
    }
    for (std::thread &thread : workers) {
        thread.join();
    }

    reporter.finish();

    // Emit a readable profile summary at end-of-run. Times are cumulative
    // worker times, not deduplicated wall-clock time, so percentages are
    // relative to wall_seconds * worker_count.
    const auto run_wallclock_end = std::chrono::steady_clock::now();
    const double wall_seconds =
        std::chrono::duration<double>(
            run_wallclock_end - run_wallclock_start).count();
    auto ns_to_s = [](std::uint64_t ns) {
        return static_cast<double>(ns) * 1e-9;
    };
    const double total_worker_seconds =
        wall_seconds * static_cast<double>(worker_count);
    auto percent_of_worker_total = [&](double seconds) {
        if (total_worker_seconds <= 0.0) {
            return 0.0;
        }
        return 100.0 * seconds / total_worker_seconds;
    };
    auto print_time_row = [&](const char *label, double seconds) {
        meshmerizer_log_detail::print_indented_status(
            "closure_report_task_times [%s]: *** %24s: %10.2f ms (%6.2f %%)\n",
            config.phase_name.c_str(),
            label,
            seconds * 1000.0,
            percent_of_worker_total(seconds));
    };
    auto print_count_row = [&](const char *label,
                               std::uint64_t count,
                               std::uint64_t /*denominator*/) {
        meshmerizer_log_detail::print_indented_status(
            "closure_report_task_counts [%s]: *** %23s: %12llu\n",
            config.phase_name.c_str(),
            label,
            static_cast<unsigned long long>(count));
    };

    meshmerizer_log_detail::print_indented_status(
        "closure_report_summary [%s]: wall=%.3fs workers=%u "
        "total_worker_s=%.3f\n",
        config.phase_name.c_str(),
        wall_seconds,
        worker_count,
        total_worker_seconds);

    meshmerizer_log_detail::print_indented_status(
        "closure_report_task_times [%s]: *** CPU time spent in "
        "closure task categories:\n",
        config.phase_name.c_str());
    print_time_row("pop wait", ns_to_s(profiler.ns_pop_wait.load()));
    print_time_row("leaf evaluation", ns_to_s(profiler.ns_leaf_eval.load()));
    print_time_row("apply split", ns_to_s(profiler.ns_apply_split.load()));
    print_time_row(
        "neighbor wake", ns_to_s(profiler.ns_neighbor_wake.load()));
    print_time_row(
        "required depth raise",
        ns_to_s(profiler.ns_required_depth_raise.load()));
    print_time_row(
        "internal descent", ns_to_s(profiler.ns_internal_descent.load()));
    print_time_row(
        "storage lock wait",
        ns_to_s(profiler.ns_storage_lock_wait.load()));
    print_time_row(
        "storage lock held",
        ns_to_s(profiler.ns_storage_lock_held.load()));
    print_time_row(
        "local DFS total", ns_to_s(profiler.ns_local_dfs_total.load()));
    print_time_row("total worker", total_worker_seconds);

    const std::uint64_t tasks_popped = profiler.tasks_popped.load();
    meshmerizer_log_detail::print_indented_status(
        "closure_report_task_counts [%s]: *** Event counts from closure "
        "scheduler:\n",
        config.phase_name.c_str());
    print_count_row("tasks popped", tasks_popped, tasks_popped);
    print_count_row(
        "retired no split",
        profiler.tasks_retired_no_split.load(),
        tasks_popped);
    print_count_row(
        "stale/skipped", profiler.tasks_stale_or_skipped.load(), tasks_popped);
    print_count_row("leaves evaluated", profiler.leaves_evaluated.load(),
                    profiler.leaves_evaluated.load());
    print_count_row(
        "splits", profiler.splits.load(), profiler.leaves_evaluated.load());
    print_count_row(
        "internal wakes", profiler.internal_wakes.load(), tasks_popped);
    print_count_row(
        "neighbor wakes", profiler.neighbor_wakes_pushed.load(), tasks_popped);
    print_count_row(
        "required depth raises",
        profiler.required_depth_raises.load(),
        tasks_popped);
    print_count_row(
        "local pushes", profiler.local_stack_pushes.load(), tasks_popped);
    print_count_row(
        "spill pushes", profiler.spill_pushes.load(), tasks_popped);
    print_count_row(
        "internal visited", profiler.internal_cells_visited.load(),
        tasks_popped);
    print_count_row(
        "non-leaf retired",
        profiler.non_leaf_retired_immediately.load(),
        tasks_popped);

    const RefinementWorkQueueStats q = queue.stats();
    print_count_row("queue pushed", q.push_count, q.push_count);
    print_count_row("queue popped", q.pop_count, q.push_count);
    print_count_row("queue peak", q.high_watermark, q.push_count);
    print_count_row("queue initial", q.initial_queue_size, q.push_count);
    print_count_row("steal attempts", q.steal_attempts, q.steal_attempts);
    print_count_row("steal successes", q.steal_successes, q.steal_attempts);
    print_count_row("idle sleeps", q.idle_sleeps, q.pop_count);
}

}  // namespace

std::pair<std::vector<OctreeCell>, std::vector<std::size_t>>
refine_with_closure(
    std::vector<OctreeCell> initial_cells,
    std::vector<std::size_t> initial_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config) {
    if (initial_cells.empty()) {
        return {{}, {}};
    }

    RefinementContext context(initial_cells, initial_contributors);
    context.sync_cell_state_size();

    RefinementWorkQueue queue;
    const std::uint32_t worker_count = std::max(1U, config.worker_count);
    queue.initialize(worker_count);
    for (std::size_t cell_index = 0; cell_index < context.size(); ++cell_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_index);
        if (context.raise_required_depth_to(
            cell_index,
            initial_cells[cell_index].depth)) {
        }
    }

    std::vector<std::size_t> seed_cells;
    seed_cells.reserve(context.size());
    for (std::size_t cell_index = 0; cell_index < context.size(); ++cell_index) {
        if (!context.cells()[cell_index].is_leaf) {
            continue;
        }
        seed_cells.push_back(cell_index);
    }
    if (!seed_queue_from_cells(
            context,
            queue,
            worker_count,
            seed_cells)) {
        return {
            std::move(initial_cells),
            std::move(initial_contributors),
        };
    }
    queue.capture_initial_queue_size();

    run_closure_queue(
        context,
        positions,
        smoothing_lengths,
        config,
        queue);

    // Materialize arena state back into the caller-visible vectors and
    // return them. Reconciles ``is_leaf`` from ``child_begin`` for any
    // cells split during the run.
    meshmerizer_log_detail::print_status(
        config.status_operation,
        config.status_function,
        "materializing refined tree (%zu cells, %zu contributors)\n",
        context.cells().size(),
        context.contributors().size());
    const auto materialize_start = std::chrono::steady_clock::now();
    std::vector<OctreeCell> out_cells;
    std::vector<std::size_t> out_contributors;
    context.materialize_into(out_cells, out_contributors);
    const double materialize_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - materialize_start).count();
    meshmerizer_log_detail::print_status(
        config.status_operation,
        config.status_function,
        "materialized refined tree in %.3f s\n",
        materialize_seconds);
    return {std::move(out_cells), std::move(out_contributors)};
}

bool refine_surface_band_with_closure(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config,
    double max_surface_leaf_size) {
    if (max_surface_leaf_size <= 0.0) {
        return false;
    }

    // Phase-A: surface-band loop fusion. Thread the target leaf size into
    // the closure config so workers self-enqueue any newly-published child
    // that still meets the surface-band criterion. With this set, a single
    // closure run reaches convergence and the outer ``while`` loop in the
    // pipeline becomes a single call.
    RefinementClosureConfig effective_config = config;
    effective_config.surface_band_target_leaf_size = max_surface_leaf_size;

    const std::size_t initial_cell_count = all_cells.size();

    RefinementContext context(all_cells, all_contributors);
    context.sync_cell_state_size();

    RefinementWorkQueue queue;
    const std::uint32_t worker_count =
        std::max(1U, effective_config.worker_count);
    queue.initialize(worker_count);
    std::vector<std::size_t> matching_cells;
    auto target_depth_for_surface_cell = [&](const OctreeCell &cell) {
        std::uint32_t target_depth = cell.depth;
        double edge = closure_cell_edge_length(cell);
        while (target_depth < effective_config.max_depth &&
               edge > max_surface_leaf_size) {
            edge *= 0.5;
            ++target_depth;
        }
        return target_depth;
    };
    for (std::size_t cell_index = 0; cell_index < context.size(); ++cell_index) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_index);
        const OctreeCell &cell = context.cells()[cell_index];
        if (!cell.is_leaf || !cell.has_surface ||
            cell.depth >= effective_config.max_depth) {
            continue;
        }
        if (closure_cell_edge_length(cell) <= max_surface_leaf_size) {
            continue;
        }
        context.raise_required_depth_to(
            cell_index, target_depth_for_surface_cell(cell));
        matching_cells.push_back(cell_index);
    }

    const std::vector<std::size_t> seed_cells =
        select_surface_band_seed_cells(
            context,
            worker_count,
            max_surface_leaf_size,
            &matching_cells);
    const bool queued_any =
        seed_queue_from_cells(context, queue, worker_count, seed_cells);

    if (!queued_any) {
        return false;
    }
    queue.capture_initial_queue_size();

    run_closure_queue(
        context,
        positions,
        smoothing_lengths,
        effective_config,
        queue);
    const std::size_t remaining_surface_eligible =
        count_remaining_surface_band_eligible(
            context, max_surface_leaf_size, effective_config.max_depth);
    context.materialize_into(all_cells, all_contributors);
    meshmerizer_log_detail::print_debug_status(
        effective_config.status_operation,
        effective_config.status_function,
        "surface-band closure summary: cells_before=%zu cells_after=%zu remaining_eligible=%zu\n",
        initial_cell_count,
        all_cells.size(),
        remaining_surface_eligible);
    return all_cells.size() > initial_cell_count;
}

bool refine_thickening_band_with_closure(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config,
    double target_leaf_size,
    const std::vector<std::size_t> &seed_cell_indices,
    const std::vector<std::uint32_t> &seed_target_depths,
    const std::vector<std::uint8_t> *initial_inside_flags,
    const std::vector<double> *initial_center_values,
    const std::vector<std::uint8_t> *initial_occupancy_states,
    std::vector<std::uint8_t> *dirty_cells,
    std::vector<std::uint8_t> *classified_inside_flags,
    std::vector<double> *classified_center_values,
    std::vector<std::uint8_t> *classified_occupancy_states) {
    if (target_leaf_size <= 0.0) {
        return false;
    }

    const std::size_t initial_cell_count = all_cells.size();

    // -------------------------------------------------------------------------
    // B2 incremental task-based path.
    //
    // When ``config.thickening_band_target_leaf_size > 0``, the thickening
    // radius and target size have been threaded into the config by the caller.
    // We seed kClassify tasks for all current leaf cells; the incremental chain
    //   kClassify → kDistanceUpdate → kRefine → kClassify(children) → ...
    // drives itself to convergence without any outer loop.
    // -------------------------------------------------------------------------
    if (config.thickening_band_target_leaf_size > 0.0) {
        RefinementContext context(all_cells, all_contributors);
        context.sync_cell_state_size();
        context.initialize_thickening_state(
            initial_inside_flags,
            initial_center_values,
            initial_occupancy_states);

        RefinementWorkQueue queue;
        const std::uint32_t worker_count = std::max(1U, config.worker_count);
        queue.initialize(worker_count);

        // Seed kClassify from the caller-provided frontier when available.
        // Fall back to the old whole-tree leaf seed only when the caller does
        // not provide a bounded frontier. The final solid classification is
        // still rebuilt by the caller after this phase, so classifying
        // max-depth / already-small leaves only creates no-op work.
        bool seeded_any = false;
        if (!seed_cell_indices.empty()) {
            for (std::size_t ordinal = 0; ordinal < seed_cell_indices.size();
                 ++ordinal) {
                const std::size_t cell_index = seed_cell_indices[ordinal];
                meshmerizer_cancel_detail::poll_for_cancellation_serial(
                    cell_index);
                if (cell_index >= context.size()) {
                    continue;
                }
                const OctreeCell &cell = context.cells()[cell_index];
                if (!cell.is_leaf) {
                    continue;
                }
                if (cell.depth >= config.max_depth ||
                    closure_cell_edge_length(cell) <= target_leaf_size) {
                    continue;
                }
                const std::uint32_t seed_depth =
                    (ordinal < seed_target_depths.size())
                        ? seed_target_depths[ordinal]
                        : static_cast<std::uint32_t>(config.max_depth);
                queue.push(
                    {cell_index, seed_depth, 0U,
                     RefinementTaskKind::kClassify},
                    static_cast<std::uint32_t>(
                        cell_index % std::max(1U, worker_count)));
                seeded_any = true;
            }
        } else {
            for (std::size_t cell_index = 0; cell_index < context.size();
                 ++cell_index) {
                meshmerizer_cancel_detail::poll_for_cancellation_serial(
                    cell_index);
                const OctreeCell &cell = context.cells()[cell_index];
                if (!cell.is_leaf) {
                    continue;
                }
                if (cell.depth >= config.max_depth ||
                    closure_cell_edge_length(cell) <= target_leaf_size) {
                    continue;
                }
                queue.push(
                    {cell_index, 0U, 0U, RefinementTaskKind::kClassify},
                    static_cast<std::uint32_t>(
                        cell_index % std::max(1U, worker_count)));
                seeded_any = true;
            }
        }

        if (!seeded_any) {
            return false;
        }
        queue.capture_initial_queue_size();

        run_closure_queue(
            context,
            positions,
            smoothing_lengths,
            config,
            queue);

        const std::size_t remaining_thickening_eligible =
            count_remaining_thickening_band_eligible(
                context,
                target_leaf_size,
                config.max_depth);

        if (classified_inside_flags != nullptr ||
            classified_center_values != nullptr ||
            classified_occupancy_states != nullptr) {
            context.materialize_thickening_state(
                classified_inside_flags,
                classified_center_values,
                classified_occupancy_states);
        }
        context.materialize_into(all_cells, all_contributors);
        meshmerizer_log_detail::print_debug_status(
            config.status_operation,
            config.status_function,
            "thickening-band closure summary: cells_before=%zu cells_after=%zu remaining_eligible=%zu\n",
            initial_cell_count,
            all_cells.size(),
            remaining_thickening_eligible);
        if (dirty_cells != nullptr) {
            dirty_cells->assign(all_cells.size(), 0U);
            if (seed_cell_indices.empty()) {
                std::fill(dirty_cells->begin(), dirty_cells->end(), 1U);
            } else {
                for (std::size_t cell_index : seed_cell_indices) {
                    if (cell_index < dirty_cells->size()) {
                        (*dirty_cells)[cell_index] = 1U;
                    }
                }
                for (std::size_t cell_index = initial_cell_count;
                     cell_index < all_cells.size();
                     ++cell_index) {
                    (*dirty_cells)[cell_index] = 1U;
                }
            }
        }
        return all_cells.size() > initial_cell_count;
    }

    // -------------------------------------------------------------------------
    // LEGACY PASS LOOP: pre-B2 seed-based path kept for fall-back / reference.
    // Reached when thickening_band_target_leaf_size == 0.0 (i.e. the caller
    // did not opt into the B2 incremental chain).
    // -------------------------------------------------------------------------

    const std::uint32_t kNoTargetDepth =
        std::numeric_limits<std::uint32_t>::max();

    RefinementContext context(all_cells, all_contributors);
    context.sync_cell_state_size();

    auto mark_dirty = [&](std::size_t cell_index) {
        if (dirty_cells == nullptr) {
            return;
        }
        if (dirty_cells->size() <= cell_index) {
            dirty_cells->resize(cell_index + 1U, 0U);
        }
        (*dirty_cells)[cell_index] = 1U;
    };

    RefinementWorkQueue queue;
    const std::uint32_t worker_count = std::max(1U, config.worker_count);
    queue.initialize(worker_count);
    std::vector<std::size_t> matching_cells;
    for (std::size_t ordinal = 0; ordinal < seed_cell_indices.size(); ++ordinal) {
        const std::size_t cell_index = seed_cell_indices[ordinal];
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_index);
        if (cell_index >= context.size()) {
            continue;
        }
        const OctreeCell &cell = context.cells()[cell_index];
        if (!cell.is_leaf || cell.depth >= config.max_depth) {
            continue;
        }
        const std::uint32_t target_depth =
            cell_index < seed_target_depths.size() ?
                seed_target_depths[cell_index] :
                kNoTargetDepth;
        if (target_depth == kNoTargetDepth) {
            continue;
        }
        if (closure_cell_edge_length(cell) <= target_leaf_size &&
            cell.depth >= target_depth) {
            continue;
        }
        context.raise_required_depth_to(cell_index, target_depth);
        matching_cells.push_back(cell_index);
        mark_dirty(cell_index);
    }

    const std::vector<std::size_t> seed_cells =
        select_targeted_seed_cells(
            context,
            worker_count,
            matching_cells,
            &matching_cells);
    const bool queued_any =
        seed_queue_from_cells(context, queue, worker_count, seed_cells);

    if (!queued_any) {
        return false;
    }
    queue.capture_initial_queue_size();

    run_closure_queue(
        context,
        positions,
        smoothing_lengths,
        config,
        queue);

    context.materialize_into(all_cells, all_contributors);
    if (dirty_cells != nullptr) {
        dirty_cells->resize(all_cells.size(), 1U);
    }
    return all_cells.size() > initial_cell_count;
}

bool refine_cells_to_next_depth_with_closure(
    std::vector<OctreeCell> &all_cells,
    std::vector<std::size_t> &all_contributors,
    const std::vector<Vector3d> &positions,
    const std::vector<double> &smoothing_lengths,
    const RefinementClosureConfig &config,
    const std::vector<std::size_t> &seed_cell_indices) {
    if (seed_cell_indices.empty()) {
        return false;
    }

    const std::size_t initial_cell_count = all_cells.size();

    RefinementContext context(all_cells, all_contributors);
    context.sync_cell_state_size();

    RefinementWorkQueue queue;
    const std::uint32_t worker_count = std::max(1U, config.worker_count);
    queue.initialize(worker_count);
    std::vector<std::size_t> matching_cells;
    for (std::size_t cell_index : seed_cell_indices) {
        meshmerizer_cancel_detail::poll_for_cancellation_serial(cell_index);
        if (cell_index >= context.size()) {
            continue;
        }
        const OctreeCell &cell = context.cells()[cell_index];
        if (!cell.is_leaf || cell.depth >= config.max_depth) {
            continue;
        }
        if (!context.raise_required_depth_to(cell_index, cell.depth + 1U)) {
            continue;
        }
        matching_cells.push_back(cell_index);
    }

    const std::vector<std::size_t> seed_cells =
        select_targeted_seed_cells(
            context,
            worker_count,
            matching_cells,
            &matching_cells);
    const bool queued_any =
        seed_queue_from_cells(context, queue, worker_count, seed_cells);

    if (!queued_any) {
        return false;
    }
    queue.capture_initial_queue_size();

    run_closure_queue(
        context,
        positions,
        smoothing_lengths,
        config,
        queue);
    context.materialize_into(all_cells, all_contributors);
    return all_cells.size() > initial_cell_count;
}
