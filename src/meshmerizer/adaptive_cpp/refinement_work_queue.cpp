#include "refinement_work_queue.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace {

inline std::uint32_t clamp_worker_id(
    std::uint32_t worker_id,
    std::size_t worker_count) {
    if (worker_count == 0U) {
        return 0U;
    }
    if (worker_id >= worker_count) {
        return static_cast<std::uint32_t>(worker_count - 1U);
    }
    return worker_id;
}

}  // namespace

void RefinementWorkQueue::initialize(std::uint32_t worker_count) {
    if (worker_count == 0U) {
        worker_count = 1U;
    }
    deques_.clear();
    deques_.reserve(worker_count);
    for (std::uint32_t i = 0; i < worker_count; ++i) {
        deques_.push_back(std::make_unique<WorkerDeque>());
    }
    queued_tasks_.store(0U, std::memory_order_relaxed);
    sleeping_workers_.store(0U, std::memory_order_relaxed);
    in_flight_tasks_.store(0U, std::memory_order_relaxed);
    shutdown_requested_.store(false, std::memory_order_relaxed);
    push_count_.store(0U, std::memory_order_relaxed);
    pop_count_.store(0U, std::memory_order_relaxed);
    high_watermark_.store(0U, std::memory_order_relaxed);
    initial_queue_size_.store(0U, std::memory_order_relaxed);
    steal_attempts_.store(0U, std::memory_order_relaxed);
    steal_successes_.store(0U, std::memory_order_relaxed);
    idle_sleeps_.store(0U, std::memory_order_relaxed);
    reset_report_slot_clock();
    next_progress_bucket_.store(1U, std::memory_order_relaxed);
    final_report_emitted_.store(false, std::memory_order_relaxed);
}

bool RefinementWorkQueue::push(const RefinementTask &task) {
    return push(task, 0U);
}

bool RefinementWorkQueue::push(
    const RefinementTask &task,
    std::uint32_t preferred_worker) {
    if (shutdown_requested_.load(std::memory_order_acquire)) {
        return false;
    }
    if (deques_.empty()) {
        initialize(1U);
    }
    const std::uint32_t worker_id = clamp_worker_id(preferred_worker, deques_.size());
    {
        std::lock_guard<std::mutex> lock(deques_[worker_id]->mutex);
        deques_[worker_id]->tasks.push_back(task);
    }
    const std::size_t qsize =
        queued_tasks_.fetch_add(1U, std::memory_order_release) + 1U;
    const std::size_t pushed =
        push_count_.fetch_add(1U, std::memory_order_relaxed) + 1U;
    (void)pushed;
    std::size_t prev = high_watermark_.load(std::memory_order_relaxed);
    while (qsize > prev && !high_watermark_.compare_exchange_weak(
               prev, qsize, std::memory_order_relaxed)) {
    }
    // Only pay the condvar wake cost if at least one worker may be
    // sleeping. Under load every worker is busy and this is a no-op,
    // which avoids the per-push thundering-herd.
    if (sleeping_workers_.load(std::memory_order_acquire) > 0U) {
        idle_condition_.notify_one();
    }
    return true;
}

std::size_t RefinementWorkQueue::push_batch(
    const std::vector<RefinementTask> &tasks,
    std::uint32_t preferred_worker) {
    if (tasks.empty()) {
        return 0U;
    }
    if (shutdown_requested_.load(std::memory_order_acquire)) {
        return 0U;
    }
    if (deques_.empty()) {
        initialize(1U);
    }
    const std::uint32_t worker_id = clamp_worker_id(preferred_worker, deques_.size());
    {
        std::lock_guard<std::mutex> lock(deques_[worker_id]->mutex);
        for (const RefinementTask &task : tasks) {
            deques_[worker_id]->tasks.push_back(task);
        }
    }
    const std::size_t qsize =
        queued_tasks_.fetch_add(tasks.size(), std::memory_order_release) +
        tasks.size();
    push_count_.fetch_add(tasks.size(), std::memory_order_relaxed);
    std::size_t prev = high_watermark_.load(std::memory_order_relaxed);
    while (qsize > prev && !high_watermark_.compare_exchange_weak(
               prev, qsize, std::memory_order_relaxed)) {
    }
    // Same elision as ``push``: skip notify if nobody is waiting.
    // ``notify_all`` is needed (rather than ``notify_one``) because a
    // batch may carry more tasks than the single waiter that ``push``
    // would wake.
    if (sleeping_workers_.load(std::memory_order_acquire) > 0U) {
        idle_condition_.notify_all();
    }
    return tasks.size();
}

std::size_t RefinementWorkQueue::approximate_queue_size_locked_unsafe() const {
    std::size_t total = 0U;
    for (const auto &deque : deques_) {
        std::lock_guard<std::mutex> lock(deque->mutex);
        total += deque->tasks.size();
    }
    return total;
}

bool RefinementWorkQueue::pop_local_(
    std::uint32_t worker_id,
    RefinementTask &task) {
    WorkerDeque &deque = *deques_[worker_id];
    std::lock_guard<std::mutex> lock(deque.mutex);
    if (deque.tasks.empty()) {
        return false;
    }
    task = deque.tasks.back();
    deque.tasks.pop_back();
    return true;
}

bool RefinementWorkQueue::steal_(
    std::uint32_t thief_id,
    RefinementTask &task) {
    const std::size_t worker_count = deques_.size();
    if (worker_count <= 1U) {
        return false;
    }
    steal_attempts_.fetch_add(1U, std::memory_order_relaxed);
    for (std::size_t offset = 1U; offset < worker_count; ++offset) {
        const std::uint32_t victim = static_cast<std::uint32_t>((thief_id + offset) % worker_count);
        WorkerDeque &deque = *deques_[victim];
        std::lock_guard<std::mutex> lock(deque.mutex);
        if (deque.tasks.empty()) {
            continue;
        }
        task = deque.tasks.front();
        deque.tasks.pop_front();
        steal_successes_.fetch_add(1U, std::memory_order_relaxed);
        return true;
    }
    return false;
}

bool RefinementWorkQueue::pop(std::uint32_t worker_id, RefinementTask &task) {
    if (deques_.empty()) {
        initialize(1U);
    }
    worker_id = clamp_worker_id(worker_id, deques_.size());

    while (true) {
        if (shutdown_requested_.load(std::memory_order_acquire) && is_idle()) {
            return false;
        }

        if (pop_local_(worker_id, task) || steal_(worker_id, task)) {
            queued_tasks_.fetch_sub(1U, std::memory_order_relaxed);
            in_flight_tasks_.fetch_add(1U, std::memory_order_relaxed);
            pop_count_.fetch_add(1U, std::memory_order_relaxed);
            return true;
        }

        std::unique_lock<std::mutex> lock(idle_mutex_);
        const std::size_t sleeping_count =
            sleeping_workers_.fetch_add(1U, std::memory_order_acq_rel) + 1U;
        idle_sleeps_.fetch_add(1U, std::memory_order_relaxed);

        const bool no_pending_work = empty();
        const bool no_active_workers =
            in_flight_tasks_.load(std::memory_order_acquire) == 0U;
        const bool all_workers_sleeping = sleeping_count == deques_.size();
        if (no_pending_work && no_active_workers && all_workers_sleeping) {
            shutdown_requested_.store(true, std::memory_order_release);
            sleeping_workers_.fetch_sub(1U, std::memory_order_acq_rel);
            idle_condition_.notify_all();
            return false;
        }

        idle_condition_.wait(lock, [&]() {
            return shutdown_requested_.load(std::memory_order_acquire) || !empty();
        });
        sleeping_workers_.fetch_sub(1U, std::memory_order_acq_rel);

        if (shutdown_requested_.load(std::memory_order_acquire) && is_idle()) {
            return false;
        }
    }
}

void RefinementWorkQueue::task_done() {
    const std::size_t previous =
        in_flight_tasks_.fetch_sub(1U, std::memory_order_acq_rel);
    // The only reason a sleeping worker cares about a ``task_done`` event
    // is the termination predicate: when ``in_flight_tasks_`` drops to 0
    // and the queue is empty, the last sleepers must wake to observe the
    // idle condition and request shutdown. Outside that case, no waiter
    // benefits from a wake (push handles waker delivery for new work).
    if (previous == 1U &&
        queued_tasks_.load(std::memory_order_acquire) == 0U &&
        sleeping_workers_.load(std::memory_order_acquire) > 0U) {
        idle_condition_.notify_all();
    }
}

void RefinementWorkQueue::begin_external_task() {
    in_flight_tasks_.fetch_add(1U, std::memory_order_acq_rel);
}

void RefinementWorkQueue::shutdown() {
    shutdown_requested_.store(true, std::memory_order_release);
    idle_condition_.notify_all();
}

bool RefinementWorkQueue::try_shutdown_if_idle() {
    if (shutdown_requested_.load(std::memory_order_acquire)) {
        return false;
    }
    if (!is_idle()) {
        return false;
    }
    bool expected = false;
    const bool did_shutdown = shutdown_requested_.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel);
    if (did_shutdown) {
        idle_condition_.notify_all();
    }
    return did_shutdown;
}

bool RefinementWorkQueue::empty() const {
    return queued_tasks_.load(std::memory_order_acquire) == 0U;
}

bool RefinementWorkQueue::is_shutdown() const {
    return shutdown_requested_.load(std::memory_order_acquire);
}

bool RefinementWorkQueue::is_idle() const {
    return empty() && in_flight_tasks_.load(std::memory_order_acquire) == 0U;
}

RefinementWorkQueueStats RefinementWorkQueue::stats() const {
    RefinementWorkQueueStats out;
    out.push_count = push_count_.load(std::memory_order_relaxed);
    out.pop_count = pop_count_.load(std::memory_order_relaxed);
    out.in_flight_count = in_flight_tasks_.load(std::memory_order_relaxed);
    out.sleeping_worker_count = sleeping_workers_.load(std::memory_order_relaxed);
    out.queue_size = queued_tasks_.load(std::memory_order_relaxed);
    out.high_watermark = high_watermark_.load(std::memory_order_relaxed);
    out.initial_queue_size = initial_queue_size_.load(std::memory_order_relaxed);
    out.steal_attempts = steal_attempts_.load(std::memory_order_relaxed);
    out.steal_successes = steal_successes_.load(std::memory_order_relaxed);
    out.idle_sleeps = idle_sleeps_.load(std::memory_order_relaxed);
    return out;
}

bool RefinementWorkQueue::try_claim_report_slot(double cadence_seconds) {
    if (cadence_seconds <= 0.0) {
        return false;
    }

    const auto now = std::chrono::steady_clock::now();
    const double elapsed_seconds =
        std::chrono::duration<double>(now - report_epoch_).count();
    if (elapsed_seconds < cadence_seconds) {
        return false;
    }

    const auto elapsed_slots = static_cast<std::uint64_t>(
        std::floor(elapsed_seconds / cadence_seconds));
    std::uint64_t expected = next_report_slot_.load(std::memory_order_acquire);

    while (expected <= elapsed_slots) {
        if (next_report_slot_.compare_exchange_weak(
                expected,
                expected + 1U,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return true;
        }
    }

    return false;
}

bool RefinementWorkQueue::try_claim_report(double cadence_seconds) {
    (void)cadence_seconds;
    const std::size_t pop_count =
        pop_count_.load(std::memory_order_acquire);
    const std::size_t high_watermark =
        high_watermark_.load(std::memory_order_acquire);
    const std::size_t work_scale = std::max(pop_count, high_watermark);
    if (work_scale == 0U) {
        return false;
    }

    const std::size_t stride = std::max<std::size_t>(1U, work_scale / 100U);
    std::size_t expected = next_progress_bucket_.load(std::memory_order_acquire);
    while (pop_count >= expected) {
        const std::size_t next_threshold = pop_count + stride;
        if (next_progress_bucket_.compare_exchange_weak(
                expected,
                next_threshold,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return true;
        }
    }

    return false;
}

void RefinementWorkQueue::reset_report_slot_clock() {
    report_epoch_ = std::chrono::steady_clock::now();
    next_report_slot_.store(1U, std::memory_order_relaxed);
    next_progress_bucket_.store(1U, std::memory_order_relaxed);
}

std::uint64_t RefinementWorkQueue::next_report_slot_index() const {
    return next_report_slot_.load(std::memory_order_acquire);
}

bool RefinementWorkQueue::try_claim_final_report() const {
    bool expected = false;
    return final_report_emitted_.compare_exchange_strong(
        expected,
        true,
        std::memory_order_acq_rel,
        std::memory_order_acquire);
}

void RefinementWorkQueue::capture_initial_queue_size() {
    initial_queue_size_.store(
        queued_tasks_.load(std::memory_order_acquire),
        std::memory_order_release);
}
