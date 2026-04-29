#ifndef MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_WORK_QUEUE_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_WORK_QUEUE_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <memory>
#include <vector>

/**
 * @file refinement_work_queue.hpp
 * @brief Thread-safe task queue for integrated octree refinement closure.
 *
 * The queue transports lightweight refinement tasks between the closure
 * scheduler and worker logic. It does not own any geometric policy; the
 * authoritative state for a cell remains in the refinement context.
 */

/**
 * @brief Discriminator for the kind of work a RefinementTask represents.
 *
 * kRefine       – default; evaluate the cell and split if needed (existing
 *                 behaviour).
 * kClassify     – determine whether the cell is inside/outside the particle
 *                 field; write result to the per-cell classification side-car
 *                 and, if the cell is in-band, enqueue a kDistanceUpdate.
 * kDistanceUpdate – compute or update the outside distance for the cell and,
 *                 if within the thickening band and above target leaf size,
 *                 enqueue a kRefine task.
 * kOccupancyUpdate – refresh inside/boundary occupancy for one leaf from the
 *                 current cell-classification state of its face neighbours.
 */
enum class RefinementTaskKind : std::uint8_t {
    kRefine = 0U,
    kClassify = 1U,
    kDistanceUpdate = 2U,
    kOccupancyUpdate = 3U,
};

struct RefinementTask {
    /** Index of the cell to process. */
    std::size_t cell_index = 0U;

    /** Optional snapshot of the demanded depth at enqueue time. */
    std::uint32_t demanded_depth = 0U;

    /** Optional bitmask describing why the task was scheduled. */
    std::uint32_t reason_flags = 0U;

    /** Discriminator for dispatch in the worker loop. Defaults to kRefine. */
    RefinementTaskKind kind = RefinementTaskKind::kRefine;
};

struct RefinementWorkQueueStats {
    /** Total tasks pushed into the queue. */
    std::size_t push_count = 0U;

    /** Total tasks popped from the queue. */
    std::size_t pop_count = 0U;

    /** Maximum observed queue size. */
    std::size_t high_watermark = 0U;

    /** Current queued task count. */
    std::size_t queue_size = 0U;

    /** Current in-flight task count (approximate under contention). */
    std::size_t in_flight_count = 0U;

    /** Current sleeping worker count. */
    std::size_t sleeping_worker_count = 0U;

    /** Queue size immediately after startup seeding. */
    std::size_t initial_queue_size = 0U;

    /** Total steal attempts made by any worker. */
    std::size_t steal_attempts = 0U;

    /** Total steal attempts that returned a task. */
    std::size_t steal_successes = 0U;

    /** Total times a worker had to enter the idle wait state. */
    std::size_t idle_sleeps = 0U;
};

/**
 * @brief Work-stealing refinement task scheduler.
 *
 * The queue is implemented as N thread-local deques with stealing.
 * Owners push/pop at the back (LIFO) for depth-first locality; thieves steal
 * from the front (FIFO) to preserve some breadth and reduce contention.
 */
class RefinementWorkQueue {
public:
    RefinementWorkQueue() = default;

    /** Initialize internal deques for worker threads. */
    void initialize(std::uint32_t worker_count);

    /** Push one task to a preferred worker and wake waiters. */
    bool push(const RefinementTask &task, std::uint32_t preferred_worker);

    /** Convenience push to worker 0 (serial / coordinator). */
    bool push(const RefinementTask &task);

    /** Push several tasks to one preferred worker. */
    std::size_t push_batch(
        const std::vector<RefinementTask> &tasks,
        std::uint32_t preferred_worker);

    /**
     * @brief Pop one task if available.
     *
     * Returns false only when shutdown has been requested and no work remains.
     */
    bool pop(std::uint32_t worker_id, RefinementTask &task);

    /** Mark one previously popped task as completed. */
    void task_done();

    /** Request queue shutdown and wake all waiters. */
    void shutdown();

    /** Request shutdown only if no queued or in-flight work remains. */
    bool try_shutdown_if_idle();

    /** Return true when the queue currently holds no tasks. */
    bool empty() const;

    /** Return whether shutdown has been requested. */
    bool is_shutdown() const;

    /** Return true when no queued or in-flight work remains. */
    bool is_idle() const;

    /** Return a snapshot of queue statistics. */
    RefinementWorkQueueStats stats() const;

    /**
     * @brief Claim one elapsed reporting cadence slot if available.
     *
     * Returns true only for the thread that successfully claims the next table
     * emission slot.
     */
    bool try_claim_report_slot(double cadence_seconds);

    /** Claim one report opportunity due to cadence or progress. */
    bool try_claim_report(double cadence_seconds);

    /** Reset report cadence state for a new run. */
    void reset_report_slot_clock();

    /** Return the currently claimed next periodic report slot. */
    std::uint64_t next_report_slot_index() const;

    /** Claim responsibility for emitting the one final completion row. */
    bool try_claim_final_report() const;

    /** Record the initial seeded queue size for progress reporting. */
    void capture_initial_queue_size();

private:
    struct WorkerDeque {
        mutable std::mutex mutex;
        std::deque<RefinementTask> tasks;
    };

    std::size_t approximate_queue_size_locked_unsafe() const;
    bool pop_local_(std::uint32_t worker_id, RefinementTask &task);
    bool steal_(std::uint32_t thief_id, RefinementTask &task);

    mutable std::mutex idle_mutex_;
    std::condition_variable idle_condition_;

    std::vector<std::unique_ptr<WorkerDeque>> deques_;
    std::atomic<std::size_t> queued_tasks_{0U};
    std::atomic<std::size_t> in_flight_tasks_{0U};
    std::atomic<std::size_t> sleeping_workers_{0U};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<std::size_t> push_count_{0U};
    std::atomic<std::size_t> pop_count_{0U};
    std::atomic<std::size_t> high_watermark_{0U};
    std::atomic<std::size_t> initial_queue_size_{0U};
    std::atomic<std::size_t> steal_attempts_{0U};
    std::atomic<std::size_t> steal_successes_{0U};
    std::atomic<std::size_t> idle_sleeps_{0U};
    std::chrono::steady_clock::time_point report_epoch_{};
    std::atomic<std::uint64_t> next_report_slot_{1U};
    std::atomic<std::size_t> next_progress_bucket_{1U};
    mutable std::atomic<bool> final_report_emitted_{false};
};

#endif  // MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_WORK_QUEUE_HPP_
