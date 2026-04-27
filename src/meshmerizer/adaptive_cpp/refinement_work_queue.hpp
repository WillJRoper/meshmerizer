#ifndef MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_WORK_QUEUE_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_WORK_QUEUE_HPP_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

/**
 * @file refinement_work_queue.hpp
 * @brief Thread-safe task queue for integrated octree refinement closure.
 *
 * The queue transports lightweight refinement tasks between the closure
 * scheduler and worker logic. It does not own any geometric policy; the
 * authoritative state for a cell remains in the refinement context.
 */

struct RefinementTask {
    /** Index of the cell to process. */
    std::size_t cell_index = 0U;

    /** Optional snapshot of the demanded depth at enqueue time. */
    std::uint32_t demanded_depth = 0U;

    /** Optional bitmask describing why the task was scheduled. */
    std::uint32_t reason_flags = 0U;
};

struct RefinementWorkQueueStats {
    /** Total tasks pushed into the queue. */
    std::size_t push_count = 0U;

    /** Total tasks popped from the queue. */
    std::size_t pop_count = 0U;

    /** Maximum observed queue size. */
    std::size_t high_watermark = 0U;
};

/**
 * @brief Thread-safe refinement task queue.
 *
 * The initial implementation deliberately favors correctness and a stable API
 * over maximum throughput. Internally it uses one mutex-protected deque with
 * condition-variable wakeups. The API is shaped so the implementation can be
 * upgraded later without changing the closure logic.
 */
class RefinementWorkQueue {
public:
    /** Push one task and wake one waiter. */
    void push(const RefinementTask &task);

    /** Push several tasks and wake all waiters. */
    void push_batch(const std::vector<RefinementTask> &tasks);

    /**
     * @brief Pop one task if available.
     *
     * Returns false only when the queue is both empty and shut down.
     */
    bool pop(RefinementTask &task);

    /** Request queue shutdown and wake all waiters. */
    void shutdown();

    /** Return true when the queue currently holds no tasks. */
    bool empty() const;

    /** Return whether shutdown has been requested. */
    bool is_shutdown() const;

    /** Return a snapshot of queue statistics. */
    RefinementWorkQueueStats stats() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<RefinementTask> tasks_;
    bool shutdown_requested_ = false;
    RefinementWorkQueueStats stats_;
};

#endif  // MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_WORK_QUEUE_HPP_
