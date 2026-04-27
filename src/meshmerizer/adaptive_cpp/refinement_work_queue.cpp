#include "refinement_work_queue.hpp"

#include <algorithm>

bool RefinementWorkQueue::push(const RefinementTask &task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_requested_) {
            return false;
        }
        tasks_.push_back(task);
        ++stats_.push_count;
        stats_.high_watermark = std::max(stats_.high_watermark, tasks_.size());
        stats_.queue_size = tasks_.size();
    }
    condition_.notify_one();
    return true;
}

std::size_t RefinementWorkQueue::push_batch(const std::vector<RefinementTask> &tasks) {
    if (tasks.empty()) {
        return 0U;
    }

    std::size_t pushed_count = 0U;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_requested_) {
            return 0U;
        }
        for (const RefinementTask &task : tasks) {
            tasks_.push_back(task);
        }
        stats_.push_count += tasks.size();
        pushed_count = tasks.size();
        stats_.high_watermark = std::max(stats_.high_watermark, tasks_.size());
        stats_.queue_size = tasks_.size();
    }
    condition_.notify_all();
    return pushed_count;
}

bool RefinementWorkQueue::pop(RefinementTask &task) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&]() {
        return shutdown_requested_ || !tasks_.empty();
    });

    if (tasks_.empty()) {
        return false;
    }

    task = tasks_.front();
    tasks_.pop_front();
    ++in_flight_tasks_;
    ++stats_.pop_count;
    stats_.queue_size = tasks_.size();
    stats_.in_flight_count = in_flight_tasks_;
    return true;
}

void RefinementWorkQueue::task_done() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (in_flight_tasks_ > 0U) {
            --in_flight_tasks_;
        }
        stats_.in_flight_count = in_flight_tasks_;
    }
    condition_.notify_all();
}

void RefinementWorkQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_requested_ = true;
    }
    condition_.notify_all();
}

bool RefinementWorkQueue::try_shutdown_if_idle() {
    bool did_shutdown = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (tasks_.empty() && in_flight_tasks_ == 0U && !shutdown_requested_) {
            shutdown_requested_ = true;
            did_shutdown = true;
        }
    }
    if (did_shutdown) {
        condition_.notify_all();
    }
    return did_shutdown;
}

bool RefinementWorkQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.empty();
}

bool RefinementWorkQueue::is_shutdown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_requested_;
}

bool RefinementWorkQueue::is_idle() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.empty() && in_flight_tasks_ == 0U;
}

RefinementWorkQueueStats RefinementWorkQueue::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}
