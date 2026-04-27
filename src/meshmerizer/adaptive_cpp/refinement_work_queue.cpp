#include "refinement_work_queue.hpp"

#include <algorithm>

void RefinementWorkQueue::push(const RefinementTask &task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push_back(task);
        ++stats_.push_count;
        stats_.high_watermark = std::max(stats_.high_watermark, tasks_.size());
    }
    condition_.notify_one();
}

void RefinementWorkQueue::push_batch(const std::vector<RefinementTask> &tasks) {
    if (tasks.empty()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const RefinementTask &task : tasks) {
            tasks_.push_back(task);
        }
        stats_.push_count += tasks.size();
        stats_.high_watermark = std::max(stats_.high_watermark, tasks_.size());
    }
    condition_.notify_all();
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
    ++stats_.pop_count;
    return true;
}

void RefinementWorkQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_requested_ = true;
    }
    condition_.notify_all();
}

bool RefinementWorkQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.empty();
}

bool RefinementWorkQueue::is_shutdown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_requested_;
}

RefinementWorkQueueStats RefinementWorkQueue::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}
