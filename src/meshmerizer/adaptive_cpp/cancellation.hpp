#ifndef MESHMERIZER_ADAPTIVE_CPP_CANCELLATION_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_CANCELLATION_HPP_

#include <Python.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>

#include "omp_config.hpp"

namespace meshmerizer_cancel_detail {

inline std::atomic<bool> &cancel_requested() {
    static std::atomic<bool> requested(false);
    return requested;
}

class OperationCancelled : public std::exception {
 public:
    const char *what() const noexcept override {
        return "operation cancelled";
    }
};

inline void reset_cancel_state() {
    cancel_requested().store(false, std::memory_order_relaxed);
}

inline void request_cancel() {
    cancel_requested().store(true, std::memory_order_relaxed);
}

inline bool is_cancel_requested() {
    return cancel_requested().load(std::memory_order_relaxed);
}

inline bool poll_python_signals() {
    if (is_cancel_requested()) {
        return true;
    }

    PyGILState_STATE gil_state = PyGILState_Ensure();
    const int signal_result = PyErr_CheckSignals();
    if (signal_result != 0) {
        PyErr_Clear();
        request_cancel();
    }
    PyGILState_Release(gil_state);
    return is_cancel_requested();
}

inline bool should_poll_counter(std::uint64_t counter,
                                std::uint64_t interval = 1024U) {
    return interval > 0U && (counter % interval) == 0U;
}

inline void throw_if_cancel_requested() {
    if (is_cancel_requested()) {
        throw OperationCancelled();
    }
}

inline void poll_for_cancellation(std::uint64_t counter = 0U,
                                  std::uint64_t interval = 1024U) {
    if (is_cancel_requested()) {
        throw OperationCancelled();
    }
    if (!should_poll_counter(counter, interval)) {
        return;
    }
    if (poll_python_signals()) {
        throw OperationCancelled();
    }
}

inline void poll_for_cancellation_serial(std::size_t counter,
                                         std::size_t interval = 1024U) {
    poll_for_cancellation(static_cast<std::uint64_t>(counter),
                          static_cast<std::uint64_t>(interval));
}

inline bool thread_should_poll(std::size_t counter,
                               std::size_t interval = 2048U) {
    return omp_get_thread_num() == 0 &&
           should_poll_counter(static_cast<std::uint64_t>(counter),
                               static_cast<std::uint64_t>(interval));
}

inline bool poll_for_cancellation_in_parallel(std::size_t counter,
                                              std::size_t interval = 2048U) {
    if (is_cancel_requested()) {
        return true;
    }
    if (!thread_should_poll(counter, interval)) {
        return false;
    }
    return poll_python_signals();
}

}  // namespace meshmerizer_cancel_detail

#endif  // MESHMERIZER_ADAPTIVE_CPP_CANCELLATION_HPP_
