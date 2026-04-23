/**
 * @file cancellation.hpp
 * @brief Cooperative cancellation helpers for long-running native routines.
 *
 * The adaptive meshing pipeline performs large serial and parallel loops inside
 * native code. This header centralizes the shared cancellation state used to
 * propagate Python keyboard interrupts into those loops without forcing every
 * call site to understand Python signal handling details.
 *
 * The general model is:
 * - poll cheaply from hot loops using counters,
 * - let one designated thread check Python signals in parallel regions,
 * - convert a detected interrupt into a lightweight C++ exception, and
 * - reset the global cancellation flag when control returns to Python.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_CANCELLATION_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_CANCELLATION_HPP_

#include <Python.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>

#include "omp_config.hpp"

namespace meshmerizer_cancel_detail {

/**
 * @brief Return the process-wide cooperative cancellation flag.
 *
 * The flag is intentionally shared across all native pipeline stages so a
 * signal detected in one loop can quickly stop subsequent work as the stack
 * unwinds.
 */
inline std::atomic<bool> &cancel_requested() {
    static std::atomic<bool> requested(false);
    return requested;
}

/**
 * @brief Exception type thrown when native work is cancelled.
 *
 * The Python binding layer catches this exception and converts it into a
 * ``KeyboardInterrupt``-style return path for Python callers.
 */
class OperationCancelled : public std::exception {
 public:
    /**
     * @brief Return the exception message.
     *
     * @return Static diagnostic string.
     */
    const char *what() const noexcept override {
        return "operation cancelled";
    }
};

/**
 * @brief Clear the shared cancellation flag.
 */
inline void reset_cancel_state() {
    cancel_requested().store(false, std::memory_order_relaxed);
}

/**
 * @brief Mark the current native operation as cancelled.
 */
inline void request_cancel() {
    cancel_requested().store(true, std::memory_order_relaxed);
}

/**
 * @brief Return whether cancellation has been requested.
 *
 * @return ``true`` when native work should stop.
 */
inline bool is_cancel_requested() {
    return cancel_requested().load(std::memory_order_relaxed);
}

/**
 * @brief Poll Python signal handlers while holding the GIL.
 *
 * This function is the boundary between Python-level interrupts and the native
 * cancellation flag. It is relatively expensive compared to a plain atomic
 * load, so hot loops call it only when counter-based polling says it is time.
 *
 * @return ``true`` when cancellation is now requested.
 */
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

/**
 * @brief Return whether a loop counter has reached a polling boundary.
 *
 * @param counter Current loop counter.
 * @param interval Polling interval in iterations.
 * @return ``true`` when the caller should perform a slower signal poll.
 */
inline bool should_poll_counter(std::uint64_t counter,
                                std::uint64_t interval = 1024U) {
    return interval > 0U && (counter % interval) == 0U;
}

/**
 * @brief Throw ``OperationCancelled`` if cancellation is already requested.
 */
inline void throw_if_cancel_requested() {
    if (is_cancel_requested()) {
        throw OperationCancelled();
    }
}

/**
 * @brief Poll for cancellation inside serial native loops.
 *
 * @param counter Current loop counter.
 * @param interval Polling interval in iterations.
 */
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

/**
 * @brief Convenience wrapper for ``std::size_t`` loop counters.
 *
 * @param counter Current loop counter.
 * @param interval Polling interval in iterations.
 */
inline void poll_for_cancellation_serial(std::size_t counter,
                                         std::size_t interval = 1024U) {
    poll_for_cancellation(static_cast<std::uint64_t>(counter),
                          static_cast<std::uint64_t>(interval));
}

/**
 * @brief Return whether the current parallel worker should poll Python.
 *
 * Only one thread performs the expensive signal poll in a parallel region.
 * Other workers only observe the shared atomic flag.
 *
 * @param counter Current loop counter.
 * @param interval Polling interval in iterations.
 * @return ``true`` when the current thread should call ``poll_python_signals``.
 */
inline bool thread_should_poll(std::size_t counter,
                               std::size_t interval = 2048U) {
    return omp_get_thread_num() == 0 &&
           should_poll_counter(static_cast<std::uint64_t>(counter),
                               static_cast<std::uint64_t>(interval));
}

/**
 * @brief Poll for cancellation inside parallel native loops.
 *
 * @param counter Current loop counter.
 * @param interval Polling interval in iterations.
 * @return ``true`` when cancellation has been requested.
 */
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
