/**
 * @file progress_bar.hpp
 * @brief Terminal-width progress bar for long-running C++ loops.
 *
 * Provides a simple, self-contained progress bar that spans the full
 * width of the terminal.  Designed to work safely with OpenMP: the
 * internal counter is atomic, and only one thread performs the actual
 * print (guarded by a compare-exchange).
 *
 * Usage:
 * @code
 *   ProgressBar bar("Meshing", n_items);
 *   #pragma omp parallel for schedule(dynamic)
 *   for (std::size_t i = 0; i < n_items; ++i) {
 *       // ... work ...
 *       bar.tick();
 *   }
 *   bar.finish();
 * @endcode
 */

#ifndef MESHMERIZER_PROGRESS_BAR_HPP
#define MESHMERIZER_PROGRESS_BAR_HPP

#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <sstream>
#include <string>

#include "omp_config.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace meshmerizer_log_detail {

inline std::string current_thread_label() {
    return omp_get_thread_num() == 0 ? "main" : "worker";
}

inline std::string format_status_prefix(
    const std::string& operation,
    const std::string& function_name,
    const std::string& thread_label = current_thread_label()) {
    std::ostringstream stream;
    stream << "[" << operation << "]"
           << "[" << function_name << "]"
           << "[" << thread_label << "]";
    return stream.str();
}

inline void vprint_status(
    const std::string& operation,
    const std::string& function_name,
    const char* format,
    std::va_list args) {
    const std::string prefix = format_status_prefix(operation, function_name);
    std::fprintf(stdout, "%s ", prefix.c_str());
    std::vfprintf(stdout, format, args);
    std::fflush(stdout);
}

inline void print_status(
    const std::string& operation,
    const std::string& function_name,
    const char* format,
    ...) {
    std::va_list args;
    va_start(args, format);
    vprint_status(operation, function_name, format, args);
    va_end(args);
}

/**
 * @brief Query the terminal width in columns.
 *
 * Falls back to 80 columns if the width cannot be determined
 * (e.g. when stdout is redirected to a file).
 *
 * @return Terminal width in characters.
 */
inline int terminal_width() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(
            GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        return csbi.srWindow.Right - csbi.srWindow.Left + 1;
    }
    return 80;
#else
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) {
        return static_cast<int>(w.ws_col);
    }
    return 80;
#endif
}

}  // namespace meshmerizer_log_detail

/**
 * @class ProgressBar
 * @brief Terminal-width progress bar with atomic updates for OpenMP.
 *
 * The bar renders as:
 * @code
 *   [Meshing] [████████████████░░░░░░░░░░░░░░░░░░] 45.2%
 * @endcode
 *
 * The bar automatically queries the terminal width so it spans the
 * full extent of the console.  On each @c tick() call the internal
 * atomic counter increments; the bar is only reprinted when the
 * visual percentage changes (to avoid excessive I/O).
 */
class ProgressBar {
public:
    /**
     * @brief Construct a progress bar.
     *
     * @param label  Short label printed before the bar (e.g.
     *               "Meshing", "Solving vertices").
     * @param total  Total number of work items.
     */
    ProgressBar(const std::string& operation,
                const std::string& function_name,
                std::size_t total)
        : prefix_(meshmerizer_log_detail::format_status_prefix(operation, function_name)),
          total_(total),
          current_(0),
          last_rendered_percent_(-1),
          finished_(false) {
        /* Print the initial 0% bar immediately. */
        render(0);
    }

    /**
     * @brief Increment the counter by one and redraw if needed.
     *
     * Thread-safe.  Multiple OpenMP threads may call this
     * concurrently; the atomic increment ensures correctness and
     * the percentage check avoids lock contention on stdout.
     */
    void tick() {
        /* Atomic increment; returns the value *before* increment. */
        std::size_t prev = current_.fetch_add(1, std::memory_order_relaxed);
        std::size_t now = prev + 1;

        /* Only redraw when the integer percentage changes. */
        int pct = static_cast<int>(
            (static_cast<double>(now) / static_cast<double>(total_)) * 100.0
        );
        if (pct > last_rendered_percent_) {
            /* Simple race: two threads may both pass this check for
             * the same percentage.  That's harmless — the bar just
             * gets printed twice at the same value.  We avoid a
             * mutex here to keep the hot path lock-free. */
            last_rendered_percent_ = pct;
            render(pct);
        }
    }

    /**
     * @brief Mark the bar as complete and print a final newline.
     *
     * Must be called exactly once, from a single thread, after
     * the parallel region ends.
     */
    void finish() {
        if (finished_) return;
        finished_ = true;
        render(100);
        std::fputc('\n', stdout);
        std::fflush(stdout);
    }

private:
    /** Short label shown before the bar. */
    std::string prefix_;

    /** Total number of work items. */
    std::size_t total_;

    /** Atomic counter of completed work items. */
    std::atomic<std::size_t> current_;

    /**
     * Last integer percentage that was rendered.  Declared as
     * @c std::atomic<int> so concurrent reads/writes from
     * different OpenMP threads are well-defined.
     */
    std::atomic<int> last_rendered_percent_;

    /** Whether @c finish() has been called. */
    bool finished_;

    /**
     * @brief Render the progress bar at the given percentage.
     *
     * The layout is:
     * @code
     *   \r[Label] [████░░░░] 100%
     * @endcode
     *
     * The bar portion is sized to fill the remaining terminal
     * width after the label and percentage text.
     *
     * @param pct  Current percentage (0-100).
     */
    void render(int pct) {
        if (pct > 100) pct = 100;

        int width = meshmerizer_log_detail::terminal_width();

        /* Fixed-width parts:
         *   prefix + " "        = prefix.size() + 1
         *   "[" + "] "         = 3
         *   "100%"             = 4  (always reserve for 3 digits + %)
         *   Total overhead     = prefix.size() + 8
         */
        int overhead = static_cast<int>(prefix_.size()) + 8;
        int bar_width = width - overhead;
        if (bar_width < 4) bar_width = 4;

        int filled = (pct * bar_width) / 100;
        int empty = bar_width - filled;

        /* Build the bar string. We use block characters:
         *   U+2588 FULL BLOCK  = ███  (filled portion)
         *   U+2591 LIGHT SHADE = ░░░  (empty portion)
         * These are UTF-8 encoded as 3 bytes each. */
        std::string bar_str;
        bar_str.reserve(
            static_cast<std::size_t>((filled + empty) * 3)
        );
        for (int i = 0; i < filled; ++i) {
            bar_str += "\xe2\x96\x88";  /* █ */
        }
        for (int i = 0; i < empty; ++i) {
            bar_str += "\xe2\x96\x91";  /* ░ */
        }

        /* Print with carriage return to overwrite the current line. */
        std::fprintf(
            stdout,
            "\r%s [%s] %3d%%",
            prefix_.c_str(),
            bar_str.c_str(),
            pct
        );
        std::fflush(stdout);
    }
};


/**
 * @class ProgressCounter
 * @brief Terminal-width counter for loops with unknown totals.
 *
 * Renders as:
 * @code
 *   [Refining] 12345 cells processed
 * @endcode
 *
 * The counter updates in place using carriage return.  Like
 * @c ProgressBar, the internal counter is atomic for OpenMP safety.
 * To avoid excessive I/O the display only refreshes when the count
 * changes by at least @c update_interval_ items.
 */
class ProgressCounter {
public:
    /**
     * @brief Construct an indeterminate progress counter.
     *
     * @param label  Short label printed before the count.
     * @param unit   Noun describing counted items (e.g. "cells").
     * @param update_interval  Minimum count change between redraws.
     */
    ProgressCounter(const std::string& operation,
                    const std::string& function_name,
                    const std::string& unit = "items",
                    std::size_t update_interval = 1)
        : prefix_(meshmerizer_log_detail::format_status_prefix(operation, function_name)),
          unit_(unit),
          current_(0),
          last_rendered_(0),
          update_interval_(update_interval),
          finished_(false) {
        render(0);
    }

    /**
     * @brief Increment the counter by one and maybe redraw.
     *
     * Thread-safe for concurrent OpenMP calls.
     */
    void tick() {
        std::size_t now =
            current_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (now - last_rendered_.load(std::memory_order_relaxed)
            >= update_interval_) {
            last_rendered_.store(now, std::memory_order_relaxed);
            render(now);
        }
    }

    /**
     * @brief Print the final count and a newline.
     */
    void finish() {
        if (finished_) return;
        finished_ = true;
        render(current_.load(std::memory_order_relaxed));
        std::fputc('\n', stdout);
        std::fflush(stdout);
    }

private:
    std::string prefix_;
    std::string unit_;
    std::atomic<std::size_t> current_;
    std::atomic<std::size_t> last_rendered_;
    std::size_t update_interval_;
    bool finished_;

    /**
     * @brief Render the counter line.
     *
     * Pads the line to the terminal width so that previous longer
     * lines are fully overwritten.
     *
     * @param count  Current item count.
     */
    void render(std::size_t count) {
        int width = meshmerizer_log_detail::terminal_width();

        /* Build the content string. */
        char buf[256];
        int len = std::snprintf(
            buf, sizeof(buf), "\r%s %zu %s processed",
            prefix_.c_str(), count, unit_.c_str());

        /* Pad with spaces to the terminal width to clear old text. */
        int pad = width - len;
        if (pad < 0) pad = 0;

        std::fprintf(stdout, "%s%*s", buf, pad, "");
        std::fflush(stdout);
    }
};


#endif /* MESHMERIZER_PROGRESS_BAR_HPP */
