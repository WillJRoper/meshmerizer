/**
 * @file omp_config.hpp
 * @brief Conditional OpenMP include and helper macros.
 *
 * When the extension is compiled with ``-DWITH_OPENMP`` (via the
 * ``WITH_OPENMP`` environment variable at build time), this header
 * includes ``<omp.h>`` and leaves OpenMP pragmas active.  Without the
 * flag, pragmas are silently ignored by the compiler (they are already
 * ``#pragma`` directives, which compilers skip when unrecognised), and
 * ``omp_get_max_threads()`` is replaced by a stub returning 1.
 *
 * This keeps the rest of the codebase free of ``#ifdef`` clutter
 * around every pragma.
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_OMP_CONFIG_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_OMP_CONFIG_HPP_

#ifdef WITH_OPENMP
#include <omp.h>
#else
/**
 * @brief Stub: return 1 thread when OpenMP is disabled.
 */
inline int omp_get_max_threads() { return 1; }

/**
 * @brief Stub: return thread 0 when OpenMP is disabled.
 */
inline int omp_get_thread_num() { return 0; }

/**
 * @brief Stub: no-op when OpenMP is disabled.
 */
inline void omp_set_num_threads(int) {}
#endif

#endif  // MESHMERIZER_ADAPTIVE_CPP_OMP_CONFIG_HPP_
