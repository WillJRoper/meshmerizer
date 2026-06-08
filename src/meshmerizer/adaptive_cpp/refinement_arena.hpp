/**
 * @file refinement_arena.hpp
 * @brief Lock-free chunked arenas for parallel adaptive octree refinement.
 *
 * The adaptive closure pipeline needs storage that:
 *
 * - Provides stable addresses across an entire refinement call so that
 *   parallel workers can hold raw references to cells without taking a lock
 *   on every access.
 * - Supports lock-free contiguous block reservation (eight cells for one
 *   octree split, or N contributor entries for one child slice) via a single
 *   atomic ``fetch_add``.
 * - Indexes by flat global offset so that existing ``child_begin`` and
 *   ``parent_index`` fields on ``OctreeCell`` continue to act as plain
 *   integer offsets, just as they did with ``std::vector<OctreeCell>``.
 *
 * The implementation is the standard chunked-vector pattern:
 *
 * - Storage is split into fixed-size chunks of ``T`` allocated on the heap.
 * - A global atomic counter ``next_index_`` is the reservation cursor.
 * - A second atomic ``committed_chunks_`` tracks the number of chunks fully
 *   allocated and visible to readers.
 * - Reservation is wait-free in the common case (atomic fetch-add). Chunk
 *   growth takes a mutex, double-checks, allocates the missing chunks, and
 *   then release-stores ``committed_chunks_``.
 * - Readers acquire-load ``committed_chunks_`` before indexing, which pairs
 *   with the release on the writer side and guarantees that any chunk
 *   pointer they observe is fully constructed.
 *
 * Chunk size is 128K entries, chosen so that very large runs (hundreds of
 * millions of particles, tens of millions of cells) only grow on the order
 * of hundreds of chunks rather than thousands while still keeping per-chunk
 * allocations bounded (a few tens of megabytes for ``OctreeCell``).
 */

#ifndef MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_ARENA_HPP_
#define MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_ARENA_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace meshmerizer_arena_detail {

/** Number of entries per arena chunk, expressed as a power of two. */
inline constexpr std::size_t kArenaChunkLog2 = 17U;
/** Entries per chunk (``1 << kArenaChunkLog2``). */
inline constexpr std::size_t kArenaChunkSize =
    static_cast<std::size_t>(1) << kArenaChunkLog2;
/** Mask for fast modulo when computing intra-chunk offsets. */
inline constexpr std::size_t kArenaChunkMask = kArenaChunkSize - 1U;

/**
 * Maximum number of chunks the spine vector is sized to hold up front.
 *
 * The spine ``std::vector<std::unique_ptr<T[]>>`` is read concurrently by
 * any worker that calls ``operator[]``. If chunk growth ever causes
 * ``std::vector::reserve`` to reallocate the spine buffer, concurrent
 * readers will race on the buffer pointer swap and may segfault when the
 * old buffer is freed.
 *
 * Pre-reserving the spine to a large fixed capacity at arena construction
 * removes the reallocation entirely: ``emplace_back`` only writes into
 * pre-allocated spine storage, never moves it. The spine slot itself is
 * synchronized via the acquire/release pair on ``committed_chunks_``.
 *
 * 65536 chunks * 128K entries = ~8.6 billion cells, well beyond any
 * realistic workload. Memory cost for the spine is
 * 65536 * sizeof(unique_ptr) = 512 KiB per arena, negligible compared to
 * the arena's actual cell payload (gigabytes).
 */
inline constexpr std::size_t kArenaSpineCapacity = 65536U;

}  // namespace meshmerizer_arena_detail

/**
 * @brief Chunked arena providing stable addresses and lock-free reservation.
 *
 * @tparam T Stored element type. Must be default-constructible.
 *
 * The arena hands out a flat global index per reserved entry. Indices are
 * monotonically increasing within one arena and map to ``T &`` references
 * that are valid for the lifetime of the arena. Reservation is lock-free
 * except when a new chunk must be allocated; in that case a single mutex
 * serializes growth, but the slow path runs at most once per ``kChunkSize``
 * reservations across all threads combined.
 */
template <typename T>
class ChunkedArena {
public:
    /**
     * @brief Default constructor; pre-reserves the spine to a fixed
     *        capacity so concurrent readers never race with spine
     *        reallocation. No chunk storage is allocated until first
     *        reserve.
     */
    ChunkedArena() {
        chunks_.reserve(meshmerizer_arena_detail::kArenaSpineCapacity);
    }

    /** Disable copy; the arena owns unique chunk allocations. */
    ChunkedArena(const ChunkedArena &) = delete;
    /** Disable copy assignment for the same reason as the copy ctor. */
    ChunkedArena &operator=(const ChunkedArena &) = delete;

    /**
     * @brief Move construction transfers chunk ownership and counters.
     */
    ChunkedArena(ChunkedArena &&other) noexcept {
        std::lock_guard<std::mutex> guard(other.grow_mutex_);
        chunks_ = std::move(other.chunks_);
        next_index_.store(
            other.next_index_.exchange(0, std::memory_order_relaxed),
            std::memory_order_relaxed);
        committed_chunks_.store(
            other.committed_chunks_.exchange(0, std::memory_order_relaxed),
            std::memory_order_relaxed);
    }

    /**
     * @brief Move assignment with the same semantics as the move ctor.
     */
    ChunkedArena &operator=(ChunkedArena &&other) noexcept {
        if (this != &other) {
            // Lock both mutexes deadlock-free. std::lock is C++11; we avoid
            // std::scoped_lock here so the file builds cleanly on any libc++
            // configuration the project supports.
            std::lock(grow_mutex_, other.grow_mutex_);
            std::lock_guard<std::mutex> guard_self(grow_mutex_,
                                                   std::adopt_lock);
            std::lock_guard<std::mutex> guard_other(other.grow_mutex_,
                                                    std::adopt_lock);
            chunks_ = std::move(other.chunks_);
            next_index_.store(
                other.next_index_.exchange(0, std::memory_order_relaxed),
                std::memory_order_relaxed);
            committed_chunks_.store(
                other.committed_chunks_.exchange(0, std::memory_order_relaxed),
                std::memory_order_relaxed);
        }
        return *this;
    }

    ~ChunkedArena() = default;

    /**
     * @brief Reserve ``count`` contiguous entries and return the first index.
     *
     * The reservation is performed by a single atomic fetch-add on the
     * global cursor, so reservation never blocks unless the call would
     * cross a chunk boundary that has not yet been allocated. In that
     * case ``ensure_capacity`` runs under ``grow_mutex_`` to allocate the
     * missing chunks; concurrent reservers race only on the mutex, not on
     * the cursor itself.
     *
     * @param count Number of contiguous entries to reserve. Must be > 0
     *              and <= ``kArenaChunkSize`` to guarantee that the slice
     *              fits inside one chunk (so the returned references are
     *              contiguous in memory, which the closure relies on for
     *              eight-child blocks).
     * @return Flat global index of the first reserved entry.
     */
    std::size_t reserve_block(std::size_t count) {
        // Round the cursor up so the requested block does not straddle a
        // chunk boundary. This keeps the eight children of one split (and
        // any contributor slice that fits in one chunk) physically
        // contiguous, which is required by callers that walk children via
        // ``child_begin + offset``.
        std::size_t begin = 0;
        for (;;) {
            std::size_t observed = next_index_.load(std::memory_order_relaxed);
            const std::size_t chunk_index =
                observed >> meshmerizer_arena_detail::kArenaChunkLog2;
            const std::size_t offset_in_chunk =
                observed & meshmerizer_arena_detail::kArenaChunkMask;
            std::size_t aligned = observed;
            if (offset_in_chunk + count >
                meshmerizer_arena_detail::kArenaChunkSize) {
                aligned = (chunk_index + 1U)
                          << meshmerizer_arena_detail::kArenaChunkLog2;
            }
            const std::size_t end = aligned + count;
            if (next_index_.compare_exchange_weak(
                    observed,
                    end,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                begin = aligned;
                ensure_capacity(end);
                return begin;
            }
            // CAS lost: retry with the updated observed value.
        }
    }

    /**
     * @brief Advance the reservation cursor to at least ``target_size``.
     *
     * Used by side-car arenas that must mirror the index space of a primary
     * arena: callers reserve their primary slot first (which may include
     * chunk-alignment rounding), then call ``reserve_to(begin + count)`` on
     * each side-car arena so that side-car index ``i`` is valid for every
     * ``i < primary.size()`` once the primary append returns.
     *
     * Multiple threads may race here harmlessly. The implementation
     * CAS-loops ``next_index_`` so the final value is the maximum of all
     * concurrent targets, then calls ``ensure_capacity`` so chunks are
     * allocated and visible. This is wait-free in the common case and only
     * takes the growth mutex when new chunks must be allocated.
     *
     * @param target_size Smallest size the arena must reach. If the arena
     *                    is already at or past this size the call is a
     *                    no-op (one acquire-load on the cursor).
     */
    void reserve_to(std::size_t target_size) {
        std::size_t observed = next_index_.load(std::memory_order_relaxed);
        while (observed < target_size) {
            if (next_index_.compare_exchange_weak(
                    observed,
                    target_size,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                break;
            }
            // CAS lost: ``observed`` was rewritten with the latest cursor;
            // re-check whether some other thread already advanced past
            // ``target_size`` and exit early if so.
        }
        ensure_capacity(target_size);
    }

    /**
     * @brief Index into the arena. Behavior is undefined for unreserved
     *        indices.
     *
     * Performs an acquire-load on ``committed_chunks_`` so that the chunk
     * pointer load synchronizes with the release-store performed by the
     * thread that allocated the chunk.
     *
     * @param index Flat global index returned by a prior ``reserve_block``.
     * @return Reference to the stored element.
     */
    T &operator[](std::size_t index) noexcept {
        const std::size_t chunk_index =
            index >> meshmerizer_arena_detail::kArenaChunkLog2;
        const std::size_t offset_in_chunk =
            index & meshmerizer_arena_detail::kArenaChunkMask;
        // Acquire-load to pair with the release-store in ensure_capacity.
        // We only need to read the spine slot; the spine vector itself is
        // stable for the lifetime of the arena once chunks are allocated
        // because we reserve enough capacity in ensure_capacity.
        (void)committed_chunks_.load(std::memory_order_acquire);
        return chunks_[chunk_index][offset_in_chunk];
    }

    /**
     * @brief Const overload of ``operator[]``.
     */
    const T &operator[](std::size_t index) const noexcept {
        const std::size_t chunk_index =
            index >> meshmerizer_arena_detail::kArenaChunkLog2;
        const std::size_t offset_in_chunk =
            index & meshmerizer_arena_detail::kArenaChunkMask;
        (void)committed_chunks_.load(std::memory_order_acquire);
        return chunks_[chunk_index][offset_in_chunk];
    }

    /**
     * @brief Return the number of reserved entries.
     */
    std::size_t size() const noexcept {
        return next_index_.load(std::memory_order_acquire);
    }

    /**
     * @brief Return whether no entries have been reserved.
     */
    bool empty() const noexcept { return size() == 0U; }

    /**
     * @brief Materialize the arena contents into a flat ``std::vector``.
     *
     * Used at refinement boundaries to hand storage back to legacy callers
     * that consume ``std::vector<T>``. The arena retains its contents.
     *
     * @return Newly allocated vector copy of every reserved entry.
     */
    std::vector<T> materialize() const {
        const std::size_t total = size();
        std::vector<T> out;
        out.reserve(total);
        for (std::size_t i = 0; i < total; ++i) {
            out.push_back((*this)[i]);
        }
        return out;
    }

    /**
     * @brief Replace arena contents with the entries of ``source``.
     *
     * Used at refinement boundaries to take ownership of an existing flat
     * vector before parallel work begins. Resets the arena, allocates the
     * required number of chunks under ``grow_mutex_``, and copies entries
     * into the freshly allocated chunks.
     *
     * @param source Source vector. Its contents are copied into the arena.
     */
    void adopt(const std::vector<T> &source) {
        std::lock_guard<std::mutex> guard(grow_mutex_);
        chunks_.clear();
        // Re-establish pre-reserved spine capacity in case ``chunks_``
        // was moved-out-of or otherwise lost its reservation. Concurrent
        // readers must never see ``chunks_`` reallocate its buffer.
        if (chunks_.capacity() <
            meshmerizer_arena_detail::kArenaSpineCapacity) {
            chunks_.reserve(
                meshmerizer_arena_detail::kArenaSpineCapacity);
        }
        next_index_.store(0, std::memory_order_relaxed);
        committed_chunks_.store(0, std::memory_order_relaxed);
        if (source.empty()) {
            return;
        }
        const std::size_t needed_chunks =
            (source.size() + meshmerizer_arena_detail::kArenaChunkSize - 1U) >>
            meshmerizer_arena_detail::kArenaChunkLog2;
        for (std::size_t c = 0; c < needed_chunks; ++c) {
            chunks_.emplace_back(std::make_unique<T[]>(
                meshmerizer_arena_detail::kArenaChunkSize));
        }
        for (std::size_t i = 0; i < source.size(); ++i) {
            const std::size_t ci =
                i >> meshmerizer_arena_detail::kArenaChunkLog2;
            const std::size_t off =
                i & meshmerizer_arena_detail::kArenaChunkMask;
            chunks_[ci][off] = source[i];
        }
        next_index_.store(source.size(), std::memory_order_relaxed);
        committed_chunks_.store(needed_chunks, std::memory_order_release);
    }

    /**
     * @brief Reset the arena to empty without releasing chunk memory.
     *
     * Useful when the same context is reused across multiple closure
     * invocations; chunks can be retained to avoid reallocation churn.
     * The contents are not destructed individually; callers that store
     * non-trivial types must reset entries through ``operator[]``.
     */
    void clear() noexcept {
        std::lock_guard<std::mutex> guard(grow_mutex_);
        next_index_.store(0, std::memory_order_relaxed);
        // Keep chunks_ allocated; they remain reusable.
    }

private:
    /**
     * @brief Ensure that all chunks needed for indices in ``[0, end_index)``
     *        are allocated and visible.
     *
     * Called after a successful ``reserve_block`` CAS. The slow path takes
     * ``grow_mutex_``, double-checks under the lock, allocates any missing
     * chunks, and finally release-stores the new ``committed_chunks_``
     * count so that readers see fully constructed chunk pointers.
     */
    void ensure_capacity(std::size_t end_index) {
        const std::size_t needed_chunks =
            (end_index + meshmerizer_arena_detail::kArenaChunkSize - 1U) >>
            meshmerizer_arena_detail::kArenaChunkLog2;
        if (committed_chunks_.load(std::memory_order_acquire) >= needed_chunks) {
            return;
        }
        std::lock_guard<std::mutex> guard(grow_mutex_);
        // Double-check under the lock; another thread may have grown the
        // arena while we were waiting for the mutex.
        std::size_t have = committed_chunks_.load(std::memory_order_relaxed);
        if (have >= needed_chunks) {
            return;
        }
        // Spine capacity is pre-reserved at construction (and re-asserted
        // in ``adopt``) to ``kArenaSpineCapacity``. ``emplace_back`` here
        // therefore writes only into pre-allocated spine slots and never
        // reallocates the spine buffer. This is essential: concurrent
        // readers index ``chunks_`` without taking ``grow_mutex_``, and a
        // spine reallocation would race with their reads of the spine
        // base pointer and free the old buffer out from under them.
        if (needed_chunks > chunks_.capacity()) {
            // The arena has exceeded its pre-reserved spine. Either the
            // workload is enormous (spine cap is 65536 chunks ~= 8.6B
            // entries) or the spine was lost. Throwing here is safer than
            // silently reallocating and racing with readers.
            throw std::runtime_error(
                "ChunkedArena: spine pre-reservation exceeded; "
                "increase kArenaSpineCapacity");
        }
        while (have < needed_chunks) {
            chunks_.emplace_back(std::make_unique<T[]>(
                meshmerizer_arena_detail::kArenaChunkSize));
            ++have;
        }
        // Release-store so that readers acquire-loading committed_chunks_
        // will see the newly appended chunk pointers.
        committed_chunks_.store(have, std::memory_order_release);
    }

    /**
     * @brief Owned chunk pointers. The vector itself is only mutated under
     *        ``grow_mutex_``; readers index it after acquire-loading
     *        ``committed_chunks_``.
     */
    std::vector<std::unique_ptr<T[]>> chunks_;
    /** Mutex serializing chunk growth. The fast path never touches it. */
    mutable std::mutex grow_mutex_;
    /** Reservation cursor. Atomic fetch-add gives a wait-free fast path. */
    std::atomic<std::size_t> next_index_{0};
    /** Number of chunks fully allocated and visible to readers. */
    std::atomic<std::size_t> committed_chunks_{0};
};

#endif  // MESHMERIZER_ADAPTIVE_CPP_REFINEMENT_ARENA_HPP_
