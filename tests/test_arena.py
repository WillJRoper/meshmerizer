"""Regression tests for the adaptive refinement chunked arena.

The adaptive closure pipeline depends on a lock-free chunked arena
(``ChunkedArena`` in ``refinement_arena.hpp``) for stable-address parallel
storage of octree cells and contributor slices. The arena is a load-bearing
primitive: a correctness bug there would silently corrupt the octree or
double-write contributor slots from multiple threads.

These tests exercise the arena via the ``_adaptive._arena_stress_test``
binding hook, which spawns many native threads, each reserving 8-cell
blocks and writing thread-id markers into them. The hook reports back
whether every invariant held:

- All reserved blocks fit inside one chunk (no straddle).
- No two reservations overlap.
- Every claimed slot retains the writing thread's marker (no torn block).
- ``arena.size()`` covers all reservations.

The hook itself is intentionally permanent in the extension so that a
future change to chunk sizing, atomics, or the growth path is caught here
before it can break the closure.
"""

from __future__ import annotations

from meshmerizer import _adaptive


def test_arena_single_thread_reservations_are_contiguous() -> None:
    """One thread should observe perfectly packed sequential reservations."""
    ok, total = _adaptive._arena_stress_test(1, 1024)
    assert ok is True
    # 1 thread * 1024 ops * 8 cells = 8192 reserved entries. With one
    # thread there is no chunk-boundary skipping until 128K, so the cursor
    # advances by exactly 8 each call.
    assert total == 1024 * 8


def test_arena_handles_modest_concurrency() -> None:
    """A handful of threads must not produce overlap or torn blocks."""
    ok, total = _adaptive._arena_stress_test(4, 2048)
    assert ok is True
    # Total may exceed 4*2048*8 due to chunk-boundary alignment skips when
    # multiple threads race across a chunk edge, but it must not be lower.
    assert total >= 4 * 2048 * 8


def test_arena_handles_aggressive_concurrency() -> None:
    """High thread counts and many ops still preserve every invariant.

    Sized to cross several 128K chunk boundaries so the chunk-growth path
    runs many times under contention.
    """
    threads = 16
    ops = 4096
    ok, total = _adaptive._arena_stress_test(threads, ops)
    assert ok is True
    # 16 * 4096 * 8 = 524288 entries, comfortably crossing four 128K chunks.
    assert total >= threads * ops * 8


def test_arena_handles_chunk_boundary_pressure() -> None:
    """Stress the chunk-growth slow path specifically.

    With many threads each performing few ops, most reservations land
    around one or two chunk boundaries simultaneously, exercising the
    grow-mutex serialization path.
    """
    threads = 32
    ops = 512
    ok, total = _adaptive._arena_stress_test(threads, ops)
    assert ok is True
    assert total >= threads * ops * 8
