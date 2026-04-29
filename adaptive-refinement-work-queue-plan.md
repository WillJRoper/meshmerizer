# Adaptive Refinement Work Queue Plan

## Purpose

This document is the implementation plan for replacing the current
"refine first, balance later" octree construction flow with a single
task-based refinement closure process.

This file is intended to live at the repository root and be updated as the
design evolves and implementation lands.

Rules for using this plan:

- Treat this file as the source of truth for the work-queue rewrite.
- Update this file in the same change whenever the design or scope changes.
- Mark obsolete checklist items explicitly rather than silently skipping them.
- Prefer one clear production path over keeping the old and new refinement
  engines alive indefinitely.
- Keep correctness and determinism requirements written down here as they are
  discovered.

## Current Status

- Current implementation phase: **Phase 6.5 - Work-stealing Scheduler Upgrade**
- Work in progress now:
  - Phase 5 is complete
  - initial refinement routes through the closure engine
  - surface-band regularization refinement routes through the closure engine
  - thickening/pre-thickening refinement routes through the closure engine
  - zero-sample incident refinement now routes through the closure engine
  - queue-phase reporting now uses periodic table rows with configurable cadence
  - the next immediate target is replacing the conservative central-queue
    worker scaffold with a genuinely parallel DFS-with-work-stealing scheduler
    (remove effective serialization, coarsen task granularity, reduce
    synchronization to atomic state transitions + final join)

Reporting-note:

- queue-driven refinement now emits coordinator-side table rows on a strict
  time cadence
- CLI/native plumbing for `table_cadence` is now present
- Not started yet:
  - production-ready parallel workers beyond the current conservative scaffold
  - work-stealing / DFS locality scheduling
  - required-depth upward propagation (max over descendants)

## Problem Statement

The current adaptive pipeline performs:

1. surface-driven refinement
2. a separate global `balance_octree(...)` pass

This works, but it has important drawbacks:

- the balance pass revisits many leaves that were not touched by the most recent
  refinement work
- refinement pressure and balance pressure are implemented as separate stages
  even though they are part of one closure problem
- the current structure makes it harder to localize work during iterative
  refinement paths such as regularization and pre-thickening
- the algorithmic boundary between "refinement" and "balancing" makes future
  parallel scheduling less natural than it should be

The goal of this rewrite is to replace the two-step process with a single
monotone worklist algorithm:

- surface splits create new children
- those new children immediately impose local 2:1 obligations on neighbors
- those obligations are pushed into the same work queue
- the queue runs until the tree is closed under both surface and balance
  requirements

## Core Idea

We do **not** want an unbalanced tree with a more complicated mesher.

We **do** want to preserve the existing meshing assumptions by ensuring that the
tree remains 2:1 balanced, but we want to enforce that balance incrementally as
part of refinement itself instead of via a separate whole-tree pass.

In other words:

- keep the 2:1 invariant
- remove the explicit global balancing stage
- integrate balance propagation into the refinement scheduler

## Goals

- Replace the global balancing pass with an integrated task-based closure
  algorithm.
- Keep the final tree 2:1 balanced.
- Preserve the current dual-contouring meshing assumptions.
- Make the new algorithm naturally parallelizable across threads.
- Avoid duplicate work when multiple nearby splits demand the same neighbor
  refinement.
- Make the new system reusable by the initial refinement pass and later local
  refinement passes.
- Keep the queue itself as a dedicated class in its own header and
  implementation file.
- Replace queue-phase progress counters with periodic table-style status output.

## Non-Goals

- Redesigning the mesher to support fully unbalanced octrees.
- Removing the 2:1 invariant from the final tree.
- Performing a speculative full rewrite of all adaptive data structures at once.
- Introducing approximate geometric neighbor searches when exact octree-lattice
  traversal is available.

## Design Decisions Recorded Up Front

These decisions should be treated as constraints unless explicitly revised here.

1. **The queue itself will be a class in its own header and implementation.**
   It must not be an ad hoc set of helper functions embedded inside
   `octree_cell.hpp`.
2. **Queue membership is not enough.** The authoritative notion of demand is a
   monotone per-cell `required_depth` value, not a boolean `in_queue` flag.
3. **Tasks must be stale-tolerant.** It must be cheap and correct to pop a task
   that has already been satisfied by another thread.
4. **Neighbor propagation should be tree-guided, not brute-force finest-grid
   raster scanning.** Use exact octree-lattice reasoning and branch pruning.
5. **Transient scheduling state should live outside serialized octree cell
   payloads where possible.** Avoid polluting persistent formats with queue-only
   bookkeeping.
6. **Queue progress reporting should be table-based and strictly time-cadenced.**
   During queue-driven refinement we want one status row every `N` seconds,
   rather than per-cell progress-counter output or count-triggered logging.

## Queue Status Reporting

The queue-driven refinement path should not use `ProgressCounter` for its main
status reporting.

Instead, it should emit a periodic table row every `N` seconds summarizing the
current refinement state.

This should be a strictly time-based cadence, not a task-count-based or hybrid
reporting scheme.

### Default cadence

- default `table_cadence = 20` seconds

### CLI surface

Add a CLI argument:

- `--table-cadence`

Intended behavior:

- controls the reporting cadence in seconds for queue-driven refinement status
  tables
- should default to `20`
- should allow users to reduce noise or increase visibility for long runs
- should be interpreted as a strict time cadence for automatic table rows

### Intended table output

The exact final columns can evolve, but the queue status table should aim to
report useful scheduler state rather than generic progress ticks.

Candidate columns:

- elapsed time
- current phase / pass name
- queue size
- total queued tasks
- total popped tasks
- stale/discarded tasks
- processed leaves
- split count
- current total cell count
- cells whose `required_depth` was increased
- optional rolling throughput metrics

### Implementation guidance

- the cadence should be time-based, not task-count-based
- the table should always be emitted for queue-driven refinement even when
  `--silent` is set
- the reporting path should be cheap relative to refinement work
- rows should be triggered only when a worker goes looking for more work and the
  cadence has elapsed
- any worker may claim responsibility for emitting the next row, but an atomic
  gate should ensure only one worker emits a given cadence slot
- queue and context stats should make this reporting straightforward
- old progress-counter output in queue-driven refinement phases should be
  removed once the table output is in place

## High-Level Algorithm

Each cell can receive refinement pressure from two sources:

1. **surface pressure**
   - the cell fails the normal refinement stop criterion
   - examples: contains surface, poor Hermite support, poor QEF fit,
     regularization-specific split request
2. **balance pressure**
   - a neighboring finer cell requires this cell to reach a minimum depth so
     that the 2:1 invariant holds

The scheduler repeatedly processes cells until both pressures are closed.

### Worker logic

For a queued cell:

1. Re-read the current cell state.
2. If the cell is no longer a leaf, discard the task as stale.
3. Read the current `required_depth` for that cell.
4. Evaluate whether the cell must split for surface reasons.
5. If neither surface pressure nor balance pressure requires a split, mark the
   task complete.
6. Otherwise split the cell.
7. Publish the children.
8. Queue/evaluate the children for future surface work.
9. Propagate balance requirements from each new child to neighboring leaves.
10. Repeat until the queue becomes idle.

### Balance propagation rule

If a cell at depth `d` is split, its children are at depth `d + 1`.

For each new child, any leaf sharing a face with that child must have depth at
least `d`.

So the propagation rule is:

- newly created child depth = `child_depth`
- neighboring leaf must satisfy `neighbor.depth >= child_depth - 1`
- if not, raise `neighbor.required_depth` to `child_depth - 1`
- if that raise increases the requirement, schedule the neighbor

This gives a monotone fixed-point algorithm.

## Why `required_depth` Matters

A boolean `in_queue` flag is not expressive enough because a cell can receive
multiple refinement requests over time.

Examples:

- one task says "evaluate this cell for surface reasons"
- another says "this neighbor now needs to reach depth 3"
- later another says "actually it must reach depth 4"

The correct abstraction is:

- each cell has a monotone `required_depth`
- updates use `max(old_required_depth, new_required_depth)`
- queueing happens only when the requirement increases or when a cell needs its
  first evaluation

This prevents duplicate work and makes pruning safe.

## Recommended Architecture

The rewrite should separate four concerns cleanly.

### 1. Queue transport

Responsibility:

- thread-safe task storage
- blocking / non-blocking pop
- push / batch push
- queue idle detection
- worker shutdown

This is the dedicated queue class the user explicitly requested.

### 2. Scheduling state

Responsibility:

- authoritative per-cell `required_depth`
- queued / processing / retired task state
- stale-task detection
- cheap deduplication of queue requests

This should likely live in a side-car refinement context rather than directly in
`OctreeCell`.

### 3. Structural octree mutation

Responsibility:

- splitting a leaf into 8 children
- allocating child cell slots
- allocating contributor spans
- updating parent/child relationships
- publishing new leaves to lookup structures

### 4. Neighbor propagation

Responsibility:

- starting from the expected adjacent branch on a given child face
- descending only through overlapping branches
- pruning immediately when the current node already satisfies the demanded
  depth
- queueing only leaves whose required depth increases

## Proposed File Layout

The exact names can still change, but the separation should look like this.

### Required new files

- `src/meshmerizer/adaptive_cpp/refinement_work_queue.hpp`
- `src/meshmerizer/adaptive_cpp/refinement_work_queue.cpp`

The queue class must live here.

### Likely supporting files

- `src/meshmerizer/adaptive_cpp/refinement_context.hpp`
- `src/meshmerizer/adaptive_cpp/refinement_context.cpp`
- `src/meshmerizer/adaptive_cpp/refinement_closure.hpp`
- `src/meshmerizer/adaptive_cpp/refinement_closure.cpp`

Whether the closure logic remains header-only is less important than the queue
separation, but the queue itself must have its own header and implementation.

### Build-system work

Because the adaptive code currently leans heavily on header-only implementation,
adding `refinement_work_queue.cpp` will require updating the extension build so
the new translation unit is compiled and linked into `_adaptive`.

This build integration must be part of the first implementation phase rather
than left for the end.

## Repository-Specific Mapping

This section maps the abstract phases above onto the actual files and functions
currently in this repository.

### Existing files that will be touched first

#### `setup.py`

Current role:

- defines the `_adaptive` extension
- currently compiles only `src/meshmerizer/_adaptive.cpp`

Required changes:

- add `src/meshmerizer/adaptive_cpp/refinement_work_queue.cpp`
- add any future `.cpp` files created for refinement context / closure logic
- keep include path handling unchanged unless the new translation units require
  it

This is the build-system choke point for the new queue class implementation.

#### `src/meshmerizer/adaptive_cpp/octree_cell.hpp`

Current role:

- defines `OctreeCell`
- owns `split_octree_leaf(...)`
- owns the current balance spatial hash and `balance_octree(...)`
- owns the current `refine_octree(...)` implementation

Functions/types that will be directly affected:

- `split_octree_leaf(...)`
- `neighbor_morton_key(...)`
- `leaf_cells_share_face(...)`
- `BalanceSpatialHash`
- `needs_balance_split(...)`
- `enqueue_balance_neighbors(...)`
- `balance_octree(...)`
- both overloads of `refine_octree(...)`

Expected direction:

- keep low-level geometry / contributor helpers here initially
- extract or replace the queue/scheduler-specific logic
- eventually make `refine_octree(...)` delegate to a closure engine rather than
  embedding BFS-plus-post-balance itself
- keep old balance helpers only as long as tests/debug validation still need
  them

#### `src/meshmerizer/adaptive_cpp/adaptive_solid.hpp`

Current role:

- owns regularization-specific local refinement entry points
- currently uses private queues and then relies on `balance_octree(...)`

Functions that will need rewriting:

- `refine_surface_band_cells(...)`
- `refine_thickening_band_cells(...)`

Expected direction:

- stop open-coding local queues and target-depth tracking here
- route both functions through the same closure-based scheduler used by initial
  refinement
- preserve the policy for deciding *which* cells need pressure, but hand the
  actual split-and-balance closure to the new engine

#### `src/meshmerizer/adaptive_cpp/dc_pipeline.hpp`

Current role:

- runs the end-to-end adaptive pipeline
- explicitly calls `refine_octree(...)`
- later explicitly calls `balance_octree(...)`
- repeats refine-then-balance loops during regularization

Functions/regions that will change:

- the initial refinement block around `run_dc_pipeline(...)`
- the regularization loop after `refine_surface_band_cells(...)`
- the pre-thickening loop after `refine_thickening_band_cells(...)`

Expected direction:

- remove production balance calls from this orchestration path
- replace status messages that refer to a separate final balance pass
- keep timing instrumentation, but shift it toward integrated closure timings
- plumb queue table-cadence configuration into the closure/reporting path

#### `src/meshmerizer/adaptive_cpp/faces.hpp`

Current role:

- contains topology-repair helpers for missing incident cells
- still calls `split_octree_leaf(...)`
- currently invokes `balance_octree(...)` after zero-sample incident refinement

Functions that will need attention:

- `refine_zero_sample_incident_cells(...)`

Expected direction:

- this helper should eventually invoke the same closure-based split path rather
  than split-then-balance manually
- this is likely a later integration step, not a phase-1 change

#### `src/meshmerizer/_adaptive.cpp`

Current role:

- Python bindings for `refine_octree(...)`
- Python binding for `run_full_pipeline(...)`
- direct use of `refine_octree(...)`, `refine_surface_band_cells(...)`, and
  `balance_octree(...)`

Functions/regions that will change:

- `refine_octree_py(...)`
- `run_full_pipeline_py(...)`

Expected direction:

- bindings should continue to expose the same Python-facing contract where
  possible
- implementation underneath should route to the new closure engine
- add plumbing for the table-cadence option through the native pipeline entry
  points
- native diagnostic/debug entry points may be added later if needed, but should
  not be phase-1 scope

#### `src/meshmerizer/cli/*.py`

Current role:

- user-facing CLI argument parsing and pipeline invocation

Expected direction:

- add `--table-cadence`
- default it to `20`
- pass it into the adaptive pipeline so queue-driven refinement uses the chosen
  reporting cadence

#### `tests/test_adaptive_core.py`

Current role:

- tests `refine_octree(...)`
- contains balance-invariant checks already useful for this rewrite

Tests/helpers already relevant:

- `_check_balance_invariant(...)`
- `test_refine_octree_satisfies_balance_invariant_with_surface`
- `test_refine_octree_balance_does_not_exceed_max_depth`
- `test_refine_octree_respects_max_depth`
- `test_refine_octree_quality_stop_preserves_mixed_leaf_depths`

Expected direction:

- extend this file first for closure-specific correctness coverage
- add tests for local propagation and stale-task safety at the Python-visible
  `refine_octree(...)` layer where possible

#### `tests/test_integration.py`

Current role:

- checks watertightness and end-to-end regularization behavior

Tests already relevant:

- `test_regularized_sphere_remains_watertight`

Expected direction:

- keep this file as the guardrail that the rewrite does not degrade final mesh
  quality

### Likely new files and their responsibilities

#### `src/meshmerizer/adaptive_cpp/refinement_work_queue.hpp`

- queue class declaration
- task payload declaration
- public queue API
- stats / shutdown / idle detection interface

#### `src/meshmerizer/adaptive_cpp/refinement_work_queue.cpp`

- queue implementation
- internal synchronization primitives
- batching / wakeup / shutdown logic

#### `src/meshmerizer/adaptive_cpp/refinement_context.hpp`

- side-car scheduling state
- `required_depth`
- task-state tracking
- optional generation counters
- storage views for cells/contributors plus helper mutation API

#### `src/meshmerizer/adaptive_cpp/refinement_context.cpp`

- implementation of context-side helpers
- resizing / state initialization routines
- maybe helper wrappers around safe child publication

#### `src/meshmerizer/adaptive_cpp/refinement_closure.hpp`

- public entry points for closure-based refinement
- serial and later parallel driver declarations
- small policy types or refinement reason flags if needed

#### `src/meshmerizer/adaptive_cpp/refinement_closure.cpp`

- worker/task processing logic
- integrated split + neighbor propagation loop
- tree-guided traversal for balance closure

## Function-by-Function Rewrite Plan

This section is intentionally concrete so implementation can proceed in small,
reviewable steps.

### A. Build-system preparation

#### `setup.py`

Change first:

- expand `Extension(..., sources=[...])` beyond `src/meshmerizer/_adaptive.cpp`

Reason:

- the queue must have its own `.cpp`
- this is a hard prerequisite for every later queue implementation commit

### B. Isolate reusable split mechanics

#### `split_octree_leaf(...)` in `octree_cell.hpp`

Current problem:

- it mutates the flat vectors directly in a way that is convenient for serial
  callers but not yet suitable for a structured closure engine

Planned changes:

- keep it available at first as the serial mutation primitive
- add a more structured return shape or helper that reports child indices
- later move storage mutation behind a context/allocator abstraction

Why first:

- every new closure worker will need one canonical split primitive

### C. Introduce the scheduling side-car

#### New refinement context

Needs to wrap or reference:

- `std::vector<OctreeCell> &all_cells`
- `std::vector<std::size_t> &all_contributors`
- positions and smoothing lengths
- refinement thresholds
- per-cell `required_depth`
- per-cell task state

Likely first users:

- a new serial closure driver
- later `refine_surface_band_cells(...)`
- later `refine_thickening_band_cells(...)`

### D. Replace `refine_octree(...)` internals

#### `refine_octree(...)` in `octree_cell.hpp`

Current structure:

- breadth-first batches
- direct split decision logic
- full post-pass `balance_octree(...)`

Planned sequence:

1. keep signature stable
2. move current split-decision logic into a helper reusable by closure workers
3. route the implementation through `refine_with_closure(...)`
4. remove the production tail call to `balance_octree(...)`

This is the main inflection point of the rewrite.

### E. Replace local regularization queues

#### `refine_surface_band_cells(...)` in `adaptive_solid.hpp`

Current structure:

- builds a local `std::queue<std::size_t>`
- splits selected leaves directly
- resamples children inline
- relies on later global balancing elsewhere

Planned changes:

- keep the surface-band *selection policy*
- replace the private queue/split loop with pressure submission into the common
  closure engine
- stop requiring a later explicit global balance step from the caller

#### `refine_thickening_band_cells(...)` in `adaptive_solid.hpp`

Current structure:

- builds a local queue
- tracks `queued_target_depths`
- open-codes a weak version of per-cell minimum required depth
- still depends on later `balance_octree(...)`

Planned changes:

- migrate its `queued_target_depths` idea into the general
  `required_depth` side-car
- use the closure engine directly rather than maintaining a second bespoke
  scheduler
- keep `dirty_cells` output behavior if later classification still needs it

This function is especially important because it already contains the beginnings
of the general idea.

### F. Remove orchestration-level balance calls

#### `run_dc_pipeline(...)` in `dc_pipeline.hpp`

Planned changes:

- replace log messages about "starting final balance"
- remove the explicit `balance_octree(...)` after surface-band refinement
- remove the explicit `balance_octree(...)` inside pre-thickening loops
- replace them with integrated closure timings and status logs

### G. Remove repair-path balance calls

#### `refine_zero_sample_incident_cells(...)` in `faces.hpp`

Current structure:

- directly splits topology-required leaves
- then calls `balance_octree(...)`

Planned changes:

- route these topology-driven splits through the closure engine as well
- ensure this path still terminates and leaves the tree 2:1 balanced

This should probably happen after the main and regularization paths are stable.

### H. Bindings and Python-visible contract

#### `refine_octree_py(...)` in `_adaptive.cpp`

Planned changes:

- keep the Python signature if possible
- let the C++ implementation switch underneath to closure-based refinement

#### `run_full_pipeline_py(...)` in `_adaptive.cpp`

Planned changes:

- remove explicit post-refinement balance sequencing from the bound native path
- keep return structure stable

## Commit-Oriented Implementation Plan

The rewrite should be broken into small commits that are individually readable
and, where possible, individually testable. The one-line commit messages below
are written in a concise style consistent with the repository's recent history.

### Commit 1

Message:

`add repo-specific plan for refinement closure rewrite`

Scope:

- update this plan with the repository mapping and commit breakdown

### Commit 2

Message:

`add debug helpers for closure rewrite validation`

Scope:

- extend `tests/test_adaptive_core.py` with reusable invariant/debug helpers if
  needed
- optionally add native debug instrumentation hooks only if they are cheap and
  clearly isolated
- no behavior change yet

### Commit 3

Message:

`build adaptive extension from multiple translation units`

Scope:

- update `setup.py` to compile new `.cpp` files
- add placeholder queue `.hpp/.cpp` files if needed to exercise the build path
- no algorithm change yet

### Commit 4

Message:

`add refinement work queue class`

Scope:

- create `refinement_work_queue.hpp`
- create `refinement_work_queue.cpp`
- implement the initial queue API
- add focused native/unit coverage if practical

### Commit 5

Message:

`add refinement context sidecar state`

Scope:

- create `refinement_context.hpp/.cpp`
- add per-cell `required_depth`
- add queue/task state tracking
- keep it unused by production code initially if that simplifies review

### Commit 6

Message:

`factor split helpers for closure-driven refinement`

Scope:

- refactor `split_octree_leaf(...)` or adjacent helpers so closure workers can
  consume them cleanly
- return/report child index ranges explicitly
- no behavior change intended

### Commit 7

Message:

`add serial refinement closure driver`

Scope:

- create `refinement_closure.hpp/.cpp`
- implement serial task processing using the new queue/context
- do not switch `refine_octree(...)` over yet

### Commit 8

Message:

`route refine_octree through serial closure engine`

Scope:

- make `refine_octree(...)` delegate to the serial closure driver
- preserve existing external API
- keep old `balance_octree(...)` available for comparison/debug only

### Commit 9

Message:

`add closure tests for local balance propagation`

Scope:

- extend `tests/test_adaptive_core.py` with cases targeting local 2:1 closure
- verify no explicit global balance pass is needed for `refine_octree(...)`

### Commit 10

Message:

`implement tree-guided neighbor closure traversal`

Scope:

- replace any coarse placeholder propagation with overlap-guided branch descent
- add pruning by demanded depth
- strengthen tests around pathological face coverage

### Commit 11

Message:

`route surface-band refinement through closure engine`

Scope:

- rewrite `refine_surface_band_cells(...)` to submit work into the common
  closure engine
- remove the need for a later caller-level balance step for this path

### Commit 12

Message:

`route thickening-band refinement through closure engine`

Scope:

- rewrite `refine_thickening_band_cells(...)` around shared `required_depth`
  machinery
- preserve `dirty_cells` behavior as needed

### Commit 13

Message:

`remove production balance calls from dc pipeline`

Scope:

- update `dc_pipeline.hpp`
- remove explicit `balance_octree(...)` calls from the main pipeline flow
- update log messages and timings
- begin replacing queue-phase progress output with periodic table reporting

### Commit 13a

Message:

`add queue table cadence reporting option`

Scope:

- add `--table-cadence` to the CLI
- plumb the cadence through Python/native pipeline entry points
- add the reporting configuration surface before or alongside the first real
  queue-driven table output

### Commit 14

Message:

`route zero-sample incident refinement through closure`

Scope:

- update `faces.hpp`
- remove split-then-balance behavior from the zero-sample topology repair path

### Commit 15

Message:

`add parallel-safe storage for closure workers`

Scope:

- introduce storage/allocator changes needed for concurrent child publication
- keep algorithm behavior equivalent in serial mode if parallel mode is not yet
  enabled

### Commit 16

Message:

`enable parallel refinement closure workers`

Scope:

- run multiple workers against the shared queue
- implement stale-task discard and queue idle handling for parallel execution
- add thread-count regression coverage where practical

### Commit 17

Message:

`validate closure determinism and mesh stability`

Scope:

- expand tests across thread counts if possible
- ensure integration coverage still checks watertightness and stability
- add any debug-only invariant verification hooks needed during development

### Commit 18

Message:

`remove obsolete post-pass balance scheduling code`

Scope:

- delete superseded balance scheduling paths
- keep only the balance code still justified for validation, fallback, or tests

## Notes on Commit Boundaries

- Commits 4 through 8 should aim to keep the code serial and easier to reason
  about.
- Commits 11 through 14 are the main integration wave where the closure engine
  becomes the single refinement path.
- Commits 15 through 18 are the concurrency and cleanup wave.
- If one of these chunks proves too large in practice, split it further rather
  than broadening the commit message beyond a single logical step.

## Test Mapping by Phase

### Early phases

- `tests/test_adaptive_core.py`
  - `test_refine_octree_returns_both_cells_and_contributors`
  - `test_refine_octree_respects_max_depth`
  - `test_refine_octree_satisfies_balance_invariant_with_surface`
  - `test_refine_octree_balance_does_not_exceed_max_depth`

### Mid phases

- add new `test_refine_octree_*` cases for local closure propagation
- add new tests covering surface-band and thickening-band integrated closure

### Late phases

- `tests/test_integration.py`
  - `test_regularization_runs_and_returns_mesh`
  - `test_regularized_sphere_remains_watertight`

## Immediate Next Files To Edit

If implementation starts immediately after this planning update, the first files
to touch should be:

1. `setup.py`
2. `src/meshmerizer/adaptive_cpp/refinement_work_queue.hpp`
3. `src/meshmerizer/adaptive_cpp/refinement_work_queue.cpp`
4. `src/meshmerizer/adaptive_cpp/refinement_context.hpp`
5. `src/meshmerizer/adaptive_cpp/refinement_context.cpp`
6. `src/meshmerizer/adaptive_cpp/refinement_closure.hpp`
7. `src/meshmerizer/adaptive_cpp/refinement_closure.cpp`
8. `src/meshmerizer/adaptive_cpp/octree_cell.hpp`
9. `tests/test_adaptive_core.py`

## Data Model Plan

### Keep `OctreeCell` focused on geometric state

Avoid adding queue-only atomics directly to `OctreeCell` unless absolutely
necessary.

Reasons:

- `OctreeCell` is copied, serialized, and exposed through existing APIs
- atomics complicate copying and storage
- transient scheduling state should not leak into persistence formats

### Introduce transient side-car state per cell

Recommended transient fields:

- `required_depth`
- `task_state` (`idle`, `queued`, `processing`, `retired` or equivalent)
- optional `generation` or `version` counter for debugging stale tasks
- optional `dirty` / `touched` markers for later incremental classification work

This side-car state should be allocated alongside the evolving cell storage and
resized as new cells are created.

### Stable cell identifiers

The new system should operate on stable cell indices / ids.

This implies:

- append-only storage during refinement
- no invalidation of previously published cell ids
- child creation that returns stable indices immediately usable by workers

## Queue Class Requirements

The queue class should be specialized enough to support the refinement engine
cleanly, but it should not own geometric policy.

### Minimum API responsibilities

- push one task
- push a small batch of tasks
- pop one task or a small batch
- report / detect global idle state
- support orderly worker shutdown
- provide cheap statistics hooks for debugging and profiling
- provide enough queue statistics for periodic table reporting

### Likely task payload

At minimum:

- `cell_index`

Optional debugging fields:

- reason flags
- enqueue generation
- demanded depth snapshot

The queue should **not** treat the task payload as the source of truth. The
source of truth remains the current side-car state for that cell.

### Implementation strategy

Start simple and correct:

- central thread-safe MPMC queue
- batched push/pop where helpful
- clear idle detection semantics

Once the algorithm is correct, the queue class can later be upgraded to
work-stealing internally if profiling shows the central queue is a bottleneck.

The public interface should make that upgrade possible without changing the
closure logic.

## Neighbor-Propagation Plan

This is the most important algorithmic piece after the queue itself.

### Required behavior

When a parent cell splits:

- examine each new child
- for each of the 6 faces, locate the adjacent octree branch
- recurse only through nodes whose bounds overlap that face region
- stop immediately when a node already satisfies the demanded minimum depth
- if a leaf is too coarse, raise its `required_depth` and schedule it

### Important constraint

Do **not** reduce this to "probe one point on the face".

That is not sufficient for arbitrary local configurations.

The traversal must reason over the whole overlapping face region, but it should
do so by descending the tree, not by scanning finest-grid samples.

### Recommended recursion shape

Given:

- a child face region
- a demanded neighbor depth
- an adjacent candidate branch

Algorithm:

1. If the branch does not overlap the face region, return.
2. If the branch is a leaf:
   - if `leaf.depth >= demanded_depth`, return
   - otherwise atomically raise `required_depth`
   - if the requirement increased enough to matter, schedule the leaf
3. If the branch is internal:
   - recurse only into overlapping children
   - prune early if an entire subtree is already known to satisfy the demand

### Starting-point optimization

Use octree-lattice adjacency to jump directly to the expected neighboring branch
at the relevant level, then descend only as needed.

Do not start from the whole tree root if a same-depth adjacent parent-aligned
candidate can be computed cheaply.

This is the part of the design closest to the original user proposal and should
be kept.

## Structural-Mutation Plan

The algorithm only works well in parallel if cell creation is made safe and
predictable.

### Requirements

- concurrent child allocation must not invalidate existing cell ids
- contributor storage must support concurrent append / reservation
- child publication must be atomic from the perspective of other workers
- lookup structures must not expose half-built children

### Recommended approach

Introduce an append-only arena or chunked storage abstraction for:

- cells
- contributor indices
- side-car refinement state

Avoid naive concurrent `std::vector::push_back(...)` on shared containers.

### Determinism

Parallel append order may vary by thread count.

We therefore need an explicit determinism strategy:

- either deterministic slot reservation / publication during build
- or a post-build canonicalization pass that reorders cells into a stable order
  such as Morton/depth order before serialization and downstream comparison

The chosen strategy must be written down here once decided.

## Implementation Phases

## Phase 0 - Baseline and Safety Rails

- [ ] Add or improve timing around initial refinement and balance work so the
      current baseline is measurable.
- [ ] Add a reusable 2:1 invariant checker for debug/testing paths.
- [ ] Record the current serial behavior of representative regression cases.

## Phase 1 - Queue and Scheduling Skeleton

- [x] Add `refinement_work_queue.hpp`.
- [x] Add `refinement_work_queue.cpp`.
- [x] Update the extension build so the queue implementation is compiled.
- [x] Add a refinement-context side-car state structure.
- [ ] Implement monotone `required_depth` updates.
- [ ] Implement queue-state transitions (`idle` / `queued` / `processing`).
- [ ] Keep this phase independent of full parallel splitting at first.
- [ ] Design queue/context statistics around periodic table reporting rather
      than progress counters.

### Phase 1 progress update

- `RefinementContext` now supports:
  - monotone required-depth raises
  - queued / processing / idle / retired task-state transitions
- `refine_octree(...)` now routes through the closure façade, but the closure
  driver still delegates to the legacy implementation to preserve behavior while
  the real worker logic is introduced incrementally.
- Compiler-backed extension rebuild succeeded after this routing change.
- Targeted `tests/test_adaptive_core.py` refinement/balance coverage remains
  green.

### Phase 1 implementation notes

- `setup.py` now builds `_adaptive` from multiple translation units.
- Added:
  - `src/meshmerizer/adaptive_cpp/refinement_work_queue.hpp`
  - `src/meshmerizer/adaptive_cpp/refinement_work_queue.cpp`
  - `src/meshmerizer/adaptive_cpp/refinement_context.hpp`
  - `src/meshmerizer/adaptive_cpp/refinement_context.cpp`
  - `src/meshmerizer/adaptive_cpp/refinement_closure.hpp`
  - `src/meshmerizer/adaptive_cpp/refinement_closure.cpp`
- The closure driver is still only a placeholder abstraction boundary. No
  production refinement logic has been rerouted yet.
- Targeted Python tests for the existing `refine_octree(...)` balance behavior
  still pass.
- `python -m build` could not be run in the current environment because the
  `build` module is not installed.

## Phase 2 - Serial Integrated Closure Prototype

- [ ] Implement a serial `refine_with_closure(...)` path using the new queue.
- [ ] Make it replace `refine_octree(...) + balance_octree(...)` logically,
      while still running on one thread.
- [ ] Propagate balance requirements immediately after each split.
- [ ] Preserve current split heuristics for surface / Hermite / QEF criteria.
- [ ] Verify that the final tree satisfies the 2:1 invariant with no extra
      global balance pass.

This phase is about algorithmic correctness, not speed.

### Phase 2 progress update

- The serial split-decision loop from the old refinement path has been migrated
  into `refine_with_closure(...)`.
- Child creation, contributor propagation, and child queueing now happen inside
  the closure driver.
- The closure driver now performs its own serial balance-closure phase using the
  existing spatial-hash neighbor logic as a stepping stone toward the final
  fully integrated scheduler.
- The final `balance_octree(...)` post-pass call has been removed from the main
  `refine_with_closure(...)` path.
- The balance path now more directly uses split-driven neighbor scheduling and
  reusable `required_depth` state instead of relying solely on a broad second
  queue seeding pass.
- The serial closure path is now effectively a single queue-driven loop in which
  surface-driven splits and balance-driven splits can both occur during the same
  refinement run.
- Local balance propagation has been strengthened to probe multiple locations
  across each face instead of only one face-center style probe.
- Local balance propagation now uses a recursive face-patch traversal inside the
  current serial closure scheduler, reducing reliance on fixed representative
  face samples.
- Added targeted adaptive-core regressions covering tougher multi-particle face
  and corner refinement layouts while continuing to assert the 2:1 invariant.
- Child publication and contributor append logic are now isolated behind
  clearer helper boundaries inside the closure engine, reducing the surface area
  that later parallel-safe storage changes must touch.
- Targeted adaptive-core refinement tests still pass.
- A watertight integration regression test still passes.

### Phase 2 completion note

- The serial closure engine now owns both surface refinement and balance
  refinement in one queue-driven flow.
- The explicit production `balance_octree(...)` post-pass has been removed from
  `refine_with_closure(...)`.
- Current targeted correctness and watertightness regressions are green.
- Phase 2 is now treated as complete enough to proceed to Phase 3 cleanup and
  traversal refinement.

## Phase 3 - Tree-Guided Neighbor Traversal

- [x] Implement the face-overlap branch traversal described above.
- [x] Avoid finest-grid raster scanning.
- [x] Add pruning based on `required_depth >= demanded_depth`.
- [x] Add tests covering pathological coarse/fine propagation layouts.
- [x] Confirm that balance closure is local and monotone.

### Phase 3 completion note

- The serial closure scheduler now uses recursive face-patch traversal rather
  than single-point face probing.
- Queue-driven balance propagation prunes duplicate work using existing
  `required_depth` state.
- Additional adaptive-core regressions were added for more difficult face and
  corner refinement layouts.
- Focused adaptive-core and watertight integration regressions are green.

## Phase 4 - Parallel Structural Mutation

- [x] Introduce a closure-side publisher abstraction.
- [x] Introduce a closure-side storage backend abstraction.
- [x] Introduce explicit reservation/publication boundaries for child slots.
- [x] Add queue in-flight accounting and atomic shutdown-if-idle semantics.
- [x] Introduce closure worker state and shared worker-step logic.
- [x] Add a conservative threaded execution scaffold behind `worker_count`.
- [x] Add coarse shared-mutation serialization for the threaded scaffold.
- [x] Add a low-level threaded sphere smoke test.
- [x] Finish the mutation/publication boundary review so child publication cannot
      expose partial state to future finer-grained workers.

### Phase 4 progress update

- Child publication and contributor append logic have been moved behind a
  closure-side publisher abstraction.
- Storage mutation has been separated from publication policy via a
  `ClosureStorage` backend.
- Reservation/publication boundaries now exist explicitly for contributors and
  child slots, with validation for child-slot publication order.
- The serial closure loop has been refactored into clearer worker/coordinator
  helpers, which should make later multi-worker execution easier to introduce.
- A conservative threaded execution scaffold now exists behind
  `RefinementClosureConfig::worker_count`, but shared mutation is still
  serialized and the path is not yet treated as production-ready parallelism.
- Queue lifecycle has been strengthened with in-flight accounting and atomic
  shutdown-if-idle semantics.
- A low-level threaded sphere smoke test has succeeded through the native
  `refine_octree` path with `worker_count=2`; this is a milestone for scaffold
  viability, but not yet a substitute for real regression coverage.
- A batched-contributor-publication experiment was attempted and then reverted
  after exposing instability in the regularization/integration path; for now,
  contributor publication remains structurally prepared for batching but not
  forced into a stronger ordering model.

### Phase 4 exit criteria

Phase 4 is complete when all of the following are true:

1. shared mutation paths have explicit ownership boundaries
2. queue lifecycle semantics are robust enough for multiple workers to drain the
   queue without obvious race/lifecycle bugs
3. the low-level threaded sphere regression passes reliably
4. focused adaptive-core and watertight integration regressions remain green

### Phase 4 completion note

- Phase 4 is now considered complete.
- The closure engine has explicit publication/storage abstractions, worker state,
  queue lifecycle hardening, conservative shared-mutation serialization, and a
  threaded sphere regression.
- The code is now ready to move into Phase 5 thread-count robustness work.

## Phase 5 - Parallel Workers

- [x] Add a direct correctness test comparing low-level refinement with
      `worker_count=1` and `worker_count=2` on the same sphere setup.
- [x] Verify both threaded and serial runs satisfy the 2:1 balance invariant.
- [x] Verify both runs respect `max_depth`.
- [x] Verify both runs complete without deadlock or livelock.
- [x] Decide and document whether exact append-order determinism is required or
      whether compatibility-level equivalence is sufficient.
- [x] Add at least one repeated threaded regression to catch flaky queue/worker
      behavior.
- [x] Measure basic contention/scaling once correctness is trusted.

### Phase 5 compatibility target

For Phase 5, the required target is **compatibility-level equivalence**, not
exact append-order determinism.

That means threaded and serial low-level refinement runs must agree on:

- validity of the 2:1 balance invariant
- respect for `max_depth`
- successful completion without deadlock/livelock
- broadly compatible refinement outcome metrics such as non-zero output and
  comparable leaf/cell counts when using the same problem setup

Exact cell append order is **not** currently required for Phase 5 completion.

### Phase 5 exit criteria

Phase 5 is complete when all of the following are true:

1. low-level threaded refinement has dedicated correctness regressions
2. thread-count robustness checks pass for at least `worker_count=1` and `2`
3. the acceptable determinism/compatibility target is explicitly documented
4. no deadlock/livelock issues are observed in the threaded regression set

### Phase 5 completion note

- Low-level threaded refinement now has a sphere smoke test, a serial-vs-threaded
  compatibility regression, and a repeated threaded regression.
- The accepted Phase 5 target is compatibility-level equivalence rather than
  exact append-order determinism.
- The conservative threaded scaffold now passes thread-count robustness checks
  for `worker_count=1` and `2` on the sphere setup.

## Phase 6.5 - Work-stealing DFS Scheduler Upgrade

Replace the conservative "shared worker state + central deque" transport with a
*genuinely parallel* scheduler:

- **Per-cell demand remains authoritative**: every cell has a monotone
  `required_depth` value stored in side-car scheduling state.
- **Demand propagation is monotone**: updates are `required_depth =
  max(required_depth, new_requirement)`.
- **Upward propagation**: `required_depth` is tracked for all cells (not just
  leaves) and parent `required_depth` is maintained as the max of its
  descendants. This supports pruning and prevents redundant re-wakes when work
  moves between leaves and their ancestors.

### New scheduling model

1. **Task granularity**

   A queued task represents "make this cell satisfy its current required depth
   and surface policy".

   - Each worker processes tasks *depth-first locally*.
   - When a task splits a cell, the worker immediately continues into one child
     (DFS), and pushes the remaining child tasks onto its own deque.

2. **Work-stealing transport**

   - Use **thread-local deques** (one per worker): owner pops/pushes at the back
     (LIFO) to preserve DFS locality.
   - Idle workers **steal from the front** of other deques (FIFO) to reduce
     contention and improve global progress.
   - Avoid a single central mutex-protected deque on the hot path.

3. **Wakeups are requirement-driven**

   - Updating `required_depth` does *not* imply queue membership; queue tasks are
     hints.
   - **Only enqueue a cell when its `required_depth` increases beyond its
     current state** (or when it first needs surface evaluation).
   - Balance neighbor propagation should therefore:
     1. compute the demanded neighbor depth,
     2. perform an atomic `max` raise on neighbor `required_depth`,
     3. enqueue only if the raise succeeded and the neighbor is currently idle.

4. **Synchronization discipline**

   - Prefer **atomic state transitions** for per-cell scheduling state:
     `idle -> queued -> processing -> (idle|retired)`.
   - Structural mutation (cell splits + child publication + spatial-hash edits)
     may still require a narrow critical section.
   - Termination should be based on "all deques empty AND no in-flight tasks"
     with one final join, not global barrier steps.

### Storage decision: flat cell array

We should avoid a single global flat `cells[]` array *if practical*, because
contiguous `push_back` mutation forces broad locking and makes lock-free reads
hard.

However, **if integration compatibility requires stable contiguous child
storage**, keep the current invariant:

- children of an internal cell occupy **8 consecutive slots**, with
  `child_begin` pointing at the first.

Justification for keeping contiguous child storage (if we do):

- downstream code assumes constant-time child indexing via
  `child_begin + octant` without indirection,
- Python bindings currently expose a flat list-like view of cells,
- changing this would be a much larger integration risk than changing the
  scheduler.

If we keep the flat array, we should mitigate it by:

- reserving capacity where feasible,
- narrowing mutation critical sections to *publication only*,
- moving expensive per-leaf evaluation outside the mutation lock.

### Phase 6.5 exit criteria

Phase 6.5 is complete when all of the following are true:

1. the production closure engine uses a work-stealing scheduler with thread-local
   deques
2. `worker_count > 1` removes effective serialization in the worker loop
3. queueing is strictly requirement-driven (enqueue only on successful
   `required_depth` raises / first evaluation)
4. the threaded adaptive-core regressions remain green

## Phase 6 - Pipeline Integration

- [x] Confirm all initial octree refinement entry points use the closure engine.
- [x] Replace surface-band regularization refinement with the closure engine.
- [x] Replace thickening/pre-thickening refinement with the closure engine.
- [ ] Keep debug-only invariant verification available after those replacements.
- [x] Replace queue-phase `ProgressCounter` output with periodic table rows.
- [x] Add and wire the `--table-cadence` CLI argument.
- [x] Verify the regularized sphere integration path still produces a watertight
      mesh after the regularization/pre-thickening routing changes.

### Phase 6 progress update

- Initial octree refinement, surface-band refinement, thickening-band
  refinement, and zero-sample incident refinement now all route through the
  closure engine.
- The zero-sample incident repair path in `faces.hpp` no longer performs
  split-then-balance manually; it now submits those leaves to the same closure
  scheduler.
- Queue-driven refinement now emits coordinator-side table rows containing
  queue size, in-flight work, pushed/popped counts, stale tasks, processed
  leaves, split count, total cell count, required-depth raises, and high-water
  mark.
- The cadence is strictly time-based, exposed through Python/native/CLI
  plumbing as `table_cadence` / `--table-cadence`.
- The regularized sphere watertightness guardrail remains green after the
  reporting and zero-sample incident-path routing changes.

### Phase 6 exit criteria

Phase 6 is complete when all of the following are true:

1. all intended production refinement paths use the closure engine
2. no production path depends on a post-refinement `balance_octree(...)` call
3. periodic table reporting replaces queue-phase progress-counter output
4. the regularized sphere integration path remains green

## Phase 7 - Cleanup

- [x] Delete obsolete balance-specific scheduling code superseded by the closure
      engine.
- [x] Retain only the old balance logic still justified for validation/tests.
- [x] Remove dead or misleading helper paths introduced during intermediate
      migration stages.
- [x] Update design docs and developer-facing architecture notes.

### Phase 7 progress update

- `_adaptive.cpp` topology/extraction helper paths now call the closure-backed
  refinement entry points directly and no longer stage an explicit production
  `balance_octree(...)` step.
- `octree_cell.hpp` no longer carries the obsolete serial balance scheduler or
  the legacy breadth-first `refine_octree_legacy_impl(...)` path.
- Removed cleanup targets include `needs_balance_split(...)`,
  `enqueue_balance_neighbors(...)`, `balance_octree(...)`, and
  `refine_octree_legacy_impl(...)`.
- Retained helpers are limited to pieces still used by the final architecture:
  `split_octree_leaf(...)` remains a structural split primitive and
  `BalanceSpatialHash` remains the closure engine's leaf lookup structure.
- Developer-facing notes in `octree_cell.hpp` now describe closure propagation
  support rather than the old refine-then-balance architecture.

### Phase 7 exit criteria

Phase 7 is complete when all of the following are true:

1. no obsolete production balance-scheduling path remains
2. remaining old balance code has an explicit validation/test justification
3. docs describe the final architecture rather than transitional states

## Phase 8 - Depth-first work-stealing redesign

This phase replaces the current queue-centric closure scheduler with the model
we agreed on during review:

- each task should refine depth-first locally rather than paying queue overhead
  per cell
- `required_depth` should live on every cell, not just leaves
- a parent's `required_depth` should be the max of the requirements in its
  descendants so any task can cheaply decide whether it must recurse further
- queueing should only wake cells whose `required_depth` increased beyond their
  currently satisfied depth/state
- workers should own thread-local deques and steal work only when their local
  deque is empty
- scheduler termination should be based on sleeping/active worker state, not on
  central queue polling loops

### Target architecture

#### Cell state model

Every cell must eventually carry enough scheduling state to answer these
questions without consulting a global scheduler lock:

1. what is the deepest refinement demanded anywhere under this cell?
2. is this cell already queued or being processed?
3. does this cell already own children?

Concretely:

- `required_depth[cell]` is authoritative and monotone
- for internal cells, `required_depth[cell] = max(required_depth[children])`
- worker/scheduler state should be atomic and monotone where possible:
  - `idle`
  - `queued`
  - `processing`
  - `retired/satisfied`

#### Task execution model

A task no longer means "process exactly one cell".

Instead, a task means:

1. claim one cell
2. evaluate whether it is already satisfied
3. if it must split, split it
4. continue locally depth-first into its children while local budget permits
5. only spill work back to a deque when:
   - local DFS budget is exceeded, or
   - a neighbor/ancestor wake-up must be handed to another worker

This is the main performance goal of Phase 8: the queue should distribute coarse
work, not micromanage every leaf.

#### Queue / scheduler model

- one deque per worker
- owner pops/pushes at the back (LIFO) for DFS locality
- thieves steal from the front (FIFO) for load balancing
- workers sleep when they have no local or stealable work
- termination happens when:
  - all deques are empty
  - no worker is actively processing
  - all workers are sleeping

### Storage direction

Longer term we want to avoid one globally mutated flat `cells[]` array because
it forces broad synchronization and makes true local ownership difficult.

Desired direction:

- each cell owns its progeny
- child publication should be one-way and stable
- positional queries should descend the tree structurally rather than relying
  on a globally rebuilt balance-era leaf map

However, to reduce integration risk, we may stage this in two steps:

1. **scheduler-first transition**
   - keep current contiguous child storage invariants temporarily
   - remove per-cell queue traffic and broad scheduler serialization first
2. **storage transition**
   - move toward cell-owned progeny / stable child blocks / positional descent

If we keep flat storage temporarily, it must be justified as a transitional
compatibility choice only, not as the intended final architecture.

### Detailed implementation steps

#### Step 1 - Record the agreed architecture in this plan

- [x] Write down the DFS work-stealing design and explicit exit criteria.

#### Step 2 - Stop using the queue as a per-cell executor

- [~] change closure tasks so one popped task continues depth-first locally
      through children
  - local DFS continuation exists now for freshly split children and for waking
    already-internal cells whose descendant `required_depth` increased
  - still not complete because neighbor wakeups and split publication still rely
    on broad shared mutation / queue-era assumptions
- [x] add a strict local DFS work budget to avoid one worker monopolizing a huge
      region forever
- [x] spill overflow work back to the owning deque only when the local budget is
      exceeded

#### Step 3 - Make required-depth propagation the main scheduling primitive

- [~] ensure `required_depth` exists meaningfully for internal cells as well as
      leaves
  - upward propagation now raises ancestors too, and internal cells can be
    re-awoken to continue DFS into descendants
  - still incomplete because the state model is not yet lock-free / atomic and
    some publication paths still assume leaf-centric handling
- [x] propagate required-depth raises upward so any ancestor can answer "do I
      need to recurse further?" cheaply
- [~] enqueue/wake a cell only when its required depth actually increased and it
      is not already awake
  - retired internal cells are now allowed to wake again when descendant demand
    rises, but queue traffic still needs further coarsening

#### Step 4 - Make worker state the basis of termination

- [x] add sleeping-worker tracking to the scheduler
- [ ] replace any remaining queue-centric termination assumptions with
      `all_deques_empty && no_active_workers && all_workers_sleeping`
- [ ] remove short-interval polling / timed wakeups from normal idle behavior

#### Step 5 - Remove effective parallel serialization

- [ ] remove or drastically narrow the global mutation critical section
- [ ] move expensive evaluation work fully outside shared-state locks
- [ ] ensure only publication / atomic state transitions require
      synchronization
- [~] remove or drastically narrow the global mutation critical section
  - queue pushes are now batched outside the mutation lock and internal-cell
    continuation can exit the lock before local DFS enqueue/spill decisions
  - publication/storage mutation and scheduler state mutation are now separated
    into distinct locks
  - still incomplete because flat global storage keeps publication itself
    serialized and balance-hash updates still ride that same storage path
- [x] move expensive evaluation work fully outside shared-state locks
- [~] ensure only publication / atomic state transitions require
      synchronization
  - much closer now, but the current transitional flat-array publication path
    still forces coarse serialized mutation
  - incremental appended-cell registration now avoids rebuilding the entire
    context lookup/state map for every published child batch
  - child-batch contributor publication is now flattened into one contiguous
    append per batch instead of many small reserve/publish steps
- [ ] verify `worker_count > 1` actually improves the threaded smoke path rather
      than merely remaining correct

#### Step 5b - Correct queue seeding and reporting pathologies

- [ ] wire every reported table column to real counters before changing output
- [ ] add a real `surface` processed-leaf counter to distinguish surface work
      from total processed leaves
- [ ] stop seeding the entire initial frontier into the queue for
      `refine_with_closure`
- [ ] replace full-frontier startup with coarse worker seeds that expand via
      local DFS before further queue growth
- [ ] verify the real example queue no longer starts with a giant stuck
      `peak_queue` equal to the initial seed count

#### Step 6 - Plumb real worker-count usage through production

- [x] forward CLI `--nthreads` / pipeline worker count into initial refinement
- [x] forward worker count into regularization / thickening closure paths
- [~] audit all remaining adaptive entry points for hardcoded `worker_count=1`
  - Python bindings for opened-surface extraction and occupied-solid
    classification now accept and forward worker count into refinement
  - remaining hardcoded internal helper paths in `_adaptive.cpp` still need
    audit

#### Step 7 - Prepare the storage transition away from flat global mutation

- [~] design per-cell child ownership / stable child-block publication
  - scheduler-side cell-local child blocks now exist in `RefinementCellState`
    and DFS continuation can consume those stable child indices instead of
    depending only on flat `child_begin` scanning
  - closure DFS continuation no longer falls back to flat `child_begin`
    scanning once a parent has published its stable child block
  - still incomplete because actual child/cell storage is still published into
    the global flat array
- [~] replace balance-hash-centric neighbor wake logic with positional query /
      structural descent where practical
  - closure neighbor wake discovery now uses Morton/depth structural lookup via
    `RefinementContext` instead of probing the balance hash
  - closure publication no longer maintains its own balance-hash state either
  - still incomplete because other adaptive subsystems still rely on spatial
    hash lookup
- [ ] document exactly which flat-array assumptions remain and why
  - current remaining flat-array assumptions:
    - published children are still appended into one global `cells` array
    - published contributor slices are still appended into one global
      `contributors` array
    - parent cells still expose `child_begin` for compatibility with older
      non-closure consumers, even though closure DFS now prefers stable child
      blocks in context state
    - true lock-free cell-owned storage has not yet replaced the shared append
      path used during publication

#### Step 8 - Revalidate performance before broad regression runs

- [ ] do not resume the broader adaptive-core regression subset until the
      threaded smoke path is back down to a few seconds
- [ ] once smoke timing is acceptable again, rerun the focused adaptive-core
      subset
- [ ] only then rerun larger integration checks

### Phase 8 exit criteria

Phase 8 is complete when all of the following are true:

1. one task processes a depth-first local refinement wave rather than a single
   cell only
2. queue traffic is requirement-driven and coarse-grained
3. workers sleep cleanly and termination is based on worker/deque state rather
   than repeated timed polling
4. `worker_count > 1` provides real speedup on the threaded smoke case
5. the threaded smoke test returns to a runtime of only a few seconds

## Validation Plan

We should not trust the rewrite until it is validated at several levels.

### Correctness

- final tree satisfies 2:1 balance invariant
- no leaf exceeds `max_depth`
- no missed propagation across coarse/fine interfaces
- meshing still produces watertight output on the existing regression suite

### Equivalence / compatibility

For a representative set of inputs, compare the new closure engine against the
old refine+balance pipeline.

We do not necessarily need identical append order, but we do need compatible
results:

- same final geometric coverage
- same 2:1 validity
- comparable leaf counts
- no regressions in mesh quality metrics already tested by the suite

### Parallel robustness

- same final invariant regardless of thread count
- no deadlocks
- no livelocks from repeated stale tasks
- bounded duplicate work under clustered refinement pressure

## Risks and Mitigations

### Risk: queue state races

Mitigation:

- keep queue transport separate from authoritative scheduling state
- use monotone `required_depth`
- treat tasks as hints, not truth

### Risk: concurrent append corruption

Mitigation:

- refactor storage before enabling real parallel splits
- use stable append-only arenas / chunked storage

### Risk: missed balance propagation on partially covered faces

Mitigation:

- use overlap-based branch traversal
- forbid one-point face probes as the production algorithm

### Risk: thread-count-dependent final ordering

Mitigation:

- define a deterministic publication or canonicalization strategy

### Risk: the queue becomes the bottleneck

Mitigation:

- start with a simple central queue for correctness
- only optimize to work-stealing if profiling demonstrates need

## Open Questions

- [ ] Should final cell order be made deterministic during build or via a final
      canonicalization pass?
- [ ] Should queue idle detection live inside the queue class or in a higher
      level scheduler wrapper?
- [ ] Which transient state should remain available for later incremental
      classification work?
- [ ] Do we want a generic queue class or a refinement-specific queue class with
      built-in stats and debug labels?

## Definition of Done

This rewrite is done when all of the following are true:

- the adaptive pipeline no longer requires a production global
  `balance_octree(...)` pass after refinement
- the final tree remains 2:1 balanced
- existing meshing assumptions remain valid
- the queue is implemented as its own class in its own header and
  implementation file
- the new refinement closure path is tested in serial and parallel modes
- this plan has been updated to reflect the final architecture rather than the
  provisional one
