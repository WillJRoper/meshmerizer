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

- Current implementation phase: **Phase 3 - Tree-Guided Neighbor Traversal**
- Work in progress now:
  - beginning Phase 4 preparation work
  - strengthening reservation tokens so they carry meaningful publication
    expectations
  - preparing storage APIs for later worker-safe fixed-slot publication

Reporting-note:

- periodic table reporting remains planned work, but is intentionally deferred
  while Phase 2 serial closure correctness continues
- Not started yet:
  - replacing regularization refinement paths
  - parallel workers

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
- rows should be emitted from one coordinator/reporting path, not by every
  worker independently
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

- [ ] Implement the face-overlap branch traversal described above.
- [ ] Avoid finest-grid raster scanning.
- [ ] Add pruning based on `required_depth >= demanded_depth`.
- [ ] Add tests covering pathological coarse/fine propagation layouts.
- [ ] Confirm that balance closure is local and monotone.

## Phase 4 - Parallel Structural Mutation

- [ ] Refactor cell storage so parallel child creation is safe.
- [ ] Refactor contributor storage so concurrent append/reservation is safe.
- [ ] Ensure side-car state grows consistently with cell creation.
- [ ] Publish child blocks atomically enough that workers never see partial
      structure.

## Phase 5 - Parallel Workers

- [ ] Run multiple worker threads over the shared queue.
- [ ] Make stale-task discard cheap and correct.
- [ ] Ensure duplicate queue requests collapse via `required_depth` updates.
- [ ] Add deterministic regression tests across thread counts.
- [ ] Measure scaling and contention hotspots.

## Phase 6 - Pipeline Integration

- [ ] Replace initial refinement entry points with the new closure-based engine.
- [ ] Replace iterative refine-then-balance loops in regularization paths with
      the same closure engine.
- [ ] Remove explicit production calls to `balance_octree(...)` from the main
      refinement flow.
- [ ] Keep debug-only invariant verification available.
- [ ] Replace queue-phase `ProgressCounter` output with periodic table rows.
- [ ] Add and wire the `--table-cadence` CLI argument.

## Phase 7 - Cleanup

- [ ] Delete obsolete balance-specific scheduling code that the new engine
      supersedes.
- [ ] Retain only the pieces of old balance logic still needed for validation or
      tests.
- [ ] Update design docs and developer-facing architecture notes.

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
