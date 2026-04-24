# Structure and Documentation Refactor Plan

## Goal

Refactor the repository aggressively to improve separation of concerns,
readability, and maintainability, then bring both Python and C++
documentation up to a consistently high standard.

This plan assumes there is **no requirement to preserve old structure,
backward compatibility, or transitional warnings**.

## Guiding principles

- Prefer a clean final architecture over incremental compatibility layers.
- Keep CLI code separate from reusable library code.
- Replace loosely structured internal payloads with explicit typed models.
- Split large orchestration files into focused modules.
- Document only after the new structure is stable.
- Bring all C++ code up to full Doxygen coverage with denser explanatory
  comments in algorithmically complex sections.

## Commit plan

## Progress updates

### 2026-04-24

- Completed the final cleanup pass by removing temporary compatibility layers
  (`meshmerizer.commands.*`, `meshmerizer.adaptive_core`,
  `meshmerizer.serialize`, and `meshmerizer.reconstruct`) and retargeting the
  remaining runtime code and tests to the canonical modules.
- Started Commit 9 in order by expanding Python module and helper docstrings in
  `meshmerizer.cli.adaptive`, `meshmerizer.adaptive.pipeline`,
  `meshmerizer.io.octree`, and the low-level adaptive/CLI layers.
- Replaced stale wording in the compatibility-facing reconstruction wrappers so
  their current role relative to the redesigned public API is explicit.
- Documented the adaptive CLI flow, print-space regularization conversion, and
  octree import/export helper contracts more fully before validation.
- Completed Commit 10 by rewriting `README.md` around the refactored package
  layout, the current public API (`generate_mesh`, `build_tree`, `regularize`,
  `extract_mesh`), the CLI workflow, and the native/core boundary.

### 2026-04-23

- Started Commit 1 work by introducing `meshmerizer.cli/`,
  `meshmerizer.io/`, `meshmerizer.mesh.operations`, and `meshmerizer.state`.
- Moved reusable mesh cleanup and mesh output helpers out of the CLI-owned
  module.
- Moved snapshot-loading code into `meshmerizer.io.swift`.
- Added compatibility wrappers for the old `meshmerizer.commands.*` import
  paths while the refactor proceeds.
- Started the documentation pass by expanding Python API/state docstrings and
  adding Doxygen coverage to key native regularization and cancellation headers.
- Updated stale native binding terminology so `_adaptive.cpp` now describes the
  current binding layer instead of historical scaffold/rewrite language.
- Continued Commit 3 and Commit 5 work by moving adaptive CLI orchestration out
  of `meshmerizer.commands.adaptive_stl` into `meshmerizer.cli.adaptive` and
  splitting the control flow into focused helpers for configuration, input
  loading, octree construction, direct pipeline execution, and mesh output.
- Continued Commit 3 in order by splitting the old monolithic
  `meshmerizer.adaptive_core` wrapper into focused modules under
  `meshmerizer.adaptive` for low-level bindings, tree/meshing wrappers,
  topology helpers, and high-level pipeline calls, while keeping a
  compatibility facade at the historical import path.
- Started Commit 4 in order by collapsing the public Python API around a
  smaller workflow vocabulary: `build_tree`, `regularize`, `extract_mesh`,
  and `generate_mesh`, with `cluster_particles` as the explicit public FOF
  helper.
- Started Commit 6 in order by moving the actual HDF5 octree schema
  implementation into `meshmerizer.io.octree`, documenting the full on-disk
  layout there, and reducing `meshmerizer.serialize` to a compatibility shim.
- Started Commit 7 in order by reorganizing `_adaptive.cpp` into clearer helper
  sections and extracting shared parsing/bridge helpers for bounds parsing,
  cell decoding, contributor decoding, and initial-cell construction.
- Started Commit 8 in order by expanding Doxygen coverage and invariant notes in
  the priority native files: `octree_cell.hpp`, `adaptive_solid.hpp`,
  `cancellation.hpp`, and `_adaptive.cpp`.

### Commit 1: Reorganize package layout around clear layers

**Goal:** establish a clean architecture before touching behavior.

**Changes:**

- Create clearer subpackages, for example:
  - `meshmerizer/api/`
  - `meshmerizer/adaptive/`
  - `meshmerizer/mesh/`
  - `meshmerizer/io/`
  - `meshmerizer/cli/`
- Move code out of `commands/` into a proper CLI package.
- Move reusable helpers out of CLI-owned modules.
- Remove redundant modules or merge overlapping ones.
- Simplify top-level exports in `__init__.py`.

**Desired result:**

- No CLI/library boundary leakage.
- Clear package layout reflecting actual responsibilities.

**Commit message:**

`refactor package structure around library, cli, and native layers`

---

### Commit 2: Replace ad-hoc dict pipelines with typed state models

**Goal:** make internal flow readable and explicit.

**Changes:**

- Introduce typed dataclasses for:
  - tree and cell state
  - topology state
  - mesh pipeline results
  - serialization payloads
- Remove string-key-heavy internal contracts where practical.
- Centralize conversions to and from native extension payloads.

**Desired result:**

- Pipeline stages are easier to follow.
- Internal contracts are explicit rather than implied by string keys.

**Commit message:**

`refactor pipeline state into explicit typed models`

---

### Commit 3: Split oversized orchestration modules

**Goal:** remove god files.

**Changes:**

- Break up current `adaptive_stl.py` responsibilities into smaller modules.
- Break up `adaptive_core.py` by abstraction level:
  - low-level bindings
  - tree operations
  - topology and regularization
  - pipeline execution
- Keep each module narrow and named by responsibility.

**Desired result:**

- Readers can navigate by responsibility instead of history.

**Commit message:**

`refactor adaptive pipeline into focused modules`

---

### Commit 4: Simplify and redesign the public Python API

**Goal:** make the intended usage obvious.

**Changes:**

- Define one clean public workflow.
- Keep staged APIs only where they are genuinely valuable.
- Remove duplicated entrypoints and overlapping wrappers.
- Make names consistent and minimal.

**Desired result:**

- Users see one coherent public API.

**Commit message:**

`redesign public api around a single coherent workflow`

---

### Commit 5: Redesign CLI around explicit command flow

**Goal:** make CLI code readable as an application layer.

**Changes:**

- Introduce a proper CLI entry module and command modules.
- Separate:
  - argument parsing
  - loading
  - execution
  - output writing
- Keep CLI-specific concerns out of library code.

**Desired result:**

- CLI becomes thin orchestration over library functions.

**Commit message:**

`refactor cli into parsing, execution, and output stages`

---

### Commit 6: Restructure serialization and I/O documentation

**Goal:** make persisted formats understandable and trustworthy.

**Changes:**

- Move serialization code into a dedicated I/O layer.
- Define a single source of truth for serialized schema.
- Document exact on-disk structure in code.
- Align code and docs.

**Desired result:**

- Serialization becomes easier to maintain and audit.

**Commit message:**

`refactor octree io and document serialized schema`

---

### Commit 7: Refactor native extension boundary for readability

**Goal:** make the C++/Python interface understandable.

**Changes:**

- Split `_adaptive.cpp` if practical, or at minimum reorganize it into
  clearly marked sections.
- Separate:
  - Python parsing and conversion helpers
  - extension method wrappers
  - pipeline bridging logic
- Reduce binding-file sprawl.

**Desired result:**

- Native bindings become navigable rather than monolithic.

**Commit message:**

`refactor native binding layer for clarity`

---

### Commit 8: Add complete Doxygen coverage to native code

**Goal:** meet the C++ documentation standard.

**Changes:**

- Add complete Doxygen `@file` docs to all C++ headers and source files.
- Add `@brief`, `@param`, `@return`, invariants, and algorithm notes.
- Prioritize:
  - `adaptive_solid.hpp`
  - `octree_cell.hpp`
  - `cancellation.hpp`
  - `_adaptive.cpp`
- Add denser inline comments in algorithmically complex sections.

**Desired result:**

- Native code can be followed without reverse-engineering behavior.

**Commit message:**

`docs add complete doxygen coverage for adaptive c++ code`

---

### Commit 9: Expand Python docstrings and module documentation

**Goal:** bring Python docs up to the same standard.

**Changes:**

- Expand module docstrings.
- Fully document public dataclasses and pipeline stages.
- Add docstrings to helper and validator functions.
- Remove stale terminology and historical wording.

**Desired result:**

- Python code explains intent, contracts, and flow clearly.

**Commit message:**

`docs expand python api and pipeline documentation`

---

### Commit 10: Rewrite README around the new architecture

**Goal:** make repository-level docs match the cleaned codebase.

**Changes:**

- Update package layout documentation.
- Document the intended API and CLI flows.
- Explain native/core boundaries.
- Add contributor-oriented architecture notes.

**Desired result:**

- The README becomes a reliable map of the project.

**Commit message:**

`docs rewrite readme for new architecture and workflows`

## Recommended execution order

Apply the commits in this order:

1. package structure
2. typed models
3. orchestration split
4. public API redesign
5. CLI redesign
6. serialization cleanup
7. native binding cleanup
8. C++ Doxygen pass
9. Python documentation pass
10. README rewrite

## Why this order

Documentation should follow structural stabilization. Writing detailed docs
too early would create unnecessary churn and duplicate work.

## Working method

For each commit:

1. change one layer only
2. run formatting and tests
3. commit immediately
4. then move to the next layer

This keeps the history readable and makes regressions easier to isolate.

## First implementation targets

The highest-value early targets are:

- `src/meshmerizer/cli/adaptive.py`
- `src/meshmerizer/adaptive/`
- `src/meshmerizer/api.py`
- `src/meshmerizer/io/octree.py`
- `src/meshmerizer/_adaptive.cpp`
- `src/meshmerizer/adaptive_cpp/adaptive_solid.hpp`
- `src/meshmerizer/adaptive_cpp/octree_cell.hpp`
- `src/meshmerizer/adaptive_cpp/cancellation.hpp`

## Success criteria

The refactor is complete when:

- library and CLI responsibilities are cleanly separated
- large modules have been decomposed into focused units
- internal pipeline contracts are typed and explicit
- native bindings are readable and easier to navigate
- all C++ files have strong Doxygen coverage
- Python public and internal modules have consistent, high-quality docstrings
- README and code structure match each other closely
