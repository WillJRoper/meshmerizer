# Adaptive Meshing Rewrite Plan

## Purpose

This document is the single source of truth for the adaptive rewrite.

Rules for implementation:

- Every meaningful implementation step must be checked against this document.
- If the design changes, update this file in the same change.
- If a checklist item becomes obsolete, mark it explicitly rather than silently
  skipping it.
- Do not preserve old behavior for backward compatibility unless it is
  re-approved here.
- Prefer deleting obsolete code over keeping parallel paths alive.

## Goals

- Represent the input field as an SPH particle field with Wendland C2 kernels.
- Generate a smooth adaptive mesh without voxel artefacts.
- Use one meshing path only: adaptive octree plus dual contouring.
- Push the heavy computation into C++ with OpenMP parallelism.
- Keep memory usage controlled and proportional to the surface, not the full
  volume, wherever possible.
- Support serializing the octree state to HDF5 so refinement/meshing can be
  resumed without reloading and reprocessing particles.
- Keep the code educational: heavily documented, explicit, and descriptive.

## Non-Goals

- Backward compatibility with the current voxel/chunk/SDF pipeline.
- Maintaining multiple extraction backends.
- Retaining the current C extension design.
- Preserving old CLI flags that no longer make sense in the new model.

## One True Pipeline

The rewrite will support one conceptual pipeline:

1. Load particles and select the working domain.
2. Build top-level adaptive cells from a user-defined base resolution.
3. Attach contributing particles to cells using kernel-overlap tests.
4. Recursively refine cells near the requested isosurface.
5. Enforce a balanced octree so neighboring leaves differ by at most one level.
6. Compute leaf-local surface constraints from the SPH field.
7. Solve one representative surface vertex per active leaf.
8. Build mesh faces from octree connectivity.
9. Assemble the final mesh, remove small disconnected solids if requested, and
   write outputs.
10. Optionally serialize or reload the octree from HDF5.

There will be no alternative dense path, chunk path, SDF path, or fallback
mesher. Future out-of-core or chunked execution must use the same adaptive core
pipeline and only change domain partitioning/work scheduling.

## Terminology

### BoundingBox

Use `BoundingBox`, not `AABB`.

Reason:

- `AABB` is common in graphics but not descriptive enough for this codebase.
- `BoundingBox` is explicit and easier to read in educational code.

Definition:

- An axis-aligned bounding box in world space with inclusive minimum corner and
  exclusive maximum corner semantics where appropriate.

### Hermite Samples

Hermite samples are the local geometric constraints used to place a surface
vertex inside a leaf.

For this project, a Hermite sample means:

- a point on the isosurface, usually found by interpolating along an edge where
  the scalar field changes sign relative to the isovalue, and
- the surface normal at that point, computed from the SPH field gradient.

These samples do not directly define triangles. They define local tangent-plane
constraints that are used to solve for a good leaf vertex.

### QEF Solve

QEF means Quadratic Error Function.

For this project, the QEF solve is the process of finding a single vertex inside
an active octree leaf that best fits all Hermite sample tangent planes gathered
for that leaf.

Intuition:

- each Hermite sample says “the leaf vertex should lie near this plane”
- the QEF finds the point that best satisfies all those plane constraints at
  once

This gives better surface vertices than simply averaging edge crossings.

## Field Model

### Particle Kernel

- Kernel: Wendland C2
- Support radius: particle smoothing length `h`
- Field value: scalar accumulation from all overlapping particle kernels
- Gradient: analytical gradient of the Wendland C2 kernel

### Smoothing-Length Modifier

Keep the existing user concept, but rename it clearly in the new CLI if needed.

Behavior:

- multiply every particle smoothing length by a user-provided scalar before any
  overlap, field, or gradient evaluation

### Field Truth

The SPH particle field is the authoritative scalar field.

Top-level cells may cache sampled scalar values, but these are approximations
used to guide refinement, not the fundamental representation of the field.

## Domain Selection

The new CLI must support:

- `--domain-center`
- `--domain-extent`
- `--tight-bounds`
- particle-type selection
- field selection
- isovalue selection
- smoothing-length modifier
- maximum refinement depth
- base cell resolution
- remove-islands-fraction on the final mesh only

Rules:

- `--tight-bounds` determines the working domain before top-level cell creation.
- domain selection is applied before contributor attachment or refinement.
- island removal happens only after the final merged mesh exists.

## Core Data Model

Use flat storage with indices where possible. Avoid pointer-heavy recursive heap
structures unless proven necessary.

### C++ Types

#### `Vector3d`

- three `double` coordinates

#### `BoundingBox`

Members:

- `Vector3d min`
- `Vector3d max`

Functions:

- containment tests
- overlap tests
- extent/center helpers
- child-box generation
- edge/corner coordinate helpers

#### `Particle`

Minimal particle payload:

- `Vector3d position`
- `float value`
- `float smoothing_length`
- optional `uint64_t id` for debugging and serialization

#### `TopLevelBin`

Spatial indexing helper for fast contributor queries.

Members:

- bounding box or integer index
- range into particle index array

#### `OctreeCell`

Required members:

- Morton key
- depth
- bounding box
- leaf/active flags
- child index or child range
- contributor index range
- sampled scalar values at corners
- sign mask relative to isovalue
- representative mesh vertex index

Optional members if profiling shows value:

- cached center value
- cached gradient estimate
- temporary refinement flags

#### `HermiteSample`

Members:

- position on the isosurface
- outward normal from the SPH gradient

#### `MeshVertex`

Members:

- position
- normal

#### `MeshTriangle`

Members:

- three vertex indices

## Morton Keys

Use Morton keys for octree indexing and locality.

Goals:

- compact cell identification
- deterministic traversal order
- easier parent/child relationships
- better cache locality
- simpler serialization

Requirements:

- stable encode/decode helpers
- explicit tests for parent/child and neighbor-related indexing logic

## Adaptive Construction Algorithm

### Stage 1: Particle Loading

- Load selected particles from the input snapshot.
- Apply domain selection.
- Apply smoothing-length modifier.
- Convert particle payloads into the minimal C++ `Particle` representation.

### Stage 2: Top-Level Grid Setup

The CLI defines a top-level cell count per axis.

Proposed flag:

- `--base-resolution`

Behavior:

- create a uniform top-level grid of `base_resolution^3` cells over the chosen
  domain

### Stage 3: Particle Binning

Build a spatial index over particles to accelerate contributor attachment.

Initial implementation:

- use uniform bins aligned with top-level cells

Later optimization is allowed, but not required initially.

### Stage 4: Contributor Attachment

For each top-level cell:

- query overlapping bins
- test each candidate particle kernel against the cell bounding box
- attach particle indices for contributors whose support overlaps the cell

Rules:

- store contributor indices, not particle copies
- contributors may be duplicated across multiple cells when support overlaps

### Stage 5: Refinement Criterion

Refine a cell if it may contain the isosurface and `depth < max_depth`.

Initial criterion:

- evaluate scalar values at the 8 corners
- optionally evaluate center value
- if sampled values straddle the isovalue, mark for split

Planned extension:

- support additional face-center or interval-based checks if corner-only tests
  miss thin structures

### Stage 6: Recursive Refinement

For each active non-terminal cell:

- split into 8 children
- compute child bounding boxes
- attach child contributors by filtering parent contributors against child-box
  overlap
- evaluate child refinement criterion
- continue breadth-first or level-by-level

Implementation preference:

- breadth-first by depth for easier balancing, memory control, and OpenMP
  scheduling

### Stage 7: Balance the Octree

Enforce the rule:

- neighboring leaves differ by at most one level

This can be done during construction or with a post-pass. The implementation may
 start with a post-pass if it is simpler to reason about.

Requirement:

- the final stored tree must be balanced before any dual contouring stage

## Surface Extraction Algorithm

### Chosen Method

Dual contouring on a balanced octree.

Reason:

- naturally adaptive
- one vertex per active leaf
- better fit for octrees than marching cubes
- avoids forcing the field into a globally refined dense lattice

### Stage 8: Edge Crossings

For each active leaf cell:

- inspect edges whose endpoints differ in sign relative to the isovalue
- interpolate a crossing point along the edge
- evaluate the SPH gradient at the crossing point
- store a `HermiteSample`

### Stage 9: Leaf Vertex Solve

For each active leaf:

- gather all Hermite samples for that leaf
- solve a QEF to place one representative vertex inside the cell
- clamp or constrain the solve if the unconstrained result leaves the cell
- compute/store the representative normal

Fallback policy:

- if the QEF is poorly conditioned, fall back to a simpler constrained point
  estimate such as the average crossing position

### Stage 10: Face Construction

Faces are built from octree connectivity, not from point-cloud reconstruction.

Required logic:

- detect sign-changing primal edges
- identify the set of incident leaf cells around each such edge
- emit a dual face connecting those leaf vertices
- triangulate quads deterministically
- maintain consistent winding

Important:

- a set of leaf vertices alone is not enough; explicit connectivity traversal is
  required

### Stage 11: Final Mesh Cleanup

- assemble all triangles into one mesh
- remove disconnected solids below `--remove-islands-fraction` if requested
- recompute/repair normals only if necessary
- export mesh

## Serialization

The octree must be serializable to HDF5 and reloadable without rerunning particle
 loading and contributor construction.

### Required HDF5 Export

Provide a function to write a documented HDF5 file containing at least:

- metadata
  - code version
  - isovalue
  - base resolution
  - max depth
  - domain center
  - domain extent
  - smoothing-length modifier
  - kernel type
- particles
  - positions
  - values
  - smoothing lengths
  - optional ids
- octree cells
  - Morton keys
  - depth
  - bounding boxes
  - child ranges
  - contributor ranges
  - sign masks
  - cached corner values if stored
- contributor index arrays
- optional mesh vertex/face arrays if extraction has already been run

### Required HDF5 Import

Provide a function to reload:

- particle payloads
- octree structure
- contributor mappings
- any cached scalar samples necessary to resume extraction

### Documentation Requirement

The HDF5 layout must be documented in this plan and later in standalone docs.

## Parallelization Strategy

Use OpenMP in C++.

Parallelization is a first-class design constraint. Every major stage must be
implemented in a way that either is already parallel or can be parallelized
without introducing a separate algorithmic code path.

### Global Parallel Rules

- Use one algorithmic path for both threaded and non-threaded execution.
- Prefer flat arrays and index ranges over pointer-heavy recursive structures.
- Prefer breadth-first, per-depth octree construction over recursive DFS.
- Prefer thread-local temporary buffers plus merge steps over shared mutable
  containers guarded by locks.
- Prefer count-then-prefix-sum-then-fill patterns over atomic push-back loops
  where feasible.
- Keep work ownership deterministic so parallel execution does not change the
  resulting mesh topology or serialization layout.

### Stage-by-Stage Parallel Plan

#### Stage 1: Particle Loading and Domain Selection

- Snapshot IO may remain partially serial depending on the reader.
- Post-load particle filtering and domain selection must be parallelizable over
  particles.
- Applying the smoothing-length modifier must be parallelizable over particles.

#### Stage 2: Top-Level Grid Setup

- Top-level cell creation is cheap, but can still be parallelized over cells if
  needed.
- The important constraint is deterministic ordering of top-level cells.

#### Stage 3: Particle Binning

- Must be parallelizable over particles.
- Implementation should use thread-local bin counts or thread-local bin lists,
  followed by a merge stage.
- Avoid shared `push_back` into one global bin structure from many threads.

#### Stage 4: Contributor Attachment

- Must be parallelizable over top-level cells.
- Each cell independently queries overlapping bins and builds its contributor
  list.
- Contributor storage should use a two-pass layout where possible:
  count contributors first, allocate once, then fill ranges.

#### Stage 5: Refinement Criterion Evaluation

- Must be parallelizable over cells within a depth level.
- This is one of the reasons the octree should be constructed breadth-first.
- Each cell evaluates the field and refinement criterion independently from its
  contributor list.

#### Stage 6: Child Creation and Octree Refinement

- Must be parallelizable over the cells selected for splitting in a given depth
  level.
- Child-cell allocation should avoid per-child heap allocations.
- Prefer batched child creation using precomputed split counts and prefix sums.
- Contributor filtering from parent to child must be parallelizable over the
  selected parent cells.

#### Stage 7: Octree Balancing

- Balancing must be designed as a batched mark-and-split process.
- Neighbor inspections may be parallelized over cells.
- Splits should be applied in synchronized batches between iterations.
- The final balanced tree must be deterministic regardless of thread count.

#### Stage 8: Hermite Sample Generation

- Must be parallelizable over active leaves.
- Each active leaf can independently inspect sign-changing edges, compute
  crossing points, and evaluate gradients.
- Use thread-local Hermite sample buffers if temporary dynamic storage is
  required.

#### Stage 9: QEF Solve

- Must be parallelizable over active leaves.
- Each active leaf solve is independent once its Hermite samples are known.
- The solve output should be written into preallocated per-leaf storage.

#### Stage 10: Face Construction

- Must be designed to be parallelizable even if the first implementation is
  validated serially.
- Face ownership must be deterministic, for example by assigning each sign-
  changing primal edge to exactly one worker domain.
- Face emission should use thread-local face buffers merged at the end.
- Parallel execution must never emit duplicate faces or skip seam faces.

#### Stage 11: Final Mesh Assembly

- Vertex and face concatenation should be compatible with parallel prefix-sum
  based assembly.
- This stage is likely cheaper than field evaluation but should still avoid
  unnecessary serial bottlenecks.

#### Stage 12: Final Island Removal

- Connected-component identification on the final mesh should be written so it
  can be parallelized later if needed, even if the first implementation is
  serial.
- This stage runs only on the final mesh, never on intermediate octree cells.

#### Stage 13: HDF5 Serialization

- HDF5 export/import does not need to be fully parallel at first, but data
  preparation for serialization must use deterministic storage layouts that are
  compatible with parallel writers later.

### Preferred Parallel Units by Stage

- particle filtering and smoothing-length modification: particles
- particle binning: particles
- contributor attachment: top-level cells
- refinement criterion evaluation: cells within one depth level
- refinement and child contributor filtering: splitting parent cells within one
  depth level
- balancing: cells within one balancing iteration
- Hermite sample evaluation: active leaves
- QEF solve: active leaves
- face generation: owned primal edges or top-level subtrees
- mesh assembly: pre-sized vertex/face output ranges

### Stages That Must Not Depend on Divergent Code Paths

The following stages must use the same implementation in serial and threaded
execution:

- contributor attachment
- octree refinement
- octree balancing
- Hermite sample generation
- QEF solve
- face generation

Rule:

- do not introduce separate algorithmic paths for threaded and non-threaded
  execution

## Python/C++ Boundary

Python should orchestrate:

- CLI parsing
- snapshot loading
- final output handling

C++ should own:

- particle compaction
- contributor attachment
- octree construction and balancing
- scalar and gradient evaluation
- Hermite sample generation
- QEF solving
- face generation
- HDF5 octree serialization

## Dependency Cleanup

Planned removals where possible:

- skimage-based meshing dependency
- dense voxel/SDF-centric helpers
- current C extension code once replaced
- legacy chunk-union meshing path

Dependencies to reassess during rewrite:

- trimesh: keep only if still useful for final IO/utility operations
- scipy: remove from the adaptive core if possible

## Deletion Policy

The rewrite should actively remove obsolete code.

Delete or replace:

- old dense voxel meshing pipeline
- old chunked overlap/union pipeline
- old SDF extraction path
- old adaptive prototype
- box-kernel evaluator/deposition code
- CLI options that only exist to support the old pipeline

Do not leave dead compatibility layers behind.

## Documentation Rules

### C++

- Every public function/class must have Doxygen comments.
- Every nontrivial block must have an explanatory comment.
- Prefer long descriptive names over short clever names.

### Python

- Every public function/class must have docstrings.
- Every nontrivial block must have an explanatory comment.
- The Python layer should explain orchestration and data movement clearly.

## Implementation Checklist

### Phase 0: Planning

- [x] Finalize this design document.
- [x] Create `adaptive-meshing` branch from `main`.
- [x] Record module layout and build-system changes needed for a C++ extension.

### Phase 1: Skeleton

- [x] Create new C++ extension scaffold.
- [x] Create minimal Python wrapper module for the new pipeline.
- [x] Add basic build/test hooks for the new extension.

### Phase 2: Core Types

- [x] Implement `Vector3d` helpers.
- [x] Implement `BoundingBox`.
- [x] Implement `Particle`.
- [x] Implement Morton key utilities.
- [x] Add unit tests for geometry and Morton indexing.

### Phase 3: Field Evaluation

- [x] Implement Wendland C2 kernel value.
- [x] Implement Wendland C2 kernel gradient.
- [ ] Implement smoothing-length modifier support.
- [x] Add field-evaluation unit tests.

### Phase 4: Spatial Indexing

- [x] Implement top-level particle binning.
- [x] Implement cell contributor attachment.
- [x] Add overlap and contributor-selection tests.

### Phase 5: Octree Construction

- [x] Implement top-level cell creation from `--base-resolution`.
- [x] Implement breadth-first refinement.
- [x] Implement child contributor filtering.
- [x] Implement refinement criterion.
- [x] Add octree construction tests.

### Phase 6: Octree Balancing

- [x] Implement balancing helper functions (neighbor key, needs_balance_split).
- [x] Implement integrated balancing in refine_octree.
- [x] Add neighbor-depth invariant tests.

### Phase 7: Dual Contouring Inputs

- [x] Implement edge sign-change detection.
- [x] Implement edge crossing interpolation.
- [x] Implement gradient evaluation at crossings.
- [x] Add Hermite sample tests.

### Phase 8: Vertex Solve

- [x] Implement QEF assembly.
- [x] Implement constrained leaf vertex solve.
- [x] Implement fallback solve for degenerate cases.
- [x] Add vertex-placement tests.

### Phase 9: Face Generation

- [x] Implement face construction from octree connectivity.
- [x] Implement deterministic triangulation and winding.
- [x] Add watertightness and manifold tests.

### Phase 10: Serialization

- [ ] Implement HDF5 octree export.
- [ ] Implement HDF5 octree import.
- [ ] Document the HDF5 schema.
- [ ] Add round-trip tests.

### Phase 11: Python Interface and CLI

- [ ] Replace the old CLI path with the adaptive pipeline.
- [ ] Keep `--domain-center` / current center equivalent.
- [ ] Keep `--domain-extent` / current extent equivalent.
- [ ] Keep `--tight-bounds`.
- [ ] Keep smoothing-length modifier support.
- [ ] Keep `--remove-islands-fraction` on final mesh only.
- [ ] Remove obsolete voxel/chunk/SDF options.

### Phase 12: Cleanup

- [ ] Remove old meshing code that no longer applies.
- [ ] Remove obsolete tests.
- [ ] Update README and docs for the new single-path design.

## Open Design Questions

- [ ] Exact refinement criterion beyond corner tests: corners only, corners plus
      centers, or interval bound estimates?
- [ ] Exact QEF solver implementation: handwritten small dense solve or library?
- [ ] Exact HDF5 library strategy for the C++ extension.
- [ ] Whether final normals come directly from leaf/QEF data or are recomputed
      from the output mesh for export.

## Implementation Notes

### Phase 5: Octree Construction

The octree construction code has an incomplete pattern for handling contributor indices
in initial cells. The current implementation assumes initial cell contributor ranges
refer to raw particle indices (0, 1, 2... n) directly. This works for
the test harness but may need review for production use with arbitrary
particle indexing from the top-level bin stage.

### Phase 6: Octree Balancing

Implemented as a post-pass inside `refine_octree`. After the main BFS loop,
`balance_octree` iteratively finds leaf pairs that violate the 2:1 rule using
bounding-box face-sharing detection and splits the shallower cell. The
`neighbor_morton_key` helper was fixed (original had wrong step size `2^depth`
instead of 1 and the wrong boundary expression). `needs_balance_split` was
rewritten to use bounding-box geometry rather than Morton-key lookup, which is
correct for cells of different sizes. A pre-existing bug was also fixed:
`refine_octree` was storing `all_contributors.size()` in `child_begin` (a
contributor offset) instead of `all_cells.size()` (the correct cell-array
offset). The file ordering was also corrected so that `neighbor_morton_key`
appears before the balance helpers and `refine_octree` that depend on it.

### Phase 8: Vertex Solve

The QEF solve is not yet implemented. This requires:
- Collecting Hermite samples per leaf
- Solving the constrained least squares problem
- Fallback handling for degenerate cases

### Phase 9: Face Generation

Face generation uses fine-grid primal edge iteration with trilinear sign
interpolation. The initial per-leaf min-corner-ownership approach failed
because quad vertices were passed in grid-layout order (c0,c1,c2,c3)
rather than cyclic order around the primal edge. This caused the quad
diagonal split to create over-shared edges. The fix was twofold:

1. **Fine-grid iteration**: Instead of iterating edges at each leaf's own
   resolution (which was also incorrect for adaptive grids), we iterate all
   primal edges at the finest grid resolution. For coarse cells, the field
   sign at interior fine-grid vertices is determined by trilinear
   interpolation of the cell's 8 corner values (`sign_at_fine_vertex`).

2. **Cyclic vertex ordering**: The 4 cells incident on each primal edge must
   be passed to `emit_quad` in cyclic (counterclockwise) order around the
   edge axis, not in grid-layout order. For X-edges: c0→c2→c3→c1; for
   Y-edges: c0→c1→c3→c2; for Z-edges: c0→c2→c3→c1.

The test helper `_build_sphere_octree` was also updated to use a larger
domain ([-1,2]^3 with base_resolution=4) so the isosurface at isovalue=0.5
is fully contained within the domain interior. Domain-boundary edges are
correctly skipped (no cell exists outside the domain), so the isosurface
must not intersect the domain boundary for a watertight mesh.

## Immediate Next Step

Phase 10: Serialization — HDF5 octree export/import and round-trip tests.
