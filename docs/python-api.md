# Python API

The public API is exported from `meshmerizer`.

For a conceptual overview of the adaptive stages, see
[Reconstruction Workflow](reconstruction-workflow.md). For practical runtime
guidance, see [Tuning Performance](tuning-performance.md).

```python
from meshmerizer import (
    build_tree,
    cluster_particles,
    compute_isovalue_from_percentile,
    extract_mesh,
    generate_mesh,
    regularize,
)
```

## Main workflows

### One-shot workflow

Use `generate_mesh(...)` when you want the simplest path from particles to a
final mesh.

```python
from meshmerizer import generate_mesh

result = generate_mesh(
    positions,
    smoothing_lengths,
    domain_min=(0.0, 0.0, 0.0),
    domain_max=(4.0, 4.0, 4.0),
    base_resolution=64,
    max_depth=4,
    isovalue=0.01,
    nthreads=8,
)

result.mesh.save("output.stl")
```

### Staged workflow

Use the staged API when you want to inspect or modify intermediate state.

```python
from meshmerizer import build_tree, extract_mesh, regularize

tree = build_tree(
    positions,
    smoothing_lengths,
    domain_min=(0.0, 0.0, 0.0),
    domain_max=(4.0, 4.0, 4.0),
    base_resolution=64,
    max_depth=4,
    isovalue=0.01,
    nthreads=8,
)

topology = regularize(tree, min_feature_thickness=0.05, nthreads=8)
result = extract_mesh(topology, nthreads=8)
```

## State objects

### `TreeState`

Use this when you want to stop after tree construction and continue later.

Important fields:

- `cells`: tuple of refined octree cell dictionaries from the native core
- `contributors`: flat NumPy index array aligned with contributor ranges stored
  on the cells
- `positions`: contiguous `float64[N, 3]` particle positions
- `smoothing_lengths`: contiguous `float64[N]` support radii
- `domain_min`, `domain_max`: working-domain bounds
- `base_resolution`, `max_depth`, `isovalue`: reconstruction controls used to
  build the tree

This is the object to keep around when the expensive part of your workflow is
the adaptive tree construction itself.

### `TopologyState`

Use this when you want regularization state before final mesh extraction.

Important fields:

- `tree`: the source `TreeState`
- `occupancy`, `depths`, `centers`, `sizes`: per-leaf reconstruction state
- `clearance`, `thickening_distance`, `dilation_distance`: distance-like arrays
  used by the regularization pass
- `thickened_inside`, `eroded_inside`, `opened_inside`: boolean masks for the
  main topology stages
- `sample_positions`, `sample_normals`: opened-surface sample buffers
- `mesh_vertices`, `mesh_faces`: extracted opened-surface mesh arrays

This is the object to inspect when you are tuning feature-thickness cleanup or
trying to understand why a structure was kept or removed.

### `MeshResult`

- `mesh`: final `Mesh` wrapper
- `isovalue`: scalar threshold used during extraction
- `n_qef_vertices`: number of active QEF vertices solved before post-processing

This is the final handoff object for one-shot or staged workflows.

## Other helpers

- `cluster_particles(...)`: FOF clustering labels
- `compute_isovalue_from_percentile(...)`: percentile-based threshold helper
- `smooth_mesh(...)`: repair/smoothing helper
- `remove_islands(...)`: connected-component filtering
- `subdivide_long_edges(...)`: mesh subdivision helper

## When to use which workflow

- Use `generate_mesh(...)` when you just want to go from particles to a final
  mesh with as little ceremony as possible.
- Use `build_tree(...) -> regularize(...) -> extract_mesh(...)` when you want
  to inspect intermediate state, save work between runs, or tune cleanup
  behavior incrementally.

## Thread control

The public Python API exposes `nthreads` on the main reconstruction entry
points:

- `build_tree(..., nthreads=...)`
- `regularize(..., nthreads=...)`
- `generate_mesh(..., nthreads=...)`
- `extract_mesh(..., nthreads=...)`

This is the knob to reach for when you want the Python interface to use more
than one native worker during queue-driven refinement and reconstruction.

## Performance guidance

If Python API runs are slower than expected:

- lower `max_depth` first,
- consider a higher `base_resolution` with a lower `max_depth`,
- crop the domain more tightly,
- increase `nthreads` if Meshmerizer was built with OpenMP,
- reuse `TreeState` instead of rebuilding from particles for every test.

See [Tuning Performance](tuning-performance.md) for more detail.
