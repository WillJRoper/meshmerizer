# Python API

The public API is exported from `meshmerizer`.

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
)

topology = regularize(tree, min_feature_thickness=0.05)
result = extract_mesh(topology)
```

## State objects

### `TreeState`

Contains:

- validated particle arrays,
- domain bounds,
- refined cells,
- contributor indices,
- refinement settings.

Use this when you want to stop after tree construction and continue later.

### `TopologyState`

Contains:

- the source `TreeState`,
- opened-solid occupancy masks,
- topology diagnostics,
- opened-surface arrays.

Use this when you want regularization state before final mesh extraction.

### `MeshResult`

Contains:

- final `Mesh`,
- `isovalue`,
- `n_qef_vertices`.

## Other helpers

- `cluster_particles(...)`: FOF clustering labels
- `compute_isovalue_from_percentile(...)`: percentile-based threshold helper
- `smooth_mesh(...)`: repair/smoothing helper
- `remove_islands(...)`: connected-component filtering
- `subdivide_long_edges(...)`: mesh subdivision helper
