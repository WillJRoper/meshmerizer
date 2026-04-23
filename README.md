# Meshmerizer

Meshmerizer converts particle-based simulation outputs into STL meshes for 3D
printing or visualization. It uses an adaptive octree, Wendland C2 SPH field
evaluation, and adaptive surface extraction to build meshes directly from
particles without a dense uniform voxel grid.

The package now supports both:

- a **CLI workflow** for snapshot-to-STL usage, and
- a **public Python API** for building custom staged pipelines in Python.

## Features

- **Adaptive octree refinement** near the surface instead of refining the full
  volume uniformly.
- **Regularized topology opening** for minimum-feature-thickness cleanup.
- **Blocky opened-solid extraction with smoothing** for a robust regularized
  path.
- **Friends-of-Friends clustering** for separating disconnected particle
  structures before meshing.
- **Mesh post-processing helpers** such as smoothing, island removal, and
  subdivision.
- **C++ core with optional OpenMP** for performance-critical stages.
- **SWIFT snapshot loading** via `swiftsimio`.
- **HDF5 octree serialization** for checkpointing CLI workflows.

## Installation

Requires Python 3.8+.

```bash
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

### Building with OpenMP

Set `WITH_OPENMP` before installation to enable threaded C++ execution.

```bash
# Linux
WITH_OPENMP=1 pip install -e .

# macOS with Homebrew libomp
WITH_OPENMP=/opt/homebrew/opt/libomp pip install -e .
```

If `WITH_OPENMP` is unset, Meshmerizer builds in serial mode.

## CLI usage

The main command is:

```bash
meshmerizer snapshot.hdf5
```

### Basic example

```bash
meshmerizer snapshot.hdf5 \
  --particle-type gas \
  --base-resolution 64 \
  --max-depth 4 \
  --surface-percentile 5 \
  --output mesh.stl
```

### Regularized print-ready example

```bash
meshmerizer snapshot.hdf5 \
  --particle-type gas \
  --base-resolution 128 \
  --max-depth 4 \
  --surface-percentile 0.1 \
  --min-feature-thickness 0.05 \
  --smoothing-iterations 10 \
  --remove-islands-fraction 0.01 \
  --target-size 15 \
  --output print_ready.stl
```

### Subregion extraction

```bash
meshmerizer snapshot.hdf5 \
  --center 60 60 60 \
  --extent 20 \
  --tight-bounds \
  --output region.stl
```

### Save and reload an octree

```bash
meshmerizer snapshot.hdf5 \
  --save-octree tree.hdf5 \
  --output first_try.stl
```

```bash
meshmerizer --load-octree tree.hdf5 \
  --remove-islands-fraction 0.0 \
  --output second_try.stl
```

## Important CLI options

| Flag | Description |
|------|-------------|
| `--base-resolution` | Number of top-level octree cells per axis |
| `--max-depth` | Maximum octree refinement depth |
| `--isovalue` / `-t` | Explicit isosurface threshold |
| `--surface-percentile` | Compute an isovalue from particle self-density |
| `--particle-type` / `-p` | Particle family to load |
| `--smoothing-factor` | Multiplier on particle smoothing lengths |
| `--center`, `--extent` | Extract a cubic subregion |
| `--tight-bounds` | Shrink the working domain to the occupied region |
| `--min-feature-thickness` | Regularize topology by removing thin features |
| `--remove-islands-fraction` | Remove components smaller than a fraction of the largest volume |
| `--target-size` / `-s` | Scale the final mesh to a target size in cm |
| `--save-octree`, `--load-octree` | Checkpoint or reload adaptive tree state |

## Public Python API

The recommended public API is imported from the package root:

```python
from meshmerizer import (
    build_and_refine_tree,
    erode_and_dilate,
    get_mesh,
    get_mesh_from_topology,
    get_mesh_from_tree,
    remove_islands,
    smooth_mesh,
)
```

### One-shot usage

```python
from meshmerizer import get_mesh

mesh_result = get_mesh(
    positions,
    smoothing_lengths,
    domain_min=(0.0, 0.0, 0.0),
    domain_max=(4.0, 4.0, 4.0),
    base_resolution=64,
    max_depth=4,
    isovalue=0.01,
    min_feature_thickness=0.05,
    smoothing_iterations=10,
    remove_islands_fraction=0.01,
)

mesh_result.mesh.save("output.stl")
```

### Staged topology-editing workflow

```python
from meshmerizer import (
    build_and_refine_tree,
    erode_and_dilate,
    get_mesh_from_topology,
)

tree = build_and_refine_tree(
    positions,
    smoothing_lengths,
    domain_min=(0.0, 0.0, 0.0),
    domain_max=(4.0, 4.0, 4.0),
    base_resolution=64,
    max_depth=4,
    isovalue=0.01,
)

topology = erode_and_dilate(tree, min_feature_thickness=0.05)

# Modify topology.opened_inside in Python here if desired.

mesh_result = get_mesh_from_topology(topology)
mesh_result.mesh.save("opened_surface.stl")
```

### Tree-stage workflow

```python
from meshmerizer import build_and_refine_tree, get_mesh_from_tree

tree = build_and_refine_tree(
    positions,
    smoothing_lengths,
    domain_min=(0.0, 0.0, 0.0),
    domain_max=(4.0, 4.0, 4.0),
    base_resolution=64,
    max_depth=4,
    isovalue=0.01,
)

mesh_result = get_mesh_from_tree(tree)
mesh_result.mesh.save("tree_mesh.stl")
```

### Post-processing helpers

```python
from meshmerizer import remove_islands, smooth_mesh, subdivide_long_edges

smoothed = smooth_mesh(mesh_result.mesh, iterations=10)
cleaned = remove_islands(smoothed, remove_islands_fraction=0.01)
denser = subdivide_long_edges(cleaned, iterations=1)
```

## Core public API objects

- `TreeState`
  - validated particle/domain inputs
  - refined octree cells
  - contributor array
  - refinement metadata
- `TopologyState`
  - opened-solid masks and diagnostics
  - boundary sample data
  - extracted opened-surface arrays
- `MeshResult`
  - `mesh` (`meshmerizer.mesh.core.Mesh`)
  - `isovalue`
  - `n_qef_vertices`

## Package layout

```text
src/meshmerizer/
  __init__.py
  api.py                    # Public Python API
  adaptive_core.py          # Thin Python wrappers over the C++ core
  _adaptive.cpp             # Python/C++ extension bindings
  reconstruct.py            # Reconstruction helpers
  serialize.py              # HDF5 octree export/import
  printing.py               # Print scaling helpers

  adaptive_cpp/
    octree_cell.hpp         # Adaptive octree build/refine/balance
    hermite.hpp             # Hermite samples and crossings
    qef.hpp                 # QEF solving
    faces.hpp               # Dual contour helpers
    adaptive_solid.hpp      # Opened-solid regularization helpers
    dc_pipeline.hpp         # Full C++ pipeline
    fof.hpp                 # Friends-of-Friends clustering

  commands/
    main.py
    args.py
    adaptive_stl.py
    loading.py

  mesh/
    core.py
```

## Testing

Run the full test suite:

```bash
pytest
```

Run focused tests:

```bash
pytest tests/test_api.py -q
pytest -k "watertight"
```
