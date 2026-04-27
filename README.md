# Meshmerizer

Meshmerizer converts particle-based simulation outputs into STL meshes for visualization and 3D printing.

Find the documentation [here](https://willjroper.github.io/meshmerizer/).

## What Meshmerizer provides

- Adaptive particles-to-mesh reconstruction of surfaces to avoid large memory footprints while maximising surface fidelity
- Optional minimum-feature-thickness regularization for print-oriented cleanup
- Friends-of-Friends clustering for disconnected structures
- Mesh post-processing helpers for smoothing, island removal, and simplification
- SWIFT snapshot loading for CLI workflows
- HDF5 octree save/load support for checkpointed reconstruction
- Native C++ acceleration, with optional OpenMP threading
- A Python API for greater control and integration into custom workflows

## Installation

Meshmerizer requires Python 3.8+.

Install the package in editable mode:

```bash
pip install -e .
```

### Building with OpenMP

Set `WITH_OPENMP` before installation to enable threaded native execution.

```bash
# Linux
WITH_OPENMP=1 pip install -e .

# Or point directly to an OpenMP install (required for macOS with Homebrew libomp)
WITH_OPENMP=/opt/homebrew/opt/libomp pip install -e .
```

If `WITH_OPENMP` is unset, Meshmerizer builds in serial mode.

### Building with native debug logging

Set `DEBUG_LOG` before installation to compile optional native debug-log file
support.

```bash
DEBUG_LOG=1 pip install -e .
```

When `DEBUG_LOG` is unset, native debug-only diagnostics are compiled out.

## CLI workflow

The main command is:

```bash
meshmerizer snapshot.hdf5
```

The CLI is intended for snapshot-to-STL workflows. It:

1. loads particles from a SWIFT snapshot or a saved octree,
2. computes or accepts an isovalue,
3. runs an adaptive octree-based reconstruction pipeline,
4. optionally regularizes thin features,
5. post-processes the mesh (e.g. smoothing, simplifying, padding), and
6. writes the final STL atomically.

### Basic example

To extract a raw surface mesh from a gas distribution without regularization or post-processing, run:

```bash
meshmerizer snapshot.hdf5 \
  --particle-type gas \
  --base-resolution 64 \
  --max-depth 4 \
  --surface-percentile 0.05 \
  --output mesh.stl
```

This will build an octree at the surface with a maximum depth of 4 (limiting the computational load and defining the smallest feature size), and extract a raw surface mesh at the 5th percentile of the particle self-density distribution.

### Regularized print-ready example

To construct a mesh with print-oriented cleanup, set `--target-size` to scale the final mesh to a target size in centimetres. This enables interpretation of print-oriented parameters such as `--min-feature-thickness` which, when defined, will make the code produce a mesh where the smallest features are at least that thick in the final print. The `--pre-thickening-radius` is in the input units and controls how much the code "puffs up" thin features before opening, which can help preserve small disconnected features that would otherwise be lost to opening.

```bash
meshmerizer snapshot.hdf5 \
  --particle-type gas \
  --base-resolution 128 \
  --max-depth 4 \
  --surface-percentile 0.1 \
  --min-feature-thickness 0.05 \
  --pre-thickening-radius 0.01 \
  --smoothing-iterations 10 \
  --remove-islands-fraction 0.01 \
  --target-size 15 \
  --output print_ready.stl
```

### Extract a subregion

We can also focus the reconstruction on a smaller region by defining a center and extent. Setting `--tight-bounds` will shrink the working domain to the bounds of the occupied particles in that region, which can help performance when the subregion is sparsely populated.

```bash
meshmerizer snapshot.hdf5 \
  --center 60 60 60 \
  --extent 20 \
  --tight-bounds \
  --output region.stl
```

### CLI Overview

| Flag                             | Purpose                                           |
| -------------------------------- | ------------------------------------------------- |
| `--base-resolution`              | Number of top-level octree cells per axis         |
| `--max-depth`                    | Maximum octree refinement depth                   |
| `--isovalue`                     | Explicit surface threshold                        |
| `--surface-percentile`           | Compute the isovalue from particle self-density   |
| `--particle-type`                | Particle family to load from a SWIFT snapshot     |
| `--center`, `--extent`           | Extract a cubic subregion                         |
| `--tight-bounds`                 | Shrink the working domain to occupied particles   |
| `--fof`                          | Reconstruct disconnected FOF groups independently |
| `--min-fof-cluster-size`         | Drop small FOF fluff populations before meshing   |
| `--min-feature-thickness`        | Remove features thinner than a target thickness   |
| `--pre-thickening-radius`        | Puff up fragile features before opening           |
| `--smoothing-iterations`         | Apply Laplacian smoothing after extraction        |
| `--remove-islands-fraction`      | Drop small connected components                   |
| `--simplify-factor`              | Simplify the final mesh by face-count fraction    |
| `--target-size`                  | Scale the final mesh to a print size in cm        |
| `--save-octree`, `--load-octree` | Save or reload octree state                       |
| `--visualise-verts`              | Save a diagnostic QEF vertex figure               |
| `--nthreads`                     | Set the OpenMP thread count                       |

For further details, see `meshmerizer --help` or the [CLI documentation](https://willjroper.github.io/meshmerizer/cli.html).

## Public Python API

The public python API can be imported from the package root:

```python
from meshmerizer import (
    MeshResult,
    TopologyState,
    TreeState,
    build_tree,
    cluster_particles,
    compute_isovalue_from_percentile,
    extract_mesh,
    generate_mesh,
    regularize,
    remove_islands,
    smooth_mesh,
    subdivide_long_edges,
)
```

The intended workflows are:

- `generate_mesh(...)` for the common one-shot path
- `build_tree(...) -> regularize(...) -> extract_mesh(...)` for staged use

### One-shot workflow

```python
from meshmerizer import generate_mesh

mesh_result = generate_mesh(
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

### Staged workflow

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

topology = regularize(
    tree,
    min_feature_thickness=0.05,
    pre_thickening_radius=0.01,
)

mesh_result = extract_mesh(topology)
mesh_result.mesh.save("opened_surface.stl")
```

For more details, see the [API documentation](https://willjroper.github.io/meshmerizer/python-api.html).
