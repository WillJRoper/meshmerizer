# Meshmerizer

Meshmerizer converts particle-based simulation outputs into STL meshes for 3D
printing or visualization. It uses an adaptive octree with a Wendland C2 SPH
kernel and adaptive reconstruction to produce smooth, resolution-adaptive
surfaces directly from particle data.

## Features

- **Adaptive octree refinement** — concentrates resolution near the
  isosurface, keeping memory proportional to the surface rather than the
  full volume.
- **Adaptive mesh reconstruction** — extracts a mesh directly from the adaptive
  implicit surface without building a dense uniform voxel grid.
- **FOF clustering** — Friends-of-Friends clustering separates distinct
  objects before reconstruction, preventing thin bridges between
  unrelated structures.
- **Wendland C2 kernel** — the same compact-support kernel used in modern
  SPH codes, with analytic gradients for accurate surface normals.
- **C++ core with optional OpenMP** — the heavy computation (refinement,
  QEF solve) runs in compiled C++ with optional multi-threaded parallelism.
- **HDF5 serialization** — save and reload the octree, contributors, and
  particle data so meshing can be resumed without rebuilding the tree.
- **SWIFT snapshot support** — load particles directly from SWIFT HDF5
  snapshots via swiftsimio.
- **Print scaling** — scale the output mesh to a target physical size for
  3D printing with `--target-size`.
- **Island removal** — discard small disconnected components with
  `--remove-islands-fraction`.
- **Density percentile isovalue** — automatically choose the isovalue
  from a density percentile with `--surface-percentile`.

## Installation

Requires Python 3.8+:

```bash
pip install -e .
```

For development tools (ruff, pytest, mypy):

```bash
pip install -e ".[dev]"
```

### Building with OpenMP

Set the `WITH_OPENMP` environment variable before installing to enable
multi-threaded parallelism in the C++ extension:

```bash
# Linux — uses system libgomp
WITH_OPENMP=1 pip install -e .

# macOS with Homebrew libomp
WITH_OPENMP=/opt/homebrew/opt/libomp pip install -e .
```

When `WITH_OPENMP` is unset or empty, the extension builds in serial mode
with no OpenMP dependency.

## CLI

The package exposes the `meshmerizer adaptive` subcommand:

```bash
meshmerizer adaptive snapshot.hdf5
```

### Common examples

Basic adaptive STL generation:

```bash
meshmerizer adaptive snapshot.hdf5 \
  --particle-type gas \
  --base-resolution 4 \
  --max-depth 4 \
  --isovalue 0.5 \
  --output mesh.stl
```

Higher resolution with island removal and print scaling:

```bash
meshmerizer adaptive snapshot.hdf5 \
  --base-resolution 4 \
  --max-depth 6 \
  --isovalue 0.3 \
  --remove-islands-fraction 0.01 \
  --target-size 15 \
  --output print_ready.stl
```

Subregion extraction:

```bash
meshmerizer adaptive snapshot.hdf5 \
  --center 60 60 60 \
  --extent 15 \
  --tight-bounds \
  --output region.stl
```

### Saving and loading the octree

Build the octree once and save it to HDF5:

```bash
meshmerizer adaptive snapshot.hdf5 \
  --max-depth 5 \
  --save-octree tree.hdf5 \
  --output first_try.stl
```

Reload the octree later to re-mesh without rebuilding the tree:

```bash
meshmerizer adaptive snapshot.hdf5 \
  --load-octree tree.hdf5 \
  --remove-islands-fraction 0.0 \
  --output second_try.stl
```

## Important options

| Flag | Description |
|------|-------------|
| `--base-resolution` | Number of top-level octree cells per axis (default: 4) |
| `--max-depth` | Maximum octree refinement depth (default: 4) |
| `--isovalue` / `-t` | Isosurface threshold (overrides `--surface-percentile`) |
| `--surface-percentile` | Density percentile for auto isovalue (default: 5) |
| `--particle-type` / `-p` | Particle type: `gas`, `dark_matter`, `stars`, `black_holes` |
| `--smoothing-factor` | Multiplier for particle smoothing lengths (default: 1.0) |
| `--box-size` / `-b` | Physical box size override |
| `--center` | Subregion center `X Y Z` in simulation units |
| `--extent` | Subregion side length in simulation units |
| `--tight-bounds` | Shrink domain to occupied particle bounds |
| `--shift` | Coordinate shift `DX DY DZ` before cropping |
| `--wrap-shift` / `--no-wrap-shift` | Wrap coordinates after shifting |
| `--no-periodic` | Disable periodic wrapping for subregion selection |
| `--linking-factor` | FOF linking length as fraction of mean separation (default: 0.2) |
| `--min-feature-thickness` | Remove features thinner than this physical scale |
| `--remove-islands-fraction` | Minimum fraction of the largest component volume to keep |
| `--target-size` / `-s` | Scale longest mesh dimension to this size (cm) |
| `--save-octree` | Save octree state to HDF5 after construction |
| `--load-octree` | Load a previously saved octree from HDF5 |
| `--output` / `-o` | Output STL filename |

## Python API

```python
import numpy as np

from meshmerizer.adaptive_core import (
    run_full_pipeline,
    fof_cluster,
)
from meshmerizer.mesh.core import Mesh

# Example: sphere of particles
n = 500
theta = np.random.uniform(0, 2 * np.pi, n)
phi = np.random.uniform(0, np.pi, n)
r = 0.3 + 0.02 * np.random.randn(n)
positions = [
    (0.5 + r[i] * np.sin(phi[i]) * np.cos(theta[i]),
     0.5 + r[i] * np.sin(phi[i]) * np.sin(theta[i]),
     0.5 + r[i] * np.cos(phi[i]))
    for i in range(n)
]
smoothing_lengths = [0.1] * n

domain_min = (0.0, 0.0, 0.0)
domain_max = (1.0, 1.0, 1.0)
base_resolution = 4
max_depth = 4
isovalue = 0.5

# Optional particle-level FOF labels.
group_labels = fof_cluster(
    positions, smoothing_lengths,
    domain_min, domain_max,
    linking_factor=1.5,
)

# Run the full C++ reconstruction pipeline.
result = run_full_pipeline(
    positions,
    smoothing_lengths,
    domain_min,
    domain_max,
    base_resolution,
    isovalue,
    max_depth,
)

# Save as STL.
mesh = Mesh(vertices=result["vertices"], faces=result["faces"])
mesh.save("output.stl")
```

## HDF5 serialization

The `serialize` module provides `export_octree` and `import_octree` for
saving and restoring the full octree state:

```python
from meshmerizer.serialize import export_octree, import_octree

# Save after building the octree.
export_octree(
    "tree.hdf5",
    isovalue=0.5,
    base_resolution=4,
    max_depth=4,
    domain_minimum=(0.0, 0.0, 0.0),
    domain_maximum=(1.0, 1.0, 1.0),
    positions=positions,
    smoothing_lengths=smoothing_lengths,
    cells=cells,
    contributors=contributors,
)

# Reload later.
state = import_octree("tree.hdf5")
cells = state["cells"]
contributors = state["contributors"]
```

## Package layout

```
src/meshmerizer/
  __init__.py
  adaptive_core.py          # Stable Python API for the C++ core
  _adaptive.cpp             # Python/C++ extension bindings
  reconstruct.py            # Grouped reconstruction wrappers over the C++ pipeline
  serialize.py              # HDF5 octree export/import
  logging.py                # Structured logging helpers
  logging_utils.py          # Logging utilities
  printing.py               # Print-scaling helper (--target-size)
  _version.py               # Auto-generated version (setuptools_scm)

  adaptive_cpp/             # Header-only C++ implementation
    omp_config.hpp          # Conditional OpenMP include with stubs
    vector3d.hpp            # Vector3d struct and helpers
    bounding_box.hpp        # BoundingBox with containment/overlap
    particle.hpp            # Minimal Particle struct
    morton.hpp              # Morton encode/decode helpers
    kernel_wendland_c2.hpp  # Wendland C2 value and gradient
    particle_grid.hpp       # Top-level particle binning
    octree_cell.hpp         # OctreeCell, refinement, balancing
    hermite.hpp             # HermiteSample, edge crossings
    qef.hpp                 # QEF accumulator, 3x3 solver
    mesh.hpp                # MeshVertex, MeshTriangle structs
    faces.hpp               # QEF vertex solving and face extraction helpers
    adaptive_solid.hpp      # Opened-solid regularization helpers
    fof.hpp                 # Friends-of-Friends clustering

  commands/                 # CLI modules
    args.py                 # Argument parser definitions
    main.py                 # CLI entry point
    adaptive_stl.py         # Adaptive subcommand implementation
    loading.py              # SWIFT particle loading

  mesh/                     # Mesh wrapper
    core.py                 # Mesh class, repair, save

tests/
  test_adaptive_core.py     # Unit tests for the C++ core
  test_reconstruct.py       # Reconstruction wrapper tests
  test_serialize.py         # HDF5 round-trip tests
  test_logging.py           # Logging test
```

## Testing

```bash
pytest
```

Run a single test by name:

```bash
pytest -k "watertightness"
```
