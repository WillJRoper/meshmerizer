# Meshmerizer

Meshmerizer converts particle-based simulation outputs into watertight STL
meshes for visualization and 3D printing.

Its current reconstruction path is built around an adaptive octree, Wendland C2
SPH field evaluation, dual-contouring-style surface extraction, and optional
minimum-thickness regularization. The project now has a clearer split between
the public Python API, the CLI application layer, the I/O layer, and the native
C++ core.

## What Meshmerizer provides

- Adaptive particles-to-mesh reconstruction without a dense uniform voxel grid
- Optional minimum-feature-thickness regularization for print-oriented cleanup
- Friends-of-Friends clustering for disconnected structures
- Mesh post-processing helpers for smoothing, island removal, and simplification
- SWIFT snapshot loading for CLI workflows
- HDF5 octree save/load support for checkpointed reconstruction workflows
- Native C++ acceleration, with optional OpenMP threading

## Installation

Meshmerizer requires Python 3.8+.

Install the package in editable mode:

```bash
pip install -e .
```

Install development tools as well:

```bash
pip install -e ".[dev]"
```

Build a wheel or sdist:

```bash
python -m build
```

### Building with OpenMP

Set `WITH_OPENMP` before installation to enable threaded native execution.

```bash
# Linux
WITH_OPENMP=1 pip install -e .

# macOS with Homebrew libomp
WITH_OPENMP=/opt/homebrew/opt/libomp pip install -e .
```

If `WITH_OPENMP` is unset, Meshmerizer builds in serial mode.

## High-level architecture

Meshmerizer is organized into four main layers:

- `meshmerizer.api` and `meshmerizer.state`
  - public Python workflow
  - staged state objects (`TreeState`, `TopologyState`, `MeshResult`)
- `meshmerizer.cli`
  - argument parsing, snapshot/octree loading, execution flow, and status output
- `meshmerizer.io`
  - SWIFT loading, octree serialization, and atomic mesh output
- `meshmerizer.adaptive` plus the native extension
  - focused Python wrappers around the C++ adaptive meshing core

The historical `meshmerizer.commands.*`, `meshmerizer.serialize`, and
`meshmerizer.adaptive_core` paths still exist as compatibility facades, but the
new code is organized around the packages above.

## CLI workflow

The main command is:

```bash
meshmerizer snapshot.hdf5
```

The CLI is intended for snapshot-to-STL workflows. It:

1. loads particles from a SWIFT snapshot or a saved octree,
2. computes or accepts an isovalue,
3. runs the adaptive reconstruction pipeline,
4. optionally regularizes thin features,
5. post-processes the mesh, and
6. writes the final STL atomically.

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
  --pre-thickening-radius 0.01 \
  --smoothing-iterations 10 \
  --remove-islands-fraction 0.01 \
  --target-size 15 \
  --output print_ready.stl
```

When `--target-size` is set, print-oriented length controls such as
`--min-feature-thickness` and `--pre-thickening-radius` are interpreted in
print centimetres and converted back to native meshing units.

### Extract a subregion

```bash
meshmerizer snapshot.hdf5 \
  --center 60 60 60 \
  --extent 20 \
  --tight-bounds \
  --output region.stl
```

### Save and reuse an octree

Build from a snapshot and save the refined tree state:

```bash
meshmerizer snapshot.hdf5 \
  --save-octree tree.hdf5 \
  --output first_try.stl
```

Reload that octree and try a different cleanup/export configuration:

```bash
meshmerizer --load-octree tree.hdf5 \
  --remove-islands-fraction 0.0 \
  --output second_try.stl
```

### Common CLI options

| Flag | Purpose |
| --- | --- |
| `--base-resolution` | Number of top-level octree cells per axis |
| `--max-depth` | Maximum octree refinement depth |
| `--isovalue` | Explicit surface threshold |
| `--surface-percentile` | Compute the isovalue from particle self-density |
| `--particle-type` | Particle family to load from a SWIFT snapshot |
| `--center`, `--extent` | Extract a cubic subregion |
| `--tight-bounds` | Shrink the working domain to occupied particles |
| `--fof` | Reconstruct disconnected FOF groups independently |
| `--min-fof-cluster-size` | Drop small FOF fluff populations before meshing |
| `--min-feature-thickness` | Remove features thinner than a target thickness |
| `--pre-thickening-radius` | Puff up fragile features before opening |
| `--smoothing-iterations` | Apply Laplacian smoothing after extraction |
| `--remove-islands-fraction` | Drop small connected components |
| `--simplify-factor` | Simplify the final mesh by face-count fraction |
| `--target-size` | Scale the final mesh to a print size in cm |
| `--save-octree`, `--load-octree` | Save or reload octree state |
| `--visualise-verts` | Save a diagnostic QEF vertex figure |
| `--nthreads` | Set the OpenMP thread count |

## Public Python API

The public API is imported from the package root:

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

You can also extract directly from a `TreeState` when you want the staged tree
workflow without the regularized topology pass.

### Clustering particles first

```python
from meshmerizer import cluster_particles

labels = cluster_particles(
    positions,
    domain_min=(0.0, 0.0, 0.0),
    domain_max=(4.0, 4.0, 4.0),
    linking_factor=0.2,
)
```

### Mesh post-processing helpers

```python
from meshmerizer import remove_islands, smooth_mesh, subdivide_long_edges

smoothed = smooth_mesh(mesh_result.mesh, iterations=10)
cleaned = remove_islands(smoothed, remove_islands_fraction=0.01)
denser = subdivide_long_edges(cleaned, iterations=1)
```

## Public state objects

- `TreeState`
  - validated particle arrays and domain metadata
  - refined octree cells and contributor indices
  - refinement parameters needed for later extraction
- `TopologyState`
  - opened-solid occupancy masks and per-leaf diagnostics
  - opened-surface sample arrays and extracted mesh buffers
  - the source `TreeState`
- `MeshResult`
  - final `Mesh`
  - `isovalue`
  - `n_qef_vertices`

## Repository layout

```text
src/meshmerizer/
  __init__.py              # Public package exports
  api.py                   # Public Python workflow
  state.py                 # Public staged state objects
  reconstruct.py           # Raw array-based reconstruction compatibility layer
  adaptive_core.py         # Compatibility facade over focused adaptive modules
  printing.py              # Print-scaling helpers

  cli/
    main.py                # Top-level CLI entrypoint
    args.py                # Argument parsing
    adaptive.py            # Adaptive CLI orchestration
    particles.py           # CLI particle loading/filtering helpers
    units.py               # Print/native unit conversion helpers
    diagnostics.py         # CLI diagnostics and visualization

  io/
    swift.py               # SWIFT snapshot loading
    octree.py              # HDF5 octree schema and import/export
    output.py              # Atomic mesh output helpers

  mesh/
    core.py                # Lightweight Mesh wrapper over trimesh
    operations.py          # Island removal and simplification helpers

  adaptive/
    bindings.py            # Low-level native helper bindings
    tree.py                # Octree build and mesh extraction wrappers
    topology.py            # Regularization/opened-solid wrappers
    pipeline.py            # Whole-pipeline helpers
    _native.py             # Native extension import boundary

  _adaptive.cpp            # Python/C++ extension bindings
  adaptive_cpp/            # Core adaptive meshing implementation
```

## Native/core boundary

The native implementation lives in `src/meshmerizer/adaptive_cpp/` and is
exposed to Python through `src/meshmerizer/_adaptive.cpp`.

The Python wrapper stack is intentionally layered:

- `meshmerizer.adaptive.bindings`
  - direct low-level helpers and geometry queries
- `meshmerizer.adaptive.tree`
  - tree construction and direct mesh extraction wrappers
- `meshmerizer.adaptive.topology`
  - regularized opened-solid operations
- `meshmerizer.adaptive.pipeline`
  - particles-to-mesh whole-pipeline entry points

This keeps the CLI and public API from depending directly on a monolithic
binding module.

## Contributor notes

If you are changing the codebase:

- keep CLI concerns in `meshmerizer.cli`
- keep reusable loading and persistence code in `meshmerizer.io`
- keep public workflow changes centered in `meshmerizer.api`
- keep the native boundary readable by routing wrappers through
  `meshmerizer.adaptive`
- treat `meshmerizer.commands.*`, `meshmerizer.serialize`, and
  `meshmerizer.adaptive_core` as compatibility layers rather than the preferred
  home for new functionality

Useful development commands:

```bash
ruff check .
ruff format .
pytest
python -m build
```

Some tests require the `_voxelize` extension to be built first, so run an
editable install before testing from a clean environment.
