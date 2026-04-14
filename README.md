# Meshmerizer

Meshmerizer converts particle-based simulation outputs into watertight STL
meshes. It is built around dense voxelization, signed-distance-field surface
extraction, and a chunked union path for cases that are too large to mesh in
one pass.

The current package focuses on STL generation from SWIFT snapshots and generic
particle data.

## Features

- C-accelerated smoothed voxelization for SPH-style particle data
- signed-distance-field meshing for watertight surfaces
- chunked meshing with a watertight `unioned` output mode
- optional preprocessing with log scaling, filament filtering, halo clipping,
  and Gaussian smoothing
- optional print scaling via `--target-size`

## Installation

```bash
pip install -e .
```

For development tools and tests:

```bash
pip install -e ".[dev]"
```

## CLI

The package currently exposes one command group:

```bash
meshmerizer stl snapshot.hdf5
```

The historical shorthand still works:

```bash
meshmerizer snapshot.hdf5
```

### Common examples

Dense STL generation:

```bash
meshmerizer stl snapshot.hdf5 \
  --particle-type gas \
  --field densities \
  --resolution 256 \
  --output dense.stl
```

Chunked watertight union:

```bash
meshmerizer stl snapshot.hdf5 \
  --particle-type gas \
  --field densities \
  --resolution 128 \
  --nchunks 2 \
  --chunk-output unioned \
  --output chunked.stl
```

Separate chunk files:

```bash
meshmerizer stl snapshot.hdf5 \
  --resolution 128 \
  --nchunks 2 \
  --chunk-output separate \
  --output chunked.stl
```

Subregion extraction with tighter bounds:

```bash
meshmerizer stl snapshot.hdf5 \
  --particle-type gas \
  --field densities \
  --center 60 60 60 \
  --extent 15 \
  --tight-bounds \
  --output region.stl
```

## Important options

- `--resolution`: voxel grid resolution per axis
- `--nchunks`: number of chunks per axis for chunked meshing
- `--chunk-output`: `unioned` or `separate`
- `--threshold`: isosurface threshold
- `--preprocess`: `none`, `log`, or `filaments`
- `--clip-halos`: percentile clip before preprocessing
- `--gaussian-sigma`: Gaussian smoothing width in voxel units
- `--target-size`: scale the longest mesh dimension to the given centimetres
- `--remove-islands`: keep only the largest connected component

## Python usage

```python
import numpy as np

from meshmerizer.mesh import voxels_to_stl_via_sdf
from meshmerizer.voxels import generate_voxel_grid

coordinates = np.random.rand(1000, 3)
data = np.ones(1000)
smoothing_lengths = np.full(1000, 0.05)

grid, voxel_size = generate_voxel_grid(
    data=data,
    coordinates=coordinates,
    resolution=128,
    smoothing_lengths=smoothing_lengths,
    box_size=1.0,
)

meshes = voxels_to_stl_via_sdf(
    grid,
    threshold=0.5,
    voxel_size=voxel_size,
)

meshes[0].save("output.stl")
```

## Package layout

- `src/meshmerizer/commands/`: CLI parsing, SWIFT loading, and STL command execution
- `src/meshmerizer/chunks/`: chunk geometry, hard-chunk meshing, union assembly, and chunk preprocessing
- `src/meshmerizer/mesh/`: mesh wrapper plus voxel-to-surface extraction helpers
- `src/meshmerizer/voxels/`: particle deposition, scalar-field preprocessing, and SWIFT voxel rendering
- `src/meshmerizer/printing.py`: print-scaling helper used by `--target-size`

More specifically:

- `src/meshmerizer/commands/args.py`: CLI argument definitions
- `src/meshmerizer/commands/loading.py`: SWIFT particle loading, subregion extraction, and voxel-preparation helpers
- `src/meshmerizer/commands/stl.py`: STL command orchestration
- `src/meshmerizer/commands/main.py`: package CLI entry point
- `src/meshmerizer/chunks/geometry.py`: virtual-grid and hard-chunk geometry types
- `src/meshmerizer/chunks/hard.py`: hard-boundary chunk voxelization and meshing
- `src/meshmerizer/chunks/assembly.py`: union assembly and connected-component helpers
- `src/meshmerizer/chunks/processing.py`: shared chunk-field preprocessing helpers
- `src/meshmerizer/mesh/core.py`: `Mesh` wrapper, repair, subdivision, and simplification
- `src/meshmerizer/mesh/volume.py`: connected-component preparation for voxel volumes
- `src/meshmerizer/mesh/extract.py`: marching-cubes and SDF extraction routines
- `src/meshmerizer/voxels/deposition.py`: dense voxel-grid generation from particles
- `src/meshmerizer/voxels/preprocess.py`: log scaling, filament filtering, halo clipping, and smoothing
- `src/meshmerizer/voxels/swift.py`: SWIFTsimIO-backed voxel rendering

## Testing

```bash
pytest
```

To do list:

- [ ] Further smoothing to remove voxelixed surface? (This must be done at the final mesh stage, after unioning, to avoid breaking watertightness.)
- [x] Progress indicators with tqdm.
- [x] Faster skipping of empty chunks.
- [ ] Make python API more flexible and user-friendly, introducing clear function entry points for the main functionality.
