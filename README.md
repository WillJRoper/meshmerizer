# Meshmerizer

**Meshmerizer** is a high-performance Python package for converting hydrodynamical simulation outputs (point clouds and SPH data) into watertight, 3D-printable STL meshes. 

It specializes in processing data from **SWIFT** simulations but is generic enough to handle any 3D point cloud data.

## Key Features

*   **Fast Voxelization**: 
    *   **C-Accelerated Smoothing**: Uses a custom C extension for high-performance SPH kernel deposition (smoothing).
    *   **Vectorized Point Cloud Deposition**: Efficient numpy-based accumulation for raw point clouds.
*   **Watertight Mesh Generation**: 
    *   **SDF (Signed Distance Field)**: Generates naturally smooth and watertight surfaces ideal for fluids.
    *   **Automatic Padding**: Ensures meshes are closed at boundaries.
*   **Physical Scaling**: 
    *   Supports `box_size` inputs to ensure the final STL mesh respects physical dimensions (e.g., parsecs, cm).
*   **Robust Repair**: 
    *   Built-in mesh cleaning to remove degenerate faces, duplicates, and unreferenced vertices.
    *   Optional Taubin smoothing for surface refinement.

## Installation

Meshmerizer requires a C compiler for the acceleration extension.

### From Source

```bash
git clone https://github.com/yourusername/meshmerizer.git
cd meshmerizer
pip install -e .
```

*Note: This will automatically compile the `_voxelize` C extension.*

### Dependencies

*   `numpy`
*   `scipy`
*   `scikit-image` (Marching Cubes)
*   `trimesh` (Mesh handling)
*   `swiftsimio` (SWIFT data loading)

## Usage

### Command Line Interface (CLI)

Meshmerizer provides a dedicated CLI for processing SWIFT simulation snapshots directly from the terminal.

**Basic Usage:**
```bash
meshmerizer snapshot_0000.hdf5
```

**Advanced Usage:**
Generate a high-resolution mesh (256^3) using SDF, project the "densities" field, and scale the final model to 15cm for printing:

```bash
meshmerizer snapshot_0000.hdf5 \
  --resolution 256 \
  --method sdf \
  --field densities \
  --target-size 15 \
  --output my_model.stl
```

**Options:**
*   `filename`: Path to the input HDF5 file.
*   `--output, -o`: Output filename (default: input + .stl).
*   `--resolution, -r`: Voxel grid resolution (default: 128).
*   `--optimize`: Automatically find the optimal threshold to maximize structure connectivity (e.g. for cosmic web). Ignores `--threshold` if set.
*   `--max-filling-factor`: When using `--optimize`, the maximum allowed volume fraction of the box (default 0.2).
*   `--preprocess`: Preprocessing step for the voxel grid: `none`, `log` (log10 scaling), `filaments` (Hessian-based structure enhancement). Recommended `log` or `filaments` for cosmic web.
*   `--particle-type, -p`: Particle type to extract: `gas`, `dark_matter`, `stars`, `black_holes` (default: `gas`).
*   `--method, -m`: Mesh generation method: `sdf` (smoother, watertight) or `standard` (default: `sdf`).
*   `--target-size, -s`: Target print size in cm.
*   `--box-size, -b`: Physical box size (overrides data bounds).
*   `--smoothing-factor`: Multiplier for SPH smoothing lengths.

### Cosmic Web Extraction Example

To extract filamentary structures from dark matter, use logarithmic scaling or the filament filter. Adding `--optimize` will automatically find the threshold that keeps the web connected while discarding noise.

```bash
meshmerizer snapshot_0000.hdf5 \
  --particle-type dark_matter \
  --preprocess log \
  --optimize \
  --max-filling-factor 0.15 \
  --resolution 256 \
  --output cosmic_web_optimized.stl
```

### Basic Example

Here is how to generate a mesh from a set of particles programmatically:

```python
import numpy as np
from meshmerizer.voxels import generate_voxel_grid
from meshmerizer.mesh import voxels_to_stl_via_sdf

# 1. Prepare your data (N, 3 coordinates)
coordinates = np.random.rand(1000, 3) 
data = np.ones(1000) # Mass or density
smoothing_lengths = np.ones(1000) * 0.05 # Optional SPH smoothing

# 2. Voxelize (C-accelerated if smoothing_lengths provided)
# box_size defines the physical scale of the grid (e.g. 1.0 kpc)
grid, voxel_size = generate_voxel_grid(
    data=data,
    coordinates=coordinates,
    resolution=128,
    smoothing_lengths=smoothing_lengths,
    box_size=1.0 
)

# 3. Generate Mesh using Signed Distance Fields (recommended for fluids)
meshes = voxels_to_stl_via_sdf(
    grid, 
    threshold=0.5, 
    voxel_size=voxel_size
)

# 4. Save
if meshes:
    meshes[0].save("output.stl")
```

### 3D Printing & Real-World Scaling

Meshmerizer includes tools to prepare your mesh for physical 3D printing by scaling it to a target real-world size.

```python
from meshmerizer.printing import scale_mesh_to_print
import unyt

# ... generate your mesh ...

# Option 1: Scale to 10 cm using a simple float
scale_mesh_to_print(meshes[0], target_size=10.0)

# Option 2: Scale using unyt quantities (e.g., 5 inches)
target_size = 5.0 * unyt.inch
scale_mesh_to_print(meshes[0], target_size=target_size)

# Save the print-ready file (units will be millimeters for slicers)
meshes[0].save("print_ready.stl")
```

### Advanced Usage

Check the `examples/` directory for a complete script generating a helical structure:

```bash
python examples/generate_example.py
```

## Development

### Running Tests

The project includes a comprehensive test suite using `pytest`.

```bash
# Install test dependencies
pip install -e .[dev]

# Run tests
pytest
```

### Project Structure

*   `src/meshmerizer/voxels.py`: Voxelization logic.
*   `src/meshmerizer/_voxelize.c`: C extension for performance.
*   `src/meshmerizer/mesh.py`: Marching cubes and mesh repair logic.
*   `tests/`: Unit tests for ensuring watertightness and scaling.