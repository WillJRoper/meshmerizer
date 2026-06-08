# Meshmerizer

Meshmerizer converts particle-based simulation outputs into STL meshes for
visualization and 3D printing.

Full documentation lives at
<https://willjroper.github.io/meshmerizer/>.

## What it does

- adaptive particles-to-mesh reconstruction without having to build a dense
  uniform voxel grid first
- optional minimum-feature-thickness regularization for print-oriented cleanup
- SWIFT snapshot loading for CLI workflows
- HDF5 octree save/load support for iterative workflows
- a staged Python API for tree construction, regularization, and extraction
- native C++ acceleration with optional OpenMP threading

## Recommended runtime

- Use standard CPython 3.11 or 3.12 when you can.
- Avoid free-threaded Python builds such as `cp313t`; some of the native
  dependencies further down the stack do not currently build there.

## Install

```bash
pip install -e .
```

For threaded native execution:

```bash
WITH_OPENMP=1 pip install -e .
```

For macOS with Homebrew `libomp`:

```bash
WITH_OPENMP=/opt/homebrew/opt/libomp pip install -e .
```

## Optional native CMake build

This is an optional convenience path for native-only iteration. It builds the
same `_adaptive` extension into `src/meshmerizer/` and does not replace the
existing `pip install -e .` workflow.

```bash
cmake -S . -B build/cmake \
  -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -DMESHMERIZER_WITH_OPENMP=ON \
  -DMESHMERIZER_OPENMP_ROOT=/opt/homebrew/opt/libomp
cmake --build build/cmake -j
```

Without OpenMP:

```bash
cmake -S . -B build/cmake \
  -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"
cmake --build build/cmake -j
```

## Quick smoke test

```bash
meshmerizer snapshot.hdf5 \
  --particle-type gas \
  --base-resolution 64 \
  --max-depth 4 \
  --surface-percentile 5 \
  --output mesh.stl
```

`--surface-percentile` uses the usual percentile scale from `0` to `100`, so
`5` really does mean the 5th percentile.

## Common next steps

- Want a first successful run? See the
  [Quickstart](https://willjroper.github.io/meshmerizer/quickstart/)
- Want CLI workflows and examples? See the
  [CLI guide](https://willjroper.github.io/meshmerizer/cli/)
- Want scripted workflows? See the
  [Python API guide](https://willjroper.github.io/meshmerizer/python-api/)
- Want to understand the adaptive meshing stages? See the
  [Reconstruction Workflow](https://willjroper.github.io/meshmerizer/reconstruction-workflow/)
- Want help balancing runtime and mesh quality? See the
  [Tuning Performance](https://willjroper.github.io/meshmerizer/tuning-performance/)

The public Python API also exposes `nthreads` on the main reconstruction entry
points if you want to control native parallelism from Python.

## Threshold histogram helper

Use `plot_mask_histogram.py` to inspect a candidate masking variable before
choosing CLI threshold values:

```bash
python plot_mask_histogram.py snapshot.hdf5 \
  --threshold-key /PartType0/Densities \
  --log-x \
  --low-thresh 1e-3 \
  --up-thresh 1e2
```

The helper prints a compact percentile summary and saves a histogram PNG with
percentile markers plus any threshold guide lines you provide.
