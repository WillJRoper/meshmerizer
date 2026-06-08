# Meshmerizer

Meshmerizer converts particle-based simulation outputs into STL meshes for
visualization and 3D printing.

It is designed for workflows where you want to go from particle data to an
inspectable or printable surface mesh without first constructing a large dense
uniform voxel grid.

Meshmerizer is built around:

- adaptive octree refinement,
- Wendland C2 SPH field evaluation,
- dual-contouring-style surface extraction,
- optional minimum-thickness regularization, and
- mesh post-processing for print-oriented cleanup.

## What Meshmerizer provides

- Adaptive particles-to-mesh reconstruction of surfaces while avoiding the
  memory cost of a dense uniform grid
- Optional minimum-feature-thickness regularization for print-oriented cleanup
- Friends-of-Friends clustering for disconnected structures
- Mesh cleanup helpers for smoothing, island removal, and simplification
- SWIFT snapshot loading for CLI workflows
- HDF5 octree save/load support for checkpointed reconstruction
- Native C++ acceleration, with optional OpenMP threading
- A Python API for staged workflows and custom integration

## Main workflow styles

Meshmerizer supports two main usage styles:

- the **CLI**, for snapshot-to-STL workflows,
- the **Python API**, for staged or scripted reconstruction workflows.

### CLI workflow

The CLI is intended for end-to-end snapshot processing. At a high level it:

1. loads particles from a SWIFT snapshot or saved octree,
2. computes or accepts an isovalue,
3. runs an adaptive octree-based reconstruction pipeline,
4. optionally regularizes thin features,
5. post-processes the mesh, and
6. writes the final STL atomically.

### Python API workflow

The public Python API is organized around:

- `generate_mesh(...)` for the common one-shot path,
- `build_tree(...) -> regularize(...) -> extract_mesh(...)` for staged use.

This makes it possible to either use Meshmerizer as a simple particles-to-mesh
tool or integrate individual pipeline stages into more custom workflows.

## Typical use cases

Meshmerizer is a good fit when you want to:

- make printable meshes from simulation particle data,
- extract surfaces from gas or other particle distributions,
- work on a smaller cropped region of a larger simulation,
- remove small detached fluff before export,
- save an adaptive octree once and reuse it for later experiments.

## Start here

Recommended reading order:

1. [Quickstart](quickstart.md) for a first successful mesh build
2. [Installation](installation.md) for build options and troubleshooting
3. [CLI](cli.md) for task-oriented usage patterns
4. [Python API](python-api.md) if you want scripted or staged workflows
5. [Reconstruction Workflow](reconstruction-workflow.md) to understand how the
   adaptive meshing stages fit together
6. [Tuning Performance](tuning-performance.md) for runtime and quality tradeoffs

Reference material:

- [Reconstruction Workflow](reconstruction-workflow.md)
- [Tuning Performance](tuning-performance.md)
- [CLI Option Reference](cli-options/index.md)
- [Architecture](architecture.md)
- [Release & Deployment](release.md)
