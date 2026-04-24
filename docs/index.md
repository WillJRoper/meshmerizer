# Meshmerizer

Meshmerizer converts particle-based simulation outputs into watertight STL
meshes for visualization and 3D printing.

It is built around:

- adaptive octree refinement,
- Wendland C2 SPH field evaluation,
- dual-contouring-style surface extraction,
- optional minimum-thickness regularization, and
- post-processing for print-oriented cleanup.

## What it is for

Meshmerizer is designed for workflows where you want to go from particle data to
a printable or inspectable triangle mesh without first constructing a dense
uniform voxel grid.

It supports two main usage styles:

- the **CLI**, for snapshot-to-STL workflows,
- the **Python API**, for staged or scripted reconstruction workflows.

## Core capabilities

- adaptive particles-to-mesh reconstruction,
- Friends-of-Friends clustering,
- optional topology regularization via minimum feature thickness,
- mesh cleanup and simplification,
- SWIFT snapshot loading,
- saved octree reuse through HDF5,
- native C++ acceleration.

## Start here

- [Installation](installation.md) for local setup and build notes
- [CLI](cli.md) for snapshot-to-STL usage
- [Python API](python-api.md) for scripted workflows
- [Architecture](architecture.md) for package structure and native boundaries
- [Release & Deployment](release.md) for build and PyPI publishing
