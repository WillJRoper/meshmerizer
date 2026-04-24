# Architecture

Meshmerizer is organized around a small number of clear layers.

This page is intended for developers who want to understand the codebase structure and where to add new features or fix bugs. If you're a user, you can safely ignore this page and think no more about the spaghetti that lies underneath the surface.

## Public workflow layer

- `meshmerizer.api`
- `meshmerizer.state`

This is the intended Python-facing surface.

## CLI layer

- `meshmerizer.cli`

This owns:

- argument parsing,
- snapshot/octree loading,
- orchestration,
- diagnostics,
- progress reporting.

## I/O layer

- `meshmerizer.io.swift`
- `meshmerizer.io.octree`
- `meshmerizer.io.output`

This layer handles:

- SWIFT snapshot loading,
- HDF5 octree import/export,
- atomic mesh output.

## Adaptive/native wrapper layer

- `meshmerizer.adaptive.bindings`
- `meshmerizer.adaptive.tree`
- `meshmerizer.adaptive.topology`
- `meshmerizer.adaptive.pipeline`

This layer provides focused Python wrappers around the compiled extension.

## Native implementation

- `src/meshmerizer/_adaptive.cpp`
- `src/meshmerizer/adaptive_cpp/`

This is where the performance-critical octree, topology, and extraction logic
lives.

## Repository shape

```text
src/meshmerizer/
  api.py
  state.py
  cli/
  io/
  mesh/
  adaptive/
  _adaptive.cpp
  adaptive_cpp/
```
