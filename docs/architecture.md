# Architecture

Meshmerizer is organized around a small number of clear layers.

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

## Compatibility modules

These still exist, but they are no longer the preferred home for new code:

- `meshmerizer.commands.*`
- `meshmerizer.serialize`
- `meshmerizer.adaptive_core`

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
