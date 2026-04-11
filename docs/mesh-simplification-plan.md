# Mesh Simplification Plan

## Goal

Add an opt-in post-processing step to reduce triangle count after mesh
generation, especially for chunked outputs where voxel-derived surfaces can
contain more detail than is useful for printing.

## CLI

- Add `--simplify-factor`
- Valid range: `0 < factor <= 1`
- Default: `1.0`
- Semantics:
  - `1.0`: no simplification
  - `0.5`: keep roughly 50% of faces
  - `0.25`: keep roughly 25% of faces

## Placement In Pipeline

Apply simplification after mesh generation, not in voxel space.

- Dense single-mesh path:
  - after final SDF mesh generation
  - before save
- Chunked `separate`:
  - simplify each chunk mesh before writing each STL
- Chunked `unioned`:
  - simplify only after chunk union assembly completes

## Implementation Notes

- Use a mesh-space decimation/simplification method from the existing stack if
  available.
- If `trimesh` does not provide a robust built-in option, consider Blender as
  the backend for simplification as well.
- Simplification must preserve watertightness as much as possible.
- Always skip simplification when factor is `1.0`.

## Testing

Add focused tests for:

- CLI parsing of `--simplify-factor`
- invalid values (`<= 0`, `> 1`)
- simplification reducing face count
- simplification preserving watertightness on representative meshes
- simplification on `unioned` output after boolean union

## Open Question

If simplification materially harms watertightness for some backends, we may
need to restrict it to:

- non-unioned outputs only, or
- Blender-backed simplification only
