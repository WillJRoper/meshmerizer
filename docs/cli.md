# CLI

The main command is:

```bash
meshmerizer snapshot.hdf5
```

The CLI is intended for end-to-end snapshot-to-STL usage.

For detailed per-option behavior, see the
[CLI option reference](cli-options.md).

## Workflow

At a high level, the CLI:

1. loads particles from a SWIFT snapshot or saved octree,
2. sets or computes an isovalue,
3. builds or reuses the adaptive octree,
4. extracts the mesh,
5. optionally regularizes and cleans it,
6. writes the final STL atomically.

## Common examples

### Basic run

```bash
meshmerizer snapshot.hdf5 \
  --particle-type gas \
  --base-resolution 64 \
  --max-depth 4 \
  --surface-percentile 5 \
  --output mesh.stl
```

### Print-oriented cleanup

```bash
meshmerizer snapshot.hdf5 \
  --base-resolution 128 \
  --max-depth 4 \
  --surface-percentile 0.1 \
  --min-feature-thickness 0.05 \
  --pre-thickening-radius 0.01 \
  --smoothing-iterations 10 \
  --remove-islands-fraction 0.01 \
  --simplify-factor 0.5 \
  --target-size 15 \
  --output print_ready.stl
```

### Subregion extraction

```bash
meshmerizer snapshot.hdf5 \
  --center 60 60 60 \
  --extent 20 \
  --tight-bounds \
  --output region.stl
```

### Save and reuse an octree

```bash
meshmerizer snapshot.hdf5 --save-octree tree.hdf5 --output first.stl
meshmerizer --load-octree tree.hdf5 --remove-islands-fraction 0.0 --output second.stl
```

## Important options

The options below link to the full
[CLI option reference](cli-options.md), where each option is described in more
detail.

### Geometry and refinement

- [`--base-resolution`](cli-options.md#base-resolution): number of top-level
  cells per axis
- [`--max-depth`](cli-options.md#max-depth): maximum octree depth
- [`--isovalue`](cli-options.md#isovalue): explicit isosurface threshold
- [`--surface-percentile`](cli-options.md#surface-percentile): derive isovalue
  from particle self-density percentile
- [`--min-usable-hermite-samples`](cli-options.md#min-usable-hermite-samples):
  controls how aggressively underconstrained cells continue refining
- [`--max-qef-rms-residual-ratio`](cli-options.md#max-qef-rms-residual-ratio):
  forces refinement when QEF fit quality is poor
- [`--min-normal-alignment-threshold`](cli-options.md#min-normal-alignment-threshold):
  forces refinement when surface normals are too inconsistent

### Region selection

- [`--center`](cli-options.md#center),
  [`--extent`](cli-options.md#extent): crop to a cubic subregion
- [`--tight-bounds`](cli-options.md#tight-bounds): shrink the working cube
  after crop/shift
- [`--shift`](cli-options.md#shift): shift coordinates before cropping
- [`--wrap-shift`](cli-options.md#wrap-shift) /
  [`--no-wrap-shift`](cli-options.md#no-wrap-shift): control periodic wrap
  after shifting
- [`--no-periodic`](cli-options.md#no-periodic): disable periodic subregion
  selection

### Topology and cleanup

- [`--min-feature-thickness`](cli-options.md#min-feature-thickness): remove
  fragile thin features
- [`--pre-thickening-radius`](cli-options.md#pre-thickening-radius): thicken
  the occupied solid before opening
- [`--smoothing-iterations`](cli-options.md#smoothing-iterations): smooth the
  extracted mesh
- [`--smoothing-strength`](cli-options.md#smoothing-strength): smoothing lambda
- [`--max-edge-ratio`](cli-options.md#max-edge-ratio): subdivide long edges
  relative to local cell size
- [`--remove-islands-fraction`](cli-options.md#remove-islands-fraction): remove
  small connected components
- [`--simplify-factor`](cli-options.md#simplify-factor): simplify the final
  mesh
- [`--target-size`](cli-options.md#target-size): scale the final mesh to a
  print size in cm

### Clustering

- [`--fof`](cli-options.md#fof): reconstruct FOF groups independently
- [`--min-fof-cluster-size`](cli-options.md#min-fof-cluster-size): drop small
  fluff populations before meshing
- [`--linking-factor`](cli-options.md#linking-factor): FOF linking-length
  multiplier

### Saved octrees and diagnostics

- [`--save-octree`](cli-options.md#save-octree): write a reusable HDF5 octree
  snapshot
- [`--load-octree`](cli-options.md#load-octree): reuse saved octree state
  instead of reloading particles
- [`--visualise-verts`](cli-options.md#visualise-verts): save QEF vertex
  diagnostics
- [`--nthreads`](cli-options.md#nthreads): set OpenMP thread count
- [`--silent`](cli-options.md#silent): reduce stdout chatter while keeping logs

## Full option reference

For per-option usage and behavior, including input/output flags and simpler
controls, see [CLI option reference](cli-options.md).

## Notes on units

When `--target-size` is provided, print-oriented controls such as
`--min-feature-thickness` and `--pre-thickening-radius` are interpreted in
print centimetres and converted back to native meshing units.
