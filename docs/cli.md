# CLI

The main CLI command is:

```bash
meshmerizer snapshot.hdf5
```

The CLI is intended for end-to-end snapshot-to-STL usage.

For detailed per-option behavior, see the
[CLI option reference](cli-options/index.md).

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

To extract a raw surface mesh from a gas distribution without regularization or
post-processing, run:

```bash
meshmerizer snapshot.hdf5 \
  --particle-type gas \
  --base-resolution 64 \
  --max-depth 4 \
  --surface-percentile 5 \
  --output mesh.stl
```

This builds an octree around the surface with a maximum depth of 4, which both
limits the computational load and sets the smallest local feature scale the
octree can represent. The isovalue is chosen automatically from the 5th
percentile of the particle self-density distribution.

### Print-oriented cleanup

To construct a mesh with print-oriented cleanup, set `--target-size` to scale
the final mesh to a target size in centimetres. This makes print-oriented
parameters such as `--min-feature-thickness` meaningful in the final printed
object rather than in the input simulation units. The `--pre-thickening-radius`
option puffs up thin features before opening so that delicate disconnected
features are less likely to disappear. In this example, `--simplify-factor 0.5`
also reduces the final face count after cleanup.

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

This is a good pattern when the goal is a printable mesh rather than a purely
diagnostic one: regularize thin structures, smooth the result, remove tiny
fragments, simplify the final surface, and then scale it into print space.

### Subregion extraction

You can also focus the reconstruction on a smaller region by defining a centre
and extent. Setting `--tight-bounds` then shrinks the working domain to the
occupied particles inside that crop, which often improves performance when the
selected region contains large empty margins.

```bash
meshmerizer snapshot.hdf5 \
  --center 60 60 60 \
  --extent 20 \
  --tight-bounds \
  --output region.stl
```

### Save and reuse an octree

When you want to experiment with cleanup settings, diagnostics, or export
choices without paying the cost of rebuilding the adaptive tree each time, save
the octree state after the initial reconstruction pass and reload it later.

```bash
meshmerizer snapshot.hdf5 --save-octree tree.hdf5 --output first.stl
meshmerizer --load-octree tree.hdf5 --remove-islands-fraction 0.0 --output second.stl
```

The first command builds the octree from the snapshot and stores the particles,
bounds, isovalue, refined cells, and contributor data in HDF5. The second
command reuses that saved state directly, which is useful when iterating on
post-processing and export behavior rather than on the tree construction.

## Important options

The options below link to the full
[CLI option reference](cli-options/index.md), where each option is described in
more detail.

### Geometry and refinement

- [`--base-resolution`](cli-options/base-resolution.md): number of top-level
  cells per axis
- [`--max-depth`](cli-options/max-depth.md): maximum octree depth
- [`--isovalue`](cli-options/isovalue.md): explicit isosurface threshold
- [`--surface-percentile`](cli-options/surface-percentile.md): derive isovalue
  from particle self-density percentile
- [`--min-usable-hermite-samples`](cli-options/min-usable-hermite-samples.md):
  controls how aggressively underconstrained cells continue refining
- [`--max-qef-rms-residual-ratio`](cli-options/max-qef-rms-residual-ratio.md):
  forces refinement when QEF fit quality is poor
- [`--min-normal-alignment-threshold`](cli-options/min-normal-alignment-threshold.md):
  forces refinement when surface normals are too inconsistent

### Region selection

- [`--center`](cli-options/center.md),
  [`--extent`](cli-options/extent.md): crop to a cubic subregion
- [`--tight-bounds`](cli-options/tight-bounds.md): shrink the working cube
  after crop/shift
- [`--shift`](cli-options/shift.md): shift coordinates before cropping
- [`--wrap-shift`](cli-options/wrap-shift.md) /
  [`--no-wrap-shift`](cli-options/no-wrap-shift.md): control periodic wrap
  after shifting
- [`--no-periodic`](cli-options/no-periodic.md): disable periodic subregion
  selection

### Topology and cleanup

- [`--min-feature-thickness`](cli-options/min-feature-thickness.md): remove
  fragile thin features
- [`--pre-thickening-radius`](cli-options/pre-thickening-radius.md): thicken
  the occupied solid before opening
- [`--smoothing-iterations`](cli-options/smoothing-iterations.md): smooth the
  extracted mesh
- [`--smoothing-strength`](cli-options/smoothing-strength.md): smoothing lambda
- [`--max-edge-ratio`](cli-options/max-edge-ratio.md): subdivide long edges
  relative to local cell size
- [`--remove-islands-fraction`](cli-options/remove-islands-fraction.md): remove
  small connected components
- [`--simplify-factor`](cli-options/simplify-factor.md): simplify the final
  mesh
- [`--target-size`](cli-options/target-size.md): scale the final mesh to a
  print size in cm

### Clustering

- [`--fof`](cli-options/fof.md): reconstruct FOF groups independently
- [`--min-fof-cluster-size`](cli-options/min-fof-cluster-size.md): drop small
  fluff populations before meshing
- [`--linking-factor`](cli-options/linking-factor.md): FOF linking-length
  multiplier

### Saved octrees and diagnostics

- [`--save-octree`](cli-options/save-octree.md): write a reusable HDF5 octree
  snapshot
- [`--load-octree`](cli-options/load-octree.md): reuse saved octree state
  instead of reloading particles
- [`--visualise-verts`](cli-options/visualise-verts.md): save QEF vertex
  diagnostics
- [`--nthreads`](cli-options/nthreads.md): set OpenMP thread count
- [`--silent`](cli-options/silent.md): hide progress bars while keeping status
  logs on stdout

## Full option reference

For per-option usage and behavior, including input/output flags and simpler
controls, see [CLI option reference](cli-options/index.md).

## Notes on units

When `--target-size` is provided, print-oriented controls such as
`--min-feature-thickness` and `--pre-thickening-radius` are interpreted in
print centimetres and converted back to native meshing units.
