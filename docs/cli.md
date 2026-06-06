# CLI

The main CLI command is:

```bash
meshmerizer snapshot.hdf5
```

The CLI is intended for end-to-end snapshot-to-STL usage.

For detailed per-option behavior, see the
[CLI option reference](cli-options/index.md).

For a conceptual explanation of the adaptive meshing stages, see
[Reconstruction Workflow](reconstruction-workflow.md). For runtime and quality
tradeoffs, see [Tuning Performance](tuning-performance.md).

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

This builds an octree around the surface with a maximum depth of 4. In practice
that means you are capping both the computational cost and the smallest local
feature scale the octree is allowed to represent. The isovalue is chosen
automatically from the 5th percentile of the particle self-density
distribution.

### Print-oriented cleanup

To construct a mesh with print-oriented cleanup, set `--target-size` to scale
the final mesh to a target size in centimetres. Once you do that,
print-oriented parameters such as `--min-feature-thickness` are interpreted in
the final printed object rather than in the input simulation units. The
`--pre-thickening-radius` option puffs up thin features before opening so that
delicate disconnected structures are less likely to disappear. In this example,
`--simplify-factor 0.5` also trims down the final face count after cleanup.

```bash
meshmerizer snapshot.hdf5 \
  --base-resolution 128 \
  --max-depth 4 \
  --surface-percentile 5 \
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

## Choosing key controls

### `--surface-percentile` vs `--isovalue`

- Use `--surface-percentile` when you want a sensible default that adapts to
  the particle distribution in each snapshot.
- Use `--isovalue` when you already know the threshold you want and need direct
  comparability across runs.

`--surface-percentile` uses the normal percentile scale from `0` to `100`.
For example, `--surface-percentile 5` means the 5th percentile.

### `--base-resolution` and `--max-depth`

- `--base-resolution` controls the coarse top-level grid.
- `--max-depth` controls the smallest local feature scale the adaptive octree
  may represent.

Increase these gradually. Larger values can improve fidelity, but they will
also push up runtime and memory use.

### Print-oriented controls

When `--target-size` is provided, print-oriented controls such as
`--min-feature-thickness` and `--pre-thickening-radius` are interpreted in
print centimetres and converted back to native meshing units.

Without `--target-size`, those controls are interpreted directly in the input
simulation units.

### Subregion extraction

You can also focus the reconstruction on a smaller region by defining a centre
and extent. Setting `--tight-bounds` then shrinks the working domain to the
occupied particles inside that crop, which often helps when the selected region
contains large empty margins.

```bash
meshmerizer snapshot.hdf5 \
  --center 60 60 60 \
  --extent 20 \
  --tight-bounds \
  --output region.stl
```

### Save and reuse an octree

When you want to experiment with cleanup settings, diagnostics, or export
choices without paying the cost of rebuilding the adaptive tree every time,
save the octree state after the initial reconstruction pass and reload it
later.

```bash
meshmerizer snapshot.hdf5 --save-octree tree.hdf5 --output first.stl
meshmerizer --load-octree tree.hdf5 --remove-islands-fraction 0.0 --output second.stl
```

The first command builds the octree from the snapshot and stores the particles,
bounds, isovalue, refined cells, and contributor data in HDF5. The second
command reuses that saved state directly, which is useful when iterating on
post-processing and export behavior rather than on the tree construction.

Saved octrees help most when you are repeating runs that keep the particle
data, domain, and refinement inputs fixed while changing cleanup and export
settings. Some workflows can reuse the saved tree directly, while others still
need a fuller reconstruction pass after loading.

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
- [`--table-cadence`](cli-options/table-cadence.md): control queue-status table
  update cadence during queue-driven refinement

## Full option reference

For per-option usage and behavior, including input/output flags and simpler
controls, see [CLI option reference](cli-options/index.md).

## Batch and quiet runs

For quieter long runs, especially on HPC:

- use `--silent` to suppress per-update progress rendering,
- use `--table-cadence` to control how often queue-status rows are emitted.

That combination is usually a good starting point for batch jobs where you want
clean logs but still want to see occasional progress updates.
