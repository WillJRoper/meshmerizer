# CLI Option Reference

This page gives per-option details for the Meshmerizer CLI.

Options that only set a simple path or scalar still have entries here, but the
most detailed explanations are reserved for the options that change meshing,
refinement, topology, clustering, or I/O behavior in important ways.

## Input and output

<a id="filename"></a>
### `filename`

**Usage:**

```bash
meshmerizer snapshot.hdf5
```

**Effect:**

This positional argument selects the SWIFT snapshot to load. If `--load-octree`
is used, the snapshot is not required for reconstruction, but a filename may
still be useful for deriving the default STL output name.

<a id="output"></a>
### `--output`, `-o`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --output mesh.stl
```

**Effect:**

Sets the final STL path. If omitted, Meshmerizer derives the filename from the
snapshot or octree path.

## Region selection and particle preparation

<a id="center"></a>
### `--center`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --center 60 60 60 --extent 20
```

**Effect:**

Selects the centre of a cubic subregion in simulation units. This only takes
effect when paired with `--extent`. The crop happens before meshing, so it
reduces both the working domain and the particles considered by the pipeline.

<a id="extent"></a>
### `--extent`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --center 60 60 60 --extent 20
```

**Effect:**

Sets the side length of the cubic subregion selected around `--center`. Smaller
extents reduce the spatial domain and usually make reconstruction cheaper.

<a id="tight-bounds"></a>
### `--tight-bounds`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --center 60 60 60 --extent 20 --tight-bounds
```

**Effect:**

After any shift or crop, shrinks the working cube to the occupied particle
bounds. This removes empty margins from the meshing domain and can improve both
performance and effective resolution. It also changes the native coordinate
frame that later print-space conversions use.

<a id="shift"></a>
### `--shift`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --shift 5 0 -3
```

**Effect:**

Applies a coordinate offset before any crop or tight-bounds step. This is most
useful when you want to recenter structures before extracting a region.

<a id="wrap-shift"></a>
### `--wrap-shift`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --shift 5 0 -3 --wrap-shift
```

**Effect:**

Wraps shifted coordinates back into `[0, box_size)` after shifting. This is the
default for SWIFT snapshots and is usually the right choice for periodic data.

<a id="no-wrap-shift"></a>
### `--no-wrap-shift`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --shift 5 0 -3 --no-wrap-shift
```

**Effect:**

Disables wrap-back after shifting. Use this when you want a literal shift in
the current coordinate frame rather than periodic remapping.

<a id="no-periodic"></a>
### `--no-periodic`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --center 60 60 60 --extent 20 --no-periodic
```

**Effect:**

Turns off periodic wrapping for region selection. Without this flag, subregion
selection treats the simulation box as periodic. Use this flag when the domain
should be treated as a fixed box with hard boundaries.

<a id="particle-type"></a>
### `--particle-type`, `-p`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --particle-type gas
```

**Effect:**

Chooses which SWIFT particle family to load. This affects both which positions
are reconstructed and which smoothing lengths are used.

<a id="box-size"></a>
### `--box-size`, `-b`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --box-size 100
```

**Effect:**

Overrides the simulation box size instead of using snapshot metadata. This is
mainly useful when metadata is missing, ambiguous, or needs manual correction.

<a id="smoothing-factor"></a>
### `--smoothing-factor`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --smoothing-factor 1.2
```

**Effect:**

Scales particle smoothing lengths before field evaluation. Increasing this makes
each particle influence a larger region and usually produces smoother, more
connected surfaces. Decreasing it sharpens features but can fragment the mesh.

## Refinement and surface definition

<a id="base-resolution"></a>
### `--base-resolution`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --base-resolution 64
```

**Effect:**

Sets how many top-level octree cells are created per axis before adaptive
refinement begins. Higher values increase the starting spatial resolution and
can better capture large-scale structure boundaries, but they also increase the
initial cost of octree construction.

<a id="max-depth"></a>
### `--max-depth`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --max-depth 5
```

**Effect:**

Sets the maximum depth that adaptive refinement can reach. This is the main cap
on the smallest spatial features the octree can represent. Higher values allow
finer local detail but increase runtime and memory use.

<a id="isovalue"></a>
### `--isovalue`, `-t`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --isovalue 0.01
```

**Effect:**

Directly sets the scalar field threshold used for surface extraction. This is
the most explicit way to control what counts as "inside" the reconstructed
surface. When provided, it overrides `--surface-percentile`.

<a id="surface-percentile"></a>
### `--surface-percentile`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --surface-percentile 5
```

**Effect:**

Computes an isovalue automatically from the particle self-density distribution.
Lower percentiles usually enclose more mass and produce a larger surface. This
is a convenient heuristic when an absolute isovalue is not known in advance.

<a id="min-usable-hermite-samples"></a>
### `--min-usable-hermite-samples`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --min-usable-hermite-samples 4
```

**Effect:**

Controls when a corner-crossing cell is considered sufficiently constrained to
stop refining. If too few usable Hermite samples are available, Meshmerizer
keeps refining that region until support improves or `--max-depth` is reached.
Increasing this value usually yields more conservative refinement.

<a id="max-qef-rms-residual-ratio"></a>
### `--max-qef-rms-residual-ratio`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --max-qef-rms-residual-ratio 0.05
```

**Effect:**

Limits how poor the local QEF fit is allowed to be before refinement continues.
Lower values force more refinement in regions where the local surface is not
well represented by the current cell.

<a id="min-normal-alignment-threshold"></a>
### `--min-normal-alignment-threshold`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --min-normal-alignment-threshold 0.99
```

**Effect:**

Checks how aligned the usable Hermite normals are in a candidate leaf. If the
normals disagree too strongly, refinement continues. Higher values force more
refinement in curved or noisy regions.

## Topology and mesh cleanup

<a id="smoothing-iterations"></a>
### `--smoothing-iterations`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --smoothing-iterations 10
```

**Effect:**

Runs Laplacian smoothing on the extracted mesh vertices. This can remove local
noise and improve appearance, but too much smoothing can soften sharp detail.

<a id="smoothing-strength"></a>
### `--smoothing-strength`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --smoothing-iterations 10 --smoothing-strength 0.3
```

**Effect:**

Sets the per-iteration Laplacian smoothing strength. Lower values move vertices
more gently; higher values smooth more aggressively.

<a id="min-feature-thickness"></a>
### `--min-feature-thickness`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --min-feature-thickness 0.05
```

**Effect:**

Enables the adaptive solid-opening regularizer and removes features thinner than
the requested thickness. This is one of the main print-preparation controls.
Without `--target-size`, the value is interpreted in native meshing units.
With `--target-size`, it is interpreted in print centimetres.

<a id="pre-thickening-radius"></a>
### `--pre-thickening-radius`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --min-feature-thickness 0.05 --pre-thickening-radius 0.01
```

**Effect:**

Applies an outward thickening step before the minimum-thickness opening. This
can help preserve fragile features that would otherwise disappear entirely under
the opening operator.

<a id="max-edge-ratio"></a>
### `--max-edge-ratio`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --max-edge-ratio 1.2
```

**Effect:**

Limits triangle edge length relative to local cell size. Long edges are
subdivided to reduce gaps and overly stretched triangles. Lower values produce
denser meshes.

<a id="remove-islands-fraction"></a>
### `--remove-islands-fraction`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --remove-islands-fraction 0.01
```

**Effect:**

Removes connected components whose reference volume falls below the specified
fraction of the largest component. Use `0.0` to keep only the largest connected
component.

<a id="simplify-factor"></a>
### `--simplify-factor`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --simplify-factor 0.5
```

**Effect:**

Simplifies the final mesh by retaining only a fraction of its faces. Smaller
values reduce mesh complexity more aggressively.

<a id="target-size"></a>
### `--target-size`, `-s`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --target-size 15
```

**Effect:**

Scales the final mesh so its longest dimension matches the requested print size
in centimetres. When this option is present, print-oriented controls such as
`--min-feature-thickness` and `--pre-thickening-radius` are interpreted in
print space rather than native simulation units.

## Clustering

<a id="fof"></a>
### `--fof`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --fof --linking-factor 0.2
```

**Effect:**

Runs Friends-of-Friends clustering and reconstructs each cluster independently.
This is useful when the domain contains genuinely disconnected structures that
should not be meshed as one continuous object.

<a id="min-fof-cluster-size"></a>
### `--min-fof-cluster-size`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --min-fof-cluster-size 500 --linking-factor 0.2
```

**Effect:**

Filters out small FOF particle groups before octree construction. Unlike
`--fof`, this does not split the scene into separate reconstructions; it simply
removes small detached fluff populations.

<a id="linking-factor"></a>
### `--linking-factor`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --fof --linking-factor 0.15
```

**Effect:**

Sets the Friends-of-Friends linking length as a multiplier on mean inter-point
separation. Smaller values split structures more aggressively; larger values
merge nearby structures more readily.

## Saved octrees and diagnostics

<a id="save-octree"></a>
### `--save-octree`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --save-octree tree.hdf5
```

**Effect:**

Saves the refined octree, particle arrays, and reconstruction settings to HDF5.
This is useful when you want to reuse the same tree for later meshing, cleanup,
or diagnostic experiments without reloading and rebuilding from the snapshot.

<a id="load-octree"></a>
### `--load-octree`

**Usage:**

```bash
meshmerizer --load-octree tree.hdf5 --output mesh.stl
```

**Effect:**

Loads a previously saved octree and reuses its particles, bounds, isovalue, and
refinement state. This bypasses snapshot loading and tree construction.

<a id="visualise-verts"></a>
### `--visualise-verts`

**Usage:**

```bash
meshmerizer --load-octree tree.hdf5 --visualise-verts qef_vertices.png
```

**Effect:**

Generates a six-panel diagnostic plot of solved QEF vertex projections. This is
primarily a debugging tool for inspecting vertex placement and octree quality.

<a id="nthreads"></a>
### `--nthreads`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --nthreads 8
```

**Effect:**

Sets the OpenMP thread count for the native extension. This only matters when
Meshmerizer was built with OpenMP support.

<a id="silent"></a>
### `--silent`

**Usage:**

```bash
meshmerizer snapshot.hdf5 --silent
```

**Effect:**

Suppresses detailed progress rendering on stdout while preserving summary output
and log-file diagnostics.
