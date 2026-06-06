# Quickstart

This page is the fastest route from a fresh checkout to a first successful
mesh.

## 1. Use a supported Python

Recommended:

- CPython 3.11
- CPython 3.12

Avoid free-threaded Python builds such as `cp313t`; some native dependencies in
the scientific stack do not currently build there.

## 2. Install Meshmerizer

```bash
pip install -e .
```

If you want OpenMP threading in the native extension:

```bash
WITH_OPENMP=1 pip install -e .
```

## 3. Run a first mesh build

```bash
meshmerizer snapshot.hdf5 \
  --particle-type gas \
  --base-resolution 64 \
  --max-depth 4 \
  --surface-percentile 5 \
  --output mesh.stl
```

This command:

1. loads gas particles from the snapshot,
2. derives an isovalue from the 5th percentile of particle self-density,
3. refines the adaptive octree near the surface,
4. extracts a mesh,
5. writes `mesh.stl`.

## 4. Check that the install is healthy

Successful runs should:

- print progress and timing information,
- produce an STL file at the requested output path,
- finish without `_adaptive` import or build errors.

If the native extension did not build, Meshmerizer will usually fail early when
the CLI tries to import `_adaptive`.

## 5. Choose how to proceed

### I want a print-ready mesh

Use the print-oriented controls:

- `--target-size`
- `--min-feature-thickness`
- `--pre-thickening-radius`
- `--remove-islands-fraction`
- `--simplify-factor`

See the [CLI guide](cli.md#print-oriented-cleanup).

### I want to crop to a smaller region

Use:

- `--center X Y Z`
- `--extent SIZE`
- optionally `--tight-bounds`

### I want to iterate on cleanup settings quickly

Use:

```bash
meshmerizer snapshot.hdf5 --save-octree tree.hdf5 --output first.stl
meshmerizer --load-octree tree.hdf5 --output second.stl
```

See the [CLI guide](cli.md#save-and-reuse-an-octree).

## 6. Troubleshooting

### Install fails on Python 3.13 free-threaded

Use standard CPython 3.11 or 3.12 instead.

### `_adaptive` fails to build

Check that you have:

- a working C++ toolchain,
- NumPy installable in the environment,
- OpenMP libraries available if you enabled `WITH_OPENMP`.

### Threading does not seem active

OpenMP builds require setting `WITH_OPENMP` at install time. Without it,
Meshmerizer will still accept `--nthreads`, but the native core will run in
serial mode.
