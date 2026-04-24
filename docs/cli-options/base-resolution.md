# `--base-resolution`

## Usage

```bash
meshmerizer snapshot.hdf5 --base-resolution 64
```

## Effect

Sets how many top-level octree cells are created per axis before adaptive
refinement begins. Higher values increase the starting spatial resolution and
can better capture large-scale structure boundaries, but they also increase the
initial octree construction cost.

## Related

- [`--max-depth`](max-depth.md)
- [`--min-usable-hermite-samples`](min-usable-hermite-samples.md)
- [`--max-qef-rms-residual-ratio`](max-qef-rms-residual-ratio.md)
- [`--min-normal-alignment-threshold`](min-normal-alignment-threshold.md)
