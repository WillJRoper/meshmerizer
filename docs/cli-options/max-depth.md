# `--max-depth`

## Usage

```bash
meshmerizer snapshot.hdf5 --max-depth 5
```

## Effect

Sets the maximum depth adaptive refinement can reach. This is the main cap on
the smallest spatial features the octree can represent. Higher values allow
finer local detail but increase runtime and memory use.

## Related

- [`--base-resolution`](base-resolution.md)
- [`--min-usable-hermite-samples`](min-usable-hermite-samples.md)
- [`--max-qef-rms-residual-ratio`](max-qef-rms-residual-ratio.md)
- [`--min-normal-alignment-threshold`](min-normal-alignment-threshold.md)
