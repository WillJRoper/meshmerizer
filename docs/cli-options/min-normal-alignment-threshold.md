# `--min-normal-alignment-threshold`

## Usage

```bash
meshmerizer snapshot.hdf5 --min-normal-alignment-threshold 0.99
```

## Effect

Checks how aligned the usable Hermite normals are in a candidate leaf. If the
normals disagree too strongly, refinement continues. Higher values force more
refinement in curved or noisy regions.

## Related

- [`--base-resolution`](base-resolution.md)
- [`--max-depth`](max-depth.md)
- [`--min-usable-hermite-samples`](min-usable-hermite-samples.md)
- [`--max-qef-rms-residual-ratio`](max-qef-rms-residual-ratio.md)
