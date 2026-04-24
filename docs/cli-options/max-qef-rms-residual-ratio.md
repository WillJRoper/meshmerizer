# `--max-qef-rms-residual-ratio`

## Usage

```bash
meshmerizer snapshot.hdf5 --max-qef-rms-residual-ratio 0.05
```

## Effect

Limits how poor the local QEF fit is allowed to be before refinement continues.
Lower values force more refinement in regions where the local surface is not
well represented by the current cell.

## Related

- [`--base-resolution`](base-resolution.md)
- [`--max-depth`](max-depth.md)
- [`--min-usable-hermite-samples`](min-usable-hermite-samples.md)
- [`--min-normal-alignment-threshold`](min-normal-alignment-threshold.md)
