# `--min-usable-hermite-samples`

## Usage

```bash
meshmerizer snapshot.hdf5 --min-usable-hermite-samples 4
```

## Effect

Controls when a corner-crossing cell is considered sufficiently constrained to
stop refining. If too few usable Hermite samples are available, Meshmerizer
keeps refining that region until support improves or
[`--max-depth`](max-depth.md) is reached. Increasing this value usually yields
more conservative refinement.

## Related

- [`--base-resolution`](base-resolution.md)
- [`--max-depth`](max-depth.md)
- [`--max-qef-rms-residual-ratio`](max-qef-rms-residual-ratio.md)
- [`--min-normal-alignment-threshold`](min-normal-alignment-threshold.md)
