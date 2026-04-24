# `--isovalue`, `-t`

## Usage

```bash
meshmerizer snapshot.hdf5 --isovalue 0.01
```

## Effect

Directly sets the scalar field threshold used for surface extraction. This is
the most explicit way to control what counts as "inside" the reconstructed
surface.

When provided, it overrides
[`--surface-percentile`](surface-percentile.md).

## Related

- [`--surface-percentile`](surface-percentile.md)
- [`--smoothing-factor`](smoothing-factor.md)
