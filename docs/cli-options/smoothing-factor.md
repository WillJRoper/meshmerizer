# `--smoothing-factor`

## Usage

```bash
meshmerizer snapshot.hdf5 --smoothing-factor 1.2
```

## Effect

Scales particle smoothing lengths before field evaluation. Increasing this makes
each particle influence a larger region and usually produces smoother, more
connected surfaces. Decreasing it sharpens features but can fragment the mesh.

## Related

- [`--particle-type`](particle-type.md)
- [`--surface-percentile`](surface-percentile.md)
- [`--isovalue`](isovalue.md)
