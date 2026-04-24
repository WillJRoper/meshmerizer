# `--surface-percentile`

## Usage

```bash
meshmerizer snapshot.hdf5 --surface-percentile 5
```

## Effect

Computes an isovalue automatically from the particle self-density distribution.
Lower percentiles usually enclose more mass and produce a larger surface. This
is a convenient heuristic when an absolute isovalue is not known in advance.

If [`--isovalue`](isovalue.md) is set explicitly, this option is ignored.

## Related

- [`--isovalue`](isovalue.md)
- [`--smoothing-factor`](smoothing-factor.md)
