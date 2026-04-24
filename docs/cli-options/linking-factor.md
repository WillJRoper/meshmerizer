# `--linking-factor`

## Usage

```bash
meshmerizer snapshot.hdf5 --fof --linking-factor 0.15
```

## Effect

Sets the Friends-of-Friends linking length as a multiplier on mean inter-point
separation. Smaller values split structures more aggressively; larger values
merge nearby structures more readily.

## Requires

- [`--fof`](fof.md) or [`--min-fof-cluster-size`](min-fof-cluster-size.md)

## Related

- [`--fof`](fof.md)
- [`--min-fof-cluster-size`](min-fof-cluster-size.md)
