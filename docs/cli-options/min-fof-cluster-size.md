# `--min-fof-cluster-size`

## Usage

```bash
meshmerizer snapshot.hdf5 --min-fof-cluster-size 500 --linking-factor 0.2
```

## Effect

Filters out small FOF particle groups before octree construction. Unlike
[`--fof`](fof.md), this does not split the scene into separate reconstructions;
it simply removes small detached fluff populations.

## Related

- [`--linking-factor`](linking-factor.md)
- [`--fof`](fof.md)
