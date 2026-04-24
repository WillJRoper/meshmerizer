# `--shift`

## Usage

```bash
meshmerizer snapshot.hdf5 --shift 5 0 -3
```

## Effect

Applies a coordinate offset before any crop or tight-bounds step. This is most
useful when you want to recenter structures before extracting a region.

## Related

- [`--wrap-shift`](wrap-shift.md)
- [`--no-wrap-shift`](no-wrap-shift.md)
- [`--center`](center.md)
- [`--tight-bounds`](tight-bounds.md)
