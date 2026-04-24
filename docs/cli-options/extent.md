# `--extent`

## Usage

```bash
meshmerizer snapshot.hdf5 --center 60 60 60 --extent 20
```

## Effect

Sets the side length of the cubic subregion centered on
[`--center`](center.md). Smaller extents reduce the spatial domain and usually
make reconstruction cheaper.

## Requires

- [`--center`](center.md)

## Related

- [`--tight-bounds`](tight-bounds.md)
- [`--no-periodic`](no-periodic.md)
