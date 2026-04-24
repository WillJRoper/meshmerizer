# `--no-periodic`

## Usage

```bash
meshmerizer snapshot.hdf5 --center 60 60 60 --extent 20 --no-periodic
```

## Effect

Turns off periodic wrapping for region selection. Without this flag, subregion
selection treats the simulation box as periodic. Use this when the domain
should be treated as a fixed box with hard boundaries.

## Related

- [`--center`](center.md)
- [`--extent`](extent.md)
- [`--shift`](shift.md)
- [`--box-size`](box-size.md)
