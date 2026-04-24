# `--tight-bounds`

## Usage

```bash
meshmerizer snapshot.hdf5 --center 60 60 60 --extent 20 --tight-bounds
```

## Effect

After any shift or crop, shrinks the working cube to the occupied particle
bounds. This removes empty margins from the meshing domain and can improve both
performance and effective resolution. It also changes the native coordinate
frame used by print-space conversions.

## Related

- [`--center`](center.md)
- [`--extent`](extent.md)
- [`--shift`](shift.md)
- [`--target-size`](target-size.md)
