# `--particle-type`, `-p`

## Usage

```bash
meshmerizer snapshot.hdf5 --particle-type gas
```

## Effect

Chooses which SWIFT particle family to load. This affects both which positions
are reconstructed and which smoothing lengths are used.

## Related

- [`filename`](filename.md)
- [`--smoothing-factor`](smoothing-factor.md)
- [`--box-size`](box-size.md)
