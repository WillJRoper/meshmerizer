# `--box-size`, `-b`

## Usage

```bash
meshmerizer snapshot.hdf5 --box-size 100
```

## Effect

Overrides the simulation box size instead of using snapshot metadata. This is
mainly useful when metadata is missing, ambiguous, or needs manual correction.

## Related

- [`filename`](filename.md)
- [`--shift`](shift.md)
- [`--no-periodic`](no-periodic.md)
