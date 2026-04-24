# `--center`

## Usage

```bash
meshmerizer snapshot.hdf5 --center 60 60 60 --extent 20
```

## Effect

Selects the centre of a cubic subregion in simulation units. The crop happens
before meshing, so it reduces both the working domain and the particles passed
to the adaptive pipeline.

## Requires

- [`--extent`](extent.md)

## Related

- [`--tight-bounds`](tight-bounds.md)
- [`--shift`](shift.md)
- [`--no-periodic`](no-periodic.md)
