# `--no-wrap-shift`

## Usage

```bash
meshmerizer snapshot.hdf5 --shift 5 0 -3 --no-wrap-shift
```

## Effect

Disables wrap-back after shifting. Use this when you want a literal coordinate
shift in the current frame rather than periodic remapping into the simulation
box.

## Related

- [`--shift`](shift.md)
- [`--wrap-shift`](wrap-shift.md)
- [`--no-periodic`](no-periodic.md)
