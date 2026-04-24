# `--wrap-shift`

## Usage

```bash
meshmerizer snapshot.hdf5 --shift 5 0 -3 --wrap-shift
```

## Effect

Wraps shifted coordinates back into `[0, box_size)` after shifting. This is the
default for periodic SWIFT snapshot workflows and is usually the correct choice
for truly periodic data.

## Related

- [`--shift`](shift.md)
- [`--no-wrap-shift`](no-wrap-shift.md)
- [`--box-size`](box-size.md)
