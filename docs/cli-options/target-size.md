# `--target-size`, `-s`

## Usage

```bash
meshmerizer snapshot.hdf5 --target-size 15
```

## Effect

Scales the final mesh so its longest dimension matches the requested print size
in centimetres.

When this option is present, print-oriented controls such as
[`--min-feature-thickness`](min-feature-thickness.md) and
[`--pre-thickening-radius`](pre-thickening-radius.md) are interpreted in print
space rather than in native simulation units.

## Related

- [`--min-feature-thickness`](min-feature-thickness.md)
- [`--pre-thickening-radius`](pre-thickening-radius.md)
- [`--tight-bounds`](tight-bounds.md)
