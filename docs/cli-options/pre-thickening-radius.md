# `--pre-thickening-radius`

## Usage

```bash
meshmerizer snapshot.hdf5 --min-feature-thickness 0.05 --pre-thickening-radius 0.01
```

## Effect

Applies an outward thickening step before the minimum-thickness opening. This
can help preserve fragile features that would otherwise disappear entirely under
the opening operator.

Without [`--target-size`](target-size.md), the value is interpreted in native
meshing units. With [`--target-size`](target-size.md), it is interpreted in
print centimetres.

## Related

- [`--min-feature-thickness`](min-feature-thickness.md)
- [`--target-size`](target-size.md)
