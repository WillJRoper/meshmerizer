# `--min-feature-thickness`

## Usage

```bash
meshmerizer snapshot.hdf5 --min-feature-thickness 0.05
```

## Effect

Enables the adaptive solid-opening regularizer and removes features thinner than
the requested thickness. This is one of the main print-preparation controls.

Without [`--target-size`](target-size.md), the value is interpreted in native
meshing units. With [`--target-size`](target-size.md), it is interpreted in
print centimetres.

## Related

- [`--pre-thickening-radius`](pre-thickening-radius.md)
- [`--target-size`](target-size.md)
- [`--smoothing-iterations`](smoothing-iterations.md)
