# `--simplify-factor`

## Usage

```bash
meshmerizer snapshot.hdf5 --simplify-factor 0.5
```

## Effect

Simplifies the final mesh by retaining only a fraction of its faces. Smaller
values reduce mesh complexity more aggressively.

## Related

- [`--remove-islands-fraction`](remove-islands-fraction.md)
- [`--smoothing-iterations`](smoothing-iterations.md)
- [`--target-size`](target-size.md)
