# `--smoothing-iterations`

## Usage

```bash
meshmerizer snapshot.hdf5 --smoothing-iterations 10
```

## Effect

Runs Laplacian smoothing on the extracted mesh vertices. This can remove local
noise and improve appearance, but too much smoothing can soften sharp detail.

## Related

- [`--smoothing-strength`](smoothing-strength.md)
- [`--simplify-factor`](simplify-factor.md)
- [`--remove-islands-fraction`](remove-islands-fraction.md)
