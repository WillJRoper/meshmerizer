# `--remove-islands-fraction`

## Usage

```bash
meshmerizer snapshot.hdf5 --remove-islands-fraction 0.01
```

## Effect

Removes connected components whose reference volume falls below the specified
fraction of the largest component. Use `0.0` to keep only the largest connected
component.

## Related

- [`--simplify-factor`](simplify-factor.md)
- [`--smoothing-iterations`](smoothing-iterations.md)
- [`--fof`](fof.md)
