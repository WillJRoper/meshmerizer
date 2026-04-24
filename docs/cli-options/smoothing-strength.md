# `--smoothing-strength`

## Usage

```bash
meshmerizer snapshot.hdf5 --smoothing-iterations 10 --smoothing-strength 0.3
```

## Effect

Sets the per-iteration Laplacian smoothing strength. Lower values move vertices
more gently; higher values smooth more aggressively.

## Requires

- [`--smoothing-iterations`](smoothing-iterations.md)

## Related

- [`--smoothing-iterations`](smoothing-iterations.md)
