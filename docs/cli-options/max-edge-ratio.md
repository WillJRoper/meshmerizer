# `--max-edge-ratio`

## Usage

```bash
meshmerizer snapshot.hdf5 --max-edge-ratio 1.2
```

## Effect

Limits triangle edge length relative to local cell size. Long edges are
subdivided to reduce gaps and overly stretched triangles. Lower values produce
denser meshes.

## Related

- [`--max-depth`](max-depth.md)
- [`--smoothing-iterations`](smoothing-iterations.md)
