# `filename`

## Usage

```bash
meshmerizer snapshot.hdf5
```

## Effect

This positional argument selects the SWIFT snapshot to load. In the normal CLI
path it is the main input dataset for particle loading, smoothing-length
handling, and domain setup.

If [`--load-octree`](load-octree.md) is used, the snapshot is not needed for
reconstruction itself, but a filename can still be useful when deriving a
default STL output name.

## Related

- [`--output`](output.md)
- [`--load-octree`](load-octree.md)
- [`--particle-type`](particle-type.md)
