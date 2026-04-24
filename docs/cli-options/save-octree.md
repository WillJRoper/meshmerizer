# `--save-octree`

## Usage

```bash
meshmerizer snapshot.hdf5 --save-octree tree.hdf5
```

## Effect

Saves the refined octree, particle arrays, and reconstruction settings to HDF5.
This is useful when you want to reuse the same tree for later meshing, cleanup,
or diagnostic experiments without rebuilding from the snapshot.

## Related

- [`--load-octree`](load-octree.md)
- [`--visualise-verts`](visualise-verts.md)
- [`--output`](output.md)
