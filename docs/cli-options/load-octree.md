# `--load-octree`

## Usage

```bash
meshmerizer --load-octree tree.hdf5 --output mesh.stl
```

## Effect

Loads a previously saved octree and reuses its particles, bounds, isovalue, and
refinement state. This bypasses snapshot loading and tree construction.

## Related

- [`--save-octree`](save-octree.md)
- [`--visualise-verts`](visualise-verts.md)
- [`--output`](output.md)
- [`filename`](filename.md)
