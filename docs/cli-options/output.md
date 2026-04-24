# `--output`, `-o`

## Usage

```bash
meshmerizer snapshot.hdf5 --output mesh.stl
```

## Effect

Sets the final STL path. If omitted, Meshmerizer derives the output filename
from the snapshot path or from [`--load-octree`](load-octree.md).

## Related

- [`filename`](filename.md)
- [`--load-octree`](load-octree.md)
- [`--save-octree`](save-octree.md)
