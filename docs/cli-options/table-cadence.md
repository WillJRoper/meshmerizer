# `--table-cadence`

Control how often Meshmerizer prints queue-driven refinement status rows.

## Usage

```bash
meshmerizer snapshot.hdf5 --table-cadence 5
```

## Effect

Sets the interval, in seconds, between status-table updates emitted by the
queue-driven refinement stages used during adaptive reconstruction and
regularization.

This cadence is still honored under `--silent`, which suppresses per-update
progress rendering but keeps coarse status output.

## Related

- [`--silent`](silent.md)
- [`--nthreads`](nthreads.md)
- [`--load-octree`](load-octree.md)
