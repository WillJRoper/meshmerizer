# `--fof`

## Usage

```bash
meshmerizer snapshot.hdf5 --fof --linking-factor 0.2
```

## Effect

Runs Friends-of-Friends clustering and reconstructs each cluster independently.
This is useful when the domain contains genuinely disconnected structures that
should not be meshed as one continuous object.

## Related

- [`--linking-factor`](linking-factor.md)
- [`--min-fof-cluster-size`](min-fof-cluster-size.md)
- [`--remove-islands-fraction`](remove-islands-fraction.md)
