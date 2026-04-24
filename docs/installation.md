# Installation

## Requirements

- Python 3.8+
- a working build environment for the native extension

## Editable install

```bash
pip install -e .
```

## Development install

```bash
pip install -e ".[dev]"
```

## Build a distribution

```bash
python -m build
```

## OpenMP builds

If you want threaded native execution, set `WITH_OPENMP` before install.

```bash
# Linux
WITH_OPENMP=1 pip install -e .

# macOS with Homebrew libomp
WITH_OPENMP=/opt/homebrew/opt/libomp pip install -e .
```

If `WITH_OPENMP` is unset, Meshmerizer builds in serial mode.

## Validation

After install, check that the package and CLI import correctly.

```bash
python -c "import meshmerizer; print(meshmerizer.__all__)"
meshmerizer --help
```

## Notes

- Some tests require the `_voxelize` extension to be built.
- For release validation, always test an installed wheel or sdist, not only an
  editable install.
