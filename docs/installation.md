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

## OpenMP builds

If you want threaded native execution, set `WITH_OPENMP` before install.

```bash
# Linux
WITH_OPENMP=1 pip install -e .

# macOS with Homebrew libomp
WITH_OPENMP=/opt/homebrew/opt/libomp pip install -e .
```

If `WITH_OPENMP` is unset, Meshmerizer builds in serial mode.

## Native debug-log builds

If you want optional native debug-log file support, set `DEBUG_LOG` before
install.

```bash
DEBUG_LOG=1 pip install -e .
```

If `DEBUG_LOG` is unset, native debug-only diagnostics are compiled out.
