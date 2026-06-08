# Installation

## Requirements

- Python 3.8+
- a working build environment for the native extension

Recommended runtimes:

- CPython 3.11
- CPython 3.12

Avoid free-threaded Python builds such as `cp313t`; some transitive native
dependencies in the scientific stack do not currently build there.

## Editable install

```bash
pip install -e .
```

## Smoke test

After installation, a minimal CLI smoke test looks like:

```bash
meshmerizer snapshot.hdf5 \
  --particle-type gas \
  --base-resolution 64 \
  --max-depth 4 \
  --surface-percentile 5 \
  --output mesh.stl
```

If `_adaptive` did not build successfully, this command will usually fail early
when the CLI imports the native extension.

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

## Native progress-counter builds

If you want per-update native progress counters for profiling or diagnostics,
set `ATOMIC_PROGRESS` before install.

```bash
ATOMIC_PROGRESS=1 pip install -e .
```

If `ATOMIC_PROGRESS` is unset, those hot-loop counters are compiled out.

## Troubleshooting

### Install fails on Python 3.13 free-threaded

Use standard CPython 3.11 or 3.12 instead.

### OpenMP build fails

Install the platform OpenMP runtime first, then rebuild with `WITH_OPENMP`.
On macOS this usually means Homebrew `libomp`.

### Native extension import fails

Check that:

- the editable install completed successfully,
- NumPy and the native build toolchain are available,
- the environment you run from is the same one you used for installation.

If you are on an HPC system, it is also worth checking that you are not mixing
module-provided Python, a user virtual environment, and compiler/runtime
libraries from different stacks.
