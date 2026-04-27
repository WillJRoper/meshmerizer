"""Setuptools build configuration for the meshmerizer extensions.

This repository uses ``pyproject.toml`` for project metadata, but the compiled
extensions still need explicit ``ext_modules`` declarations so editable
installs rebuild the shared objects from source.

OpenMP support
--------------
Set the ``WITH_OPENMP`` environment variable to enable OpenMP parallelism
in the C++ extension.  The variable accepts two forms:

- A truthy value (e.g. ``WITH_OPENMP=1``) enables OpenMP using the
  system default compiler/linker search paths.
- A filesystem path (e.g. ``WITH_OPENMP=/opt/homebrew/opt/libomp``)
  additionally adds ``<path>/include`` to the include path and
  ``-L<path>/lib`` to the linker flags.  This is typically needed on
  macOS when OpenMP is installed via Homebrew.

Example usage::

    WITH_OPENMP=1 pip install -e .
    WITH_OPENMP=/opt/homebrew/opt/libomp pip install -e .
"""

from __future__ import annotations

import os
import sys

import numpy
from setuptools import Extension, setup

# Read build-time feature flags from the environment.
#
# WITH_OPENMP:
#   An empty string or unset means "no OpenMP". Any non-empty value enables
#   it; if that value is an existing directory path it is treated as the
#   OpenMP install prefix.
# DEBUG_LOG:
#   Any non-empty value enables compilation of optional native debug-log file
#   support. When unset, native debug-only diagnostics are compiled out.
WITH_OPENMP = os.environ.get("WITH_OPENMP", "")
DEBUG_LOG = os.environ.get("DEBUG_LOG", "")


def _build_adaptive_extension() -> Extension:
    """Create the ``meshmerizer._adaptive`` extension definition.

    Reads the module-level ``WITH_OPENMP`` flag to decide whether to add
    OpenMP compiler and linker flags.

    Returns:
        Configured setuptools extension for the adaptive meshing core.
    """
    include_dirs = [numpy.get_include()]
    compile_flags: list[str] = []
    link_flags: list[str] = []

    if sys.platform == "win32":
        compile_flags.append("/std:c++20")
    else:
        compile_flags.append("-std=c++20")

    if len(WITH_OPENMP) > 0:
        # If the value is a directory, add its include/ and lib/ paths
        # so the compiler can find <omp.h> and libomp/libgomp.
        if os.path.isdir(WITH_OPENMP):
            include_dirs.append(os.path.join(WITH_OPENMP, "include"))
            link_flags.append("-L" + os.path.join(WITH_OPENMP, "lib"))

        # Platform-specific compiler and linker flags.
        if sys.platform == "darwin":
            # Apple Clang needs -Xpreprocessor to forward -fopenmp.
            compile_flags.append("-Xpreprocessor")
            compile_flags.append("-fopenmp")
            link_flags.append("-lomp")
        elif sys.platform == "win32":
            compile_flags.append("/openmp")
        else:
            # Linux / other POSIX: GCC-style flags.
            compile_flags.append("-fopenmp")
            link_flags.append("-lgomp")

        # Define a preprocessor macro so C++ code can guard OpenMP
        # pragmas behind ``#ifdef WITH_OPENMP``.
        compile_flags.append("-DWITH_OPENMP")

    if len(DEBUG_LOG) > 0:
        compile_flags.append("-DDEBUG_LOG")

    return Extension(
        "meshmerizer._adaptive",
        sources=["src/meshmerizer/_adaptive.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=compile_flags,
        extra_link_args=link_flags,
    )


setup(ext_modules=[_build_adaptive_extension()])
