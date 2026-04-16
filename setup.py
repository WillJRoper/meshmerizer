"""Setuptools build configuration for the meshmerizer extensions.

This repository uses ``pyproject.toml`` for project metadata, but the compiled
extensions still need explicit ``ext_modules`` declarations so editable
installs rebuild the shared objects from source.
"""

from __future__ import annotations

import numpy
from setuptools import Extension, setup


def _build_voxelize_extension() -> Extension:
    """Create the ``meshmerizer._voxelize`` extension definition.

    Returns:
        Configured setuptools extension for the local voxelization helpers.
    """
    # Build a simple serial extension. Chunk-level parallelism is handled in
    # Python, so the helper itself does not depend on OpenMP.
    return Extension(
        "meshmerizer._voxelize",
        sources=["src/meshmerizer/_voxelize.c"],
        include_dirs=[numpy.get_include()],
    )


def _build_adaptive_extension() -> Extension:
    """Create the ``meshmerizer._adaptive`` extension definition.

    Returns:
        Configured setuptools extension for the new adaptive meshing core.
    """
    return Extension(
        "meshmerizer._adaptive",
        sources=["src/meshmerizer/_adaptive.cpp"],
        include_dirs=[numpy.get_include()],
        language="c++",
    )


setup(ext_modules=[_build_voxelize_extension(), _build_adaptive_extension()])
