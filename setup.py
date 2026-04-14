"""Setuptools build configuration for the meshmerizer C extension.

This repository uses ``pyproject.toml`` for project metadata, but the compiled
``meshmerizer._voxelize`` extension still needs an explicit ``ext_modules``
declaration so editable installs rebuild the shared object from source.
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


setup(ext_modules=[_build_voxelize_extension()])
