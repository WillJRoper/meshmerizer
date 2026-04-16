"""Setuptools build configuration for the meshmerizer extensions.

This repository uses ``pyproject.toml`` for project metadata, but the compiled
extensions still need explicit ``ext_modules`` declarations so editable
installs rebuild the shared objects from source.
"""

from __future__ import annotations

import numpy
from setuptools import Extension, setup


def _build_adaptive_extension() -> Extension:
    """Create the ``meshmerizer._adaptive`` extension definition.

    Returns:
        Configured setuptools extension for the adaptive meshing core.
    """
    return Extension(
        "meshmerizer._adaptive",
        sources=["src/meshmerizer/_adaptive.cpp"],
        include_dirs=[numpy.get_include()],
        language="c++",
    )


setup(ext_modules=[_build_adaptive_extension()])
