"""Mesh wrapper package.

Exposes the lightweight :class:`Mesh` wrapper around ``trimesh.Trimesh``
used by the adaptive meshing pipeline for STL export and post-processing.
"""

from .core import Mesh

__all__ = ["Mesh"]
