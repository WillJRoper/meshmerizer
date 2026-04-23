"""Mesh wrapper package.

Exposes the lightweight :class:`Mesh` wrapper around ``trimesh.Trimesh``
used by the adaptive meshing pipeline for STL export and post-processing.
"""

from .core import Mesh
from .operations import remove_islands, simplify_mesh

__all__ = ["Mesh", "remove_islands", "simplify_mesh"]
