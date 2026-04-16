"""Meshmerizer package.

Meshmerizer converts particle-based simulation outputs into 3D-printable
STL meshes using an adaptive octree and dual contouring.

The package is organized into focused subpackages:

- ``meshmerizer.adaptive_core``: Python wrapper for the adaptive C++ core.
- ``meshmerizer.serialize``: HDF5 octree serialization.
- ``meshmerizer.commands``: CLI parsing, loading, and command execution.
- ``meshmerizer.mesh``: lightweight trimesh wrapper for STL export.
"""
