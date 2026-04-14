"""Meshmerizer package.

Meshmerizer converts particle-based simulation outputs into voxel grids and STL
meshes, with support for dense workflows and chunked watertight unioning.

The package is organized into focused subpackages:

- ``meshmerizer.commands``: CLI parsing, loading, and command execution.
- ``meshmerizer.chunks``: virtual-grid geometry and hard-chunk meshing.
- ``meshmerizer.mesh``: mesh wrapper, repair, and extraction routines.
- ``meshmerizer.voxels``: deposition and scalar-field preprocessing helpers.
"""
