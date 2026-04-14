"""Voxelization and scalar-field preprocessing package.

This package groups the dense particle-deposition and scalar-field
preprocessing helpers used by the current public workflows.
"""

from .deposition import generate_voxel_grid
from .preprocess import (
    process_filament_filter,
    process_gaussian_smoothing,
    process_log_scale,
    process_remove_halos,
)

__all__ = [
    "generate_voxel_grid",
    "process_filament_filter",
    "process_gaussian_smoothing",
    "process_log_scale",
    "process_remove_halos",
]
