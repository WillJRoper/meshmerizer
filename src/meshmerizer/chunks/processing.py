"""Shared chunk-field preprocessing helpers."""

import math

import numpy as np

from meshmerizer.voxels import (
    process_filament_filter,
    process_gaussian_smoothing,
    process_log_scale,
    process_remove_halos,
)


def chunk_halo_voxels(gaussian_sigma: float) -> int:
    """Return the halo size needed for chunk-local processing.

    Args:
        gaussian_sigma: Gaussian smoothing width in voxel units.

    Returns:
        Number of halo voxels required around each chunk.

    Raises:
        ValueError: If ``gaussian_sigma`` is negative.
    """
    # Use a conservative four-sigma radius.
    # This gives chunk-local Gaussian smoothing enough neighbouring support
    # before the field is cropped back down.
    if gaussian_sigma < 0:
        raise ValueError("gaussian_sigma must be >= 0")
    return max(1, int(math.ceil(4.0 * gaussian_sigma)))


def preprocess_chunk_grid(
    grid: np.ndarray,
    *,
    preprocess: str,
    clip_halos: float | None,
    gaussian_sigma: float,
) -> np.ndarray:
    """Apply scalar-field preprocessing to a chunk-local grid.

    Args:
        grid: Chunk-local dense scalar field.
        preprocess: Named preprocessing mode.
        clip_halos: Optional clipping percentile.
        gaussian_sigma: Gaussian smoothing width in voxel units.

    Returns:
        Preprocessed chunk-local scalar field.

    Raises:
        ValueError: If the preprocessing mode or Gaussian width is invalid.
    """
    # Work on a float64 array so preprocessing operations behave consistently
    # regardless of the incoming grid dtype.
    out = np.asarray(grid, dtype=np.float64)

    if clip_halos is not None:
        out = process_remove_halos(out, threshold_percentile=clip_halos)

    # Apply the named transform explicitly so invalid mode names fail early.
    if preprocess == "log":
        out = process_log_scale(out)
    elif preprocess == "filaments":
        out = process_filament_filter(out)
    elif preprocess != "none":
        raise ValueError(f"Unknown preprocess mode: {preprocess}")

    # Finish with Gaussian smoothing because it is spatial rather than a field
    # remapping step.
    if gaussian_sigma < 0:
        raise ValueError("gaussian_sigma must be >= 0")
    out = process_gaussian_smoothing(out, sigma=gaussian_sigma)
    return out
