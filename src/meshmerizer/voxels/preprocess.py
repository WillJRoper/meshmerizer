"""Voxel-grid preprocessing helpers.

This module contains the scalar-field transforms used before mesh extraction,
including logarithmic compression, halo clipping, Gaussian smoothing, Hessian-
based filament enhancement, and a threshold search helper based on connected-
component dominance.
"""

import numpy as np
from scipy import ndimage
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from meshmerizer.logging_utils import log_status


def process_log_scale(grid: np.ndarray) -> np.ndarray:
    """Apply logarithmic scaling to the grid to compress dynamic range.

    Args:
        grid: Input scalar field.

    Returns:
        Log-scaled scalar field.
    """
    # Add a tiny epsilon so zero-valued voxels do not produce ``-inf``.
    epsilon = 1e-10 * np.max(grid)
    if epsilon == 0:
        epsilon = 1e-10
    return np.log10(grid + epsilon)


def process_remove_halos(
    grid: np.ndarray,
    threshold_percentile: float = 99.0,
) -> np.ndarray:
    """Clip the highest density peaks to reveal fainter structures.

    Args:
        grid: Input scalar field.
        threshold_percentile: Percentile above which values are clipped.

    Returns:
        Grid with the highest peaks clipped.
    """
    # Clip only the extreme high tail so bright compact structures do not hide
    # lower-contrast extended features.
    limit = np.percentile(grid, threshold_percentile)
    log_status(
        "Cleaning",
        f"Clipping halos > {limit:.4e} ({threshold_percentile}th percentile)",
    )
    return np.clip(grid, a_min=None, a_max=limit)


def process_gaussian_smoothing(
    grid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Smooth a voxel grid with a Gaussian kernel in voxel units.

    Args:
        grid: Input scalar field.
        sigma: Gaussian sigma in voxel units.

    Returns:
        Smoothed scalar field.

    Raises:
        ValueError: If ``sigma`` is negative.
    """
    # Treat zero smoothing as a no-op so callers can pass through user settings
    # without special casing.
    if sigma < 0:
        raise ValueError(f"gaussian sigma must be >= 0, got {sigma}")
    if sigma == 0:
        return grid

    log_status(
        "Cleaning",
        f"Applying Gaussian smoothing (sigma={sigma:.3g} voxels)",
    )
    return ndimage.gaussian_filter(grid, sigma=sigma)


def process_filament_filter(
    grid: np.ndarray, sigma: float = 2.0
) -> np.ndarray:
    """Enhance filamentary and halo-like structures in a scalar field.

    Args:
        grid: Input scalar field.
        sigma: Smoothing scale used for Hessian derivatives.

    Returns:
        Normalized filament-and-halo response field.
    """
    # Compute the Hessian tensor of the field at the requested scale.
    log_status("Cleaning", f"Computing Hessian features (sigma={sigma})...")
    hessian = hessian_matrix(
        grid,
        sigma=sigma,
        mode="reflect",
        use_gaussian_derivatives=False,
    )
    # Use the magnitude of the second ordered eigenvalue so filaments and halos
    # are emphasized while sheet-like structures are suppressed.
    eigvals = hessian_matrix_eigvals(hessian)
    response = np.abs(eigvals[1])

    # Normalize to a 0-1 range so downstream thresholding behaves more
    # predictably.
    max_val = np.max(response)
    if max_val > 0:
        response /= max_val

    return response
