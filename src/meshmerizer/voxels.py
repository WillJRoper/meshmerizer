"""Voxelization and scalar-field preprocessing helpers.

This module converts particle data into dense voxel grids and provides the
preprocessing operations used before isosurface extraction. It supports both
plain point deposition and C-accelerated smoothed deposition for SPH-like
particle data.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from swiftsimio import SWIFTDataset, cosmo_array
from swiftsimio.visualisation.volume_render import render_gas

from .logging_utils import log_status

# Import the C extension for accelerated voxelization
try:
    from . import _voxelize
except ImportError:
    # Keep a soft failure here so non-smoothed workflows can still run even if
    # the extension was not built.
    _voxelize = None
    log_status(
        "Voxelising",
        "Warning: _voxelize C extension not found. "
        "Box deposition with smoothing_lengths will be slow.",
    )


def process_log_scale(grid: np.ndarray) -> np.ndarray:
    """Apply logarithmic scaling to the grid to compress dynamic range.

    Useful for scalar fields whose values span many orders of magnitude.

    Args:
        grid: Input scalar field.

    Returns:
        Log-scaled scalar field.
    """
    # Avoid log(0) by adding a tiny epsilon relative to the field maximum.
    epsilon = 1e-10 * np.max(grid)
    if epsilon == 0:
        # Fall back to an absolute epsilon when the grid is entirely zero.
        epsilon = 1e-10
    return np.log10(grid + epsilon)


def process_remove_halos(
    grid: np.ndarray,
    threshold_percentile: float = 99.0,
) -> np.ndarray:
    """Clip the highest density peaks (halos) to reveal fainter structures.

    Args:
        grid: Input grid.
        threshold_percentile: Percentile above which to clip (e.g. 99.0).

    Returns:
        The grid with peaks clipped.
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
        grid: Input grid.
        sigma: Gaussian sigma in voxel units. Must be >= 0.

    Returns:
        Smoothed grid.

    Raises:
        ValueError: If ``sigma`` is negative.
    """
    # Reject negative widths because scipy's Gaussian filter does not interpret
    # them meaningfully for this workflow.
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
    """Enhance the 'Cosmic Web Skeleton' (Halos + Filaments).

    This method computes the Hessian matrix eigenvalues (l1 < l2 < l3) to
    identify local morphology.

    - Halos (3D curvature): l1, l2, l3 are all large negative.
    - Filaments (2D curvature): l1, l2 large negative, l3 small.
    - Walls/Sheets (1D curvature): l1 large negative, l2, l3 small.
    - Voids: All small.

    By selecting the magnitude of the *second* eigenvalue (|l2|), we capture
    structures with high curvature in at least two dimensions:
    -> Halos AND Filaments are kept.
    -> Walls (Sheets) and Voids are suppressed.

    Args:
        grid: Input 3D voxel grid, usually a density-like field.
        sigma: Smoothing scale for derivative calculation in voxel units.

    Returns:
        Normalized scalar field representing the filament-and-halo response.
    """
    log_status("Cleaning", f"Computing Hessian features (sigma={sigma})...")
    # Compute the Hessian tensor of the scalar field at the requested scale.
    hessian = hessian_matrix(
        grid,
        sigma=sigma,
        mode="reflect",
        use_gaussian_derivatives=False,
    )

    # Convert the tensor field into ordered eigenvalues. Scikit-image returns
    # them sorted as l1 <= l2 <= l3.
    eigvals = hessian_matrix_eigvals(hessian)

    # Use |l2| to keep filamentary and halo-like structures while suppressing
    # sheet-like features.
    response = np.abs(eigvals[1])

    # Normalize to 0-1 so downstream thresholding is easier to interpret.
    max_val = np.max(response)
    if max_val > 0:
        response /= max_val

    return response


def optimize_threshold_connectivity(
    grid: np.ndarray,
    max_filling_factor: float = 0.20,
    min_filling_factor: float = 0.01,
    steps: int = 20,
) -> float:
    """Find the threshold that maximizes the connectivity of the structure.

    This uses the 'Giant Component Fraction' metric:
    (Volume of Largest Component) / (Total Volume of All Components).

    We search for a threshold that yields a highly connected structure
    (high Giant Component Fraction) while keeping the total volume
    within reasonable bounds (between min and max filling factors).

    Args:
        grid: The input 3D grid (e.g., processed density or filament score).
        max_filling_factor: Maximum allowed fraction of the box volume
            (0.0-1.0). Prevents selecting the whole box.
        min_filling_factor: Minimum allowed fraction (0.0-1.0).
            Prevents selecting just a few peak voxels.
        steps: Number of threshold steps to test.

    Returns:
        Threshold that maximizes the giant connected component score within the
        requested filling-factor range.
    """
    log_status(
        "Cleaning",
        "Optimizing threshold "
        f"(Target Volume: {min_filling_factor:.1%} - "
        f"{max_filling_factor:.1%})...",
    )

    # Build candidate thresholds from percentiles that correspond to the target
    # filling-factor range.
    p_start = (1.0 - max_filling_factor) * 100
    p_end = (1.0 - min_filling_factor) * 100
    percentiles = np.linspace(p_start, p_end, steps)
    thresholds = np.percentile(grid, percentiles)

    # Initialize the best score so the first valid threshold always wins.
    best_score = -1.0
    best_threshold = thresholds[0]

    # Use 26-connectivity so filamentary structures are treated as connected
    # whenever they touch by faces, edges, or corners.
    structure = ndimage.generate_binary_structure(3, 3)

    log_status(
        "Cleaning",
        f"{'Threshold':>12} | {'Vol%':>7} | {'Giant Comp%':>10} | {'Status'}",
    )
    log_status("Cleaning", "-" * 50)

    for t in thresholds:
        mask = grid > t
        total_voxels = np.sum(mask)

        # Skip thresholds whose occupied volume falls outside the requested
        # range, even if percentile rounding put them here.
        filling_factor = total_voxels / grid.size
        if (
            filling_factor < min_filling_factor
            or filling_factor > max_filling_factor
        ):
            # Should be handled by percentile choice, but precision varies.
            continue

        # Measure connected components to score how much of the structure sits
        # in one dominant island.
        labeled, n_components = ndimage.label(mask, structure=structure)

        if n_components == 0:
            continue

        # Background occupies label 0, so ignore it when computing component
        # sizes.
        component_sizes = np.bincount(labeled.ravel())
        if len(component_sizes) < 2:  # Only background found
            continue

        largest_comp_size = component_sizes[1:].max()
        giant_fraction = largest_comp_size / total_voxels

        # Prefer thresholds that maximize the giant component fraction.
        score = giant_fraction

        is_best = ""
        if score > best_score:
            best_score = score
            best_threshold = t
            is_best = "*"

        log_status(
            "Cleaning",
            f"{t:12.4e} | {filling_factor:7.2%} | "
            f"{giant_fraction:10.1%} | {is_best}",
        )

    log_status(
        "Cleaning",
        "Optimization Complete. "
        f"Best Threshold: {best_threshold:.4e} "
        f"(Connectivity: {best_score:.1%})",
    )
    return best_threshold


def generate_voxel_grid_swift(
    data: SWIFTDataset,
    resolution: int,
    project: str = "masses",
    parallel: bool = False,
    rotation_matrix: Optional[np.ndarray] = None,
    rotation_center: Optional[cosmo_array] = None,
    region: Optional[cosmo_array] = None,
    periodic: bool = True,
) -> np.ndarray:
    """Use SWIFTsimIO's volume rendering to produce a 3D voxel grid.

    Args:
        data (SWIFTDataset): Loaded SWIFT dataset.
        resolution (int): Number of voxels along each axis.
        project (str): Data field to project (e.g. 'masses', 'temperatures',
            or your custom field).
        parallel (bool): Whether to use parallel rendering.
        rotation_matrix (Optional[np.ndarray]): 3×3 rotation matrix for
            arbitrary viewing angles.
        rotation_center (Optional[cosmo_array]): Center point for rotations.
        region (Optional[cosmo_array]): 6-element array [x_min, x_max, y_min,
            y_max, z_min, z_max].
        periodic (bool): Account for periodic boundaries.

    Returns:
        Dense voxel grid with shape ``(resolution, resolution, resolution)``.
    """
    # Delegate rendering to SWIFTsimIO, which returns a cosmo_array-like
    # object.
    grid = render_gas(
        data,
        resolution=resolution,
        project=project,
        parallel=parallel,
        rotation_matrix=rotation_matrix,
        rotation_center=rotation_center,
        region=region,
        periodic=periodic,
    )
    # Convert the result to a plain NumPy array for downstream processing.
    try:
        return grid.to_value()
    except AttributeError:
        return np.array(grid)


def generate_voxel_grid(
    data: np.ndarray,
    coordinates: np.ndarray,
    resolution: int,
    smoothing_lengths: Optional[np.ndarray] = None,
    parallel: bool = False,  # Not used for this implementation
    box_size: Optional[float] = None,
    nthreads: int = 1,
) -> Tuple[np.ndarray, float]:
    """Generate a dense voxel grid from particle samples.

    Args:
        data: Scalar value attached to each particle.
        coordinates: Particle coordinates with shape ``(N, 3)``.
        resolution: Number of voxels along each axis.
        smoothing_lengths: Optional per-particle smoothing lengths. When
            provided, box deposition is used instead of point deposition.
        parallel: Retained for compatibility. Currently unused by this helper.
        box_size: Physical size of the cubic volume. When omitted, a cubic box
            is inferred from the coordinate ranges.
        nthreads: Number of threads requested for the C smoothing kernel.

    Returns:
        Tuple containing the dense voxel grid and the physical voxel size.

    Raises:
        ValueError: If the requested resolution or input shapes are invalid.
        RuntimeError: If smoothing lengths are supplied but the C extension is
            unavailable.
    """
    # Allocate the dense grid in float64 so the C deposition kernel can write
    # into it directly without dtype conversion.
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float64)

    # Validate threading early so both the C and NumPy paths see the same error
    # behaviour.
    if nthreads < 1:
        raise ValueError(f"nthreads must be >= 1, got {nthreads}")

    coords = np.asarray(coordinates)
    # Require explicit ``(N, 3)`` coordinates because the later voxel mapping
    # assumes three spatial axes throughout.
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coordinates must be an array of shape (N, 3), got {coords.shape}"
        )

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins

    # Map coordinates onto a cubic voxel lattice, either using the supplied box
    # size or inferring one from the particle bounds.
    if box_size is not None:
        # When a box size is supplied, preserve that physical scale exactly
        # instead of inferring one from the occupied particle bounds.
        if box_size <= 0:
            raise ValueError(f"box_size must be > 0, got {box_size}")
        voxel_size = box_size / resolution

        # If the coordinates already lie in the box, keep the origin at zero.
        # Otherwise treat the minimum corner as the translated local origin.
        eps = 1e-6 * box_size
        origin = np.where(
            (mins >= -eps) & (maxs <= box_size + eps),
            0.0,
            mins,
        )

        scaled = (coords - origin) / box_size * resolution
        vox_indices = np.floor(scaled).astype(np.int64)
    else:
        # Without an explicit box size, infer a cubic box from the occupied
        # particle range so all three axes share the same voxel size.
        box_size_inferred = float(np.max(ranges))
        voxel_size = (
            box_size_inferred / resolution if box_size_inferred > 0 else 1.0
        )

        vox_indices = np.zeros_like(coords, dtype=np.int64)
        # Rescale each axis independently into the inferred cubic lattice while
        # handling degenerate single-valued axes safely.
        for axis in range(3):
            axis_range = ranges[axis]
            if axis_range > 0:
                scaled = (
                    (coords[:, axis] - mins[axis]) / axis_range * resolution
                )
                vox_indices[:, axis] = np.floor(scaled).astype(np.int64)
            else:
                vox_indices[:, axis] = 0

    # Clip indices before deposition so particles on the upper boundary stay in
    # range.
    vox_indices = np.clip(vox_indices, 0, resolution - 1)

    # Use the C extension for smoothed deposition and NumPy accumulation for
    # the simpler point-deposition path.
    if smoothing_lengths is not None:
        if _voxelize is None:
            raise RuntimeError(
                "Smoothing with Python loops is deprecated due to "
                "performance. Please ensure the _voxelize C extension "
                "is built correctly."
            )

        # Convert smoothing lengths from physical units into voxel radii before
        # passing them to the deposition kernel.
        if voxel_size > 0:
            smoothing_lengths_vox = (smoothing_lengths / voxel_size).astype(
                np.int64
            )
        else:
            smoothing_lengths_vox = np.zeros_like(
                smoothing_lengths, dtype=np.int64
            )

        # Delegate the smoothed deposition to the compiled kernel because a
        # Python implementation is too slow for the intended data sizes.
        _voxelize.box_deposition(
            grid,
            # Ensure data is float64 for C
            data.astype(np.float64),
            vox_indices,
            smoothing_lengths_vox,
            resolution,
            int(nthreads),
        )

    else:
        # Use vectorised accumulation for the simpler point-deposition path.
        x_indices = vox_indices[:, 0]
        y_indices = vox_indices[:, 1]
        z_indices = vox_indices[:, 2]
        np.add.at(grid, (x_indices, y_indices, z_indices), data)

    return grid, voxel_size
