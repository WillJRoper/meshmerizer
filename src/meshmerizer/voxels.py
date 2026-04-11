"""A module with functions for voxelization of input point clouds."""

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from swiftsimio import SWIFTDataset, cosmo_array
from swiftsimio.visualisation.volume_render import render_gas

# Import the C extension for accelerated voxelization
try:
    from . import _voxelize
except ImportError:
    _voxelize = None
    print(
        "Warning: _voxelize C extension not found. "
        "Box deposition with smoothing_lengths will be slow."
    )


def process_log_scale(grid: np.ndarray) -> np.ndarray:
    """Apply logarithmic scaling to the grid to compress dynamic range.

    Useful for cosmic web visualization where density spans many orders of
    magnitude.
    """
    # Avoid log(0) by adding a tiny epsilon relative to the max value
    epsilon = 1e-10 * np.max(grid)
    if epsilon == 0:
        # Handle empty grid
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
    limit = np.percentile(grid, threshold_percentile)
    print(
        f"Clipping halos > {limit:.4e} ({threshold_percentile}th percentile)"
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
    """
    if sigma < 0:
        raise ValueError(f"gaussian sigma must be >= 0, got {sigma}")
    if sigma == 0:
        return grid

    print(f"Applying Gaussian smoothing (sigma={sigma:.3g} voxels)")
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
        grid (np.ndarray): Input 3D voxel grid (usually density).
        sigma (float): Smoothing scale for derivative calculation.
                       Matches the width of filaments to detect (in voxels).

    Returns:
        np.ndarray: A normalized (0-1) scalar field representing the web
            skeleton.
    """
    print(f"Computing Hessian features (sigma={sigma})...")
    # Compute Hessian (returns list of gradients)
    hessian = hessian_matrix(
        grid,
        sigma=sigma,
        mode="reflect",
        use_gaussian_derivatives=False,
    )

    # Compute Eigenvalues
    # Scikit-image sorts values: l1 <= l2 <= l3
    # For density peaks, eigenvalues are negative.
    # l1 is the most negative (largest magnitude curvature).
    # l2 is the second most negative.
    eigvals = hessian_matrix_eigvals(hessian)

    # Use |l2| to target filaments + halos, ignoring sheets.
    response = np.abs(eigvals[1])

    # Normalize to 0-1 for easier thresholding
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
        float: The optimal threshold value.
    """
    print(
        "Optimizing threshold "
        f"(Target Volume: {min_filling_factor:.1%} - "
        f"{max_filling_factor:.1%})..."
    )

    # 1. Determine candidate thresholds based on percentiles
    # We want to keep between min and max factor of the voxels.
    # So we look at percentiles (1 - max) to (1 - min).
    p_start = (1.0 - max_filling_factor) * 100
    p_end = (1.0 - min_filling_factor) * 100
    percentiles = np.linspace(p_start, p_end, steps)
    thresholds = np.percentile(grid, percentiles)

    best_score = -1.0
    best_threshold = thresholds[0]

    # Structuring element for connectivity (26-connectivity is good for
    # filaments).
    structure = ndimage.generate_binary_structure(3, 3)

    print(
        f"{'Threshold':>12} | {'Vol%':>7} | {'Giant Comp%':>10} | {'Status'}"
    )
    print("-" * 50)

    for t in thresholds:
        mask = grid > t
        total_voxels = np.sum(mask)

        # Check volume constraints
        filling_factor = total_voxels / grid.size
        if (
            filling_factor < min_filling_factor
            or filling_factor > max_filling_factor
        ):
            # Should be handled by percentile choice, but precision varies.
            continue

        # Label connected components
        labeled, n_components = ndimage.label(mask, structure=structure)

        if n_components == 0:
            continue

        # Get sizes of components
        # 0 is background, so we skip it. But bincount includes it.
        component_sizes = np.bincount(labeled.ravel())
        if len(component_sizes) < 2:  # Only background found
            continue

        largest_comp_size = component_sizes[1:].max()
        giant_fraction = largest_comp_size / total_voxels

        # Score: Primarily Giant Fraction.
        # We prefer higher connectivity.
        score = giant_fraction

        is_best = ""
        if score > best_score:
            best_score = score
            best_threshold = t
            is_best = "*"

        print(
            f"{t:12.4e} | {filling_factor:7.2%} | "
            f"{giant_fraction:10.1%} | {is_best}"
        )

    print(
        "Optimization Complete. "
        f"Best Threshold: {best_threshold:.4e} "
        f"(Connectivity: {best_score:.1%})"
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
        np.ndarray: 3D voxel grid of shape (resolution, resolution,
            resolution).
    """
    # Perform the volume render, returns a cosmo_array
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
    # Convert to a plain numpy array
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
    """Generate a 3D voxel grid from a 2D array.

    Args:
        data (np.ndarray): Input data array to sort into voxels.
        coordinates (np.ndarray): Coordinates of the data points.
        smoothing_lengths (np.ndarray, optional): Smoothing lengths per data
            point.
            If provided, a box-deposition kernel is used.
        resolution (int): Number of voxels along each axis.
        parallel (bool): Whether to use parallel rendering. (Currently ignored;
            parallelism is handled by underlying C code if available, or
            implicitly by numpy operations.)
        box_size (float, optional): The physical size of the cubic volume. If
            provided, the returned `voxel_size` will be
            `box_size / resolution`.
            Otherwise, `box_size` is inferred from coordinate bounds and
            `voxel_size` is set accordingly.
        nthreads (int): Number of threads to request for the C smoothing
            deposition kernel. Ignored for non-smoothed voxelization.

    Returns:
        tuple:
            - np.ndarray: 3D voxel grid of shape (resolution, resolution,
              resolution).
            - float: The physical size of a single voxel in the grid.
    """
    # Create a 3D grid of zeros, ensure it's float64 for C extension
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float64)

    if nthreads < 1:
        raise ValueError(f"nthreads must be >= 1, got {nthreads}")

    coords = np.asarray(coordinates)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coordinates must be an array of shape (N, 3), got {coords.shape}"
        )

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins

    # Determine voxel size and map coordinates to voxel indices.
    #
    # - If box_size is provided, we assume coordinates are in [0, box_size]
    #   (typical SWIFT) or [min, min + box_size] (translated subvolume).
    # - If box_size is not provided, infer an approximate cubic box size from
    #   the coordinate bounds and return a meaningful voxel_size.
    if box_size is not None:
        if box_size <= 0:
            raise ValueError(f"box_size must be > 0, got {box_size}")
        voxel_size = box_size / resolution

        eps = 1e-6 * box_size
        origin = np.where(
            (mins >= -eps) & (maxs <= box_size + eps),
            0.0,
            mins,
        )

        scaled = (coords - origin) / box_size * resolution
        vox_indices = np.floor(scaled).astype(np.int64)
    else:
        box_size_inferred = float(np.max(ranges))
        voxel_size = (
            box_size_inferred / resolution if box_size_inferred > 0 else 1.0
        )

        vox_indices = np.zeros_like(coords, dtype=np.int64)
        for axis in range(3):
            axis_range = ranges[axis]
            if axis_range > 0:
                scaled = (
                    (coords[:, axis] - mins[axis]) / axis_range * resolution
                )
                vox_indices[:, axis] = np.floor(scaled).astype(np.int64)
            else:
                vox_indices[:, axis] = 0

    # Clip to ensure valid indices before passing to C or np.add.at.
    vox_indices = np.clip(vox_indices, 0, resolution - 1)

    # Convert smoothing lengths to voxel units (if provided)
    if smoothing_lengths is not None:
        if _voxelize is None:
            raise RuntimeError(
                "Smoothing with Python loops is deprecated due to "
                "performance. Please ensure the _voxelize C extension "
                "is built correctly."
            )

        # Ensure smoothing lengths are int64 and convert to voxel units
        if voxel_size > 0:
            smoothing_lengths_vox = (smoothing_lengths / voxel_size).astype(
                np.int64
            )
        else:
            smoothing_lengths_vox = np.zeros_like(
                smoothing_lengths, dtype=np.int64
            )

        # Call the C extension
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
        # Use numpy.add.at for efficient non-smoothed deposition
        x_indices = vox_indices[:, 0]
        y_indices = vox_indices[:, 1]
        z_indices = vox_indices[:, 2]
        np.add.at(grid, (x_indices, y_indices, z_indices), data)

    return grid, voxel_size
