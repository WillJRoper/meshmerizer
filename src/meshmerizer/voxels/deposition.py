"""Particle-to-voxel deposition helpers."""

from typing import Optional, Tuple

import numpy as np

from meshmerizer.logging_utils import log_status

try:
    from meshmerizer import _voxelize
except ImportError:
    _voxelize = None
    log_status(
        "Voxelising",
        "Warning: _voxelize C extension not found. "
        "Box deposition with smoothing_lengths will be slow.",
    )


def generate_voxel_grid(
    data: np.ndarray,
    coordinates: np.ndarray,
    resolution: int,
    smoothing_lengths: Optional[np.ndarray] = None,
    parallel: bool = False,
    box_size: Optional[float] = None,
    nthreads: int = 1,
) -> Tuple[np.ndarray, float]:
    """Generate a dense voxel grid from particle samples.

    Args:
        data: Scalar value attached to each particle.
        coordinates: Particle coordinates with shape ``(N, 3)``.
        resolution: Number of voxels along each axis.
        smoothing_lengths: Optional per-particle smoothing lengths.
        parallel: Retained for compatibility. Currently unused.
        box_size: Optional physical size of the cubic volume.
        nthreads: Number of threads requested for the C smoothing kernel.

    Returns:
        Tuple containing the dense voxel grid and the physical voxel size.

    Raises:
        ValueError: If the inputs are malformed.
        RuntimeError: If smoothing is requested but the C extension is missing.
    """
    # The historical ``parallel`` argument is retained for compatibility but is
    # not used by the current implementation.
    del parallel

    # Allocate the dense output grid in float64 so the compiled deposition code
    # can write to it directly.
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float64)

    if nthreads < 1:
        raise ValueError(f"nthreads must be >= 1, got {nthreads}")

    # Validate the coordinate array early because every later step assumes a
    # strict ``(N, 3)`` shape.
    coords = np.asarray(coordinates)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coordinates must be an array of shape (N, 3), got {coords.shape}"
        )

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins

    # Map particle coordinates onto a cubic lattice. Use the explicit box size
    # when supplied, otherwise infer one from the occupied coordinate range.
    if box_size is not None:
        if box_size <= 0:
            raise ValueError(f"box_size must be > 0, got {box_size}")
        voxel_size = box_size / resolution
        # Keep the origin at zero when the coordinates already lie inside the
        # supplied box; otherwise use the minimum corner as the local origin.
        eps = 1e-6 * box_size
        origin = np.where(
            (mins >= -eps) & (maxs <= box_size + eps),
            0.0,
            mins,
        )
        scaled = (coords - origin) / box_size * resolution
        vox_indices = np.floor(scaled).astype(np.int64)
    else:
        # Without an explicit box size, infer a cubic box from the largest axis
        # span so all three axes share the same voxel scale.
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

    # Clip the mapped indices so upper-boundary particles stay inside the grid.
    vox_indices = np.clip(vox_indices, 0, resolution - 1)

    # Use the compiled kernel for smoothed deposition and vectorized NumPy
    # accumulation for the simpler point-deposition path.
    if smoothing_lengths is not None:
        if _voxelize is None:
            raise RuntimeError(
                "Smoothing with Python loops is deprecated due to "
                "performance. Please ensure the _voxelize C extension "
                "is built correctly."
            )

        # Convert smoothing lengths into voxel radii before passing them to the
        # compiled deposition kernel.
        if voxel_size > 0:
            smoothing_lengths_vox = (smoothing_lengths / voxel_size).astype(
                np.int64
            )
        else:
            smoothing_lengths_vox = np.zeros_like(
                smoothing_lengths, dtype=np.int64
            )

        _voxelize.box_deposition(
            grid,
            data.astype(np.float64),
            vox_indices,
            smoothing_lengths_vox,
            resolution,
        )
    else:
        # Point deposition is just indexed accumulation on the voxel lattice.
        x_indices = vox_indices[:, 0]
        y_indices = vox_indices[:, 1]
        z_indices = vox_indices[:, 2]
        np.add.at(grid, (x_indices, y_indices, z_indices), data)

    return grid, voxel_size
