"""A module with functions for voxelization of input point clouds."""

from typing import Optional

import numpy as np
from swiftsimio import SWIFTDataset, cosmo_array
from swiftsimio.visualisation.volume_render import render_gas


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
    smoothing_lengths: np.ndarray = None,
    parallel: bool = False,
) -> np.ndarray:
    """Generate a 3D voxel grid from a 2D array.

    Args:
        data (np.ndarray): Input data array to sort into voxels.
        coordinates (np.ndarray): Coordinates of the data points.
        smoothing_lengths (np.ndarray): Smoothing lengths for each data point.
        resolution (int): Number of voxels along each axis.
        parallel (bool): Whether to use parallel rendering.

    Returns:
        np.ndarray: 3D voxel grid of shape (resolution, resolution,
            resolution).
    """
    # Create a 3D grid of zeros
    grid = np.zeros((resolution, resolution, resolution), dtype=data.dtype)

    # Normalize coordinates to the range [0, resolution)
    vox_indices = (
        (coordinates - np.min(coordinates))
        / (np.max(coordinates) - np.min(coordinates))
        * resolution
    ).astype(int)

    # Convert smoothing lengths to voxel units (if provided)
    if smoothing_lengths is not None:
        smoothing_lengths = (
            smoothing_lengths
            / (np.max(coordinates) - np.min(coordinates))
            / resolution
        )
        smoothing_lengths = smoothing_lengths.astype(int)

    # Fill the grid with the data
    for i in range(data.shape[0]):
        x, y, z = vox_indices[i]
        if smoothing_lengths is not None:
            for ii in range(
                x - smoothing_lengths[i], x + smoothing_lengths[i] + 1
            ):
                for jj in range(
                    y - smoothing_lengths[i], y + smoothing_lengths[i] + 1
                ):
                    for kk in range(
                        z - smoothing_lengths[i], z + smoothing_lengths[i] + 1
                    ):
                        if (
                            0 <= ii < resolution
                            and 0 <= jj < resolution
                            and 0 <= kk < resolution
                        ):
                            grid[ii, jj, kk] += data[i]
        else:
            if (
                0 <= x < resolution
                and 0 <= y < resolution
                and 0 <= z < resolution
            ):
                grid[x, y, z] += data[i]
            else:
                print(
                    f"Warning: Data point {i} with coordinates ({x}, {y}, {z}) "
                    "is out of bounds and will be ignored."
                )

    return grid
