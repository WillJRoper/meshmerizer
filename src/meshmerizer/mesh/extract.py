"""Voxel-to-mesh extraction routines."""

import time
from typing import List, Optional

import numpy as np
from scipy import ndimage
from skimage import measure

from meshmerizer.logging import record_elapsed

from .core import Mesh
from .volume import prepare_volume


def voxels_to_stl(
    volume: np.ndarray,
    threshold: float,
    closing_radius: int = 1,
    split_islands: bool = False,
    remove_islands: Optional[int] = None,
    mesh_index: Optional[int] = None,
    voxel_size: float = 1.0,
) -> List[Mesh]:
    """Convert a voxel volume to meshes with standard marching cubes.

    Args:
        volume: Input scalar field.
        threshold: Threshold used to binarize the field.
        closing_radius: Radius for binary closing.
        split_islands: Whether to keep connected components separate.
        remove_islands: Optional island-removal mode.
        mesh_index: Optional specific component label to extract.
        voxel_size: Physical size of one voxel.

    Returns:
        List of extracted meshes.

    Raises:
        ValueError: If no mesh is produced for the requested configuration.
    """
    # Keep one end-to-end timer so extraction cost is reported for the whole
    # workflow rather than only the marching-cubes kernel.
    start_time = time.perf_counter()

    labeled, island_ids = prepare_volume(
        volume,
        threshold,
        closing_radius,
        split_islands,
        remove_islands,
        mesh_index,
    )

    # Extract each requested connected component independently so callers can
    # keep islands separated when desired.
    mesh_start = time.perf_counter()
    meshes: List[Mesh] = []
    for idx in island_ids:
        mask = labeled == idx
        if not mask.any():
            continue
        # Skip tiny components that are almost always numerical debris.
        if np.sum(mask) < 10:
            continue

        # March directly on the binary component mask. ``level=0.5`` places the
        # surface between empty and filled voxels.
        verts, faces, normals, _ = measure.marching_cubes(
            mask, level=0.5, spacing=(voxel_size, voxel_size, voxel_size)
        )
        meshes.append(
            Mesh(vertices=verts, faces=faces, vertex_normals=normals)
        )

    record_elapsed("Marching cubes", mesh_start, operation="Meshing")

    # Distinguish between an actually empty field and a field whose candidate
    # components were filtered out.
    if not meshes:
        if volume.max() <= threshold:
            msg = "Volume max value below threshold."
        else:
            msg = "Meshes removed by size filtering or invalid index."
        raise ValueError(f"No meshes created. {msg}")

    record_elapsed(
        "Marching-cubes extraction", start_time, operation="Meshing"
    )
    return meshes


def voxels_to_stl_via_sdf(
    volume: np.ndarray,
    threshold: float,
    closing_radius: int = 1,
    split_islands: bool = False,
    remove_islands: Optional[int] = None,
    mesh_index: Optional[int] = None,
    voxel_size: float = 1.0,
) -> List[Mesh]:
    """Convert voxels to meshes using a signed distance field.

    Args:
        volume: Input scalar field.
        threshold: Threshold used to binarize the field.
        closing_radius: Radius for binary closing.
        split_islands: Whether to keep connected components separate.
        remove_islands: Optional island-removal mode.
        mesh_index: Optional specific component label to extract.
        voxel_size: Physical size of one voxel.

    Returns:
        List of extracted meshes.

    Raises:
        ValueError: If no mesh is produced for the requested configuration.
    """
    # As in the standard extraction path, measure the full end-to-end cost of
    # the SDF workflow.
    start_time = time.perf_counter()

    labeled, island_ids = prepare_volume(
        volume,
        threshold,
        closing_radius,
        split_islands,
        remove_islands,
        mesh_index,
    )

    # Convert each component independently so the caller can still request
    # separate islands.
    meshes: List[Mesh] = []
    for idx in island_ids:
        mask = labeled == idx
        if not mask.any():
            continue

        # Build a signed distance field from the binary component and extract
        # the zero level-set as the final surface.
        d_in = ndimage.distance_transform_edt(mask)
        d_out = ndimage.distance_transform_edt(~mask)
        sdf = d_in.astype(float) - d_out.astype(float)

        verts, faces, normals, _ = measure.marching_cubes(
            volume=sdf,
            level=0.0,
            spacing=(voxel_size, voxel_size, voxel_size),
            gradient_direction="ascent",
        )
        meshes.append(
            Mesh(vertices=verts, faces=faces, vertex_normals=normals)
        )

    if not meshes:
        raise ValueError(
            "No meshes created via SDF. Check threshold and input."
        )

    record_elapsed("SDF extraction", start_time, operation="Meshing")
    return meshes
