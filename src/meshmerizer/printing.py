"""Utilities for preparing meshes for physical 3D printing."""

from typing import Any, Union

import numpy as np

from .mesh import Mesh


def scale_mesh_to_print(
    mesh: Mesh,
    target_size: Union[float, Any],
) -> Mesh:
    """Scale the mesh to a specific real-world size.

    This assumes the output unit of the STL is millimeters (standard for
    slicers).

    Args:
        mesh (Mesh): The mesh to scale.
        target_size (float or unyt_quantity): The target size for the longest
            dimension. If float, assumed to be centimeters. If unyt_quantity,
            it will be converted to centimeters.

    Returns:
        Mesh: The scaled mesh object (modified in-place, but returned for
            convenience).
    """
    # Current extents
    extents = mesh.to_trimesh().extents
    max_dimension = np.max(extents)

    if max_dimension == 0:
        print("Warning: Mesh has zero extent. Cannot scale.")
        return mesh

    # Handle unyt quantities
    if hasattr(target_size, "units"):
        # Convert to cm then take value
        try:
            target_size_cm = target_size.to("cm").value
        except Exception as e:
            print(
                f"Error converting unyt quantity: {e}. Assuming value is cm."
            )
            target_size_cm = target_size.value
    else:
        # Assume cm
        target_size_cm = float(target_size)

    # Target size in mm (standard for STL)
    target_size_mm = target_size_cm * 10.0

    # Calculate scale factor
    scale_factor = target_size_mm / max_dimension

    # Apply scale
    # Mesh wrapper doesn't have apply_scale, need to use underlying trimesh
    mesh.to_trimesh().apply_scale(scale_factor)

    print(
        f"Scaled mesh to {target_size_cm} cm "
        f"(max dimension: {target_size_mm} mm)."
    )
    print(f"Scale factor applied: {scale_factor:.4f}")

    return mesh
