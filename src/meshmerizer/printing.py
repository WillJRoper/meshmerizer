"""Helpers for preparing generated meshes for physical printing.

This module contains lightweight utilities that convert simulation-scale meshes
into dimensions that are convenient for slicers and physical printers. The
current branch uses this module to implement the CLI's ``--target-size``
option.
"""

from typing import Any, Union

import numpy as np

from .logging_utils import log_status
from .mesh import Mesh


def scale_mesh_to_print(
    mesh: Mesh,
    target_size: Union[float, Any],
) -> Mesh:
    """Scale a mesh so its longest dimension matches a print target.

    This assumes the output unit of the STL is millimeters (standard for
    slicers).

    Args:
        mesh: Mesh to scale in place.
        target_size: Target size for the longest mesh dimension. Plain floats
            are interpreted as centimetres. Quantity-like objects with a
            ``units`` attribute are converted to centimetres when possible.

    Returns:
        The same mesh instance after in-place scaling.
    """
    # Measure the current longest dimension before computing the scale factor.
    # Measure the current mesh extents so scaling is based on the longest side.
    extents = mesh.to_trimesh().extents
    max_dimension = np.max(extents)

    # Refuse to scale degenerate geometry because the scale factor would be
    # undefined.
    if max_dimension == 0:
        log_status("Cleaning", "Warning: Mesh has zero extent. Cannot scale.")
        return mesh

    # Convert quantity-like inputs to centimetres so the CLI accepts both plain
    # numbers and unit-aware values.
    if hasattr(target_size, "units"):
        try:
            target_size_cm = target_size.to("cm").value
        except Exception as e:
            log_status(
                "Cleaning",
                f"Error converting unyt quantity: {e}. Assuming value is cm.",
            )
            target_size_cm = target_size.value
    else:
        target_size_cm = float(target_size)

    # Convert from centimetres to millimetres because that is what most slicers
    # assume for STL geometry.
    target_size_mm = target_size_cm * 10.0

    # Apply a uniform scale so the longest dimension lands exactly on the
    # requested physical size.
    scale_factor = target_size_mm / max_dimension
    mesh.to_trimesh().apply_scale(scale_factor)

    log_status(
        "Cleaning",
        f"Scaled mesh to {target_size_cm} cm "
        f"(max dimension: {target_size_mm} mm).",
    )
    log_status("Cleaning", f"Scale factor applied: {scale_factor:.4f}")

    return mesh
