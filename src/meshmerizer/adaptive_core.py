"""Python wrapper for the adaptive meshing C++ core.

This module is the stable Python entry point for the new adaptive rewrite. The
initial implementation only exposes scaffold-level functionality while the C++
core types and algorithms are introduced in later phases.
"""

from __future__ import annotations

from importlib import import_module

_adaptive = import_module("meshmerizer._adaptive")


def adaptive_status() -> str:
    """Return the current adaptive core scaffold status string.

    Returns:
        Short human-readable status from the C++ adaptive extension.
    """
    # Delegate to the compiled module so tests verify the C++ extension is the
    # real implementation entry point.
    return _adaptive.adaptive_status()


def morton_encode_3d(x: int, y: int, z: int) -> int:
    """Encode three integer coordinates into a 3-D Morton key.

    Args:
        x: X-axis integer coordinate.
        y: Y-axis integer coordinate.
        z: Z-axis integer coordinate.

    Returns:
        Bit-interleaved Morton key.
    """
    return _adaptive.morton_encode_3d(x, y, z)


def morton_decode_3d(key: int) -> tuple[int, int, int]:
    """Decode a 3-D Morton key back into integer coordinates.

    Args:
        key: Bit-interleaved Morton key.

    Returns:
        Tuple containing the decoded `(x, y, z)` coordinates.
    """
    return _adaptive.morton_decode_3d(key)


def bounding_box_contains(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    point: tuple[float, float, float],
) -> bool:
    """Return whether a point lies within a half-open bounding box.

    Args:
        minimum: Inclusive lower corner of the bounding box.
        maximum: Exclusive upper corner of the bounding box.
        point: Point to test.

    Returns:
        `True` if the point lies inside the bounding box.
    """
    return _adaptive.bounding_box_contains(minimum, maximum, point)


def bounding_box_overlaps(
    left_minimum: tuple[float, float, float],
    left_maximum: tuple[float, float, float],
    right_minimum: tuple[float, float, float],
    right_maximum: tuple[float, float, float],
) -> bool:
    """Return whether two half-open bounding boxes overlap.

    Args:
        left_minimum: Inclusive lower corner of the left bounding box.
        left_maximum: Exclusive upper corner of the left bounding box.
        right_minimum: Inclusive lower corner of the right bounding box.
        right_maximum: Exclusive upper corner of the right bounding box.

    Returns:
        `True` when the boxes overlap with positive volume.
    """
    return _adaptive.bounding_box_overlaps(
        left_minimum,
        left_maximum,
        right_minimum,
        right_maximum,
    )


def particle_fields() -> tuple[str, str, str, str]:
    """Return the documented field names of the adaptive particle payload.

    Returns:
        Tuple containing the current adaptive `Particle` member names.
    """
    return _adaptive.particle_fields()
