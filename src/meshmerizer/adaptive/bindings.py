"""Low-level Python wrappers for native adaptive-core utilities."""

from __future__ import annotations

from ._native import _adaptive


def adaptive_status() -> str:
    """Return a short status string from the compiled adaptive extension.

    Returns:
        Short human-readable status from the C++ adaptive extension.
    """
    return _adaptive.adaptive_status()


def morton_encode_3d(x: int, y: int, z: int) -> int:
    """Encode three integer coordinates into a 3-D Morton key.

    Args:
        x: X-axis integer coordinate.
        y: Y-axis integer coordinate.
        z: Z-axis integer coordinate.

    Returns:
        Interleaved 3-D Morton key.
    """
    # Keep the wrapper tiny so tests can reach the native implementation
    # directly while still exposing a documented Python signature.
    return _adaptive.morton_encode_3d(x, y, z)


def morton_decode_3d(key: int) -> tuple[int, int, int]:
    """Decode a 3-D Morton key back into integer coordinates.

    Args:
        key: Morton key to decode.

    Returns:
        Tuple of ``(x, y, z)`` integer coordinates.
    """
    # Decoding is exposed for diagnostics and tests that inspect octree layout
    # without reimplementing the bit-interleaving logic in Python.
    return _adaptive.morton_decode_3d(key)


def bounding_box_contains(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    point: tuple[float, float, float],
) -> bool:
    """Return whether a point lies within a half-open bounding box.

    Args:
        minimum: Lower corner of the box.
        maximum: Upper corner of the box.
        point: Query point.

    Returns:
        ``True`` if the point lies inside the half-open interval.
    """
    # Bounding-box predicates are used heavily in tests and debugging helpers,
    # so they stay available as thin documented wrappers.
    return _adaptive.bounding_box_contains(minimum, maximum, point)


def bounding_box_overlaps(
    left_minimum: tuple[float, float, float],
    left_maximum: tuple[float, float, float],
    right_minimum: tuple[float, float, float],
    right_maximum: tuple[float, float, float],
) -> bool:
    """Return whether two half-open bounding boxes overlap.

    Args:
        left_minimum: Lower corner of the left box.
        left_maximum: Upper corner of the left box.
        right_minimum: Lower corner of the right box.
        right_maximum: Upper corner of the right box.

    Returns:
        ``True`` if the two half-open boxes overlap.
    """
    # Overlap testing is delegated to the native implementation so it matches
    # the exact interval semantics used during contributor queries.
    return _adaptive.bounding_box_overlaps(
        left_minimum,
        left_maximum,
        right_minimum,
        right_maximum,
    )


def particle_fields() -> tuple[str, str, str, str]:
    """Return the documented field names of the adaptive particle payload.

    Returns:
        Tuple of field names expected by the native particle bridge.
    """
    # Exposing the field order here helps tests and diagnostics stay aligned
    # with the native payload contract.
    return _adaptive.particle_fields()


def wendland_c2_value(
    radius: float,
    smoothing_length: float,
    normalize: bool = False,
) -> float:
    """Evaluate the Wendland C2 kernel at one radius.

    Args:
        radius: Radial distance from the particle centre.
        smoothing_length: Particle smoothing length.
        normalize: Whether to apply the kernel normalization constant.

    Returns:
        Kernel value at the requested radius.
    """
    # Keep the Python wrapper minimal so kernel probes used in tests still hit
    # the exact production implementation.
    return _adaptive.wendland_c2_value(radius, smoothing_length, normalize)


def wendland_c2_gradient(
    displacement: tuple[float, float, float],
    smoothing_length: float,
    normalize: bool = False,
) -> tuple[float, float, float]:
    """Evaluate the Wendland C2 kernel gradient for one displacement.

    Args:
        displacement: Vector from the particle centre to the query point.
        smoothing_length: Particle smoothing length.
        normalize: Whether to apply the kernel normalization constant.

    Returns:
        Gradient vector at the requested displacement.
    """
    # The gradient wrapper mirrors the scalar kernel helper so low-level tests
    # can validate both value and derivative behavior consistently.
    return _adaptive.wendland_c2_gradient(
        displacement,
        smoothing_length,
        normalize,
    )


def top_level_bin_counts(
    positions: list[tuple[float, float, float]],
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    resolution: int,
) -> tuple[int, ...]:
    """Count particles in the flattened top-level bins.

    Args:
        positions: Particle positions with shape ``(N, 3)``.
        domain_minimum: Lower corner of the working domain.
        domain_maximum: Upper corner of the working domain.
        resolution: Number of top-level bins per axis.

    Returns:
        Flattened tuple of per-bin particle counts.
    """
    # Bin counting is exposed for diagnostics so users can inspect how
    # particles distribute across the initial octree lattice.
    return _adaptive.top_level_bin_counts(
        positions,
        domain_minimum,
        domain_maximum,
        resolution,
    )


def query_cell_contributors(
    positions: list[tuple[float, float, float]],
    smoothing_lengths: list[float],
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    resolution: int,
    cell_minimum: tuple[float, float, float],
    cell_maximum: tuple[float, float, float],
) -> tuple[int, ...]:
    """Return candidate contributor indices for one query cell.

    Args:
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths.
        domain_minimum: Lower corner of the working domain.
        domain_maximum: Upper corner of the working domain.
        resolution: Number of top-level bins per axis.
        cell_minimum: Lower corner of the query cell.
        cell_maximum: Upper corner of the query cell.

    Returns:
        Tuple of candidate contributor indices.
    """
    # This helper exposes the native contributor query used during
    # refinement so tests can reason about bridge behavior explicitly.
    return _adaptive.query_cell_contributors(
        positions,
        smoothing_lengths,
        domain_minimum,
        domain_maximum,
        resolution,
        cell_minimum,
        cell_maximum,
    )


def cell_may_contain_isosurface(
    corner_values: list[float],
    isovalue: float,
) -> bool:
    """Return whether eight corner values can contain the isosurface.

    Args:
        corner_values: Scalar field values at the eight cell corners.
        isovalue: Scalar field threshold.

    Returns:
        ``True`` if the isosurface can pass through the cell.
    """
    # This predicate is used in diagnostics and tests that inspect refinement
    # decisions without running the full pipeline.
    return _adaptive.cell_may_contain_isosurface(corner_values, isovalue)


def corner_sign_mask(corner_values: list[float], isovalue: float) -> int:
    """Return the bit mask of corner values relative to the isovalue.

    Args:
        corner_values: Scalar field values at the eight cell corners.
        isovalue: Scalar field threshold.

    Returns:
        Bit mask describing which corners lie above the isovalue.
    """
    # The sign mask is a compact summary used throughout octree meshing, so it
    # is exposed directly for low-level debugging and assertions.
    return _adaptive.corner_sign_mask(corner_values, isovalue)


__all__ = [
    "adaptive_status",
    "bounding_box_contains",
    "bounding_box_overlaps",
    "cell_may_contain_isosurface",
    "corner_sign_mask",
    "morton_decode_3d",
    "morton_encode_3d",
    "particle_fields",
    "query_cell_contributors",
    "top_level_bin_counts",
    "wendland_c2_gradient",
    "wendland_c2_value",
]
