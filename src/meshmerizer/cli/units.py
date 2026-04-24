"""Helpers for converting print-space controls into meshing units."""

from __future__ import annotations

import numpy as np


def compute_print_scale_factor_cm(
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    target_size_cm: float,
) -> float:
    """Return the scale factor from native units to print millimetres.

    Args:
        domain_min: Lower corner of the working domain.
        domain_max: Upper corner of the working domain.
        target_size_cm: Desired longest physical dimension in centimetres.

    Returns:
        Multiplicative factor that maps native coordinates to millimetres.

    Raises:
        ValueError: If the working domain has zero extent.
    """
    # Compute extents from the domain corners rather than from mesh vertices so
    # print-unit conversion can happen before any mesh exists.
    extents = np.asarray(domain_max, dtype=np.float64) - np.asarray(
        domain_min, dtype=np.float64
    )
    max_dimension = float(np.max(extents))
    if max_dimension <= 0.0:
        raise ValueError("working domain has zero extent")
    # STL workflows usually interpret coordinates as millimetres, so
    # convert the centimetre target before computing the scale factor.
    target_size_mm = float(target_size_cm) * 10.0
    return target_size_mm / max_dimension


def convert_print_length_to_native_units(
    length_cm: float,
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    target_size_cm: float,
) -> float:
    """Convert a print-space centimetre length to native meshing units.

    Args:
        length_cm: Physical length in centimetres.
        domain_min: Lower corner of the working domain.
        domain_max: Upper corner of the working domain.
        target_size_cm: Desired longest physical dimension in centimetres.

    Returns:
        Equivalent length in native meshing units.
    """
    # Reuse the same domain-derived scale factor as final print scaling so the
    # regularization controls remain consistent with the exported mesh size.
    scale_factor = compute_print_scale_factor_cm(
        domain_min, domain_max, target_size_cm
    )
    return (float(length_cm) * 10.0) / scale_factor


__all__ = [
    "compute_print_scale_factor_cm",
    "convert_print_length_to_native_units",
]
