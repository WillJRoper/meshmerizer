"""Topology and regularization wrappers around the native adaptive core."""

from __future__ import annotations

import numpy as np

from ._native import _adaptive


def extract_opened_surface_mesh(
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    base_resolution: int,
    isovalue: float,
    max_depth: int,
    opened_inside: np.ndarray,
    minimum_usable_hermite_samples: int = 3,
    max_qef_rms_residual_ratio: float = 0.1,
    min_normal_alignment_threshold: float = 0.97,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a blocky opened-surface mesh from an editable opened mask."""
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    sml = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
    opened = np.ascontiguousarray(opened_inside, dtype=np.uint8)
    vertices, faces = _adaptive.extract_opened_surface_mesh(
        pos,
        sml,
        domain_minimum,
        domain_maximum,
        base_resolution,
        isovalue,
        max_depth,
        opened,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
    )
    return np.asarray(vertices, dtype=np.float64), np.asarray(
        faces, dtype=np.uint32
    )


def classify_occupied_solid(
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    base_resolution: int,
    isovalue: float,
    max_depth: int,
    minimum_usable_hermite_samples: int = 3,
    max_qef_rms_residual_ratio: float = 0.1,
    min_normal_alignment_threshold: float = 0.97,
    max_surface_leaf_size: float = 0.0,
    erosion_radius: float = 0.0,
    pre_thickening_radius: float = 0.0,
) -> dict:
    """Classify the adaptive occupied solid on octree leaves."""
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    sml = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
    return _adaptive.classify_occupied_solid(
        pos,
        sml,
        tuple(domain_minimum),
        tuple(domain_maximum),
        base_resolution,
        isovalue,
        max_depth,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
        max_surface_leaf_size,
        erosion_radius,
        pre_thickening_radius,
    )


__all__ = ["classify_occupied_solid", "extract_opened_surface_mesh"]
