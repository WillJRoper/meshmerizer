"""High-level adaptive pipeline wrappers and helper calculations."""

from __future__ import annotations

import numpy as np

from ._native import _adaptive


def compute_isovalue_from_percentile(
    smoothing_lengths: np.ndarray,
    percentile: float,
) -> float:
    """Compute an isovalue from a density percentile of the particles."""
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    h = np.asarray(smoothing_lengths, dtype=np.float64)
    if h.size == 0:
        raise ValueError("smoothing_lengths array is empty")

    self_density = 21.0 / (2.0 * np.pi * h**3)
    return float(np.percentile(self_density, percentile))


def fof_cluster(
    positions: np.ndarray,
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    linking_factor: float = 1.5,
) -> np.ndarray:
    """Cluster points using a friends-of-friends algorithm."""
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    return _adaptive.fof_cluster(pos, domain_min, domain_max, linking_factor)


def run_full_pipeline(
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_min: tuple,
    domain_max: tuple,
    base_resolution: int,
    isovalue: float,
    max_depth: int,
    smoothing_iterations: int = 0,
    smoothing_strength: float = 0.5,
    max_edge_ratio: float = 1.5,
    minimum_usable_hermite_samples: int = 3,
    max_qef_rms_residual_ratio: float = 0.1,
    min_normal_alignment_threshold: float = 0.97,
    min_feature_thickness: float = 0.0,
    pre_thickening_radius: float = 0.0,
) -> dict:
    """Run the full particles-to-mesh pipeline in C++."""
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    sml = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
    return _adaptive.run_full_pipeline(
        pos,
        sml,
        tuple(domain_min),
        tuple(domain_max),
        base_resolution,
        isovalue,
        max_depth,
        smoothing_iterations,
        smoothing_strength,
        max_edge_ratio,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
        min_feature_thickness,
        pre_thickening_radius,
    )


__all__ = [
    "compute_isovalue_from_percentile",
    "fof_cluster",
    "run_full_pipeline",
]
