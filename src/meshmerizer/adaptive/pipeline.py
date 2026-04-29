"""High-level adaptive pipeline wrappers and helper calculations.

This module contains the smallest possible Python wrappers around the native
adaptive pipeline entry points. It intentionally groups together
operations that conceptually run on whole particle sets rather than on
pre-built octree state.
"""

from __future__ import annotations

import numpy as np

from ._native import _adaptive


def compute_isovalue_from_percentile(
    smoothing_lengths: np.ndarray,
    percentile: float,
) -> float:
    """Compute an isovalue from a density percentile of the particles.

    The adaptive pipeline often chooses an isovalue from the percentile of the
    particles' self-density proxy rather than requiring callers to guess a raw
    threshold.

    Args:
        smoothing_lengths: Per-particle smoothing lengths.
        percentile: Percentile of the self-density proxy in ``[0, 100]``.

    Returns:
        Isovalue suitable for passing to the adaptive pipeline.

    Raises:
        ValueError: If ``percentile`` is outside ``[0, 100]`` or the input is
            empty.
    """
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
    """Cluster points using a friends-of-friends algorithm.

    Args:
        positions: Particle positions with shape ``(N, 3)``.
        domain_min: Lower corner of the nominal domain.
        domain_max: Upper corner of the nominal domain.
        linking_factor: Multiplicative factor applied to the characteristic
            linking length.

    Returns:
        ``(N,)`` array of integer cluster labels.
    """
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
    worker_count: int = 1,
    smoothing_iterations: int = 0,
    smoothing_strength: float = 0.5,
    max_edge_ratio: float = 1.5,
    minimum_usable_hermite_samples: int = 3,
    max_qef_rms_residual_ratio: float = 0.1,
    min_normal_alignment_threshold: float = 0.97,
    min_feature_thickness: float = 0.0,
    pre_thickening_radius: float = 0.0,
    table_cadence: float = 10.0,
) -> dict:
    """Run the full particles-to-mesh pipeline in C++.

    Args:
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths with shape ``(N,)``.
        domain_min: Lower corner of the working domain.
        domain_max: Upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.
        isovalue: Scalar field threshold for reconstruction.
        max_depth: Maximum octree refinement depth.
        table_cadence: Strict time cadence in seconds for queue-status table
            rows emitted by queue-driven refinement. Defaults to ``10.0``.
        smoothing_iterations: Number of smoothing iterations.
        smoothing_strength: Laplacian smoothing strength in ``(0, 1]``.
        max_edge_ratio: Maximum permitted edge length relative to local cell
            size.
        minimum_usable_hermite_samples: Minimum usable Hermite sample count.
        max_qef_rms_residual_ratio: Maximum QEF RMS residual ratio.
        min_normal_alignment_threshold: Minimum usable-normal alignment.
        min_feature_thickness: Minimum preserved feature thickness.
        pre_thickening_radius: Optional outward pre-thickening radius.

    Returns:
        Native result dictionary containing mesh arrays and lightweight
        metadata.
    """
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
        worker_count,
        table_cadence,
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
