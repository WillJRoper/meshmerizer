"""Topology and regularization wrappers around the native adaptive core.

This module exposes the native operations that classify the adaptive occupied
solid and extract a mesh from an editable opened-solid mask. These wrappers are
used by the staged Python API when callers want to inspect or modify
regularization state before final mesh extraction.
"""

from __future__ import annotations

import numpy as np

from ._native import _adaptive


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
    worker_count: int = 1,
    table_cadence: float = 0.0,
) -> dict:
    """Classify the adaptive occupied solid on octree leaves.

    Args:
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths with shape ``(N,)``.
        domain_minimum: Lower corner of the working domain.
        domain_maximum: Upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.
        isovalue: Scalar field threshold.
        max_depth: Maximum octree refinement depth.
        minimum_usable_hermite_samples: Minimum usable Hermite sample count.
        max_qef_rms_residual_ratio: Maximum acceptable RMS QEF residual ratio.
        min_normal_alignment_threshold: Minimum acceptable normal alignment.
        max_surface_leaf_size: Optional upper bound on surface-leaf size during
            the topology pass.
        erosion_radius: Erosion radius used by the opening operator.
        pre_thickening_radius: Optional outward thickening radius applied
            before erosion.
        worker_count: Number of native closure workers to use during topology
            refinement.
        table_cadence: Queue-status table cadence in seconds for queue-driven
            refinement used by this topology path. Defaults to ``0.0``.

    Returns:
        Native result dictionary containing occupancy masks, diagnostics, and
        opened-surface sample buffers.
    """
    # Normalize the particle arrays before entering C++ so the native topology
    # pass receives the same stable layout as the meshing pipeline.
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    sml = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
    # Return the native dictionary unchanged because the staged public API
    # wraps its fields into ``TopologyState`` at a higher layer.
    return _adaptive.classify_occupied_solid(
        pos,
        sml,
        tuple(domain_minimum),
        tuple(domain_maximum),
        int(base_resolution),
        isovalue,
        int(max_depth),
        worker_count,
        int(minimum_usable_hermite_samples),
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
        max_surface_leaf_size,
        erosion_radius,
        pre_thickening_radius,
        table_cadence,
    )


def classify_occupied_solid_from_tree(
    cells: list,
    contributors: np.ndarray,
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    base_resolution: int,
    isovalue: float,
    max_depth: int,
    erosion_radius: float = 0.0,
    pre_thickening_radius: float = 0.0,
    worker_count: int = 1,
) -> dict:
    """Classify the adaptive occupied solid from an already-built tree.

    Skips particle re-insertion by reusing the existing tree cells and
    contributor lists produced by ``build_refined_tree``.

    Args:
        cells: Octree cell list from ``build_refined_tree``.
        contributors: Contributor index array from ``build_refined_tree``.
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths with shape ``(N,)``.
        domain_minimum: Lower corner of the working domain.
        domain_maximum: Upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.
        isovalue: Scalar field threshold.
        max_depth: Maximum octree refinement depth.
        erosion_radius: Erosion radius used by the opening operator.
        pre_thickening_radius: Optional outward thickening radius applied
            before erosion.
        worker_count: Number of native closure workers to use during topology
            refinement.

    Returns:
        Native result dictionary with the same structure as
        ``classify_occupied_solid``.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    sml = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
    return _adaptive.classify_occupied_solid_from_tree(
        cells,
        contributors,
        pos,
        sml,
        isovalue,
        tuple(domain_minimum),
        tuple(domain_maximum),
        int(max_depth),
        int(base_resolution),
        float(erosion_radius),
        float(pre_thickening_radius),
        int(worker_count),
    )


def classify_occupied_solid_from_handle(
    native_handle: object,
    erosion_radius: float = 0.0,
    pre_thickening_radius: float = 0.0,
    worker_count: int = 1,
) -> dict:
    """Classify the occupied solid using an opaque native tree handle."""
    return _adaptive.classify_occupied_solid_from_handle(
        native_handle,
        float(erosion_radius),
        float(pre_thickening_radius),
        int(worker_count),
    )


__all__ = [
    "classify_occupied_solid",
    "classify_occupied_solid_from_handle",
    "classify_occupied_solid_from_tree",
]
