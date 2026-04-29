"""Topology and regularization wrappers around the native adaptive core.

This module exposes the native operations that classify the adaptive occupied
solid and extract a mesh from an editable opened-solid mask. These wrappers are
used by the staged Python API when callers want to inspect or modify
regularization state before final mesh extraction.
"""

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
    table_cadence: float = 0.0,
    minimum_usable_hermite_samples: int = 3,
    max_qef_rms_residual_ratio: float = 0.1,
    min_normal_alignment_threshold: float = 0.97,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a blocky opened-surface mesh from an editable opened mask.

    Args:
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths with shape ``(N,)``.
        domain_minimum: Lower corner of the working domain.
        domain_maximum: Upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.
        isovalue: Scalar field threshold.
        max_depth: Maximum octree refinement depth.
        opened_inside: Editable opened-solid occupancy mask on octree leaves.
        table_cadence: Queue-status table cadence in seconds for any closure
            refinement used during the native preparation path. Defaults to
            ``0.0`` for this diagnostic/extraction route.
        minimum_usable_hermite_samples: Minimum usable Hermite sample count.
        max_qef_rms_residual_ratio: Maximum acceptable RMS QEF residual ratio.
        min_normal_alignment_threshold: Minimum acceptable normal alignment.

    Returns:
        Tuple of ``(vertices, faces)`` arrays for the opened surface.
    """
    # Normalize to contiguous arrays so the editable opened mask and particle
    # inputs can be passed to C++ without extra copies inside the extension.
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    sml = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
    opened = np.ascontiguousarray(opened_inside, dtype=np.uint8)
    # The native layer returns plain buffers; convert them to explicit NumPy
    # dtypes here so downstream code sees consistent array types.
    vertices, faces = _adaptive.extract_opened_surface_mesh(
        pos,
        sml,
        domain_minimum,
        domain_maximum,
        base_resolution,
        isovalue,
        max_depth,
        opened,
        table_cadence,
        1,
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
        # worker_count: topology path is always serial.
        1,
        int(minimum_usable_hermite_samples),
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
        max_surface_leaf_size,
        erosion_radius,
        pre_thickening_radius,
        table_cadence,
    )


__all__ = ["classify_occupied_solid", "extract_opened_surface_mesh"]
