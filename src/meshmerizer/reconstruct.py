"""Adaptive mesh reconstruction wrappers.

This module provides thin Python wrappers around the native adaptive meshing
pipeline for callers that still want raw ``(vertices, faces)`` arrays rather
than the newer higher-level public API objects.

It remains useful for:

- CLI orchestration that wants direct NumPy arrays,
- tests that assert on raw mesh buffers, and
- internal compatibility during the staged API redesign.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from meshmerizer.adaptive_core import run_full_pipeline


def reconstruct_group(
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct a mesh for one particle group.

    Args:
        positions: (N, 3) float64 array of particle positions.
        smoothing_lengths: (N,) float64 array of smoothing lengths.
        domain_min: (x, y, z) lower corner of the domain.
        domain_max: (x, y, z) upper corner of the domain.
        base_resolution: Number of top-level cells per axis.
        isovalue: Density isovalue for octree refinement.
        max_depth: Maximum octree refinement depth.
        smoothing_iterations: Number of Laplacian smoothing iterations.
        smoothing_strength: Smoothing lambda in (0, 1].
        max_edge_ratio: Maximum edge length as a multiple of local cell size.
        minimum_usable_hermite_samples: Minimum usable Hermite sample count
            required before a corner-crossing cell may stop refining.
        max_qef_rms_residual_ratio: Maximum RMS QEF plane residual as a
            fraction of the local cell radius before a split is required.
        min_normal_alignment_threshold: Minimum alignment between usable
            Hermite normals and their mean direction before a split is
            required.
        min_feature_thickness: Minimum physical feature thickness to preserve
            via adaptive implicit opening.
        pre_thickening_radius: Optional outward thickening radius applied
            before the opening stage.

    Returns:
        Tuple of ``(vertices, faces)`` where ``vertices`` is a ``(V, 3)``
        float64 array and ``faces`` is an ``(F, 3)`` int64 array.

    Raises:
        ValueError: If positions has fewer than 3 points.
    """
    # Keep the compatibility layer defensive: it accepts direct NumPy
    # inputs and normalizes them before crossing into the native
    # full-pipeline wrapper.
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    smoothing_lengths = np.ascontiguousarray(
        smoothing_lengths, dtype=np.float64
    )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {positions.shape}")
    if positions.shape[0] < 3:
        raise ValueError(
            "Need at least 3 points for reconstruction, "
            f"got {positions.shape[0]}"
        )
    if smoothing_lengths.ndim != 1:
        raise ValueError(
            "smoothing_lengths must be a 1-D array, "
            f"got {smoothing_lengths.shape}"
        )
    if smoothing_lengths.shape[0] != positions.shape[0]:
        raise ValueError(
            "smoothing_lengths must have the same length as positions, "
            f"got {smoothing_lengths.shape[0]} and {positions.shape[0]}"
        )
    if not np.all(np.isfinite(smoothing_lengths)):
        raise ValueError("smoothing_lengths must be finite")
    if np.any(smoothing_lengths < 0.0):
        raise ValueError("smoothing_lengths must be non-negative")

    result = run_full_pipeline(
        positions,
        smoothing_lengths,
        domain_min,
        domain_max,
        base_resolution,
        isovalue,
        max_depth,
        smoothing_iterations=smoothing_iterations,
        smoothing_strength=smoothing_strength,
        max_edge_ratio=max_edge_ratio,
        minimum_usable_hermite_samples=minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio=max_qef_rms_residual_ratio,
        min_normal_alignment_threshold=min_normal_alignment_threshold,
        min_feature_thickness=min_feature_thickness,
        pre_thickening_radius=pre_thickening_radius,
    )

    verts = result["vertices"]
    faces = result["faces"].astype(np.int64)
    return verts, faces


def reconstruct_mesh(
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_min: tuple,
    domain_max: tuple,
    base_resolution: int,
    isovalue: float,
    max_depth: int,
    group_labels: Optional[np.ndarray] = None,
    smoothing_iterations: int = 0,
    smoothing_strength: float = 0.5,
    max_edge_ratio: float = 1.5,
    minimum_usable_hermite_samples: int = 3,
    max_qef_rms_residual_ratio: float = 0.1,
    min_normal_alignment_threshold: float = 0.97,
    min_feature_thickness: float = 0.0,
    pre_thickening_radius: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct meshes for one or more particle groups and merge them.

    Args:
        positions: (N, 3) float64 array of particle positions.
        smoothing_lengths: (N,) float64 array of smoothing lengths.
        domain_min: (x, y, z) lower corner of the domain.
        domain_max: (x, y, z) upper corner of the domain.
        base_resolution: Number of top-level cells per axis.
        isovalue: Density isovalue for octree refinement.
        max_depth: Maximum octree refinement depth.
        group_labels: Optional ``(N,)`` int64 array of group labels.
        smoothing_iterations: Number of Laplacian smoothing iterations.
        smoothing_strength: Smoothing lambda in (0, 1].
        max_edge_ratio: Maximum edge length as a multiple of local cell size.
        minimum_usable_hermite_samples: Minimum usable Hermite sample count
            required before a corner-crossing cell may stop refining.
        max_qef_rms_residual_ratio: Maximum RMS QEF plane residual as a
            fraction of the local cell radius before a split is required.
        min_normal_alignment_threshold: Minimum alignment between usable
            Hermite normals and their mean direction before a split is
            required.
        min_feature_thickness: Minimum physical feature thickness to preserve
            via adaptive implicit opening.
        pre_thickening_radius: Optional outward thickening radius applied
            before the opening stage.

    Returns:
        Tuple of ``(vertices, faces)`` merged across all groups.
    """
    # This compatibility helper reconstructs each group independently and then
    # concatenates the resulting vertex/face buffers. That preserves the
    # historical behavior of avoiding spurious bridges between disconnected FOF
    # groups without exposing the group loop at the public API level.
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    smoothing_lengths = np.ascontiguousarray(
        smoothing_lengths, dtype=np.float64
    )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {positions.shape}")
    if smoothing_lengths.ndim != 1:
        raise ValueError(
            "smoothing_lengths must be a 1-D array, "
            f"got {smoothing_lengths.shape}"
        )
    if smoothing_lengths.shape[0] != positions.shape[0]:
        raise ValueError(
            "smoothing_lengths must have the same length as positions, "
            f"got {smoothing_lengths.shape[0]} and {positions.shape[0]}"
        )

    if group_labels is None:
        group_labels = np.zeros(len(positions), dtype=np.int64)
    else:
        group_labels = np.asarray(group_labels, dtype=np.int64)
        if group_labels.shape != (positions.shape[0],):
            raise ValueError(
                f"group_labels must have shape (N,), got {group_labels.shape}"
            )

    unique_groups = np.unique(group_labels)

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for group_id in unique_groups:
        mask = group_labels == group_id
        group_pos = positions[mask]
        group_sml = smoothing_lengths[mask]

        if group_pos.shape[0] < 3:
            continue

        verts, faces = reconstruct_group(
            group_pos,
            group_sml,
            domain_min,
            domain_max,
            base_resolution,
            isovalue,
            max_depth,
            smoothing_iterations=smoothing_iterations,
            smoothing_strength=smoothing_strength,
            max_edge_ratio=max_edge_ratio,
            minimum_usable_hermite_samples=minimum_usable_hermite_samples,
            max_qef_rms_residual_ratio=max_qef_rms_residual_ratio,
            min_normal_alignment_threshold=min_normal_alignment_threshold,
            min_feature_thickness=min_feature_thickness,
            pre_thickening_radius=pre_thickening_radius,
        )

        if verts.shape[0] == 0:
            continue

        all_faces.append(faces + vertex_offset)
        all_vertices.append(verts)
        vertex_offset += verts.shape[0]

    if len(all_vertices) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
        )

    return (
        np.vstack(all_vertices),
        np.vstack(all_faces),
    )
