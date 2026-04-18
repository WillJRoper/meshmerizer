"""Poisson surface reconstruction via the C++ backend.

This module provides a thin Python wrapper around the C++ full
pipeline (``run_full_pipeline``) which performs screened Poisson
surface reconstruction entirely in C++.  The previous Open3D-based
implementation has been removed because Open3D crashes on large
datasets (``Assertion failed: (idx < size())``).

The two entry points are:

- ``poisson_reconstruct_group``: reconstruct a single cluster.
- ``poisson_reconstruct``: reconstruct all clusters and merge.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from meshmerizer.adaptive_core import run_full_pipeline


def poisson_reconstruct_group(
    positions: np.ndarray,
    normals: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_min: tuple,
    domain_max: tuple,
    base_resolution: int,
    isovalue: float,
    max_depth: int,
    screening_weight: float = 4.0,
    max_iters: int = 1000,
    tol: float = 1e-6,
    smoothing_iterations: int = 0,
    smoothing_strength: float = 0.5,
    max_edge_ratio: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct a mesh from particles via Poisson in C++.

    Runs the full particles-to-mesh pipeline (octree + QEF +
    Poisson + Marching Cubes) entirely in C++.

    Args:
        positions: (N, 3) float64 array of particle positions.
        normals: Unused (kept for API compatibility).  Normals
            are computed internally from the density field.
        smoothing_lengths: (N,) float64 array of smoothing lengths.
        domain_min: (x, y, z) lower corner of the domain.
        domain_max: (x, y, z) upper corner of the domain.
        base_resolution: Number of top-level cells per axis.
        isovalue: Density isovalue for octree refinement.
        max_depth: Maximum octree refinement depth.
        screening_weight: Poisson screening weight alpha.
        max_iters: Maximum PCG iterations.
        tol: PCG relative residual tolerance.
        smoothing_iterations: Number of Laplacian smoothing
            iterations (0 = disabled).
        smoothing_strength: Smoothing lambda in (0, 1].
        max_edge_ratio: Maximum edge length as a multiple of
            local cell size for gap filling.  Default 1.5.

    Returns:
        Tuple of ``(vertices, faces)`` where ``vertices`` is an
        (V, 3) float64 array and ``faces`` is an (F, 3) int64
        array of triangle vertex indices.

    Raises:
        ValueError: If positions has fewer than 3 points.
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {positions.shape}")
    if positions.shape[0] < 3:
        raise ValueError(
            "Need at least 3 points for reconstruction, "
            f"got {positions.shape[0]}"
        )

    result = run_full_pipeline(
        positions,
        smoothing_lengths,
        domain_min,
        domain_max,
        base_resolution,
        isovalue,
        max_depth,
        screening_weight=screening_weight,
        max_iters=max_iters,
        tol=tol,
        smoothing_iterations=smoothing_iterations,
        smoothing_strength=smoothing_strength,
        max_edge_ratio=max_edge_ratio,
    )

    verts = result["vertices"]
    faces = result["faces"].astype(np.int64)

    return verts, faces


def poisson_reconstruct(
    positions: np.ndarray,
    normals: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_min: tuple,
    domain_max: tuple,
    base_resolution: int,
    isovalue: float,
    max_depth: int,
    group_labels: Optional[np.ndarray] = None,
    screening_weight: float = 4.0,
    max_iters: int = 1000,
    tol: float = 1e-6,
    smoothing_iterations: int = 0,
    smoothing_strength: float = 0.5,
    max_edge_ratio: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct meshes for all FOF groups and merge them.

    If ``group_labels`` is None, treats all particles as one group.

    Args:
        positions: (N, 3) float64 array of particle positions.
        normals: Unused (kept for API compatibility).
        smoothing_lengths: (N,) float64 array of smoothing lengths.
        domain_min: (x, y, z) lower corner of the domain.
        domain_max: (x, y, z) upper corner of the domain.
        base_resolution: Number of top-level cells per axis.
        isovalue: Density isovalue for octree refinement.
        max_depth: Maximum octree refinement depth.
        group_labels: (N,) int64 array of FOF group labels.
        screening_weight: Poisson screening weight alpha.
        max_iters: Maximum PCG iterations.
        tol: PCG relative residual tolerance.
        smoothing_iterations: Number of Laplacian smoothing
            iterations (0 = disabled).
        smoothing_strength: Smoothing lambda in (0, 1].
        max_edge_ratio: Maximum edge length as a multiple of
            local cell size for gap filling.  Default 1.5.

    Returns:
        Tuple of ``(vertices, faces)`` merged across all groups.
    """
    if group_labels is None:
        group_labels = np.zeros(len(positions), dtype=np.int64)
    else:
        group_labels = np.asarray(group_labels, dtype=np.int64)

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

        verts, faces = poisson_reconstruct_group(
            group_pos,
            None,
            group_sml,
            domain_min,
            domain_max,
            base_resolution,
            isovalue,
            max_depth,
            screening_weight=screening_weight,
            max_iters=max_iters,
            tol=tol,
            smoothing_iterations=smoothing_iterations,
            smoothing_strength=smoothing_strength,
            max_edge_ratio=max_edge_ratio,
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
