"""Poisson surface reconstruction via Open3D.

This module wraps Open3D's screened Poisson surface reconstruction
to convert oriented point clouds (positions + normals) into
watertight triangle meshes.  It is designed to be called after the
adaptive octree has produced QEF vertex positions and normals, and
after FOF clustering has assigned group labels to those vertices.

The two main entry points are:

- ``poisson_reconstruct_group``: reconstruct a single cluster.
- ``poisson_reconstruct``: reconstruct all clusters and merge them
  into a single mesh.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import open3d as o3d


def poisson_reconstruct_group(
    positions: np.ndarray,
    normals: np.ndarray,
    poisson_depth: int = 8,
    density_quantile: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct a watertight mesh from oriented points via Poisson.

    Constructs an Open3D point cloud from the given positions and
    normals, runs screened Poisson surface reconstruction, and trims
    low-density vertices to remove spurious geometry far from the
    input points.

    Args:
        positions: (N, 3) float64 array of point positions.
        normals: (N, 3) float64 array of outward-facing unit normals
            at each point.
        poisson_depth: Octree depth for the Poisson solver.  Higher
            values produce finer detail but take longer.  Typical
            range is 6--12.
        density_quantile: Fraction of lowest-density vertices to
            remove after reconstruction.  A value of 0.1 discards
            the bottom 10% of the density distribution, which
            removes the thin membranes that Poisson creates in
            regions far from any input point.  Set to 0.0 to keep
            everything.

    Returns:
        Tuple of ``(vertices, faces)`` where ``vertices`` is an
        (V, 3) float64 array and ``faces`` is an (F, 3) int64 array
        of triangle vertex indices.

    Raises:
        ValueError: If positions and normals have mismatched shapes,
            or if the input has fewer than 3 points (the minimum
            needed for a triangle).
    """
    # ── Input validation ──────────────────────────────────────────
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)

    if positions.shape != normals.shape:
        raise ValueError(
            f"positions shape {positions.shape} != "
            f"normals shape {normals.shape}"
        )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions must be (N, 3), got shape {positions.shape}"
        )
    if positions.shape[0] < 3:
        raise ValueError(
            "Need at least 3 points for Poisson "
            f"reconstruction, got {positions.shape[0]}"
        )

    # ── Build Open3D point cloud ──────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # ── Run screened Poisson reconstruction ───────────────────────
    # The depth parameter controls the octree resolution of the
    # Poisson solver (not our adaptive octree).  A depth of 8 gives
    # 256^3 effective resolution, which is a good balance between
    # detail and speed.
    mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth
        )
    )

    # ── Density-based trimming ────────────────────────────────────
    # The Poisson solver produces a closed surface that extends well
    # beyond the input points.  Vertices with low density values are
    # in regions where the solver extrapolated without evidence.
    # Removing them trims the spurious membranes.
    if density_quantile > 0.0:
        densities_arr = np.asarray(densities)
        threshold = np.quantile(densities_arr, density_quantile)
        # Keep vertices whose density exceeds the threshold.
        vertices_to_remove = densities_arr < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

    # ── Extract arrays ────────────────────────────────────────────
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)

    return vertices, faces


def poisson_reconstruct(
    positions: np.ndarray,
    normals: np.ndarray,
    group_labels: np.ndarray,
    poisson_depth: int = 8,
    density_quantile: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct meshes for all FOF groups and merge them.

    Iterates over the unique group labels, reconstructs each group
    independently via ``poisson_reconstruct_group``, re-indexes the
    face arrays so vertex indices are globally unique, and
    concatenates everything into a single mesh.

    Reconstructing groups independently prevents thin bridges between
    distinct objects (e.g. separate galaxies) that a single Poisson
    solve would create.

    Args:
        positions: (N, 3) float64 array of all QEF vertex positions.
        normals: (N, 3) float64 array of all QEF vertex normals.
        group_labels: (N,) int64 array of FOF group labels (one per
            vertex, 0-based contiguous integers).
        poisson_depth: Octree depth for the Poisson solver.
        density_quantile: Fraction of lowest-density vertices to
            trim per group.

    Returns:
        Tuple of ``(vertices, faces)`` where ``vertices`` is an
        (V_total, 3) float64 array and ``faces`` is an
        (F_total, 3) int64 array with globally consistent vertex
        indices.
    """
    group_labels = np.asarray(group_labels, dtype=np.int64)
    unique_groups = np.unique(group_labels)

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for group_id in unique_groups:
        mask = group_labels == group_id
        group_pos = positions[mask]
        group_nrm = normals[mask]

        # Skip groups that are too small for Poisson.
        if group_pos.shape[0] < 3:
            continue

        verts, faces = poisson_reconstruct_group(
            group_pos,
            group_nrm,
            poisson_depth=poisson_depth,
            density_quantile=density_quantile,
        )

        if verts.shape[0] == 0:
            continue

        # Re-index faces to account for previously added vertices.
        all_faces.append(faces + vertex_offset)
        all_vertices.append(verts)
        vertex_offset += verts.shape[0]

    # Handle case where all groups were too small or empty.
    if len(all_vertices) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
        )

    return (
        np.vstack(all_vertices),
        np.vstack(all_faces),
    )
