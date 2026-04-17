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
    poisson_depth: int = 9,
    density_quantile: float = 0.02,
    scale: float = 1.2,
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
            remove after reconstruction.  A value of 0.02 discards
            the bottom 2% of the density distribution, which
            removes only the thinnest extrapolated membranes
            without cutting into boundary geometry.  Set to 0.0 to
            keep everything.
        scale: Ratio of the Poisson solver bounding box to the
            input point cloud bounding box.  Values above 1.0 add
            padding so the reconstructed surface is not clipped at
            the edges.  Default 1.2 adds 20% padding on each side.

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

    # ── Sanitize input points ────────────────────────────────────
    # QEF solving can produce degenerate vertices: zero-length
    # normals from ill-conditioned cells, near-duplicate positions
    # from adjacent cells at different octree depths, and normals
    # that are not exactly unit length.  All of these confuse the
    # Poisson solver and can cause it to hang on isosurface
    # extraction ("Failed to close loop") or produce garbage.

    # 1. Remove points whose normals have near-zero magnitude.
    #    These carry no orientation information and inject "bad
    #    data" into the Poisson indicator function.
    nrm_lengths = np.linalg.norm(normals, axis=1)
    valid_mask = nrm_lengths > 1e-12
    positions = positions[valid_mask]
    normals = normals[valid_mask]
    nrm_lengths = nrm_lengths[valid_mask]

    if positions.shape[0] < 3:
        raise ValueError(
            "Fewer than 3 points remain after removing zero-normal vertices"
        )

    # 2. Re-normalize normals to exact unit length.  Small
    #    deviations from unit length accumulate into the Poisson
    #    gradient field and degrade surface quality.
    normals = normals / nrm_lengths[:, np.newaxis]

    # 3. Remove near-duplicate points.  Open3D's
    #    remove_duplicated_points uses an exact-match grid, so
    #    we use a voxel-based downsampling with a tiny voxel
    #    size to merge points that are effectively coincident.
    #    The voxel size is set to 1e-8 times the bounding box
    #    diagonal, which is far below any meaningful feature
    #    but catches true duplicates from floating-point
    #    coincidence.
    bbox_diag = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
    dedup_voxel = max(bbox_diag * 1e-8, 1e-15)

    # ── Build Open3D point cloud ──────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Deduplicate via voxel downsampling.
    pcd = pcd.voxel_down_sample(voxel_size=dedup_voxel)

    # ── Run screened Poisson reconstruction ───────────────────────
    # The depth parameter controls the octree resolution of the
    # Poisson solver (not our adaptive octree).  A depth of 9 gives
    # 512^3 effective resolution, which is a good balance between
    # detail and speed for typical SPH particle counts.
    #
    # Suppress the extremely verbose "getValue assumes leaf node"
    # warning that PoissonRecon prints for every non-leaf evaluation.
    prev_verbosity = o3d.utility.get_verbosity_level()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    try:
        mesh, densities = (
            o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth, scale=scale
            )
        )
    finally:
        o3d.utility.set_verbosity_level(prev_verbosity)

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
    poisson_depth: int = 9,
    density_quantile: float = 0.02,
    scale: float = 1.2,
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
        scale: Ratio of the Poisson solver bounding box to the
            input point cloud bounding box.

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
            scale=scale,
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
