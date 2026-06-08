"""Mesh post-processing operations shared by the CLI and API."""

from __future__ import annotations

from typing import Optional

import numpy as np
from trimesh import graph as trimesh_graph

from meshmerizer.logging import log_status

from .core import Mesh


def remove_islands(
    mesh: Mesh,
    remove_islands_fraction: Optional[float],
) -> Mesh:
    """Remove small disconnected components from a mesh.

    Args:
        mesh: Input mesh.
        remove_islands_fraction: Fraction of the largest component volume below
            which a connected component is discarded. ``0.0`` keeps only the
            largest component. ``None`` disables island removal entirely.

    Returns:
        A new ``Mesh`` with small islands removed, or the original mesh
        unchanged when removal is disabled.
    """
    # Skip all connected-component work when the feature is disabled so callers
    # do not pay for an unnecessary mesh split.
    if remove_islands_fraction is None:
        return mesh

    def _signed_triangle_soup_volume(
        vertices: np.ndarray, faces: np.ndarray
    ) -> float:
        """Estimate enclosed volume directly from triangle geometry.

        Args:
            vertices: Vertex positions for one component.
            faces: Triangle indices for one component.

        Returns:
            Absolute signed volume estimate, or ``0.0`` when unavailable.
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            return 0.0
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            return 0.0

        triangles = vertices[faces]
        signed_six_volume = np.einsum(
            "ij,ij->i",
            triangles[:, 0, :],
            np.cross(triangles[:, 1, :], triangles[:, 2, :]),
        )
        volume = abs(float(np.sum(signed_six_volume) / 6.0))
        return volume if np.isfinite(volume) else 0.0

    def _component_reference_volume(face_indices: np.ndarray) -> float:
        """Return a robust size estimate for island filtering.

        Args:
            face_indices: Face indices for one connected component.

        Returns:
            Non-negative reference volume used for island ranking.
        """
        component_faces = faces[face_indices]
        if len(component_faces) == 0:
            return 0.0

        component_triangles = vertices[component_faces]
        unique_vertices = component_triangles.reshape(-1, 3)
        if len(unique_vertices) < 4:
            return 0.0

        bounds_min = unique_vertices.min(axis=0)
        bounds_max = unique_vertices.max(axis=0)
        extents = np.asarray(bounds_max - bounds_min, dtype=np.float64)
        if extents.shape != (3,) or not np.all(np.isfinite(extents)):
            return 0.0
        if np.count_nonzero(extents > 0.0) < 3:
            return 0.0

        volume = _signed_triangle_soup_volume(vertices, component_faces)
        if volume > 0.0:
            return volume

        return float(np.prod(extents))

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return mesh

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    component_faces = trimesh_graph.connected_components(
        mesh.mesh.face_adjacency, nodes=np.arange(len(faces), dtype=np.int64)
    )
    if len(component_faces) <= 1:
        return mesh

    component_faces = [
        np.asarray(group, dtype=np.int64) for group in component_faces
    ]

    # Treat zero as the special "keep only the largest component" mode.
    if remove_islands_fraction == 0.0:
        largest = max(component_faces, key=_component_reference_volume)
        log_status(
            "Cleaning",
            f"Kept largest of {len(component_faces)} components.",
        )
        keep_faces = np.zeros(len(faces), dtype=bool)
        keep_faces[largest] = True
        mesh.mesh.update_faces(keep_faces)
        mesh.mesh.remove_unreferenced_vertices()
        return mesh

    # Compute the per-component reference sizes once so they can drive both the
    # keep mask and the status message.
    volumes = [_component_reference_volume(group) for group in component_faces]
    largest_volume = max(volumes, default=0.0)
    if largest_volume == 0.0:
        return mesh

    # Keep every component whose reference volume is large enough relative to
    # the largest observed component.
    kept = [
        face_group
        for face_group, volume in zip(component_faces, volumes)
        if volume / largest_volume >= remove_islands_fraction
    ]
    if not kept:
        # Always keep at least one component so filtering never returns
        # an empty mesh purely because the threshold was too aggressive.
        kept = [max(component_faces, key=_component_reference_volume)]

    log_status(
        "Cleaning",
        f"Kept {len(kept)} of {len(component_faces)} components "
        f"(fraction >= {remove_islands_fraction} of largest volume).",
    )

    keep_faces = np.zeros(len(faces), dtype=bool)
    for face_group in kept:
        keep_faces[face_group] = True

    # Update the existing mesh in place so island filtering does not
    # temporarily duplicate a very large extracted surface.
    mesh.mesh.update_faces(keep_faces)
    mesh.mesh.remove_unreferenced_vertices()
    return mesh


def simplify_mesh(mesh: Mesh, simplify_factor: float) -> Mesh:
    """Optionally simplify the mesh after extraction and cleanup.

    Args:
        mesh: Mesh to simplify in place.
        simplify_factor: Fraction of faces to retain in ``(0, 1]``.

    Returns:
        Simplified mesh instance.
    """
    # Treat a factor of 1.0 as an explicit no-op so callers can pass
    # through CLI input directly without branching.
    if simplify_factor == 1.0:
        return mesh

    # Log before and after face counts so simplification strength is visible in
    # CLI output and tests.
    log_status(
        "Cleaning",
        f"Simplifying mesh to retain factor {simplify_factor:.6g}...",
    )
    before_faces = len(mesh.faces)
    mesh.simplify(factor=simplify_factor)
    after_faces = len(mesh.faces)
    log_status(
        "Cleaning",
        f"Simplified mesh faces: {before_faces} -> {after_faces}",
    )
    return mesh


__all__ = ["remove_islands", "simplify_mesh"]
