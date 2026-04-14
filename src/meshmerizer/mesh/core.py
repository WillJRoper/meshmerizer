"""Core mesh wrapper and repair helpers.

This module defines the lightweight :class:`Mesh` wrapper used throughout the
package. It also contains the in-place repair, subdivision, simplification, and
local broken-face stitching helpers that prepare extracted surfaces for export.
"""

from typing import Optional

import numpy as np
import trimesh
import trimesh.remesh as remesh
import trimesh.smoothing as smoothing
from trimesh import repair as trimesh_repair

from meshmerizer.logging_utils import log_status


class Mesh:
    """A lightweight wrapper around :class:`trimesh.Trimesh`.

    Attributes:
        mesh: Underlying ``trimesh.Trimesh`` instance.
    """

    def __init__(
        self,
        vertices: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
        vertex_normals: Optional[np.ndarray] = None,
        mesh: Optional[trimesh.Trimesh] = None,
    ) -> None:
        """Initialize the Mesh object.

        Args:
            vertices: Optional ``(N, 3)`` vertex array.
            faces: Optional ``(M, 3)`` triangle index array.
            vertex_normals: Optional ``(N, 3)`` per-vertex normals.
            mesh: Optional existing ``trimesh.Trimesh`` to wrap.

        Returns:
            ``None``. The wrapper is initialized in place.

        Raises:
            ValueError: If neither a ``mesh`` nor raw vertex/face arrays are
                provided.
        """
        # Prefer wrapping an existing trimesh so geometry is not rebuilt when a
        # caller already has one.
        if mesh is not None:
            self.mesh = mesh
        elif vertices is not None and faces is not None:
            # Disable trimesh auto-processing here because Mesh controls the
            # repair order explicitly.
            self.mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_normals=vertex_normals,
                process=False,
            )
        else:
            raise ValueError(
                "Must provide either 'mesh' or 'vertices' and 'faces'."
            )

    @property
    def vertices(self) -> np.ndarray:
        """Return vertex positions.

        Returns:
            Vertex positions as an ``(N, 3)`` array.
        """
        return self.mesh.vertices

    @property
    def faces(self) -> np.ndarray:
        """Return triangle indices.

        Returns:
            Triangle indices as an ``(M, 3)`` array.
        """
        return self.mesh.faces

    @property
    def vertex_normals(self) -> np.ndarray:
        """Return per-vertex normals.

        Returns:
            Per-vertex normals as an ``(N, 3)`` array.
        """
        return self.mesh.vertex_normals

    def __repr__(self) -> str:
        """Return a developer-facing representation of the mesh.

        Returns:
            Representation containing vertex and face array shapes.
        """
        return (
            f"Mesh(vertices={self.vertices.shape}, faces={self.faces.shape})"
        )

    def __str__(self) -> str:
        """Return a human-readable description of the mesh.

        Returns:
            Short description of the mesh vertex and face counts.
        """
        return (
            f"Mesh with {self.vertices.shape[0]} "
            f"vertices and {self.faces.shape[0]} faces"
        )

    def __len__(self) -> int:
        """Return the number of vertices in the mesh.

        Returns:
            Vertex count.
        """
        return self.vertices.shape[0]

    def __add__(self, other: "Mesh") -> "Mesh":
        """Add two Mesh objects together.

        Args:
            other: Mesh to concatenate with this mesh.

        Returns:
            New mesh containing both triangle soups.

        Raises:
            TypeError: If ``other`` is not a :class:`Mesh` instance.
        """
        # Restrict addition to Mesh objects so the result type stays
        # predictable throughout the package.
        if not isinstance(other, Mesh):
            raise TypeError("Can only add another Mesh object.")

        combined = trimesh.util.concatenate([self.mesh, other.mesh])
        return Mesh(mesh=combined)

    def translate(self, offset: np.ndarray) -> None:
        """Translate the mesh in place.

        Args:
            offset: Translation vector with shape ``(3,)``.

        Returns:
            ``None``. The mesh is updated in place.
        """
        # Translate the wrapped mesh in place so existing references to the
        # underlying trimesh remain valid.
        self.mesh.vertices[:] = self.mesh.vertices + np.asarray(offset)

    def repair(self, smoothing_iters: int = 0) -> None:
        """Repair the mesh for 3D printing.

        Args:
            smoothing_iters: Number of Taubin smoothing iterations to apply
                after the initial cleanup pass.

        Returns:
            ``None``. The mesh is repaired in place.
        """
        # Run trimesh cleanup first so subsequent local repair and smoothing
        # see a sane starting topology.
        self.mesh.process()
        _repair_local_broken_faces(self.mesh)

        # Apply smoothing only when the caller asks for it because smoothing
        # can slightly alter the extracted surface.
        if smoothing_iters > 0:
            smoothing.filter_taubin(
                self.mesh,
                lamb=0.5,
                nu=0.5,
                iterations=smoothing_iters,
            )

        # Re-run cleanup after smoothing because smoothing can invalidate
        # cached topology state and introduce small local defects.
        self.mesh.process()
        _repair_local_broken_faces(self.mesh)
        self.mesh.fix_normals()
        if not self.mesh.is_watertight:
            log_status("Cleaning", "⚠️ Mesh still not watertight after repair.")

    def subdivide(self, iterations: int = 1) -> None:
        """Subdivide the mesh surface using Loop subdivision.

        Args:
            iterations: Number of subdivision iterations.

        Returns:
            ``None``. The mesh is replaced in place.

        Raises:
            ValueError: If ``iterations`` is negative.
        """
        # Treat zero iterations as a no-op so callers can pass user input
        # through directly.
        if iterations < 0:
            raise ValueError("iterations must be >= 0")
        if iterations == 0:
            return

        # Rebuild the wrapped trimesh from the subdivided arrays so Mesh keeps
        # a coherent single underlying object.
        vertices, faces = remesh.subdivide_loop(
            self.mesh.vertices,
            self.mesh.faces,
            iterations=iterations,
        )
        self.mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )
        self.mesh.process()
        self.mesh.fix_normals()

    def simplify(self, factor: float = 1.0) -> None:
        """Simplify the mesh by targeting a fraction of the current faces.

        Args:
            factor: Fraction of faces to keep. ``1.0`` disables
                simplification.

        Returns:
            ``None``. The mesh is simplified in place.

        Raises:
            ValueError: If ``factor`` is outside ``(0, 1]``.
            RuntimeError: If simplification fails or produces no faces.
        """
        # Treat a factor of one as an explicit no-op so callers do not need to
        # special-case the CLI default.
        if not (0.0 < factor <= 1.0):
            raise ValueError("simplify factor must satisfy 0 < factor <= 1")
        if factor == 1.0:
            return

        face_count = len(self.mesh.faces)
        if face_count < 4:
            return

        # Convert the retention factor into an absolute face budget while
        # avoiding requests that would create degenerate meshes.
        target_faces = max(4, int(round(face_count * factor)))
        if target_faces >= face_count:
            return

        try:
            simplified = self.mesh.simplify_quadric_decimation(
                face_count=target_faces
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "mesh simplification failed. Install the "
                "'fast-simplification' package and reinstall meshmerizer. "
                f"Original error: {exc}"
            ) from exc

        # Clean the simplified result immediately so later operations see the
        # same invariants as meshes created elsewhere in the package.
        if simplified is None or len(simplified.faces) == 0:
            raise RuntimeError("mesh simplification produced no faces")
        self.mesh = simplified
        self.mesh.process()
        _repair_local_broken_faces(self.mesh)
        self.mesh.fix_normals()

    def to_trimesh(self) -> trimesh.Trimesh:
        """Return the underlying trimesh object.

        Returns:
            Underlying ``trimesh.Trimesh`` instance.
        """
        return self.mesh

    def save(self, filename: str) -> None:
        """Save the mesh to a file.

        Args:
            filename: Output mesh filename.

        Returns:
            ``None``. The mesh is written to disk.
        """
        # Delegate serialization to trimesh while keeping the status message in
        # one place.
        self.mesh.export(filename)
        log_status("Saving", f"Mesh saved to {filename}")


def _repair_local_broken_faces(mesh: trimesh.Trimesh) -> None:
    """Repair tiny local broken-face regions in place.

    Args:
        mesh: Mesh to repair in place.

    Returns:
        ``None``. The input mesh is modified in place when repair succeeds.
    """
    # Ask trimesh for locally broken faces and ignore the common no-op case
    # immediately.
    broken = trimesh_repair.broken_faces(mesh)
    if len(broken) == 0:
        return

    broken = np.asarray(broken, dtype=np.int64)
    if broken.size > 128:
        return

    # Remove only the broken patch and ask trimesh to stitch that local hole
    # rather than attempting a global remesh.
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[broken] = False

    repaired = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        process=False,
    )
    repaired.update_faces(keep)
    repaired.remove_unreferenced_vertices()

    try:
        stitch_faces = trimesh_repair.stitch(repaired, insert_vertices=False)
    except ValueError:
        return
    if len(stitch_faces) == 0:
        return

    repaired = trimesh.Trimesh(
        vertices=repaired.vertices.copy(),
        faces=np.vstack([repaired.faces, stitch_faces]),
        process=False,
    )
    repaired.merge_vertices()
    repaired.update_faces(repaired.unique_faces())
    repaired.update_faces(repaired.nondegenerate_faces())
    repaired.remove_unreferenced_vertices()

    # Replace the original mesh arrays in place so existing references to the
    # wrapped trimesh remain valid.
    mesh.vertices = repaired.vertices.copy()
    mesh.faces = repaired.faces.copy()
