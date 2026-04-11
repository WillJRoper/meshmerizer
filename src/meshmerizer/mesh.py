"""Mesh wrappers and voxel-to-surface conversion helpers.

This module provides the lightweight :class:`Mesh` wrapper used throughout the
package, along with helpers that convert dense voxel grids into triangle meshes
via standard marching cubes or signed-distance-field extraction. It also
contains repair and subdivision helpers used by the CLI after mesh generation.
"""

import time
from typing import Iterable, List, Optional, Tuple

import numpy as np
import trimesh
import trimesh.remesh as remesh
import trimesh.smoothing as smoothing
from scipy import ndimage
from skimage import measure


class Mesh:
    """A lightweight wrapper around :class:`trimesh.Trimesh`.

    The wrapper keeps mesh-specific utilities in one place and provides a small
    package-local API used by the CLI and chunking code.

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

        The mesh can be created either from raw vertex/face arrays or by
        wrapping an existing ``trimesh.Trimesh`` instance.

        Args:
            vertices: Optional ``(N, 3)`` vertex array.
            faces: Optional ``(M, 3)`` triangle index array.
            vertex_normals: Optional ``(N, 3)`` per-vertex normals.
            mesh: Optional existing ``trimesh.Trimesh`` to wrap.

        Raises:
            ValueError: If neither a ``mesh`` nor raw vertex/face arrays are
                provided.
        """
        # Prefer wrapping an existing trimesh when one is already available so
        # we do not rebuild geometry unnecessarily.
        if mesh is not None:
            self.mesh = mesh
        elif vertices is not None and faces is not None:
            # Create the underlying trimesh without automatic processing so the
            # package stays in control of cleanup and repair order.
            self.mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_normals=vertex_normals,
                process=False,  # We handle processing/repair manually
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
            String representation with vertex and face array shapes.
        """
        return (
            f"Mesh(vertices={self.vertices.shape}, faces={self.faces.shape})"
        )

    def __str__(self) -> str:
        """Return a human-readable description of the mesh.

        Returns:
            Short description of the vertex and face counts.
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
        # Restrict addition to package-local Mesh objects so the return type
        # and downstream API remain predictable.
        if not isinstance(other, Mesh):
            raise TypeError("Can only add another Mesh object.")

        combined = trimesh.util.concatenate([self.mesh, other.mesh])
        return Mesh(mesh=combined)

    def translate(self, offset: np.ndarray) -> None:
        """Translate the mesh in place.

        Args:
            offset: Translation vector with shape ``(3,)``.
        """
        # Update vertices in place so all other mesh properties remain attached
        # to the same underlying trimesh object.
        self.mesh.vertices[:] = self.mesh.vertices + np.asarray(offset)

    def repair(self, smoothing_iters: int = 0) -> None:
        """Repair the mesh for 3D printing.

        Operations:
        1. Remove degenerate faces and unreferenced vertices.
        2. Fix normals.
        3. Optional smoothing.

        Args:
            smoothing_iters: Number of Taubin smoothing iterations to apply
                after the initial cleanup pass.
        """
        # Run trimesh cleanup before any optional smoothing so the surface
        # starts from a sane topology.
        self.mesh.process()

        if smoothing_iters > 0:
            # Smooth only when requested; the default path keeps the extracted
            # geometry unchanged apart from topology cleanup.
            smoothing.filter_taubin(
                self.mesh,
                lamb=0.5,
                nu=0.5,
                iterations=smoothing_iters,
            )

        # Re-run cleanup after smoothing because smoothing can introduce small
        # inconsistencies in normals and cached topology state.
        self.mesh.process()
        self.mesh.fix_normals()
        if not self.mesh.is_watertight:
            print("⚠️ Mesh still not watertight after repair.")

    def subdivide(self, iterations: int = 1) -> None:
        """Subdivide the mesh surface using Loop subdivision.

        Args:
            iterations: Number of subdivision iterations.

        Raises:
            ValueError: If ``iterations`` is negative.
        """
        # Treat zero iterations as a no-op so the caller can pass through user
        # input without special handling.
        if iterations < 0:
            raise ValueError("iterations must be >= 0")
        if iterations == 0:
            return

        # Run Loop subdivision on the raw vertex/face arrays, then rebuild the
        # trimesh so the wrapper continues to own a consistent object.
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

    def to_trimesh(self) -> trimesh.Trimesh:
        """Return the underlying trimesh object.

        Returns:
            Underlying ``trimesh.Trimesh`` instance.

        Note:
            This method does not perform auto-repair. Call :meth:`repair`
            explicitly when cleanup is required.
        """
        return self.mesh

    def save(self, filename: str) -> None:
        """Save the mesh to a file.

        Args:
            filename: Output mesh filename.
        """
        # Delegate serialization to trimesh and keep the wrapper responsible
        # for user-facing status output.
        self.mesh.export(filename)
        print(f"Mesh saved to {filename}")

    def show(self) -> None:
        """Display the mesh using trimesh's built-in viewer."""
        self.mesh.show()
        print("Mesh displayed.")


def _prepare_volume(
    volume: np.ndarray,
    threshold: float,
    closing_radius: int,
    split_islands: bool,
    remove_islands: bool,
    mesh_index: Optional[int],
    padding: int = 1,
) -> Tuple[np.ndarray, Iterable[int]]:
    """Internal helper to binarize, close, and label a voxel volume.

    Args:
        volume (np.ndarray): 3D voxel array.
        threshold (float): Binarization threshold.
        closing_radius (int): Radius for binary closing.
        split_islands (bool): Whether to split connected components.
        remove_islands (bool): Whether to keep only the largest connected
            component.
        mesh_index (Optional[int]): Specific island to extract.
        padding (int): Number of voxels to pad the volume with zeros.

    Returns:
        Tuple containing:

        - Integer volume where ``0`` is background and positive values are
          component labels.
        - Iterable of label IDs to extract.
    """
    # Pad the field before thresholding so surfaces on the volume boundary can
    # close cleanly during extraction.
    if padding > 0:
        volume = np.pad(
            volume,
            pad_width=padding,
            mode="constant",
            constant_values=0,
        )

    # Binarise the scalar field at the requested isovalue.
    bin_start = time.perf_counter()
    bin_vol = volume > threshold
    bin_end = time.perf_counter()
    print(f"Binarization took {bin_end - bin_start:.4f} seconds.")

    # Close small gaps in the binary mask before connected-component analysis.
    if closing_radius > 0:
        close_start = time.perf_counter()
        base_struct = ndimage.generate_binary_structure(3, 1)
        closing_struct = ndimage.iterate_structure(base_struct, closing_radius)
        bin_vol = ndimage.binary_closing(bin_vol, structure=closing_struct)
        close_end = time.perf_counter()
        print(f"Binary closing took {close_end - close_start:.4f} seconds.")

    # Label islands only when the caller wants separate components or wants to
    # keep just the largest one.
    label_struct = ndimage.generate_binary_structure(3, 1)
    if split_islands or remove_islands:
        split_start = time.perf_counter()
        labeled, num = ndimage.label(bin_vol, structure=label_struct)
        island_ids = range(1, num + 1)
        split_end = time.perf_counter()
        print(
            f"Labeling took {split_end - split_start:.4f} seconds. "
            f"Found {num} islands."
        )

        if remove_islands and num > 0:
            # Keep only the dominant component when requested so small floating
            # islands do not survive into the final mesh.
            component_sizes = np.bincount(labeled.ravel())
            component_sizes[0] = 0
            largest_label = int(np.argmax(component_sizes))
            largest_size = int(component_sizes[largest_label])
            removed_count = num - 1
            labeled = np.where(labeled == largest_label, largest_label, 0)
            island_ids = [largest_label]
            print(
                "Removing disconnected islands. "
                f"Keeping largest component with {largest_size} voxels "
                f"and discarding {removed_count} island(s)."
            )
    else:
        # In the simple path we treat the whole mask as one component labelled
        # 1 if any voxels survive thresholding.
        labeled = bin_vol.astype(np.int32)
        if not np.any(labeled):
            island_ids = []
        else:
            island_ids = [1]

    # Restrict extraction to a specific connected-component label when the
    # caller requests a single island.
    if mesh_index is not None:
        all_ids = list(island_ids)
        if mesh_index not in all_ids:
            raise ValueError(
                f"Mesh index {mesh_index} not found in the volume."
            )
        island_ids = [mesh_index]

    return labeled, island_ids


def voxels_to_stl(
    volume: np.ndarray,
    threshold: float,
    closing_radius: int = 1,
    split_islands: bool = False,
    remove_islands: bool = False,
    mesh_index: Optional[int] = None,
    voxel_size: float = 1.0,
) -> List[Mesh]:
    """Convert a voxel volume to meshes with standard marching cubes.

    Args:
        volume (np.ndarray): 3D voxel array representing the volume.
        threshold (float): Threshold value for binarization.
        closing_radius (int): Radius for binary closing to fill small gaps.
        split_islands (bool): If True, split the volume into separate islands.
        remove_islands (bool): If True, keep only the largest connected
            component.
        mesh_index (Optional[int]): Index of the mesh to extract. If None,
            all meshes are extracted.
        voxel_size (float): The physical size of a single voxel. Used for
            scaling the output mesh.

    Returns:
        List of extracted meshes.
    """
    # Keep an end-to-end timer so the caller can see the full extraction cost,
    # not just the marching-cubes kernel time.
    start_time = time.perf_counter()

    labeled, island_ids = _prepare_volume(
        volume,
        threshold,
        closing_radius,
        split_islands,
        remove_islands,
        mesh_index,
    )

    mesh_start = time.perf_counter()
    meshes: List[Mesh] = []

    # Extract each requested connected component independently so callers can
    # keep islands split if they want to post-process them separately.
    for idx in island_ids:
        mask = labeled == idx
        if not mask.any():
            continue

        # Skip tiny components that are almost always numerical debris.
        if np.sum(mask) < 10:
            continue

        # Run marching cubes directly on the binary mask. ``level=0.5`` places
        # the surface midway between empty and filled voxels.
        verts, faces, normals, _ = measure.marching_cubes(
            mask, level=0.5, spacing=(voxel_size, voxel_size, voxel_size)
        )

        meshes.append(
            Mesh(vertices=verts, faces=faces, vertex_normals=normals)
        )

    mesh_end = time.perf_counter()
    print(
        f"Marching cubes took {mesh_end - mesh_start:.4f} seconds. "
        f"Created {len(meshes)} meshes."
    )

    if not meshes:
        # Distinguish between an empty field and a field whose candidate meshes
        # were filtered out so callers get a more useful error.
        if volume.max() <= threshold:
            msg = "Volume max value below threshold."
        else:
            msg = "Meshes removed by size filtering or invalid index."
        raise ValueError(f"No meshes created. {msg}")

    end_time = time.perf_counter()
    print(
        f"Converted volume to {len(meshes)} meshes in "
        f"{end_time - start_time:.4f} seconds."
    )
    return meshes


def voxels_to_stl_via_sdf(
    volume: np.ndarray,
    threshold: float,
    closing_radius: int = 1,
    split_islands: bool = False,
    remove_islands: bool = False,
    mesh_index: Optional[int] = None,
    voxel_size: float = 1.0,
) -> List[Mesh]:
    """Convert voxels to meshes using a signed distance field.

    This method often produces more watertight meshes than standard
    binary mask marching cubes.

    Args:
        volume (np.ndarray): 3D voxel array representing the volume.
        threshold (float): Threshold value for binarization.
        closing_radius (int): Radius for binary closing to fill small gaps.
        split_islands (bool): If True, split the volume into separate islands.
        remove_islands (bool): If True, keep only the largest connected
            component.
        mesh_index (Optional[int]): Index of the mesh to extract. If None,
            all meshes are extracted.
        voxel_size (float): The physical size of a single voxel. Used for
            scaling the output mesh.

    Returns:
        List of extracted meshes.
    """
    # As with the standard path, keep a total timer for the full SDF workflow.
    start_time = time.perf_counter()

    labeled, island_ids = _prepare_volume(
        volume,
        threshold,
        closing_radius,
        split_islands,
        remove_islands,
        mesh_index,
    )

    meshes: List[Mesh] = []
    # Convert each connected component independently so the caller can still
    # get separate meshes when island splitting is enabled.
    for idx in island_ids:
        mask = labeled == idx
        if not mask.any():
            continue

        # Build a signed distance field from the binary component mask.
        d_in = ndimage.distance_transform_edt(mask)
        d_out = ndimage.distance_transform_edt(~mask)
        sdf = d_in.astype(float) - d_out.astype(float)

        # Extract the zero level-set, which usually gives a smoother and more
        # watertight surface than marching cubes on the raw mask.
        verts, faces, normals, _ = measure.marching_cubes(
            volume=sdf,
            level=0.0,
            spacing=(voxel_size, voxel_size, voxel_size),
            gradient_direction="ascent",
        )

        meshes.append(
            Mesh(vertices=verts, faces=faces, vertex_normals=normals)
        )

    if not meshes:
        raise ValueError(
            "No meshes created via SDF. Check threshold and input."
        )

    end_time = time.perf_counter()
    print(f"SDF Conversion finished in {end_time - start_time:.4f} seconds.")

    return meshes
