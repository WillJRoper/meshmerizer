"""A module containing the Mesh class and its methods."""

import time
from typing import Iterable, List, Optional, Tuple

import numpy as np
import trimesh
import trimesh.remesh as remesh
import trimesh.smoothing as smoothing
from scipy import ndimage
from skimage import measure


class Mesh:
    """A wrapper around trimesh.Trimesh for 3D printing workflows.

    Attributes:
        mesh (trimesh.Trimesh): The underlying trimesh object.
    """

    def __init__(
        self,
        vertices: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
        vertex_normals: Optional[np.ndarray] = None,
        mesh: Optional[trimesh.Trimesh] = None,
    ):
        """Initialize the Mesh object.

        Can be initialized either with raw data (vertices, faces) or
        an existing trimesh object.

        Args:
            vertices (np.ndarray, optional): The vertices of the mesh.
            faces (np.ndarray, optional): The faces of the mesh.
            vertex_normals (np.ndarray, optional): The normals of the vertices.
            mesh (trimesh.Trimesh, optional): An existing trimesh object.
        """
        if mesh is not None:
            self.mesh = mesh
        elif vertices is not None and faces is not None:
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
        """Vertex positions as an (N, 3) array."""
        return self.mesh.vertices

    @property
    def faces(self) -> np.ndarray:
        """Triangle indices as an (M, 3) array."""
        return self.mesh.faces

    @property
    def vertex_normals(self) -> np.ndarray:
        """Per-vertex normals as an (N, 3) array."""
        return self.mesh.vertex_normals

    def __repr__(self):
        """Return a string representation of the Mesh object."""
        return (
            f"Mesh(vertices={self.vertices.shape}, faces={self.faces.shape})"
        )

    def __str__(self):
        """Return a string describing the Mesh object."""
        return (
            f"Mesh with {self.vertices.shape[0]} "
            f"vertices and {self.faces.shape[0]} faces"
        )

    def __len__(self):
        """Return the number of vertices in the mesh."""
        return self.vertices.shape[0]

    def __add__(self, other):
        """Add two Mesh objects together.

        Args:
            other (Mesh): The other Mesh object to add.

        Returns:
            Mesh: A new Mesh object containing the combined vertices and faces.
        """
        if not isinstance(other, Mesh):
            raise TypeError("Can only add another Mesh object.")

        combined = trimesh.util.concatenate([self.mesh, other.mesh])
        return Mesh(mesh=combined)

    def translate(self, offset: np.ndarray) -> None:
        """Translate the mesh in place by the given offset."""
        self.mesh.vertices[:] = self.mesh.vertices + np.asarray(offset)

    def repair(self, smoothing_iters: int = 0):
        """Repair the mesh for 3D printing.

        Operations:
        1. Remove degenerate faces and unreferenced vertices.
        2. Fix normals.
        3. Optional smoothing.

        Args:
            smoothing_iters (int): Number of Taubin smoothing iterations.
        """
        self.mesh.process()

        if smoothing_iters > 0:
            smoothing.filter_taubin(
                self.mesh,
                lamb=0.5,
                nu=0.5,
                iterations=smoothing_iters,
            )

        # Re-run cleanup after smoothing so topology and normals stay sane.
        self.mesh.process()
        self.mesh.fix_normals()
        if not self.mesh.is_watertight:
            print("⚠️ Mesh still not watertight after repair.")

    def subdivide(self, iterations: int = 1):
        """Subdivide the mesh surface using Loop subdivision.

        Args:
            iterations (int): Number of subdivision iterations.
        """
        if iterations < 0:
            raise ValueError("iterations must be >= 0")
        if iterations == 0:
            return

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

        Note: This no longer performs auto-repair. Use `.repair()` explicitly.
        """
        return self.mesh

    def save(self, filename: str):
        """Save the mesh to a file.

        Parameters:
            filename (str): The name of the file to save the mesh to.
        """
        self.mesh.export(filename)
        print(f"Mesh saved to {filename}")

    def show(self):
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
        labeled_volume (np.ndarray): Integer array where 0 is background and
            values > 0 are object labels.
        island_ids (iterable): A list or range of IDs to process.
    """
    # Pad the volume to ensure watertightness
    if padding > 0:
        volume = np.pad(
            volume,
            pad_width=padding,
            mode="constant",
            constant_values=0,
        )

    # 1) Binarise
    bin_start = time.perf_counter()
    bin_vol = volume > threshold
    bin_end = time.perf_counter()
    print(f"Binarization took {bin_end - bin_start:.4f} seconds.")

    # 2) Binary Closing
    if closing_radius > 0:
        close_start = time.perf_counter()
        base_struct = ndimage.generate_binary_structure(3, 1)
        closing_struct = ndimage.iterate_structure(base_struct, closing_radius)
        bin_vol = ndimage.binary_closing(bin_vol, structure=closing_struct)
        close_end = time.perf_counter()
        print(f"Binary closing took {close_end - close_start:.4f} seconds.")

    # 3) Label islands
    label_struct = ndimage.generate_binary_structure(3, 1)
    # If we are splitting islands, we label connected components.
    # If not, we treat the whole thing as one big island (label=1).
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
        labeled = bin_vol.astype(np.int32)
        # If there are no voxels, island_ids should be empty
        if not np.any(labeled):
            island_ids = []
        else:
            island_ids = [1]

    # Filter for specific mesh index if requested
    if mesh_index is not None:
        # Check availability
        # Note: island_ids might be a range, so we convert to list to check
        all_ids = list(island_ids)
        if mesh_index not in all_ids:
            # For backwards compatibility with the original logic which allowed
            # mesh_index to be 1-based or 0-based, we assume the user passes
            # the specific integer label they want.
            # The original code did `island_ids = [mesh_index]`.
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
    """Convert a 3D voxel array to one or more STL meshes using standard MC.

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
        meshes : list of Mesh
            List of Mesh objects representing the solid regions.
    """
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

    for idx in island_ids:
        mask = labeled == idx
        if not mask.any():
            continue

        # Remove small objects (heuristic from original code)
        if np.sum(mask) < 10:
            continue

        # Marching cubes on the mask directly
        # level=0.5 works because mask is 0 or 1
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
        # Check if the volume was actually empty or if filtering removed
        # everything.
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
    """Convert voxels to mesh using Signed Distance Fields (SDF).

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
        meshes : list of Mesh
            List of Mesh objects representing the solid regions.
    """
    start_time = time.perf_counter()

    # Note: original code expected mesh_index to be 0-based for the list of
    # islands in this function, but the island labels are 1-based.
    # We will adhere to the new standard: mesh_index refers to the LABEL ID.

    labeled, island_ids = _prepare_volume(
        volume,
        threshold,
        closing_radius,
        split_islands,
        remove_islands,
        mesh_index,
    )

    meshes: List[Mesh] = []
    for idx in island_ids:
        mask = labeled == idx
        if not mask.any():
            continue

        # Compute SDF
        d_in = ndimage.distance_transform_edt(mask)
        d_out = ndimage.distance_transform_edt(~mask)
        sdf = d_in.astype(float) - d_out.astype(float)

        # Marching cubes on the zero level set of the SDF
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
