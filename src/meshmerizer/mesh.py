"""A module containing the Mesh class and its methods."""

import time
from typing import List, Optional

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure


class Mesh:
    """A class representing a 3D mesh using trimesh.

    Attributes:
        vertices (np.ndarray): The vertices of the mesh.
        faces (np.ndarray): The faces of the mesh.
        vertex_normals (np.ndarray): The normals of the vertices.
    """

    def __init__(self, vertices, faces, vertex_normals):
        """Initialize the Mesh object.

        Args:
            vertices (np.ndarray): The vertices of the mesh.
            faces (np.ndarray): The faces of the mesh.
            vertex_normals (np.ndarray): The normals of the vertices.
        """
        self.vertices = vertices
        self.faces = faces
        self.vertex_normals = vertex_normals

        self.trimesh = None

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
        return Mesh(
            np.concatenate((self.vertices, other.vertices)),
            np.concatenate((self.faces, other.faces + len(self.vertices))),
            np.concatenate((self.vertex_normals, other.vertex_normals)),
        )

    def to_trimesh(self):
        """Convert the Mesh object to a trimesh object.

        Returns:
            trimesh.Trimesh: The trimesh object.
        """
        if self.trimesh is None:
            self.trimesh = trimesh.Trimesh(
                vertices=self.vertices,
                faces=self.faces,
                vertex_normals=self.vertex_normals,
                process=False,  # we’ll call repair ourselves
            )

        # 1) Fill tiny holes
        trimesh.repair.fill_holes(self.trimesh)

        # 2) Remove stray faces/vertices
        self.trimesh.remove_duplicate_faces()
        self.trimesh.remove_degenerate_faces()
        self.trimesh.remove_unreferenced_vertices()

        # 3) Run MeshFix for stubborn leaks (needs pymeshfix installed)
        try:
            from pymeshfix import MeshFix

            mf = MeshFix(self.trimesh.vertices, self.trimesh.faces)
            mf.repair()
            self.trimesh = trimesh.Trimesh(mf.v, mf.f)
        except ImportError:
            pass

        # 4) Recompute normals & check watertightness
        self.trimesh.fix_normals()
        if not self.trimesh.is_watertight:
            print("⚠️ Mesh still not watertight after repair.")

        # 5) Optional smoothing
        import trimesh.smoothing as smoothing

        smoothing.filter_taubin(self.trimesh, lamb=0.5, nu=-0.5, iterations=5)

        return self.trimesh

    def save(self, filename):
        """Save the mesh to a file.

        Parameters:
            filename (str): The name of the file to save the mesh to.
        """
        self.to_trimesh().export(filename)
        print(f"Mesh saved to {filename}")

    def show(self):
        """Display the mesh using trimesh's built-in viewer."""
        self.to_trimesh().show()
        print("Mesh displayed.")


def voxels_to_stl(
    volume: np.ndarray,
    threshold: float,
    closing_radius: int = 1,
    split_islands: bool = False,
    mesh_index: Optional[int] = None,
):
    """Convert a 3D voxel array to one or more STL meshes.

    Args:
        volume (np.ndarray): 3D voxel array representing the volume.
        threshold (float): Threshold value for binarization.
        closing_radius (int): Radius for binary closing to fill small gaps.
        split_islands (bool): If True, split the volume into separate islands.
        mesh_index (Optional[int]): Index of the mesh to extract. If None,
            all meshes are extracted.

    Returns:
        meshes : list of Mesh
            List of Mesh objects representing the solid regions.
    """
    start_time = time.perf_counter()

    # 1) Binarise
    bin_start = time.perf_counter()
    bin_vol = volume > threshold
    bin_end = time.perf_counter()
    print(f"Binarization took {bin_end - bin_start:.4f} seconds.")

    # 2) Define structuring elements
    # For closing: larger spherical element
    closing_struct = None
    if closing_radius > 0:
        close_start = time.perf_counter()
        base_struct = ndimage.generate_binary_structure(3, 1)
        closing_struct = ndimage.iterate_structure(base_struct, closing_radius)
        bin_vol = ndimage.binary_closing(bin_vol, structure=closing_struct)
        close_end = time.perf_counter()
        print(f"Binary closing took {close_end - close_start:.4f} seconds.")

    # For labeling: use connectivity-1 (3x3x3)
    label_start = time.perf_counter()
    label_struct = ndimage.generate_binary_structure(3, 1)
    label_struct = ndimage.iterate_structure(label_struct, 1)
    label_end = time.perf_counter()
    print(
        f"Labeling structure created in {label_end - label_start:.4f} seconds."
    )

    # 3) Label islands
    if split_islands:
        split_start = time.perf_counter()
        labeled, num = ndimage.label(bin_vol, structure=label_struct)
        island_ids = range(1, num + 1)
        split_end = time.perf_counter()
        print(
            f"Labeling took {split_end - split_start:.4f} seconds. "
            f"Found {num} islands."
        )
    else:
        labeled = bin_vol.astype(np.int32)
        island_ids = [1]

    # If we are targeting a specific mesh, filter the island IDs
    # to only include the one we want
    if mesh_index is not None:
        island_ids = [mesh_index]
        if mesh_index not in island_ids:
            raise ValueError(
                f"Mesh index {mesh_index} not found in the volume."
            )

    mesh_start = time.perf_counter()
    meshes: List[trimesh.Trimesh] = []
    for idx in island_ids:
        mask = labeled == idx
        if not mask.any():
            continue

        # 3) Remove small objects
        if np.sum(mask) < 100:
            continue

        # 4) Marching cubes
        verts, faces, normals, _ = measure.marching_cubes(
            mask, level=0.5, spacing=(1.0, 1.0, 1.0)
        )

        # Create a Mesh object
        mesh = Mesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
        )

        meshes.append(mesh)

    mesh_end = time.perf_counter()
    print(
        f"Marching cubes took {mesh_end - mesh_start:.4f} seconds. "
        f"Created {len(meshes)} meshes."
    )

    # Raise an error if no meshes were created
    if len(meshes) == 0:
        raise ValueError(
            "No meshes were created. Check the threshold and input grid."
        )

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
    mesh_index: Optional[int] = None,
):
    """Analogous to voxels_to_stl, but generates a watertight mesh via
    a signed distance field (SDF) instead of a binary mask.

    Steps:
    1. Binarise at `threshold` and close gaps
    2. (Optional) split islands as in voxels_to_stl
    3. Compute signed distance field
    4. Marching cubes on level=0.0 of the SDF
    5. Wrap results in Mesh objects
    """
    # 1) Binarise and close
    bin_vol = volume > threshold
    if closing_radius > 0:
        struct = ndimage.iterate_structure(
            ndimage.generate_binary_structure(3, 1), closing_radius
        )
        bin_vol = ndimage.binary_closing(bin_vol, structure=struct)

    # 2) Label islands if needed
    label_struct = ndimage.generate_binary_structure(3, 1)
    if split_islands:
        labeled, num = ndimage.label(bin_vol, structure=label_struct)
        island_ids = range(1, num + 1)
    else:
        labeled = bin_vol.astype(np.int32)
        island_ids = [1]

    # Optionally filter by mesh_index
    if mesh_index is not None:
        if mesh_index < 0 or mesh_index >= len(list(island_ids)):
            raise ValueError(f"Mesh index {mesh_index} not in islands.")
        island_ids = [mesh_index + 1]  # label IDs start at 1

    meshes: List[Mesh] = []
    for idx in island_ids:
        mask = labeled == idx
        if not mask.any():
            continue

        # 3) Compute SDF
        d_in = ndimage.distance_transform_edt(mask)
        d_out = ndimage.distance_transform_edt(~mask)
        sdf = d_in.astype(float) - d_out.astype(float)

        # 4) Marching cubes on the zero level set
        verts, faces, normals, _ = measure.marching_cubes(
            volume=sdf,
            level=0.0,
            spacing=(1.0, 1.0, 1.0),
            gradient_direction="ascent",
        )

        # 5) Wrap in Mesh
        mesh = Mesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
        )
        meshes.append(mesh)

    if not meshes:
        raise ValueError(
            "No meshes created via SDF. Check threshold and input."
        )

    return meshes
