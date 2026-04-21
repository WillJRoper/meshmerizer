"""High-level Python API for composing meshmerizer pipelines.

This module provides the public, Python-friendly API for building adaptive
meshing workflows in stages. The goal is to let users either run the whole
pipeline with a single call or stop at well-defined intermediate products and
modify them in Python before continuing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from meshmerizer.adaptive_core import (
    build_refined_tree,
    classify_occupied_solid,
    compute_isovalue_from_percentile,
    fof_cluster,
    generate_mesh,
    run_full_pipeline,
)
from meshmerizer.commands.adaptive_stl import _remove_islands
from meshmerizer.mesh.core import Mesh

Vec3 = Tuple[float, float, float]


@dataclass
class TreeState:
    """Adaptive tree plus the particle/domain metadata needed downstream."""

    cells: tuple[dict[str, object], ...]
    contributors: tuple[int, ...]
    positions: np.ndarray
    smoothing_lengths: np.ndarray
    domain_min: Vec3
    domain_max: Vec3
    base_resolution: int
    max_depth: int
    isovalue: float
    minimum_usable_hermite_samples: int = 3
    max_qef_rms_residual_ratio: float = 0.1
    min_normal_alignment_threshold: float = 0.97


@dataclass
class TopologyState:
    """Regularized opened-solid topology derived from a tree state."""

    tree: TreeState
    occupancy: np.ndarray
    depths: np.ndarray
    centers: np.ndarray
    sizes: np.ndarray
    clearance: np.ndarray
    eroded_inside: np.ndarray
    dilation_distance: np.ndarray
    opened_inside: np.ndarray
    sample_positions: np.ndarray
    sample_normals: np.ndarray
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray
    min_feature_thickness: float


@dataclass
class MeshResult:
    """Mesh plus lightweight metadata from the reconstruction pipeline."""

    mesh: Mesh
    isovalue: float
    n_qef_vertices: int


def _as_positions(positions: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.ascontiguousarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    return arr


def _as_smoothing_lengths(smoothing_lengths: Sequence[float]) -> np.ndarray:
    arr = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("smoothing_lengths must have shape (N,)")
    return arr


def _validate_particle_arrays(
    positions: Sequence[Sequence[float]],
    smoothing_lengths: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    pos = _as_positions(positions)
    sml = _as_smoothing_lengths(smoothing_lengths)
    if pos.shape[0] != sml.shape[0]:
        raise ValueError(
            "positions and smoothing_lengths must contain the same number "
            "of particles"
        )
    return pos, sml


def _mesh_result_from_pipeline_dict(result: dict) -> MeshResult:
    return MeshResult(
        mesh=Mesh(vertices=result["vertices"], faces=result["faces"]),
        isovalue=float(result["isovalue"]),
        n_qef_vertices=int(result["n_qef_vertices"]),
    )


def build_and_refine_tree(
    positions: Sequence[Sequence[float]],
    smoothing_lengths: Sequence[float],
    domain_min: Vec3,
    domain_max: Vec3,
    base_resolution: int,
    isovalue: float,
    max_depth: int,
    minimum_usable_hermite_samples: int = 3,
    max_qef_rms_residual_ratio: float = 0.1,
    min_normal_alignment_threshold: float = 0.97,
) -> TreeState:
    """Build and refine the adaptive tree without extracting a final mesh."""
    pos, sml = _validate_particle_arrays(positions, smoothing_lengths)
    cells, contributors = build_refined_tree(
        pos,
        sml,
        tuple(domain_min),
        tuple(domain_max),
        base_resolution,
        isovalue,
        max_depth,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
    )
    return TreeState(
        cells=cells,
        contributors=tuple(int(v) for v in contributors.tolist()),
        positions=pos,
        smoothing_lengths=sml,
        domain_min=tuple(domain_min),
        domain_max=tuple(domain_max),
        base_resolution=base_resolution,
        max_depth=max_depth,
        isovalue=float(isovalue),
        minimum_usable_hermite_samples=minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio=max_qef_rms_residual_ratio,
        min_normal_alignment_threshold=min_normal_alignment_threshold,
    )


def erode_and_dilate(
    tree: TreeState,
    min_feature_thickness: float,
) -> TopologyState:
    """Build the opened-solid topology used by the regularized pipeline."""
    erosion_radius = 0.5 * float(min_feature_thickness)
    result = classify_occupied_solid(
        tree.positions,
        tree.smoothing_lengths,
        tree.domain_min,
        tree.domain_max,
        tree.base_resolution,
        tree.isovalue,
        tree.max_depth,
        tree.minimum_usable_hermite_samples,
        tree.max_qef_rms_residual_ratio,
        tree.min_normal_alignment_threshold,
        max_surface_leaf_size=erosion_radius,
        erosion_radius=erosion_radius,
    )
    return TopologyState(
        tree=tree,
        occupancy=result["occupancy"],
        depths=result["depths"],
        centers=result["center_values"],
        sizes=result["cell_sizes"],
        clearance=result["clearance"],
        eroded_inside=result["eroded_inside"],
        dilation_distance=result["dilation_distance"],
        opened_inside=result["opened_inside"],
        sample_positions=result["opened_boundary_positions"],
        sample_normals=result["opened_boundary_normals"],
        mesh_vertices=result["opened_surface_vertices"],
        mesh_faces=result["opened_surface_faces"],
        min_feature_thickness=float(min_feature_thickness),
    )


def get_mesh_from_tree(
    tree: TreeState,
    *,
    smoothing_iterations: int = 0,
    smoothing_strength: float = 0.5,
    max_edge_ratio: float = 1.5,
    min_feature_thickness: float = 0.0,
    remove_islands_fraction: Optional[float] = None,
) -> MeshResult:
    """Extract a mesh from an already-built tree state."""
    if tree.cells and tree.contributors and min_feature_thickness <= 0.0:
        vertices, _, faces = generate_mesh(
            tree.cells,
            tree.contributors,
            tree.positions,
            tree.smoothing_lengths,
            tree.isovalue,
            tree.domain_min,
            tree.domain_max,
            tree.max_depth,
            tree.base_resolution,
        )
        mesh_result = MeshResult(
            mesh=Mesh(vertices=vertices, faces=faces.astype(np.uint32)),
            isovalue=tree.isovalue,
            n_qef_vertices=int(vertices.shape[0]),
        )
    else:
        result = run_full_pipeline(
            tree.positions,
            tree.smoothing_lengths,
            tree.domain_min,
            tree.domain_max,
            tree.base_resolution,
            tree.isovalue,
            tree.max_depth,
            smoothing_iterations=smoothing_iterations,
            smoothing_strength=smoothing_strength,
            max_edge_ratio=max_edge_ratio,
            minimum_usable_hermite_samples=tree.minimum_usable_hermite_samples,
            max_qef_rms_residual_ratio=tree.max_qef_rms_residual_ratio,
            min_normal_alignment_threshold=tree.min_normal_alignment_threshold,
            min_feature_thickness=min_feature_thickness,
        )
        mesh_result = _mesh_result_from_pipeline_dict(result)

    if remove_islands_fraction is not None:
        mesh_result.mesh = remove_islands(
            mesh_result.mesh,
            remove_islands_fraction=remove_islands_fraction,
        )
    return mesh_result


def get_mesh(
    positions: Sequence[Sequence[float]],
    smoothing_lengths: Sequence[float],
    domain_min: Vec3,
    domain_max: Vec3,
    base_resolution: int,
    max_depth: int,
    isovalue: float,
    smoothing_iterations: int = 0,
    smoothing_strength: float = 0.5,
    max_edge_ratio: float = 1.5,
    minimum_usable_hermite_samples: int = 3,
    max_qef_rms_residual_ratio: float = 0.1,
    min_normal_alignment_threshold: float = 0.97,
    min_feature_thickness: float = 0.0,
    remove_islands_fraction: Optional[float] = None,
) -> MeshResult:
    """Run the full public particles-to-mesh workflow."""
    pos, sml = _validate_particle_arrays(positions, smoothing_lengths)
    result = run_full_pipeline(
        pos,
        sml,
        tuple(domain_min),
        tuple(domain_max),
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
    )
    mesh_result = _mesh_result_from_pipeline_dict(result)
    if remove_islands_fraction is not None:
        mesh_result.mesh = remove_islands(
            mesh_result.mesh,
            remove_islands_fraction=remove_islands_fraction,
        )
    return mesh_result


def smooth_mesh(
    mesh: Mesh,
    iterations: int = 10,
    *,
    inplace: bool = False,
) -> Mesh:
    """Apply the existing mesh-repair smoothing workflow to a mesh."""
    target = mesh if inplace else Mesh(mesh=mesh.mesh.copy())
    target.repair(smoothing_iters=iterations)
    return target


def remove_islands(
    mesh: Mesh,
    remove_islands_fraction: Optional[float],
) -> Mesh:
    """Remove connected components below a fraction of the largest volume."""
    return _remove_islands(mesh, remove_islands_fraction)


def subdivide_long_edges(mesh: Mesh, iterations: int = 1) -> Mesh:
    """Subdivide mesh edges using the existing trimesh-based subdivision."""
    result = Mesh(mesh=mesh.mesh.copy())
    result.subdivide(iterations=iterations)
    return result


__all__ = [
    "MeshResult",
    "TopologyState",
    "TreeState",
    "build_and_refine_tree",
    "compute_isovalue_from_percentile",
    "erode_and_dilate",
    "fof_cluster",
    "get_mesh",
    "get_mesh_from_tree",
    "remove_islands",
    "smooth_mesh",
    "subdivide_long_edges",
]
