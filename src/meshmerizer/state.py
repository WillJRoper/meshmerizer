"""Typed state objects shared by the staged public API.

The staged Python API passes explicit state objects between pipeline phases so
callers can inspect, persist, or modify intermediate results without relying on
loosely structured dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meshmerizer.mesh import Mesh

Vec3 = tuple[float, float, float]


@dataclass
class TreeState:
    """Adaptive tree plus downstream particle and domain metadata.

    Attributes:
        cells: Refined octree cells returned by the adaptive core.
        contributors: Flat contributor index array aligned with ``cells``.
        positions: Contiguous particle positions with shape ``(N, 3)``.
        smoothing_lengths: Contiguous per-particle support radii with shape
            ``(N,)``.
        domain_min: Inclusive lower corner of the working domain.
        domain_max: Exclusive upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.
        max_depth: Maximum allowed octree refinement depth.
        isovalue: Scalar field threshold used for surface extraction.
        minimum_usable_hermite_samples: Minimum usable Hermite sample count
            required before a corner-crossing cell may stop refining.
        max_qef_rms_residual_ratio: Maximum acceptable RMS QEF residual as a
            fraction of the local cell radius.
        min_normal_alignment_threshold: Minimum alignment required between
            usable Hermite normals and their mean direction.
    """

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
    """Regularized opened-solid topology derived from a tree state.

    Attributes:
        tree: Source tree state used to compute the topology.
        occupancy: Per-leaf occupancy classification values.
        depths: Per-leaf octree depth values.
        centers: Scalar field values sampled at leaf centres.
        sizes: Edge length for each leaf cell.
        clearance: Estimated distance from each leaf to the boundary.
        thickening_distance: Outward thickening distance applied per leaf.
        thickened_inside: Boolean mask after optional pre-thickening.
        eroded_inside: Boolean mask after erosion.
        dilation_distance: Dilation distance used to reopen the solid.
        opened_inside: Final opened-solid boolean mask.
        sample_positions: Boundary sample positions from the opened solid.
        sample_normals: Boundary sample normals from the opened solid.
        mesh_vertices: Vertices of the extracted opened-surface mesh.
        mesh_faces: Faces of the extracted opened-surface mesh.
        min_feature_thickness: Requested minimum preserved feature thickness.
    """

    tree: TreeState
    occupancy: np.ndarray
    depths: np.ndarray
    centers: np.ndarray
    sizes: np.ndarray
    clearance: np.ndarray
    thickening_distance: np.ndarray
    thickened_inside: np.ndarray
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
    """Mesh plus lightweight metadata from the reconstruction pipeline.

    Attributes:
        mesh: Resulting mesh wrapper.
        isovalue: Scalar field threshold used during extraction.
        n_qef_vertices: Number of QEF vertices solved before post-processing.
    """

    mesh: Mesh
    isovalue: float
    n_qef_vertices: int


__all__ = ["MeshResult", "TopologyState", "TreeState", "Vec3"]
