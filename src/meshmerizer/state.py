"""Typed state objects shared by the staged public API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meshmerizer.mesh import Mesh

Vec3 = tuple[float, float, float]


@dataclass
class TreeState:
    """Adaptive tree plus downstream particle and domain metadata."""

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
    """Mesh plus lightweight metadata from the reconstruction pipeline."""

    mesh: Mesh
    isovalue: float
    n_qef_vertices: int


__all__ = ["MeshResult", "TopologyState", "TreeState", "Vec3"]
