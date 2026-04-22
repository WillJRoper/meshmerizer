"""Tests for the public Python API layer."""

import numpy as np

from meshmerizer import (
    MeshResult,
    TopologyState,
    TreeState,
    build_and_refine_tree,
    erode_and_dilate,
    get_mesh,
    get_mesh_from_topology,
    get_mesh_from_tree,
    remove_islands,
    smooth_mesh,
    subdivide_long_edges,
)
from meshmerizer.mesh.core import Mesh


def _simple_particles() -> tuple[np.ndarray, np.ndarray]:
    positions = np.array(
        [
            [0.9, 0.9, 0.9],
            [1.1, 0.9, 0.9],
            [0.9, 1.1, 0.9],
            [1.1, 1.1, 0.9],
            [0.9, 0.9, 1.1],
            [1.1, 0.9, 1.1],
            [0.9, 1.1, 1.1],
            [1.1, 1.1, 1.1],
        ],
        dtype=np.float64,
    )
    smoothing_lengths = np.full(positions.shape[0], 0.45, dtype=np.float64)
    return positions, smoothing_lengths


def test_build_and_refine_tree_returns_tree_state() -> None:
    positions, smoothing_lengths = _simple_particles()
    tree = build_and_refine_tree(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        isovalue=0.01,
        max_depth=2,
    )
    assert isinstance(tree, TreeState)
    assert tree.positions.shape == positions.shape
    assert tree.smoothing_lengths.shape == smoothing_lengths.shape


def test_erode_and_dilate_returns_topology_state() -> None:
    positions, smoothing_lengths = _simple_particles()
    tree = build_and_refine_tree(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        isovalue=0.01,
        max_depth=2,
    )
    topology = erode_and_dilate(tree, min_feature_thickness=0.2)
    assert isinstance(topology, TopologyState)
    assert topology.opened_inside.ndim == 1


def test_erode_and_dilate_supports_pre_thickening() -> None:
    positions, smoothing_lengths = _simple_particles()
    tree = build_and_refine_tree(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        isovalue=0.01,
        max_depth=2,
    )
    baseline = erode_and_dilate(tree, min_feature_thickness=0.2)
    thickened = erode_and_dilate(
        tree,
        min_feature_thickness=0.2,
        pre_thickening_radius=0.2,
    )
    assert thickened.thickening_distance.shape == thickened.opened_inside.shape
    assert thickened.thickened_inside.shape == thickened.opened_inside.shape
    assert np.count_nonzero(thickened.thickened_inside) >= np.count_nonzero(
        baseline.thickened_inside
    )


def test_get_mesh_returns_mesh_result() -> None:
    positions, smoothing_lengths = _simple_particles()
    result = get_mesh(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        max_depth=2,
        isovalue=0.01,
    )
    assert isinstance(result, MeshResult)
    assert result.mesh.vertices.shape[1] == 3
    assert result.mesh.faces.shape[1] == 3


def test_get_mesh_supports_pre_thickening() -> None:
    positions, smoothing_lengths = _simple_particles()
    result = get_mesh(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        max_depth=2,
        isovalue=0.01,
        min_feature_thickness=0.2,
        pre_thickening_radius=0.2,
    )
    assert isinstance(result, MeshResult)
    assert result.mesh.faces.shape[1] == 3


def test_get_mesh_from_tree_returns_mesh_result() -> None:
    positions, smoothing_lengths = _simple_particles()
    tree = build_and_refine_tree(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        isovalue=0.01,
        max_depth=2,
    )
    result = get_mesh_from_tree(tree)
    assert isinstance(result, MeshResult)
    assert result.mesh.faces.shape[1] == 3


def test_get_mesh_from_topology_returns_mesh_result() -> None:
    positions, smoothing_lengths = _simple_particles()
    tree = build_and_refine_tree(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        isovalue=0.01,
        max_depth=2,
    )
    topology = erode_and_dilate(tree, min_feature_thickness=0.2)
    result = get_mesh_from_topology(topology)
    assert isinstance(result, MeshResult)
    assert result.mesh.faces.shape[1] == 3


def test_mesh_helpers_return_mesh_instances() -> None:
    box = Mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
    )
    smoothed = smooth_mesh(box, iterations=0)
    subdivided = subdivide_long_edges(box, iterations=1)
    cleaned = remove_islands(box, remove_islands_fraction=None)
    assert isinstance(smoothed, Mesh)
    assert isinstance(subdivided, Mesh)
    assert isinstance(cleaned, Mesh)
