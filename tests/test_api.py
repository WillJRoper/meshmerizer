"""Tests for the public Python API layer."""

import numpy as np
import pytest

from meshmerizer import (
    MeshResult,
    TopologyState,
    TreeState,
    build_tree,
    cluster_particles,
    extract_mesh,
    generate_mesh,
    regularize,
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


def test_build_tree_returns_tree_state() -> None:
    positions, smoothing_lengths = _simple_particles()
    tree = build_tree(
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
    assert isinstance(tree.contributors, np.ndarray)


def test_build_tree_passes_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions, smoothing_lengths = _simple_particles()
    captured = {}

    def fake_build_refined_tree(
        pos,
        sml,
        domain_min,
        domain_max,
        base_resolution,
        isovalue,
        max_depth,
        worker_count,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
    ):
        captured["worker_count"] = worker_count
        return (), np.array([], dtype=np.int64)

    monkeypatch.setattr(
        "meshmerizer.api.build_refined_tree",
        fake_build_refined_tree,
    )

    tree = build_tree(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        isovalue=0.01,
        max_depth=2,
        worker_count=3,
    )

    assert captured["worker_count"] == 3
    assert isinstance(tree, TreeState)


def test_regularize_returns_topology_state() -> None:
    positions, smoothing_lengths = _simple_particles()
    tree = build_tree(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        isovalue=0.01,
        max_depth=2,
    )
    topology = regularize(tree, min_feature_thickness=0.2)
    assert isinstance(topology, TopologyState)
    assert topology.opened_inside.ndim == 1


def test_regularize_passes_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions, smoothing_lengths = _simple_particles()
    tree = TreeState(
        cells=(),
        contributors=np.array([], dtype=np.int64),
        positions=positions,
        smoothing_lengths=smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        max_depth=2,
        isovalue=0.01,
    )
    captured = {}

    def fake_classify_occupied_solid(*args, **kwargs):
        captured["worker_count"] = kwargs["worker_count"]
        return {
            "occupancy": np.zeros(1, dtype=np.uint8),
            "depths": np.zeros(1, dtype=np.uint32),
            "center_values": np.zeros(1, dtype=np.float64),
            "cell_sizes": np.ones(1, dtype=np.float64),
            "clearance": np.zeros(1, dtype=np.float64),
            "thickening_distance": np.zeros(1, dtype=np.float64),
            "thickened_inside": np.zeros(1, dtype=np.uint8),
            "eroded_inside": np.zeros(1, dtype=np.uint8),
            "dilation_distance": np.zeros(1, dtype=np.float64),
            "opened_inside": np.zeros(1, dtype=np.uint8),
            "opened_boundary_positions": np.zeros((0, 3), dtype=np.float64),
            "opened_boundary_normals": np.zeros((0, 3), dtype=np.float64),
            "opened_surface_vertices": np.zeros((0, 3), dtype=np.float64),
            "opened_surface_faces": np.zeros((0, 3), dtype=np.uint32),
        }

    monkeypatch.setattr(
        "meshmerizer.api.classify_occupied_solid",
        fake_classify_occupied_solid,
    )

    topology = regularize(
        tree,
        min_feature_thickness=0.2,
        worker_count=4,
    )

    assert captured["worker_count"] == 4
    assert isinstance(topology, TopologyState)


def test_regularize_supports_pre_thickening() -> None:
    positions, smoothing_lengths = _simple_particles()
    tree = build_tree(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        isovalue=0.01,
        max_depth=2,
    )
    baseline = regularize(tree, min_feature_thickness=0.2)
    thickened = regularize(
        tree,
        min_feature_thickness=0.2,
        pre_thickening_radius=0.2,
    )
    assert thickened.thickening_distance.shape == thickened.opened_inside.shape
    assert thickened.thickened_inside.shape == thickened.opened_inside.shape
    assert np.count_nonzero(thickened.thickened_inside) >= np.count_nonzero(
        baseline.thickened_inside
    )


def test_generate_mesh_returns_mesh_result() -> None:
    positions, smoothing_lengths = _simple_particles()
    result = generate_mesh(
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


def test_generate_mesh_returns_compacted_native_geometry() -> None:
    """Native extraction should not leave repeated or unreferenced geometry."""
    positions, smoothing_lengths = _simple_particles()
    result = generate_mesh(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        max_depth=2,
        isovalue=0.01,
    )

    faces = np.asarray(result.mesh.faces)
    vertices = np.asarray(result.mesh.vertices)
    assert np.all(faces[:, 0] != faces[:, 1])
    assert np.all(faces[:, 1] != faces[:, 2])
    assert np.all(faces[:, 0] != faces[:, 2])

    referenced = np.unique(faces.reshape(-1))
    assert referenced.size == vertices.shape[0]
    assert np.array_equal(referenced, np.arange(vertices.shape[0]))


def test_generate_mesh_supports_pre_thickening() -> None:
    positions, smoothing_lengths = _simple_particles()
    result = generate_mesh(
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


def test_extract_mesh_from_tree_returns_mesh_result() -> None:
    positions, smoothing_lengths = _simple_particles()
    tree = build_tree(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        isovalue=0.01,
        max_depth=2,
    )
    result = extract_mesh(tree)
    assert isinstance(result, MeshResult)
    assert result.mesh.faces.shape[1] == 3


def test_extract_mesh_from_topology_returns_mesh_result() -> None:
    positions, smoothing_lengths = _simple_particles()
    tree = build_tree(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        isovalue=0.01,
        max_depth=2,
    )
    topology = regularize(tree, min_feature_thickness=0.2)
    result = extract_mesh(topology)
    assert isinstance(result, MeshResult)
    assert result.mesh.faces.shape[1] == 3


def test_extract_mesh_from_topology_reuses_cached_mesh() -> None:
    positions, smoothing_lengths = _simple_particles()
    tree = build_tree(
        positions,
        smoothing_lengths,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
        base_resolution=2,
        isovalue=0.01,
        max_depth=2,
    )
    topology = regularize(tree, min_feature_thickness=0.2)
    cached_vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    cached_faces = np.array([[0, 1, 2]], dtype=np.uint32)
    topology.mesh_vertices = cached_vertices
    topology.mesh_faces = cached_faces

    result = extract_mesh(topology)

    assert np.array_equal(result.mesh.vertices, cached_vertices)
    assert np.array_equal(result.mesh.faces, cached_faces)


def test_cluster_particles_returns_labels() -> None:
    positions, _ = _simple_particles()
    labels = cluster_particles(
        positions,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.0, 2.0, 2.0),
    )
    assert labels.shape == (positions.shape[0],)


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


def test_smooth_mesh_skips_second_process_without_smoothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = Mesh(
        mesh=Mesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.uint32),
        ).mesh.copy()
    )

    process_calls = 0
    fix_normals_calls = 0

    def count_process() -> None:
        nonlocal process_calls
        process_calls += 1

    def count_fix_normals() -> None:
        nonlocal fix_normals_calls
        fix_normals_calls += 1

    monkeypatch.setattr(mesh.mesh, "process", count_process)
    monkeypatch.setattr(mesh.mesh, "fix_normals", count_fix_normals)
    monkeypatch.setattr(
        "meshmerizer.mesh.core._repair_local_broken_faces",
        lambda _: None,
    )

    smooth_mesh(mesh, iterations=0, inplace=True)

    assert process_calls == 1
    assert fix_normals_calls == 1
