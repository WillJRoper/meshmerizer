"""Tests for Meshmerizer core functionality."""

import numpy as np
import pytest
import trimesh

from meshmerizer.cli import _build_parser, _run_stl
from meshmerizer.mesh import Mesh, voxels_to_stl, voxels_to_stl_via_sdf
from meshmerizer.voxels import generate_voxel_grid, process_gaussian_smoothing


# --- Fixtures / Helper Functions ---
@pytest.fixture
def simple_cube_volume():
    """Generates a 3D binary cube volume."""
    resolution = 20
    volume = np.zeros((resolution, resolution, resolution))
    # Create a 10x10x10 cube in the center
    volume[5:15, 5:15, 5:15] = 1.0
    return volume


@pytest.fixture
def sphere_volume():
    """Generates a 3D binary sphere volume."""
    x, y, z = np.ogrid[-10:10, -10:10, -10:10]
    volume = (x**2 + y**2 + z**2) <= 8**2
    return volume.astype(float)


# --- Tests for Watertightness and Scaling ---
@pytest.mark.parametrize(
    "volume_fixture",
    ["simple_cube_volume", "sphere_volume"],
)
@pytest.mark.parametrize("method", ["standard", "sdf"])
@pytest.mark.parametrize("box_size", [1.0, 10.0, 0.5])
def test_watertightness_and_scaling(volume_fixture, method, box_size, request):
    volume = request.getfixturevalue(volume_fixture)
    resolution = volume.shape[0]

    # Generate voxel grid with box_size to get the correct voxel_size
    # The actual grid for mesh generation is the 'volume' fixture itself.
    _dummy_data = np.array([1.0])
    _dummy_coordinates = np.array([[0.0, 0.0, 0.0]])
    _, voxel_size = generate_voxel_grid(
        data=_dummy_data,
        coordinates=_dummy_coordinates,
        resolution=resolution,
        box_size=box_size,
    )

    if method == "standard":
        meshes = voxels_to_stl(volume, threshold=0.5, voxel_size=voxel_size)
    else:  # sdf
        meshes = voxels_to_stl_via_sdf(
            volume,
            threshold=0.5,
            voxel_size=voxel_size,
        )

    assert len(meshes) > 0, (
        f"No meshes generated for {volume_fixture} with {method}"
    )

    for mesh_obj in meshes:
        assert mesh_obj.to_trimesh().is_watertight, (
            "Mesh not watertight for "
            f"{volume_fixture} with {method}, box_size={box_size}"
        )

        # Test scaling: Bounding box should match box_size (approximately)
        # Marching cubes output vertices are already scaled

        bbox_max = np.max(mesh_obj.vertices, axis=0)
        bbox_min = np.min(mesh_obj.vertices, axis=0)

        # The object should occupy space proportional to the box_size. The
        # actual span will be slightly less than box_size if the object doesn't
        # fill the whole grid.

        # Check that the overall span of the mesh is within the expected
        # box_size range.
        total_span = bbox_max - bbox_min

        # Allow some tolerance for floating point and marching cubes
        # discretization.
        # Check max dimension of the span against box_size

        # Upper bound: Should not exceed box_size (plus slight MC padding).
        assert total_span.max() <= box_size + voxel_size * 2, (
            "Scaling too large for "
            f"{volume_fixture} with {method}, box_size={box_size}. "
            f"Span max: {total_span.max()}"
        )

        # Lower bound: Should be substantial (e.g. > 20% of box size) to prove
        # it wasn't shrunk to near zero.
        # The test shapes (sphere, cube) are roughly 50-80% of the volume.
        assert total_span.max() > 0.2 * box_size, (
            "Scaling too small for "
            f"{volume_fixture} with {method}, box_size={box_size}. "
            f"Span max: {total_span.max()}"
        )

        # Check that the mesh doesn't extend significantly beyond the box.
        assert np.all(bbox_min >= -voxel_size * 2)
        assert np.all(bbox_max <= box_size + voxel_size * 2)


# --- Test C Extension and generate_voxel_grid ---
def test_generate_voxel_grid_no_smoothing():
    data = np.array([1.0, 2.0, 3.0])
    coordinates = np.array([[0.2, 0.2, 0.2], [1.8, 1.8, 1.8], [1.0, 1.0, 1.0]])
    resolution = 4
    box_size = 2.0

    grid, voxel_size = generate_voxel_grid(
        data,
        coordinates,
        resolution,
        box_size=box_size,
    )

    expected_grid = np.zeros(
        (resolution, resolution, resolution),
        dtype=np.float64,
    )
    expected_grid[0, 0, 0] += 1.0  # 0.2 / 2.0 * 4 = 0.4 -> 0
    expected_grid[3, 3, 3] += 2.0  # 1.8 / 2.0 * 4 = 3.6 -> 3
    expected_grid[2, 2, 2] += 3.0  # 1.0 / 2.0 * 4 = 2.0 -> 2

    assert np.allclose(grid, expected_grid)
    assert voxel_size == box_size / resolution


def test_generate_voxel_grid_infers_voxel_size_from_bounds():
    data = np.array([1.0, 1.0])
    coordinates = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    resolution = 4

    _grid, voxel_size = generate_voxel_grid(
        data=data,
        coordinates=coordinates,
        resolution=resolution,
        box_size=None,
    )

    assert voxel_size == 0.5


def test_generate_voxel_grid_with_smoothing_c_extension():
    # Define points: Two corners to define the box [0,1], and one center point
    # Corners have data=0 so they don't affect the check, just the bounds
    test_data = np.array([0.0, 0.0, 10.0])
    test_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # Corner 1
            [1.0, 1.0, 1.0],  # Corner 2
            [0.5, 0.5, 0.5],  # Center point
        ]
    )

    # Smoothing: Corners=0, Center=0.2
    test_smoothing_physical = np.array([0.0, 0.0, 0.2])

    resolution = 5
    box_size = 1.0

    grid, voxel_size = generate_voxel_grid(
        test_data,
        test_coords,
        resolution,
        smoothing_lengths=test_smoothing_physical,
        box_size=box_size,
    )

    expected_grid = np.zeros(
        (resolution, resolution, resolution),
        dtype=np.float64,
    )

    # Center point is at 0.5, which maps to index 2 (in 0..4).
    # Smoothing 0.2 physical = 1 voxel unit.
    # So it should deposit 10.0 into [2-1 : 2+1] -> [1:4, 1:4, 1:4]
    expected_grid[1:4, 1:4, 1:4] += 10.0

    # Corner points (indices 0 and 4) have data=0, so they deposit nothing.

    assert np.allclose(grid, expected_grid)
    assert voxel_size == box_size / resolution


def test_generate_voxel_grid_with_smoothing_respects_nthreads():
    test_data = np.array([0.0, 0.0, 10.0])
    test_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ]
    )
    test_smoothing_physical = np.array([0.0, 0.0, 0.2])

    grid_single, _ = generate_voxel_grid(
        test_data,
        test_coords,
        resolution=5,
        smoothing_lengths=test_smoothing_physical,
        box_size=1.0,
        nthreads=1,
    )
    grid_multi, _ = generate_voxel_grid(
        test_data,
        test_coords,
        resolution=5,
        smoothing_lengths=test_smoothing_physical,
        box_size=1.0,
        nthreads=2,
    )

    assert np.allclose(grid_single, grid_multi)


def test_process_gaussian_smoothing_reduces_peak_and_preserves_mass():
    grid = np.zeros((9, 9, 9), dtype=float)
    grid[4, 4, 4] = 1.0

    smoothed = process_gaussian_smoothing(grid, sigma=1.0)

    assert smoothed[4, 4, 4] < 1.0
    assert smoothed[4, 4, 4] > 0.0
    assert np.isclose(smoothed.sum(), grid.sum())


# --- Test Mesh class functionality ---
def test_mesh_add():
    mesh1 = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
        vertex_normals=np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]),
    )
    mesh2 = Mesh(
        vertices=np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
        vertex_normals=np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]),
    )

    combined_mesh = mesh1 + mesh2

    assert isinstance(combined_mesh, Mesh)
    assert len(combined_mesh) == 6  # 3 from mesh1 + 3 from mesh2
    # Check if trimesh object has correctly combined
    assert combined_mesh.to_trimesh().vertices.shape[0] == 6
    assert combined_mesh.to_trimesh().faces.shape[0] == 2


def test_mesh_repair_no_errors():
    # Create a mesh that might have issues
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    faces = np.array([[0, 1, 2], [0, 1, 2]])
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])

    mesh = Mesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Should not raise any exceptions
    try:
        mesh.repair(smoothing_iters=1)
    except Exception as e:
        pytest.fail(f"Mesh repair raised an exception: {e}")

    # Basic check: should have fewer vertices/faces after repair
    # Original mesh has 4 vertices (0,0,0 is duplicated but explicit).
    # After repair, it should identify and remove duplicate vertex and faces.
    # trimesh.Trimesh(vertices=verts, faces=faces, process=True) would result
    # in 3 unique vertices, 1 face.
    # Our Mesh.repair doesn't re-process from scratch, it cleans the existing.
    # It removes duplicate faces, degenerate faces, unreferenced vertices.
    # Let's make this test more robust by creating a trimesh and then checking.
    test_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    test_mesh.process()

    assert mesh.to_trimesh().vertices.shape[0] == test_mesh.vertices.shape[0]
    assert mesh.to_trimesh().faces.shape[0] == test_mesh.faces.shape[0]


def test_mesh_repair_smoothing_preserves_watertightness():
    volume = np.zeros((32, 32, 32), dtype=float)
    volume[8:24, 8:24, 8:24] = 1.0
    mesh = voxels_to_stl_via_sdf(volume, threshold=0.5)[0]

    mesh.repair(smoothing_iters=10)
    trimesh_mesh = mesh.to_trimesh()

    assert np.isfinite(trimesh_mesh.vertices).all()
    assert trimesh_mesh.is_watertight


def test_mesh_subdivide_then_repair_preserves_watertightness():
    volume = np.zeros((32, 32, 32), dtype=float)
    volume[8:24, 8:24, 8:24] = 1.0
    mesh = voxels_to_stl_via_sdf(volume, threshold=0.5)[0]

    n_vertices_before = mesh.to_trimesh().vertices.shape[0]
    mesh.subdivide(iterations=1)
    mesh.repair(smoothing_iters=5)
    trimesh_mesh = mesh.to_trimesh()

    assert trimesh_mesh.vertices.shape[0] > n_vertices_before
    assert np.isfinite(trimesh_mesh.vertices).all()
    assert trimesh_mesh.is_watertight


# --- Edge Cases ---
def test_empty_volume_handling():
    volume = np.zeros((5, 5, 5))
    with pytest.raises(ValueError, match="No meshes created"):
        voxels_to_stl(volume, threshold=0.5)

    with pytest.raises(ValueError, match="No meshes created"):
        voxels_to_stl_via_sdf(volume, threshold=0.5)


def test_sub_threshold_volume_handling():
    volume = np.full((5, 5, 5), 0.1)  # All values below threshold
    with pytest.raises(ValueError, match="No meshes created"):
        voxels_to_stl(volume, threshold=0.5)


@pytest.mark.parametrize("method", ["standard", "sdf"])
def test_remove_islands_keeps_largest_component(method):
    volume = np.zeros((24, 24, 24), dtype=float)
    volume[4:14, 4:14, 4:14] = 1.0
    volume[18:20, 18:20, 18:20] = 1.0

    if method == "standard":
        meshes = voxels_to_stl(
            volume,
            threshold=0.5,
            remove_islands=True,
        )
    else:
        meshes = voxels_to_stl_via_sdf(
            volume,
            threshold=0.5,
            remove_islands=True,
        )

    assert len(meshes) == 1

    bbox_min = np.min(meshes[0].vertices, axis=0)
    bbox_max = np.max(meshes[0].vertices, axis=0)

    assert np.all(bbox_min < 15.0)
    assert np.all(bbox_max < 16.0)


def test_cli_parses_smooth_iters():
    parser = _build_parser()
    args = parser.parse_args(["stl", "snapshot.hdf5", "--smooth-iters", "8"])

    assert args.smooth_iters == 8


def test_cli_parses_nthreads():
    parser = _build_parser()
    args = parser.parse_args(["stl", "snapshot.hdf5", "--nthreads", "4"])

    assert args.nthreads == 4


def test_cli_parses_gaussian_sigma():
    parser = _build_parser()
    args = parser.parse_args(
        ["stl", "snapshot.hdf5", "--gaussian-sigma", "1.5"]
    )

    assert args.gaussian_sigma == 1.5


def test_cli_parses_subdivide_iters():
    parser = _build_parser()
    args = parser.parse_args(
        ["stl", "snapshot.hdf5", "--subdivide-iters", "1"]
    )

    assert args.subdivide_iters == 1


def test_cli_parses_tight_bounds():
    parser = _build_parser()
    args = parser.parse_args(["stl", "snapshot.hdf5", "--tight-bounds"])

    assert args.tight_bounds is True


def test_cli_parses_nchunks():
    parser = _build_parser()
    args = parser.parse_args(["stl", "snapshot.hdf5", "--nchunks", "3"])

    assert args.nchunks == 3


def test_cli_parses_chunk_output():
    parser = _build_parser()
    args = parser.parse_args(
        ["stl", "snapshot.hdf5", "--chunk-output", "separate"]
    )

    assert args.chunk_output == "separate"


def test_cli_parses_chunk_output_unioned():
    parser = _build_parser()
    args = parser.parse_args(
        ["stl", "snapshot.hdf5", "--chunk-output", "unioned"]
    )

    assert args.chunk_output == "unioned"


def test_run_stl_uses_chunked_mesh_path(monkeypatch, tmp_path):
    parser = _build_parser()
    out_path = tmp_path / "chunked.stl"
    args = parser.parse_args(
        [
            "stl",
            "snapshot.hdf5",
            "--nchunks",
            "2",
            "--output",
            str(out_path),
        ]
    )

    called = {"chunked": False}

    def fake_load_particles(**_kwargs):
        return (
            np.array([1.0]),
            np.array([[0.1, 0.1, 0.1]]),
            None,
            1.0,
            np.zeros(3),
        )

    def fake_generate_hard_chunk_meshes(*_args, **_kwargs):
        called["chunked"] = True
        mesh = Mesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2]]),
        )
        return [(object(), [mesh])]

    monkeypatch.setattr(
        "meshmerizer.cli._load_swift_particles", fake_load_particles
    )
    monkeypatch.setattr(
        "meshmerizer.cli.generate_hard_chunk_meshes",
        fake_generate_hard_chunk_meshes,
    )

    _run_stl(args)

    assert called["chunked"] is True
    assert out_path.exists()


def test_run_stl_writes_separate_chunk_directory(monkeypatch, tmp_path):
    parser = _build_parser()
    out_path = tmp_path / "chunked.stl"
    args = parser.parse_args(
        [
            "stl",
            "snapshot.hdf5",
            "--nchunks",
            "2",
            "--chunk-output",
            "separate",
            "--output",
            str(out_path),
        ]
    )

    def fake_load_particles(**_kwargs):
        return (
            np.array([1.0]),
            np.array([[0.1, 0.1, 0.1]]),
            None,
            1.0,
            np.zeros(3),
        )

    mesh = Mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
    )

    def fake_generate_hard_chunk_meshes(*_args, **_kwargs):
        return [(object(), [mesh]), (object(), [mesh])]

    monkeypatch.setattr(
        "meshmerizer.cli._load_swift_particles", fake_load_particles
    )
    monkeypatch.setattr(
        "meshmerizer.cli.generate_hard_chunk_meshes",
        fake_generate_hard_chunk_meshes,
    )

    _run_stl(args)

    output_dir = tmp_path / "chunked"
    assert output_dir.is_dir()
    assert (output_dir / "chunked_1.stl").exists()
    assert (output_dir / "chunked_2.stl").exists()


def test_run_stl_uses_unioned_chunk_output(monkeypatch, tmp_path):
    parser = _build_parser()
    out_path = tmp_path / "chunked.stl"
    args = parser.parse_args(
        [
            "stl",
            "snapshot.hdf5",
            "--nchunks",
            "2",
            "--chunk-output",
            "unioned",
            "--output",
            str(out_path),
        ]
    )

    called = {"unioned": False}

    def fake_load_particles(**_kwargs):
        return (
            np.array([1.0]),
            np.array([[0.1, 0.1, 0.1]]),
            None,
            1.0,
            np.zeros(3),
        )

    mesh = Mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
    )

    def fake_generate_hard_chunk_meshes(*_args, **_kwargs):
        return [(object(), [mesh]), (object(), [mesh])]

    def fake_union_hard_chunk_meshes(*_args, **_kwargs):
        called["unioned"] = True
        return mesh

    monkeypatch.setattr(
        "meshmerizer.cli._load_swift_particles", fake_load_particles
    )
    monkeypatch.setattr(
        "meshmerizer.cli.generate_hard_chunk_meshes",
        fake_generate_hard_chunk_meshes,
    )
    monkeypatch.setattr(
        "meshmerizer.cli.union_hard_chunk_meshes",
        fake_union_hard_chunk_meshes,
    )

    _run_stl(args)

    assert called["unioned"] is True
    assert out_path.exists()


def test_generate_voxel_grid_coords_range_zero():
    data = np.array([1.0])
    coordinates = np.array([[0.0, 0.0, 0.0]])  # All coords the same
    resolution = 5
    box_size = 1.0

    grid, voxel_size = generate_voxel_grid(
        data,
        coordinates,
        resolution,
        box_size=box_size,
    )

    expected_grid = np.zeros(
        (resolution, resolution, resolution),
        dtype=np.float64,
    )
    # The single point should map to 0,0,0
    # Note: `np.clip(vox_indices, 0, resolution - 1)` will make this (0,0,0)
    expected_grid[0, 0, 0] = 1.0

    assert np.allclose(grid, expected_grid)
    assert voxel_size == box_size / resolution
