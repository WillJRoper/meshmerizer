"""Tests for Meshmerizer core functionality."""

import numpy as np
import pytest
import trimesh

from meshmerizer.chunks import HardChunkBounds
from meshmerizer.commands.args import build_parser
from meshmerizer.commands.stl import run_stl
from meshmerizer.mesh import Mesh, voxels_to_stl, voxels_to_stl_via_sdf
from meshmerizer.voxels import generate_voxel_grid, process_gaussian_smoothing

try:
    import fast_simplification  # noqa: F401

    HAS_FAST_SIMPLIFICATION = True
except ImportError:
    HAS_FAST_SIMPLIFICATION = False


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


def test_mesh_repair_handles_small_broken_patch() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, -1.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [1, 2, 3],
            [1, 3, 5],
            [5, 3, 4],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 4],
            [1, 4, 5],
        ],
        dtype=np.int64,
    )
    mesh = Mesh(vertices=vertices, faces=faces)

    assert not mesh.to_trimesh().is_watertight

    mesh.repair(smoothing_iters=0)

    repaired = mesh.to_trimesh()
    assert repaired.is_winding_consistent
    assert repaired.faces.shape[0] > 0


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


@pytest.mark.skipif(
    not HAS_FAST_SIMPLIFICATION,
    reason="fast-simplification is not installed",
)
def test_mesh_simplify_reduces_face_count_and_preserves_watertightness():
    mesh = Mesh(mesh=trimesh.creation.icosphere(subdivisions=3, radius=1.0))
    original_faces = len(mesh.faces)

    mesh.simplify(0.5)

    assert len(mesh.faces) < original_faces
    assert mesh.to_trimesh().is_watertight


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
            remove_islands=0,
        )
    else:
        meshes = voxels_to_stl_via_sdf(
            volume,
            threshold=0.5,
            remove_islands=0,
        )

    assert len(meshes) == 1

    bbox_min = np.min(meshes[0].vertices, axis=0)
    bbox_max = np.max(meshes[0].vertices, axis=0)

    assert np.all(bbox_min < 15.0)
    assert np.all(bbox_max < 16.0)


@pytest.mark.parametrize("method", ["standard", "sdf"])
def test_remove_islands_threshold_discards_only_small_components(method):
    volume = np.zeros((24, 24, 24), dtype=float)
    volume[2:12, 2:12, 2:12] = 1.0
    volume[14:17, 14:17, 14:17] = 1.0
    volume[20:22, 20:22, 20:22] = 1.0

    if method == "standard":
        meshes = voxels_to_stl(
            volume,
            threshold=0.5,
            split_islands=True,
            remove_islands=10,
        )
    else:
        meshes = voxels_to_stl_via_sdf(
            volume,
            threshold=0.5,
            split_islands=True,
            remove_islands=10,
        )

    assert len(meshes) == 2

    max_spans = sorted(
        [np.ptp(mesh.vertices, axis=0).max() for mesh in meshes],
        reverse=True,
    )
    assert max_spans[0] > 8.0
    assert max_spans[1] > 2.0


def test_cli_parses_remove_islands_flag_as_largest_only():
    parser = build_parser()
    args = parser.parse_args(["stl", "snapshot.hdf5", "--remove-islands"])

    assert args.remove_islands == 0


def test_cli_parses_remove_islands_threshold():
    parser = build_parser()
    args = parser.parse_args(
        ["stl", "snapshot.hdf5", "--remove-islands", "10"]
    )

    assert args.remove_islands == 10


def test_cli_parses_smooth_iters():
    parser = build_parser()
    args = parser.parse_args(["stl", "snapshot.hdf5", "--smooth-iters", "8"])

    assert args.smooth_iters == 8


def test_cli_parses_nthreads():
    parser = build_parser()
    args = parser.parse_args(["stl", "snapshot.hdf5", "--nthreads", "4"])

    assert args.nthreads == 4


def test_cli_parses_gaussian_sigma():
    parser = build_parser()
    args = parser.parse_args(
        ["stl", "snapshot.hdf5", "--gaussian-sigma", "1.5"]
    )

    assert args.gaussian_sigma == 1.5


def test_cli_parses_subdivide_iters():
    parser = build_parser()
    args = parser.parse_args(
        ["stl", "snapshot.hdf5", "--subdivide-iters", "1"]
    )

    assert args.subdivide_iters == 1


def test_cli_parses_simplify_factor():
    parser = build_parser()
    args = parser.parse_args(
        ["stl", "snapshot.hdf5", "--simplify-factor", "0.5"]
    )

    assert args.simplify_factor == 0.5


def test_run_stl_rejects_invalid_simplify_factor(monkeypatch, tmp_path):
    parser = build_parser()
    out_path = tmp_path / "invalid_simplify.stl"
    args = parser.parse_args(
        [
            "stl",
            "snapshot.hdf5",
            "--simplify-factor",
            "1.5",
            "--output",
            str(out_path),
        ]
    )

    with pytest.raises(SystemExit):
        run_stl(args)


def test_cli_parses_tight_bounds():
    parser = build_parser()
    args = parser.parse_args(["stl", "snapshot.hdf5", "--tight-bounds"])

    assert args.tight_bounds is True


def test_cli_parses_nchunks():
    parser = build_parser()
    args = parser.parse_args(["stl", "snapshot.hdf5", "--nchunks", "3"])

    assert args.nchunks == 3


def test_cli_parses_chunk_output():
    parser = build_parser()
    args = parser.parse_args(
        ["stl", "snapshot.hdf5", "--chunk-output", "separate"]
    )

    assert args.chunk_output == "separate"


def test_cli_parses_chunk_output_unioned():
    parser = build_parser()
    args = parser.parse_args(
        ["stl", "snapshot.hdf5", "--chunk-output", "unioned"]
    )

    assert args.chunk_output == "unioned"


def test_cli_parses_chunk_overlap_percent():
    parser = build_parser()
    args = parser.parse_args(
        ["stl", "snapshot.hdf5", "--chunk-overlap-percent", "12.5"]
    )

    assert args.chunk_overlap_percent == 12.5


def test_run_stl_uses_chunked_mesh_path(monkeypatch, tmp_path):
    parser = build_parser()
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

    called = {"chunked": False}

    def fake_load_particles(**_kwargs):
        return (
            np.array([1.0]),
            np.array([[0.1, 0.1, 0.1]]),
            None,
            1.0,
            np.zeros(3),
        )

    bounds = HardChunkBounds(
        index=(0, 0, 0),
        nchunks=1,
        sample_start=np.array([0, 0, 0]),
        sample_stop=np.array([2, 2, 2]),
        local_start=np.array([0.0, 0.0, 0.0]),
        local_stop=np.array([1.0, 1.0, 1.0]),
        world_start=np.array([0.0, 0.0, 0.0]),
        world_stop=np.array([1.0, 1.0, 1.0]),
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
        return [(bounds, [mesh])]

    monkeypatch.setattr(
        "meshmerizer.commands.stl.load_swift_particles", fake_load_particles
    )
    monkeypatch.setattr(
        "meshmerizer.commands.stl.generate_hard_chunk_meshes",
        fake_generate_hard_chunk_meshes,
    )

    run_stl(args)

    assert called["chunked"] is True
    assert out_path.exists()


def test_run_stl_writes_separate_chunk_directory(monkeypatch, tmp_path):
    parser = build_parser()
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
        "meshmerizer.commands.stl.load_swift_particles", fake_load_particles
    )
    monkeypatch.setattr(
        "meshmerizer.commands.stl.generate_hard_chunk_meshes",
        fake_generate_hard_chunk_meshes,
    )

    run_stl(args)

    output_dir = tmp_path / "chunked"
    assert output_dir.is_dir()
    assert (output_dir / "chunked_1.stl").exists()
    assert (output_dir / "chunked_2.stl").exists()


def test_run_stl_separate_rejects_target_size(monkeypatch, tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "stl",
            "snapshot.hdf5",
            "--nchunks",
            "2",
            "--chunk-output",
            "separate",
            "--target-size",
            "12",
            "--output",
            str(tmp_path / "chunked.stl"),
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

    def fake_generate_hard_chunk_meshes(*_args, **_kwargs):
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
        "meshmerizer.commands.stl.load_swift_particles", fake_load_particles
    )
    monkeypatch.setattr(
        "meshmerizer.commands.stl.generate_hard_chunk_meshes",
        fake_generate_hard_chunk_meshes,
    )

    with pytest.raises(SystemExit):
        run_stl(args)


def test_run_stl_separate_rejects_remove_islands(monkeypatch, tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "stl",
            "snapshot.hdf5",
            "--nchunks",
            "2",
            "--chunk-output",
            "separate",
            "--remove-islands",
            "--output",
            str(tmp_path / "chunked.stl"),
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

    def fake_generate_hard_chunk_meshes(*_args, **_kwargs):
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
        "meshmerizer.commands.stl.load_swift_particles", fake_load_particles
    )
    monkeypatch.setattr(
        "meshmerizer.commands.stl.generate_hard_chunk_meshes",
        fake_generate_hard_chunk_meshes,
    )

    with pytest.raises(SystemExit):
        run_stl(args)


def test_run_stl_chunked_handles_runtime_error(monkeypatch, tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "stl",
            "snapshot.hdf5",
            "--nchunks",
            "2",
            "--chunk-output",
            "unioned",
            "--output",
            str(tmp_path / "chunked.stl"),
        ]
    )

    def fake_load_particles(**_kwargs):
        return (
            np.array([1.0]),
            np.array([[0.1, 0.1, 0.1]]),
            np.array([0.05]),
            1.0,
            np.zeros(3),
        )

    def fake_generate_hard_chunk_meshes(*_args, **_kwargs):
        raise RuntimeError("chunk-local smoothing requires _voxelize")

    monkeypatch.setattr(
        "meshmerizer.commands.stl.load_swift_particles", fake_load_particles
    )
    monkeypatch.setattr(
        "meshmerizer.commands.stl.generate_hard_chunk_meshes",
        fake_generate_hard_chunk_meshes,
    )

    with pytest.raises(SystemExit):
        run_stl(args)


def test_run_stl_uses_unioned_chunk_output(monkeypatch, tmp_path):
    parser = build_parser()
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
        "meshmerizer.commands.stl.load_swift_particles", fake_load_particles
    )
    monkeypatch.setattr(
        "meshmerizer.commands.stl.generate_hard_chunk_meshes",
        fake_generate_hard_chunk_meshes,
    )
    monkeypatch.setattr(
        "meshmerizer.commands.stl.union_hard_chunk_meshes",
        fake_union_hard_chunk_meshes,
    )

    run_stl(args)

    assert called["unioned"] is True
    assert out_path.exists()


def test_run_stl_unioned_remove_islands_keeps_largest(monkeypatch, tmp_path):
    parser = build_parser()
    out_path = tmp_path / "largest_only.stl"
    args = parser.parse_args(
        [
            "stl",
            "snapshot.hdf5",
            "--nchunks",
            "2",
            "--chunk-output",
            "unioned",
            "--remove-islands",
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

    def fake_generate_hard_chunk_meshes(*_args, **_kwargs):
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

    large = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    small = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    small.apply_translation([5.0, 0.0, 0.0])
    union_mesh = Mesh(mesh=trimesh.util.concatenate([large, small]))

    def fake_union_hard_chunk_meshes(*_args, **_kwargs):
        return union_mesh

    monkeypatch.setattr(
        "meshmerizer.commands.stl.load_swift_particles", fake_load_particles
    )
    monkeypatch.setattr(
        "meshmerizer.commands.stl.generate_hard_chunk_meshes",
        fake_generate_hard_chunk_meshes,
    )
    monkeypatch.setattr(
        "meshmerizer.commands.stl.union_hard_chunk_meshes",
        fake_union_hard_chunk_meshes,
    )

    run_stl(args)

    written = trimesh.load(out_path, force="mesh")
    assert len(written.split(only_watertight=False)) == 1
    assert written.bounds[1][0] < 2.0


def test_run_stl_unioned_remove_islands_threshold(monkeypatch, tmp_path):
    parser = build_parser()
    out_path = tmp_path / "thresholded.stl"
    args = parser.parse_args(
        [
            "stl",
            "snapshot.hdf5",
            "--nchunks",
            "2",
            "--chunk-output",
            "unioned",
            "--remove-islands",
            "200000",
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

    def fake_generate_hard_chunk_meshes(*_args, **_kwargs):
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

    large = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    medium = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    tiny = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    medium.apply_translation([5.0, 0.0, 0.0])
    tiny.apply_translation([8.0, 0.0, 0.0])
    union_mesh = Mesh(mesh=trimesh.util.concatenate([large, medium, tiny]))

    def fake_union_hard_chunk_meshes(*_args, **_kwargs):
        return union_mesh

    monkeypatch.setattr(
        "meshmerizer.commands.stl.load_swift_particles", fake_load_particles
    )
    monkeypatch.setattr(
        "meshmerizer.commands.stl.generate_hard_chunk_meshes",
        fake_generate_hard_chunk_meshes,
    )
    monkeypatch.setattr(
        "meshmerizer.commands.stl.union_hard_chunk_meshes",
        fake_union_hard_chunk_meshes,
    )

    run_stl(args)

    written = trimesh.load(out_path, force="mesh")
    components = written.split(only_watertight=False)
    assert len(components) == 2
    assert max(c.bounds[1][0] for c in components) < 6.0


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
