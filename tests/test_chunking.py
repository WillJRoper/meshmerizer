from __future__ import annotations

import numpy as np
import trimesh

from meshmerizer.chunks import (
    VirtualGrid,
    chunk_world_bounds,
    clip_mesh_to_hard_chunk,
    crop_grid_to_chunk_bounds,
    expand_hard_chunk_bounds,
    generate_hard_chunk_meshes,
    iter_hard_chunk_bounds,
    keep_largest_mesh_component,
    mesh_hard_chunk_sdf,
    preprocess_chunk_grid,
    select_particles_in_hard_chunk,
    union_hard_chunk_meshes,
    voxelize_hard_chunk,
)
from meshmerizer.logging import cli_logging_context
from meshmerizer.mesh import Mesh


def test_virtual_grid_voxel_size() -> None:
    grid = VirtualGrid(
        origin=np.array([1.0, 2.0, 3.0]),
        box_size=8.0,
        resolution=16,
        nchunks=3,
    )

    assert grid.voxel_size == 0.5
    assert grid.cell_resolution == 15


def test_chunk_partition_covers_cells_exactly_once() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=9,
        nchunks=3,
    )

    chunks = list(grid.iter_chunks())
    assert len(chunks) == 27

    covered = set()
    for chunk in chunks:
        for i in range(chunk.x.cell_start, chunk.x.cell_stop):
            for j in range(chunk.y.cell_start, chunk.y.cell_stop):
                for k in range(chunk.z.cell_start, chunk.z.cell_stop):
                    covered.add((i, j, k))

    expected = {
        (i, j, k)
        for i in range(grid.cell_resolution)
        for j in range(grid.cell_resolution)
        for k in range(grid.cell_resolution)
    }
    assert covered == expected


def test_chunk_partition_handles_non_divisible_resolution() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=11,
        nchunks=3,
    )

    axis_widths = sorted(
        [
            chunk.x.cell_stop - chunk.x.cell_start
            for chunk in grid.iter_chunks()
            if chunk.index[1] == 0 and chunk.index[2] == 0
        ]
    )

    assert axis_widths == [3, 3, 4]


def test_chunk_sample_ranges_cover_owned_cells() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=10,
        nchunks=4,
    )

    for chunk in grid.iter_chunks():
        assert chunk.x.sample_start == chunk.x.cell_start
        assert chunk.x.sample_stop == chunk.x.cell_stop + 1
        assert chunk.y.sample_start == chunk.y.cell_start
        assert chunk.y.sample_stop == chunk.y.cell_stop + 1
        assert chunk.z.sample_start == chunk.z.cell_start
        assert chunk.z.sample_stop == chunk.z.cell_stop + 1


def test_chunk_world_bounds_match_sample_ranges() -> None:
    grid = VirtualGrid(
        origin=np.array([5.0, 6.0, 7.0]),
        box_size=8.0,
        resolution=9,
        nchunks=2,
    )
    chunk = next(c for c in grid.iter_chunks() if c.index == (1, 0, 0))

    bounds = chunk_world_bounds(grid, chunk)

    np.testing.assert_array_equal(bounds.sample_start, np.array([4, 0, 0]))
    np.testing.assert_array_equal(bounds.sample_stop, np.array([9, 5, 5]))
    np.testing.assert_allclose(
        bounds.local_start,
        np.array([4, 0, 0]) * grid.voxel_size,
    )
    np.testing.assert_allclose(
        bounds.local_stop,
        np.array([9, 5, 5]) * grid.voxel_size,
    )
    np.testing.assert_allclose(
        bounds.world_start,
        np.array([5.0, 6.0, 7.0]) + np.array([4, 0, 0]) * grid.voxel_size,
    )
    np.testing.assert_allclose(
        bounds.world_stop,
        np.array([5.0, 6.0, 7.0]) + np.array([9, 5, 5]) * grid.voxel_size,
    )


def test_iter_hard_chunk_bounds_cover_domain() -> None:
    grid = VirtualGrid(
        origin=np.array([1.0, 2.0, 3.0]),
        box_size=9.0,
        resolution=10,
        nchunks=3,
    )

    bounds = list(iter_hard_chunk_bounds(grid))

    assert len(bounds) == 27
    mins = np.min(np.array([b.world_start for b in bounds]), axis=0)
    maxs = np.max(np.array([b.world_stop for b in bounds]), axis=0)
    np.testing.assert_allclose(mins, grid.origin)
    np.testing.assert_allclose(maxs, grid.origin + grid.box_size)


def test_select_particles_in_hard_chunk_without_smoothing() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    bounds = next(
        b for b in iter_hard_chunk_bounds(grid) if b.index == (0, 0, 0)
    )
    coords = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.49, 0.49, 0.49],
            [0.70, 0.10, 0.10],
        ],
        dtype=np.float64,
    )

    indices = select_particles_in_hard_chunk(coords, bounds)

    np.testing.assert_array_equal(indices, np.array([0, 1]))


def test_select_particles_in_hard_chunk_with_smoothing_overlap() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    bounds = next(
        b for b in iter_hard_chunk_bounds(grid) if b.index == (0, 0, 0)
    )
    coords = np.array(
        [
            [0.60, 0.10, 0.10],
            [0.80, 0.80, 0.80],
        ],
        dtype=np.float64,
    )
    smoothing = np.array([0.15, 0.05], dtype=np.float64)

    indices = select_particles_in_hard_chunk(
        coords,
        bounds,
        smoothing_lengths=smoothing,
    )

    np.testing.assert_array_equal(indices, np.array([0]))


def test_voxelize_hard_chunk_uses_local_chunk_bounds() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    bounds = next(
        b for b in iter_hard_chunk_bounds(grid) if b.index == (0, 0, 0)
    )
    coords = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.20, 0.20, 0.20],
        ],
        dtype=np.float64,
    )
    data = np.array([1.0, 2.0], dtype=np.float64)

    chunk_grid, voxel_size = voxelize_hard_chunk(data, coords, bounds)

    assert chunk_grid.shape == bounds.shape
    assert np.isclose(voxel_size, bounds.extent[0] / bounds.shape[0])
    assert np.isclose(chunk_grid.sum(), data.sum())


def test_voxelize_hard_chunk_handles_non_cubic_chunk_shapes() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=11,
        nchunks=3,
    )
    bounds = next(
        b for b in iter_hard_chunk_bounds(grid) if b.index == (1, 0, 0)
    )
    coords = np.array(
        [
            bounds.local_start + 0.25 * bounds.extent,
            bounds.local_start + 0.75 * bounds.extent,
        ],
        dtype=np.float64,
    )
    data = np.array([1.0, 2.0], dtype=np.float64)

    chunk_grid, voxel_size = voxelize_hard_chunk(data, coords, bounds)

    assert chunk_grid.shape == bounds.shape
    assert np.isclose(chunk_grid.sum(), data.sum())
    np.testing.assert_allclose(
        bounds.extent / np.asarray(bounds.shape, dtype=np.float64),
        voxel_size,
    )


def test_crop_grid_to_chunk_bounds_restores_target_shape() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    target = next(
        b for b in iter_hard_chunk_bounds(grid) if b.index == (0, 0, 0)
    )
    expanded = expand_hard_chunk_bounds(grid, target, overlap_voxels=1)
    expanded_grid = np.zeros(expanded.shape, dtype=float)

    cropped = crop_grid_to_chunk_bounds(expanded_grid, expanded, target)

    assert cropped.shape == target.shape


def test_mesh_hard_chunk_sdf_places_mesh_in_world_space() -> None:
    grid = VirtualGrid(
        origin=np.array([10.0, 20.0, 30.0]),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    bounds = next(
        b for b in iter_hard_chunk_bounds(grid) if b.index == (0, 0, 0)
    )
    chunk_grid = np.zeros(bounds.shape, dtype=float)
    chunk_grid[1:4, 1:4, 1:4] = 1.0

    meshes = mesh_hard_chunk_sdf(chunk_grid, bounds, threshold=0.5)

    assert meshes
    trimesh_mesh = meshes[0].to_trimesh()
    assert trimesh_mesh.is_watertight
    assert np.all(trimesh_mesh.bounds[0] >= bounds.world_start)
    assert np.all(trimesh_mesh.bounds[1] <= bounds.world_stop + 1e-8)


def test_clip_mesh_to_hard_chunk_enforces_exact_bounds() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    bounds = next(
        b for b in iter_hard_chunk_bounds(grid) if b.index == (0, 0, 0)
    )
    raw = Mesh(mesh=trimesh.creation.box(extents=(0.8, 0.8, 0.8)))
    raw.vertices[:] = raw.vertices + np.array([0.4, 0.4, 0.4])

    clipped = clip_mesh_to_hard_chunk(raw, bounds).to_trimesh()

    assert clipped.is_watertight
    assert np.all(clipped.bounds[0] >= bounds.hard_world_start - 1e-8)
    assert np.all(clipped.bounds[1] <= bounds.hard_world_stop + 1e-8)


def test_clipped_overlap_chunk_reaches_owned_boundary() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=16,
        nchunks=2,
    )
    coords = np.array(
        [
            [(i + 0.5) / 16.0, (j + 0.5) / 16.0, (k + 0.5) / 16.0]
            for i in range(4, 12)
            for j in range(4, 12)
            for k in range(4, 12)
        ],
        dtype=np.float64,
    )
    data = np.ones(coords.shape[0], dtype=np.float64)

    chunk_meshes = generate_hard_chunk_meshes(
        data,
        coords,
        None,
        grid,
        threshold=0.5,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=0.0,
        overlap_voxels=1,
        clip_to_bounds=True,
    )

    bounds, meshes = next(
        (b, meshes) for b, meshes in chunk_meshes if b.index == (1, 0, 0)
    )
    clipped = meshes[0].to_trimesh()

    assert np.isclose(clipped.bounds[0, 0], bounds.hard_world_start[0])


def test_mesh_hard_chunk_sdf_returns_empty_for_empty_chunk() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    bounds = next(
        b for b in iter_hard_chunk_bounds(grid) if b.index == (0, 0, 0)
    )
    chunk_grid = np.zeros(bounds.shape, dtype=float)

    meshes = mesh_hard_chunk_sdf(chunk_grid, bounds, threshold=0.5)

    assert meshes == []


def test_generate_hard_chunk_meshes_returns_chunk_mesh_pairs() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    coords = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.20, 0.20, 0.20],
            [0.80, 0.80, 0.80],
        ],
        dtype=np.float64,
    )
    data = np.ones(coords.shape[0], dtype=np.float64)

    chunk_meshes = generate_hard_chunk_meshes(
        data,
        coords,
        None,
        grid,
        threshold=0.5,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=0.0,
    )

    assert chunk_meshes
    assert all(meshes for _bounds, meshes in chunk_meshes)
    for _bounds, meshes in chunk_meshes:
        for mesh in meshes:
            assert mesh.to_trimesh().is_watertight


def test_union_hard_chunk_meshes_returns_combined_mesh() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    bounds = list(iter_hard_chunk_bounds(grid))[:2]
    chunk_grid_1 = np.zeros(bounds[0].shape, dtype=float)
    chunk_grid_1[1:4, 1:4, 1:4] = 1.0
    chunk_grid_2 = np.zeros(bounds[1].shape, dtype=float)
    chunk_grid_2[1:4, 1:4, 1:4] = 1.0
    mesh1 = mesh_hard_chunk_sdf(chunk_grid_1, bounds[0], threshold=0.5)[0]
    mesh2 = mesh_hard_chunk_sdf(chunk_grid_2, bounds[1], threshold=0.5)[0]

    combined = union_hard_chunk_meshes(
        [(bounds[0], [mesh1]), (bounds[1], [mesh2])]
    ).to_trimesh()

    assert combined.vertices.shape[0] > 0


def test_union_hard_chunk_meshes_is_watertight_for_split_volume() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=16,
        nchunks=2,
    )
    coords = np.array(
        [
            [(i + 0.5) / 16.0, (j + 0.5) / 16.0, (k + 0.5) / 16.0]
            for i in range(4, 12)
            for j in range(4, 12)
            for k in range(4, 12)
        ],
        dtype=np.float64,
    )
    data = np.ones(coords.shape[0], dtype=np.float64)
    chunk_meshes = generate_hard_chunk_meshes(
        data,
        coords,
        None,
        grid,
        threshold=0.5,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=0.0,
        overlap_voxels=1,
    )

    combined = union_hard_chunk_meshes(chunk_meshes).to_trimesh()

    assert combined.is_watertight


def test_union_hard_chunk_meshes_is_watertight_after_seam_ownership() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=16,
        nchunks=2,
    )
    coords = np.array(
        [
            [(i + 0.5) / 16.0, (j + 0.5) / 16.0, (k + 0.5) / 16.0]
            for i in range(4, 12)
            for j in range(4, 12)
            for k in range(4, 12)
        ],
        dtype=np.float64,
    )
    data = np.ones(coords.shape[0], dtype=np.float64)

    chunk_meshes = generate_hard_chunk_meshes(
        data,
        coords,
        None,
        grid,
        threshold=0.5,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=0.0,
        overlap_voxels=1,
    )
    combined = union_hard_chunk_meshes(chunk_meshes).to_trimesh()

    assert combined.is_watertight
    assert len(combined.split(only_watertight=False)) == 1


def test_generate_hard_chunk_meshes_parallel_matches_serial() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=16,
        nchunks=2,
    )
    coords = np.array(
        [
            [(i + 0.5) / 16.0, (j + 0.5) / 16.0, (k + 0.5) / 16.0]
            for i in range(4, 12)
            for j in range(4, 12)
            for k in range(4, 12)
        ],
        dtype=np.float64,
    )
    data = np.ones(coords.shape[0], dtype=np.float64)

    serial = generate_hard_chunk_meshes(
        data,
        coords,
        None,
        grid,
        threshold=0.5,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=0.0,
        nthreads=1,
        overlap_voxels=1,
        clip_to_bounds=True,
    )
    parallel = generate_hard_chunk_meshes(
        data,
        coords,
        None,
        grid,
        threshold=0.5,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=0.0,
        nthreads=2,
        overlap_voxels=1,
        clip_to_bounds=True,
    )

    assert [bounds.index for bounds, _ in serial] == [
        bounds.index for bounds, _ in parallel
    ]
    assert [len(meshes) for _, meshes in serial] == [
        len(meshes) for _, meshes in parallel
    ]


def test_generate_hard_chunk_meshes_uses_single_deposition_thread_per_chunk(
    monkeypatch,
) -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=16,
        nchunks=2,
    )
    coords = np.array(
        [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        dtype=np.float64,
    )
    data = np.ones(coords.shape[0], dtype=np.float64)
    seen_nthreads = []

    def fake_voxelize_hard_chunk(
        data,
        coordinates,
        chunk_bounds,
        *,
        smoothing_lengths=None,
        nthreads=1,
    ):
        seen_nthreads.append(nthreads)
        return np.ones(chunk_bounds.shape, dtype=np.float64), 1.0 / 16.0

    def fake_mesh_hard_chunk_sdf(chunk_grid, chunk_bounds, *, threshold):
        mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        return [Mesh(mesh=mesh)]

    monkeypatch.setattr(
        "meshmerizer.chunks.hard.voxelize_hard_chunk",
        fake_voxelize_hard_chunk,
    )
    monkeypatch.setattr(
        "meshmerizer.chunks.hard.mesh_hard_chunk_sdf",
        fake_mesh_hard_chunk_sdf,
    )

    generate_hard_chunk_meshes(
        data,
        coords,
        None,
        grid,
        threshold=0.5,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=0.0,
        nthreads=4,
        overlap_voxels=1,
        clip_to_bounds=False,
    )

    assert seen_nthreads
    assert all(thread_count == 1 for thread_count in seen_nthreads)


def test_generate_hard_chunk_meshes_parallel_uses_completion_order(
    monkeypatch,
) -> None:
    """Ensure threaded chunk progress follows completion order.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        ``None``. Assertions verify completion-order result handling.
    """
    # Use deterministic per-chunk delays so the test can observe whether the
    # threaded path handles finished work or submission order.
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=4,
        nchunks=2,
    )
    coords = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    data = np.ones(coords.shape[0], dtype=np.float64)
    handled_indices = []

    def fake_log_debug_status(operation, message, *, thread=None):
        """Capture chunk-level debug messages.

        Args:
            operation: Operation name passed by the implementation.
            message: Status message text.
            thread: Optional worker identifier.

        Returns:
            ``None``. Per-chunk indices are recorded for later assertions.
        """
        del thread
        if operation != "Meshing" or not message.startswith("("):
            return
        handled_indices.append(message.split(":", 1)[0])

    def fake_voxelize_hard_chunk(
        data,
        coordinates,
        chunk_bounds,
        *,
        smoothing_lengths=None,
        nthreads=1,
    ):
        """Return a trivial occupied chunk grid after a staged delay.

        Args:
            data: Chunk-local particle data.
            coordinates: Chunk-local particle coordinates.
            chunk_bounds: Bounds for the requested chunk.
            smoothing_lengths: Optional chunk-local smoothing lengths.
            nthreads: Number of local deposition threads.

        Returns:
            Tuple containing the trivial chunk grid and voxel size.
        """
        import time

        del data, coordinates, smoothing_lengths, nthreads
        if chunk_bounds.index == (0, 0, 0):
            time.sleep(0.05)
        else:
            time.sleep(0.0)
        return np.ones(chunk_bounds.shape, dtype=np.float64), 0.25

    def fake_mesh_hard_chunk_sdf(chunk_grid, chunk_bounds, *, threshold):
        """Return one trivial mesh for every processed chunk.

        Args:
            chunk_grid: Chunk voxel grid.
            chunk_bounds: Bounds for the processed chunk.
            threshold: Extraction threshold.

        Returns:
            List containing one trivial mesh.
        """
        del chunk_grid, chunk_bounds, threshold
        mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        return [Mesh(mesh=mesh)]

    monkeypatch.setattr(
        "meshmerizer.chunks.hard.voxelize_hard_chunk",
        fake_voxelize_hard_chunk,
    )
    monkeypatch.setattr(
        "meshmerizer.chunks.hard.mesh_hard_chunk_sdf",
        fake_mesh_hard_chunk_sdf,
    )
    monkeypatch.setattr(
        "meshmerizer.chunks.hard.log_debug_status",
        fake_log_debug_status,
    )

    with cli_logging_context():
        generate_hard_chunk_meshes(
            data,
            coords,
            None,
            grid,
            threshold=0.5,
            preprocess="none",
            clip_halos=None,
            gaussian_sigma=0.0,
            nthreads=2,
            overlap_voxels=0,
            clip_to_bounds=False,
        )

    assert handled_indices
    assert handled_indices[0] != "(0, 0, 0)"


def test_bucket_point_particles_into_hard_chunks_matches_chunk_selection() -> (
    None
):
    """Ensure direct point bucketing matches per-chunk hard selection.

    Returns:
        ``None``. Assertions verify the fast path matches the existing logic.
    """
    # Compare the direct ownership bucketing against the slower chunk-by-chunk
    # selector in the simple point-particle case where the optimization
    # applies.
    from meshmerizer.chunks.hard import (
        _bucket_point_particles_into_hard_chunks,
    )

    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    coords = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.20, 0.20, 0.20],
            [0.74, 0.10, 0.10],
            [0.80, 0.80, 0.80],
            [0.999, 0.999, 0.999],
        ],
        dtype=np.float64,
    )

    buckets = _bucket_point_particles_into_hard_chunks(coords, grid)
    expected = {}
    for bounds in iter_hard_chunk_bounds(grid):
        indices = select_particles_in_hard_chunk(coords, bounds)
        if indices.size > 0:
            expected[bounds.index] = indices

    assert buckets.keys() == expected.keys()
    for chunk_index, indices in expected.items():
        assert np.array_equal(buckets[chunk_index], indices)


def test_generate_hard_chunk_meshes_empty_input_returns_empty_list(
    monkeypatch,
) -> None:
    """Ensure completely empty particle input short-circuits chunk processing.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        ``None``. Assertions verify no chunk work is attempted.
    """
    # The global empty-input fast path should return before any chunk-level
    # voxelization or meshing helpers are called.
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    called = {"voxelize": False}

    def fake_voxelize_hard_chunk(*args, **kwargs):
        """Fail if chunk voxelization is reached unexpectedly.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Never returns.

        Raises:
            AssertionError: Always, because this path should not run.
        """
        del args, kwargs
        called["voxelize"] = True
        raise AssertionError("voxelize_hard_chunk should not be called")

    monkeypatch.setattr(
        "meshmerizer.chunks.hard.voxelize_hard_chunk",
        fake_voxelize_hard_chunk,
    )

    chunk_meshes = generate_hard_chunk_meshes(
        np.array([], dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
        None,
        grid,
        threshold=0.5,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=0.0,
        nthreads=2,
        overlap_voxels=0,
        clip_to_bounds=False,
    )

    assert chunk_meshes == []
    assert called["voxelize"] is False


def test_occupied_chunk_mask_marks_only_touched_chunks() -> None:
    """Ensure the occupancy prepass excludes obviously untouched chunks.

    Returns:
        ``None``. Assertions verify occupied chunks are conservatively marked.
    """
    # The conservative occupancy prepass should mark chunks touched by particle
    # support while leaving distant chunks unmarked.
    from meshmerizer.chunks.hard import _occupied_chunk_mask

    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    coords = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.80, 0.80, 0.80],
        ],
        dtype=np.float64,
    )
    smoothing_lengths = np.array([0.01, 0.01], dtype=np.float64)

    occupied = _occupied_chunk_mask(
        coords,
        grid,
        smoothing_lengths=smoothing_lengths,
        overlap_voxels=0,
    )

    assert occupied[0, 0, 0]
    assert occupied[1, 1, 1]
    assert not occupied[0, 0, 1]
    assert not occupied[1, 0, 0]


def test_build_chunk_particle_index_matches_exact_selection() -> None:
    """Ensure the one-pass chunk index matches exact per-chunk selection.

    Returns:
        ``None``. Assertions verify the precomputed lists match the exact
        geometric selector.
    """
    # Compare the new one-pass chunk association against the historical exact
    # per-chunk selector for a smoothed-overlap configuration.
    from meshmerizer.chunks.hard import _build_chunk_particle_index

    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    coords = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.49, 0.49, 0.49],
            [0.60, 0.10, 0.10],
            [0.80, 0.80, 0.80],
        ],
        dtype=np.float64,
    )
    smoothing = np.array([0.01, 0.06, 0.15, 0.05], dtype=np.float64)
    overlap_voxels = 1

    offsets, particle_indices = _build_chunk_particle_index(
        coords,
        grid,
        smoothing_lengths=smoothing,
        overlap_voxels=overlap_voxels,
    )

    expected = {}
    for bounds in iter_hard_chunk_bounds(grid):
        expanded = expand_hard_chunk_bounds(
            grid,
            bounds,
            overlap_voxels=overlap_voxels,
        )
        indices = select_particles_in_hard_chunk(
            coords,
            expanded,
            smoothing_lengths=smoothing,
        )
        if indices.size > 0:
            expected[bounds.index] = indices

    actual_keys = set()
    for ix in range(grid.nchunks):
        for iy in range(grid.nchunks):
            for iz in range(grid.nchunks):
                flat_index = (ix * grid.nchunks + iy) * grid.nchunks + iz
                if offsets[flat_index + 1] > offsets[flat_index]:
                    actual_keys.add((ix, iy, iz))

    assert actual_keys == set(expected.keys())
    for chunk_key, indices in expected.items():
        flat_index = (
            chunk_key[0] * grid.nchunks + chunk_key[1]
        ) * grid.nchunks + chunk_key[2]
        actual_indices = particle_indices[
            offsets[flat_index] : offsets[flat_index + 1]
        ]
        assert np.array_equal(actual_indices, indices)


def test_build_chunk_particle_index_returns_csr_layout() -> None:
    """Ensure the chunk index builder returns consistent CSR arrays.

    Returns:
        ``None``. Assertions verify the returned CSR layout.
    """
    # The chunk association builder should return offsets and payload arrays in
    # a deterministic CSR-style layout for later chunk lookup.
    from meshmerizer.chunks.hard import _build_chunk_particle_index

    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    coords = np.array([[0.10, 0.10, 0.10], [0.80, 0.80, 0.80]])

    offsets, particle_indices = _build_chunk_particle_index(
        coords,
        grid,
        smoothing_lengths=None,
        overlap_voxels=0,
    )

    assert offsets.shape == (grid.nchunks**3 + 1,)
    assert particle_indices.ndim == 1
    assert np.all(offsets[1:] >= offsets[:-1])


def test_build_chunk_particle_index_uses_c_extension_when_available(
    monkeypatch,
) -> None:
    """Ensure the wrapper uses the C builder when the extension exposes it.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        ``None``. Assertions verify the C builder is called.
    """
    # The Python wrapper should prefer the extension-backed builder when it is
    # available, while preserving the same returned CSR shape.
    from meshmerizer.chunks import hard as hard_module

    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    coords = np.array([[0.10, 0.10, 0.10]], dtype=np.float64)
    called = {"value": False}

    class FakeVoxelizeModule:
        """Stub extension exposing the chunk-index builder."""

        def build_chunk_particle_index(
            self,
            coordinates,
            support_radius,
            lower_bounds,
            upper_bounds,
            nchunks,
        ):
            """Return a trivial CSR layout.

            Args:
                coordinates: Particle coordinates.
                support_radius: Support radius array.
                lower_bounds: Lower chunk bounds.
                upper_bounds: Upper chunk bounds.
                nchunks: Number of chunks per axis.

            Returns:
                Tuple of CSR arrays.
            """
            del (
                coordinates,
                support_radius,
                lower_bounds,
                upper_bounds,
                nchunks,
            )
            called["value"] = True
            offsets = np.zeros(grid.nchunks**3 + 1, dtype=np.int64)
            offsets[1:] = 1
            return offsets, np.array([0], dtype=np.int64)

    monkeypatch.setattr(hard_module, "_voxelize", FakeVoxelizeModule())

    offsets, particle_indices = hard_module._build_chunk_particle_index(
        coords,
        grid,
        smoothing_lengths=np.array([0.01], dtype=np.float64),
        overlap_voxels=0,
    )

    assert called["value"] is True
    assert offsets.shape == (grid.nchunks**3 + 1,)
    assert particle_indices.shape == (1,)


def test_generate_hard_chunk_meshes_avoids_exact_selector_scan(
    monkeypatch,
) -> None:
    """Ensure chunk meshing no longer calls the exact global selector.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        ``None``. Assertions verify the old exact selector path is bypassed.
    """
    # The chunk pipeline should now rely on the precomputed chunk index rather
    # than re-running the exact global selector for each chunk.
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    coords = np.array([[0.10, 0.10, 0.10]], dtype=np.float64)
    data = np.ones(coords.shape[0], dtype=np.float64)

    def fail_select_particles_in_hard_chunk(
        coordinates,
        chunk_bounds,
        *,
        smoothing_lengths=None,
    ):
        """Fail if the historical exact selector is reached.

        Args:
            coordinates: Particle coordinates.
            chunk_bounds: Chunk being processed.
            smoothing_lengths: Optional smoothing lengths.

        Raises:
            AssertionError: Always, because this path should not be used.
        """
        del coordinates, chunk_bounds, smoothing_lengths
        raise AssertionError("select_particles_in_hard_chunk should not run")

    def fake_voxelize_hard_chunk(
        data,
        coordinates,
        chunk_bounds,
        *,
        smoothing_lengths=None,
        nthreads=1,
    ):
        """Return one trivial occupied chunk grid.

        Args:
            data: Chunk-local particle data.
            coordinates: Chunk-local particle coordinates.
            chunk_bounds: Bounds for the requested chunk.
            smoothing_lengths: Optional chunk-local smoothing lengths.
            nthreads: Number of local deposition threads.

        Returns:
            Tuple containing the trivial chunk grid and voxel size.
        """
        del data, coordinates, smoothing_lengths, nthreads
        return np.ones(chunk_bounds.shape, dtype=np.float64), 0.25

    def fake_mesh_hard_chunk_sdf(chunk_grid, chunk_bounds, *, threshold):
        """Return no mesh so the test stays lightweight.

        Args:
            chunk_grid: Chunk voxel grid.
            chunk_bounds: Bounds for the processed chunk.
            threshold: Extraction threshold.

        Returns:
            Empty list.
        """
        del chunk_grid, chunk_bounds, threshold
        return []

    monkeypatch.setattr(
        "meshmerizer.chunks.hard.select_particles_in_hard_chunk",
        fail_select_particles_in_hard_chunk,
    )
    monkeypatch.setattr(
        "meshmerizer.chunks.hard.voxelize_hard_chunk",
        fake_voxelize_hard_chunk,
    )
    monkeypatch.setattr(
        "meshmerizer.chunks.hard.mesh_hard_chunk_sdf",
        fake_mesh_hard_chunk_sdf,
    )

    generate_hard_chunk_meshes(
        data,
        coords,
        np.array([0.01], dtype=np.float64),
        grid,
        threshold=0.5,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=0.0,
        nthreads=1,
        overlap_voxels=0,
        clip_to_bounds=False,
    )


def test_generate_hard_chunk_meshes_with_overlap_returns_meshes() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=16,
        nchunks=2,
    )
    coords = np.array(
        [
            [(i + 0.5) / 16.0, (j + 0.5) / 16.0, (k + 0.5) / 16.0]
            for i in range(4, 12)
            for j in range(4, 12)
            for k in range(4, 12)
        ],
        dtype=np.float64,
    )
    data = np.ones(coords.shape[0], dtype=np.float64)

    chunk_meshes = generate_hard_chunk_meshes(
        data,
        coords,
        None,
        grid,
        threshold=0.5,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=0.0,
        overlap_voxels=1,
    )

    assert chunk_meshes


def test_preprocess_chunk_grid_applies_gaussian_smoothing() -> None:
    grid = np.zeros((9, 9, 9), dtype=float)
    grid[4, 4, 4] = 1.0

    processed = preprocess_chunk_grid(
        grid,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=1.0,
    )

    assert processed[4, 4, 4] < 1.0
    assert processed[4, 4, 4] > 0.0
    assert np.isclose(processed.sum(), grid.sum())


def test_keep_largest_mesh_component_discards_small_island() -> None:
    large = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    small = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    small.apply_translation([5.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([large, small])

    filtered = keep_largest_mesh_component(Mesh(mesh=combined)).to_trimesh()

    assert len(filtered.split(only_watertight=False)) == 1
    assert filtered.bounds[1][0] < 2.0
