from __future__ import annotations

import numpy as np
import trimesh

from meshmerizer.chunking import (
    VirtualGrid,
    assign_particles_to_chunks,
    chunk_origin,
    chunk_samples,
    chunk_sdf_to_mesh,
    chunk_world_bounds,
    combine_chunk_meshes,
    generate_chunk_grid,
    generate_chunked_mesh,
    iter_hard_chunk_bounds,
    keep_largest_mesh_component,
    particle_support_bounds,
    particle_voxel_indices,
    preprocess_chunk_grid,
    select_particles_in_hard_chunk,
    voxelize_hard_chunk,
)
from meshmerizer.mesh import Mesh
from meshmerizer.voxels import generate_voxel_grid


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


def test_chunk_samples_expand_with_halo() -> None:
    grid = VirtualGrid(
        origin=np.array([10.0, 20.0, 30.0]),
        box_size=8.0,
        resolution=9,
        nchunks=2,
    )
    chunk = next(c for c in grid.iter_chunks() if c.index == (1, 0, 0))

    samples = chunk_samples(grid, chunk, halo=2)

    np.testing.assert_array_equal(samples.owned_start, np.array([4, 0, 0]))
    np.testing.assert_array_equal(samples.owned_stop, np.array([9, 5, 5]))
    np.testing.assert_array_equal(samples.start, np.array([2, 0, 0]))
    np.testing.assert_array_equal(samples.stop, np.array([9, 7, 7]))
    np.testing.assert_allclose(
        chunk_origin(grid, samples),
        [10.0 + 2.0 * grid.voxel_size, 20.0, 30.0],
    )


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


def test_generate_chunk_grid_matches_full_grid_owned_region() -> None:
    coords = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.20, 0.20, 0.20],
            [0.55, 0.55, 0.55],
            [0.80, 0.80, 0.80],
        ],
        dtype=np.float64,
    )
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    full_grid, _ = generate_voxel_grid(
        data=data,
        coordinates=coords,
        resolution=8,
        box_size=1.0,
    )
    virtual_grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    chunk = next(c for c in virtual_grid.iter_chunks() if c.index == (0, 0, 0))

    local_grid, samples = generate_chunk_grid(
        data,
        coords,
        virtual_grid,
        chunk,
    )

    expected = full_grid[
        samples.owned_start[0] : samples.owned_stop[0],
        samples.owned_start[1] : samples.owned_stop[1],
        samples.owned_start[2] : samples.owned_stop[2],
    ]

    np.testing.assert_allclose(local_grid, expected)


def test_chunk_grid_smoothing_matches_region() -> None:
    coords = np.array(
        [
            [0.20, 0.20, 0.20],
            [0.45, 0.45, 0.45],
            [0.80, 0.80, 0.80],
        ],
        dtype=np.float64,
    )
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    smoothing_lengths = np.array([0.0, 0.15, 0.0], dtype=np.float64)

    full_grid, _ = generate_voxel_grid(
        data=data,
        coordinates=coords,
        resolution=8,
        smoothing_lengths=smoothing_lengths,
        box_size=1.0,
    )
    virtual_grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    chunk = next(c for c in virtual_grid.iter_chunks() if c.index == (0, 0, 0))

    local_grid, samples = generate_chunk_grid(
        data,
        coords,
        virtual_grid,
        chunk,
        halo=2,
        smoothing_lengths=smoothing_lengths,
    )

    expected = full_grid[
        samples.start[0] : samples.stop[0],
        samples.start[1] : samples.stop[1],
        samples.start[2] : samples.stop[2],
    ]

    np.testing.assert_allclose(local_grid, expected)


def test_particle_voxel_indices_map_to_virtual_grid() -> None:
    grid = VirtualGrid(
        origin=np.array([1.0, 2.0, 3.0]),
        box_size=8.0,
        resolution=8,
        nchunks=2,
    )
    coords = np.array(
        [
            [1.1, 2.1, 3.1],
            [8.9, 9.9, 10.9],
        ]
    )

    vox = particle_voxel_indices(coords, grid)

    np.testing.assert_array_equal(vox[0], np.array([0, 0, 0]))
    np.testing.assert_array_equal(vox[1], np.array([7, 7, 7]))


def test_particle_support_bounds_expand_with_smoothing_lengths() -> None:
    vox = np.array(
        [
            [0, 1, 2],
            [4, 5, 6],
        ],
        dtype=np.int64,
    )
    mins, maxs = particle_support_bounds(
        vox,
        np.array([1, 2], dtype=np.int64),
        resolution=8,
    )

    np.testing.assert_array_equal(mins[0], np.array([0, 0, 1]))
    np.testing.assert_array_equal(maxs[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(mins[1], np.array([2, 3, 4]))
    np.testing.assert_array_equal(maxs[1], np.array([6, 7, 7]))


def test_assign_particles_to_chunks_for_points() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    coords = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.80, 0.10, 0.10],
        ],
        dtype=np.float64,
    )

    assignments = assign_particles_to_chunks(coords, grid)

    np.testing.assert_array_equal(assignments[(0, 0, 0)], np.array([0]))
    np.testing.assert_array_equal(assignments[(1, 0, 0)], np.array([1]))
    assert assignments[(0, 1, 0)].size == 0


def test_assign_particles_to_chunks_with_smoothing_overlap() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=8,
        nchunks=2,
    )
    coords = np.array([[0.49, 0.10, 0.10]], dtype=np.float64)

    assignments = assign_particles_to_chunks(
        coords,
        grid,
        smoothing_lengths_vox=np.array([1], dtype=np.int64),
    )

    np.testing.assert_array_equal(assignments[(0, 0, 0)], np.array([0]))
    np.testing.assert_array_equal(assignments[(1, 0, 0)], np.array([0]))


def test_assign_particles_to_chunks_spans_multiple_chunk_ranges() -> None:
    grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=16,
        nchunks=4,
    )
    coords = np.array([[0.49, 0.49, 0.49]], dtype=np.float64)

    assignments = assign_particles_to_chunks(
        coords,
        grid,
        smoothing_lengths_vox=np.array([2], dtype=np.int64),
    )

    touched = {
        chunk_index
        for chunk_index, indices in assignments.items()
        if len(indices)
    }
    assert touched == {
        (1, 1, 1),
        (1, 1, 2),
        (1, 1, 3),
        (1, 2, 1),
        (1, 2, 2),
        (1, 2, 3),
        (1, 3, 1),
        (1, 3, 2),
        (1, 3, 3),
        (2, 1, 1),
        (2, 1, 2),
        (2, 1, 3),
        (2, 2, 1),
        (2, 2, 2),
        (2, 2, 3),
        (2, 3, 1),
        (2, 3, 2),
        (2, 3, 3),
        (3, 1, 1),
        (3, 1, 2),
        (3, 1, 3),
        (3, 2, 1),
        (3, 2, 2),
        (3, 2, 3),
        (3, 3, 1),
        (3, 3, 2),
        (3, 3, 3),
    }


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


def test_chunk_meshes_combine_into_watertight_surface() -> None:
    virtual_grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=16,
        nchunks=2,
    )
    volume = np.zeros((16, 16, 16), dtype=float)
    volume[3:13, 3:13, 3:13] = 1.0

    meshes = []
    for chunk in virtual_grid.iter_chunks():
        samples = chunk_samples(virtual_grid, chunk, halo=0)
        chunk_grid = volume[
            samples.start[0] : samples.stop[0],
            samples.start[1] : samples.stop[1],
            samples.start[2] : samples.stop[2],
        ]
        mesh = chunk_sdf_to_mesh(
            chunk_grid,
            samples,
            virtual_grid,
            threshold=0.5,
        )
        if mesh is not None:
            meshes.append(mesh)

    combined = combine_chunk_meshes(meshes).to_trimesh()

    assert combined.vertices.shape[0] > 0
    assert np.isfinite(combined.vertices).all()


def test_keep_largest_mesh_component_discards_small_island() -> None:
    large = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    small = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    small.apply_translation([5.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([large, small])

    filtered = keep_largest_mesh_component(Mesh(mesh=combined)).to_trimesh()

    assert len(filtered.split(only_watertight=False)) == 1
    assert filtered.bounds[1][0] < 2.0


def test_generate_chunked_mesh_remove_islands_keeps_main_component() -> None:
    main = [
        [x, y, z]
        for x in (0.18, 0.20, 0.22)
        for y in (0.18, 0.20, 0.22)
        for z in (0.18, 0.20, 0.22)
    ]
    island = [[0.80, 0.80, 0.80], [0.82, 0.80, 0.80], [0.80, 0.82, 0.80]]
    coords = np.array(main + island, dtype=np.float64)
    data = np.ones(coords.shape[0], dtype=np.float64)
    virtual_grid = VirtualGrid(
        origin=np.zeros(3),
        box_size=1.0,
        resolution=24,
        nchunks=2,
    )

    mesh = generate_chunked_mesh(
        data,
        coords,
        None,
        virtual_grid,
        threshold=0.5,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=0.5,
        remove_islands=True,
    ).to_trimesh()

    assert len(mesh.split(only_watertight=False)) == 1
    assert mesh.bounds[1][0] < 0.6


def test_chunked_mesh_bounds_match_across_chunk_counts() -> None:
    resolution = 24
    coords = np.array(
        [
            [
                (i + 0.5) / resolution,
                (j + 0.5) / resolution,
                (k + 0.5) / resolution,
            ]
            for i in range(6, 14)
            for j in range(6, 14)
            for k in range(6, 14)
        ],
        dtype=np.float64,
    )
    data = np.ones(coords.shape[0], dtype=np.float64)

    mesh_single = generate_chunked_mesh(
        data,
        coords,
        None,
        VirtualGrid(
            origin=np.zeros(3),
            box_size=1.0,
            resolution=resolution,
            nchunks=1,
        ),
        threshold=0.5,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=0.0,
    ).to_trimesh()
    mesh_split = generate_chunked_mesh(
        data,
        coords,
        None,
        VirtualGrid(
            origin=np.zeros(3),
            box_size=1.0,
            resolution=resolution,
            nchunks=2,
        ),
        threshold=0.5,
        preprocess="none",
        clip_halos=None,
        gaussian_sigma=0.0,
    ).to_trimesh()

    assert len(mesh_single.split(only_watertight=False)) == 1
    assert len(mesh_split.split(only_watertight=False)) == 1
    np.testing.assert_allclose(
        mesh_split.bounds, mesh_single.bounds, atol=0.05
    )
