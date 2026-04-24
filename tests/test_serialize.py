"""Tests for HDF5 serialization of the adaptive octree."""

import os
import tempfile

import numpy as np

from meshmerizer.adaptive_core import (
    create_top_level_cells,
    refine_octree,
    solve_vertices,
)
from meshmerizer.io.octree import export_octree, import_octree


def _build_sphere_octree(
    base_resolution=4,
    max_depth=2,
    isovalue=0.5,
):
    """Build a refined octree from a cluster of particles.

    Mirrors the helper in test_adaptive_core.py.
    """
    positions = [
        (0.45, 0.5, 0.5),
        (0.55, 0.5, 0.5),
        (0.5, 0.45, 0.5),
        (0.5, 0.55, 0.5),
        (0.5, 0.5, 0.45),
        (0.5, 0.5, 0.55),
    ]
    smoothing_lengths = [0.8] * len(positions)
    domain_min = (-1.0, -1.0, -1.0)
    domain_max = (2.0, 2.0, 2.0)

    top_cells = create_top_level_cells(domain_min, domain_max, base_resolution)
    initial_cells = []
    for cell in top_cells:
        cell_dict = dict(cell)
        cell_dict["contributor_begin"] = 0
        cell_dict["contributor_end"] = len(positions)
        initial_cells.append(cell_dict)

    cells, contributors = refine_octree(
        initial_cells,
        positions,
        smoothing_lengths,
        isovalue,
        max_depth,
        domain=(domain_min, domain_max),
        base_resolution=base_resolution,
    )

    return (
        cells,
        contributors,
        positions,
        smoothing_lengths,
        isovalue,
        domain_min,
        domain_max,
        max_depth,
        base_resolution,
    )


def test_round_trip_without_mesh():
    """Export then import an octree without mesh data."""
    (
        cells,
        contributors,
        positions,
        smoothing_lengths,
        isovalue,
        domain_min,
        domain_max,
        max_depth,
        base_resolution,
    ) = _build_sphere_octree()

    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        path = f.name

    try:
        export_octree(
            path,
            isovalue=isovalue,
            base_resolution=base_resolution,
            max_depth=max_depth,
            domain_minimum=domain_min,
            domain_maximum=domain_max,
            positions=positions,
            smoothing_lengths=smoothing_lengths,
            cells=cells,
            contributors=contributors,
            version="test-0.1",
        )

        result = import_octree(path)

        # Metadata
        assert result["isovalue"] == isovalue
        assert result["base_resolution"] == base_resolution
        assert result["max_depth"] == max_depth
        assert result["domain_minimum"] == domain_min
        assert result["domain_maximum"] == domain_max
        assert result["kernel_type"] == "wendland_c2"
        assert result["version"] == "test-0.1"

        # Particles
        assert len(result["positions"]) == len(positions)
        for orig, loaded in zip(positions, result["positions"]):
            assert orig == loaded

        assert result["smoothing_lengths"] == list(smoothing_lengths)

        # Contributors
        assert result["contributors"] == list(contributors)

        # Cells
        assert len(result["cells"]) == len(cells)
        for orig, loaded in zip(cells, result["cells"]):
            assert loaded["morton_key"] == orig["morton_key"]
            assert loaded["depth"] == orig["depth"]
            assert loaded["bounds"] == orig["bounds"]
            assert bool(loaded["is_leaf"]) == bool(orig["is_leaf"])
            assert bool(loaded["is_active"]) == bool(orig["is_active"])
            assert bool(loaded["has_surface"]) == bool(orig["has_surface"])
            assert loaded["child_begin"] == orig["child_begin"]
            assert loaded["corner_sign_mask"] == orig["corner_sign_mask"]
            assert loaded["corner_values"] == orig["corner_values"]
            assert loaded["contributors"] == orig["contributors"]

        # No QEF vertices
        assert result["vertices"] is None
        assert result["normals"] is None
        assert result["group_labels"] is None
    finally:
        os.unlink(path)


def test_round_trip_with_mesh():
    """Export then import an octree with QEF vertex data."""
    (
        cells,
        contributors,
        positions,
        smoothing_lengths,
        isovalue,
        domain_min,
        domain_max,
        max_depth,
        base_resolution,
    ) = _build_sphere_octree()

    vert_positions, vert_normals = solve_vertices(
        cells,
        contributors,
        positions,
        smoothing_lengths,
        isovalue,
        domain_min,
        domain_max,
        max_depth,
        base_resolution,
    )

    # Fake group labels for testing.
    n_verts = len(vert_positions)
    group_labels = np.zeros(n_verts, dtype=np.int64)

    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        path = f.name

    try:
        export_octree(
            path,
            isovalue=isovalue,
            base_resolution=base_resolution,
            max_depth=max_depth,
            domain_minimum=domain_min,
            domain_maximum=domain_max,
            positions=positions,
            smoothing_lengths=smoothing_lengths,
            cells=cells,
            contributors=contributors,
            vertices=vert_positions,
            normals=vert_normals,
            group_labels=group_labels,
            version="test-0.2",
        )

        result = import_octree(path)

        # Check vertices are present as NumPy arrays.
        assert result["vertices"] is not None
        assert result["normals"] is not None
        assert result["group_labels"] is not None
        assert len(result["vertices"]) == n_verts

        # Verify positions and normals are close.
        np.testing.assert_allclose(
            result["vertices"],
            np.asarray(vert_positions),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result["normals"],
            np.asarray(vert_normals),
            atol=1e-12,
        )
        np.testing.assert_array_equal(result["group_labels"], group_labels)
    finally:
        os.unlink(path)


def test_round_trip_mesh_usable_for_solve_vertices():
    """Imported octree can be fed back into solve_vertices."""
    (
        cells,
        contributors,
        positions,
        smoothing_lengths,
        isovalue,
        domain_min,
        domain_max,
        max_depth,
        base_resolution,
    ) = _build_sphere_octree()

    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        path = f.name

    try:
        export_octree(
            path,
            isovalue=isovalue,
            base_resolution=base_resolution,
            max_depth=max_depth,
            domain_minimum=domain_min,
            domain_maximum=domain_max,
            positions=positions,
            smoothing_lengths=smoothing_lengths,
            cells=cells,
            contributors=contributors,
        )

        result = import_octree(path)

        # Use imported data to solve vertices
        verts_p, verts_n = solve_vertices(
            result["cells"],
            result["contributors"],
            result["positions"],
            result["smoothing_lengths"],
            result["isovalue"],
            result["domain_minimum"],
            result["domain_maximum"],
            result["max_depth"],
            result["base_resolution"],
        )

        # Original vertices
        orig_verts_p, orig_verts_n = solve_vertices(
            cells,
            contributors,
            positions,
            smoothing_lengths,
            isovalue,
            domain_min,
            domain_max,
            max_depth,
            base_resolution,
        )

        # Must produce identical results.
        assert len(verts_p) == len(orig_verts_p)
    finally:
        os.unlink(path)


def test_empty_octree_round_trip():
    """An octree with no cells round-trips correctly."""
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        path = f.name

    try:
        export_octree(
            path,
            isovalue=0.5,
            base_resolution=2,
            max_depth=1,
            domain_minimum=(0.0, 0.0, 0.0),
            domain_maximum=(1.0, 1.0, 1.0),
            positions=[],
            smoothing_lengths=[],
            cells=[],
            contributors=[],
        )

        result = import_octree(path)
        assert len(result["cells"]) == 0
        assert len(result["contributors"]) == 0
        assert len(result["positions"]) == 0
    finally:
        os.unlink(path)
