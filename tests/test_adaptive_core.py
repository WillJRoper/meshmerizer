"""Tests for the adaptive C++ core scaffold and core utilities."""

from meshmerizer.adaptive_core import (
    adaptive_status,
    bounding_box_contains,
    bounding_box_overlaps,
    cell_may_contain_isosurface,
    corner_sign_mask,
    create_child_cells,
    create_top_level_cells,
    filter_child_contributors,
    morton_decode_3d,
    morton_encode_3d,
    particle_fields,
    query_cell_contributors,
    refine_octree,
    top_level_bin_counts,
    wendland_c2_gradient,
    wendland_c2_value,
)


def test_adaptive_extension_scaffold_imports() -> None:
    """The adaptive C++ extension should import through its Python wrapper."""
    assert adaptive_status() == "adaptive core scaffold ready"


def test_morton_encode_decode_round_trip() -> None:
    """Morton helpers should round-trip representative coordinates exactly."""
    key = morton_encode_3d(7, 11, 13)
    assert morton_decode_3d(key) == (7, 11, 13)


def test_bounding_box_contains_uses_half_open_bounds() -> None:
    """Bounding-box containment should include min and exclude max."""
    assert bounding_box_contains(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5)
    )
    assert not bounding_box_contains(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.5, 0.5),
    )


def test_bounding_box_overlap_requires_positive_volume() -> None:
    """Boxes that only touch at a face should not count as overlapping."""
    assert bounding_box_overlaps(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.5, 0.5, 0.5),
        (1.5, 1.5, 1.5),
    )
    assert not bounding_box_overlaps(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (2.0, 1.0, 1.0),
    )


def test_particle_fields_document_minimal_payload() -> None:
    """The adaptive particle payload should expose its documented fields."""
    assert particle_fields() == ("position", "value", "smoothing_length", "id")


def test_wendland_c2_value_matches_expected_profile() -> None:
    """The unnormalized Wendland value should peak at one and vanish at `h`."""
    assert wendland_c2_value(0.0, 2.0) == 1.0
    assert wendland_c2_value(2.0, 2.0) == 0.0
    assert wendland_c2_value(3.0, 2.0) == 0.0


def test_wendland_c2_gradient_points_inward() -> None:
    """The kernel gradient should point back toward the particle center."""
    gradient = wendland_c2_gradient((0.5, 0.0, 0.0), 1.0)
    assert gradient[0] < 0.0
    assert gradient[1] == 0.0
    assert gradient[2] == 0.0


def test_wendland_c2_gradient_vanishes_at_center_and_support() -> None:
    """The kernel gradient should vanish at the center and support edge."""
    assert wendland_c2_gradient((0.0, 0.0, 0.0), 1.0) == (0.0, 0.0, 0.0)
    assert wendland_c2_gradient((1.0, 0.0, 0.0), 1.0) == (0.0, 0.0, 0.0)


def test_top_level_bin_counts_tracks_particle_ownership() -> None:
    """Top-level particle binning should count particles in row-major bins."""
    counts = top_level_bin_counts(
        positions=[(0.1, 0.1, 0.1), (0.6, 0.1, 0.1), (0.6, 0.6, 0.6)],
        domain_minimum=(0.0, 0.0, 0.0),
        domain_maximum=(1.0, 1.0, 1.0),
        resolution=2,
    )
    assert counts == (1, 0, 0, 0, 1, 0, 0, 1)


def test_query_cell_contributors_filters_by_support_overlap() -> None:
    """Contributor queries should return only overlapping particles."""
    contributors = query_cell_contributors(
        positions=[(0.1, 0.1, 0.1), (0.75, 0.75, 0.75), (0.9, 0.1, 0.1)],
        smoothing_lengths=[0.05, 0.2, 0.05],
        domain_minimum=(0.0, 0.0, 0.0),
        domain_maximum=(1.0, 1.0, 1.0),
        resolution=2,
        cell_minimum=(0.5, 0.5, 0.5),
        cell_maximum=(1.0, 1.0, 1.0),
    )
    assert contributors == (1,)


def test_query_cell_contributors_includes_wide_support_particles() -> None:
    """Contributor queries should include particles outside the cell bins."""
    contributors = query_cell_contributors(
        positions=[(0.2, 0.2, 0.2), (0.1, 0.5, 0.5)],
        smoothing_lengths=[0.05, 0.45],
        domain_minimum=(0.0, 0.0, 0.0),
        domain_maximum=(1.0, 1.0, 1.0),
        resolution=4,
        cell_minimum=(0.5, 0.4, 0.4),
        cell_maximum=(0.75, 0.6, 0.6),
    )
    assert contributors == (1,)


def test_cell_may_contain_isosurface_detects_corner_straddle() -> None:
    """Corner values crossing the isovalue should trigger refinement."""
    assert cell_may_contain_isosurface(
        [0.0, 0.2, 0.1, 0.3, 0.8, 0.9, 1.1, 1.2],
        0.5,
    )
    assert not cell_may_contain_isosurface([0.1] * 8, 0.5)


def test_corner_sign_mask_tracks_positive_corners() -> None:
    """Corner sign masks should set bits for samples above the isovalue."""
    assert (
        corner_sign_mask([0.0, 0.6, 0.7, 0.2, 0.8, 0.9, 0.1, 0.0], 0.5) == 54
    )


def test_create_top_level_cells_returns_row_major_cells() -> None:
    """Top-level cell creation should emit deterministic row-major cells."""
    cells = create_top_level_cells((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2)
    assert len(cells) == 8
    assert cells[0]["morton_key"] == morton_encode_3d(0, 0, 0)
    assert cells[-1]["morton_key"] == morton_encode_3d(1, 1, 1)
    assert cells[0]["bounds"] == ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
    assert cells[-1]["bounds"] == ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))


def test_create_top_level_cells_rejects_zero_resolution() -> None:
    """Top-level cell creation should reject a zero base resolution."""
    import pytest

    with pytest.raises(ValueError, match="base_resolution"):
        create_top_level_cells((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0)


def test_create_child_cells_splits_bounds_and_depth_consistently() -> None:
    """Child creation should halve the bounds and increment the depth."""
    children = create_child_cells(
        morton_key=morton_encode_3d(1, 2, 3),
        bounds=((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
        depth=4,
    )

    assert len(children) == 8
    assert children[0]["depth"] == 5
    assert children[0]["morton_key"] == morton_encode_3d(2, 4, 6)
    assert children[-1]["morton_key"] == morton_encode_3d(3, 5, 7)
    assert children[0]["bounds"] == ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert children[-1]["bounds"] == ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))


def test_filter_child_contributors_preserves_overlap_across_children() -> None:
    """Wide-support parent contributors should attach to multiple children."""
    child_contributors = filter_child_contributors(
        parent_contributors=[0, 1],
        positions=[(0.25, 0.25, 0.25), (0.5, 0.5, 0.5)],
        smoothing_lengths=[0.05, 0.45],
        parent_bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )

    assert child_contributors[0] == (0, 1)
    assert child_contributors[7] == (1,)
    assert (
        sum(1 for contributors in child_contributors if 1 in contributors) > 1
    )


def test_refine_octree_returns_both_cells_and_contributors() -> None:
    """Refinement should return a tuple of cells and contributor indices."""
    cells = create_top_level_cells((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2)

    for cell in cells:
        cell["contributor_begin"] = 0
        cell["contributor_end"] = 0

    result = refine_octree(
        cells,
        positions=[],
        smoothing_lengths=[],
        isovalue=0.5,
        max_depth=3,
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    refined_cells, contributors = result
    assert isinstance(refined_cells, (list, tuple))
    assert isinstance(contributors, (list, tuple))


def test_refine_octree_empty_particles_no_surface() -> None:
    """Empty particle list should produce no surface cells."""
    cells = create_top_level_cells((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2)

    for cell in cells:
        cell["contributor_begin"] = 0
        cell["contributor_end"] = 0

    refined_cells, contributors = refine_octree(
        cells,
        positions=[],
        smoothing_lengths=[],
        isovalue=0.5,
        max_depth=3,
    )

    active_cells = [c for c in refined_cells if c.get("has_surface")]
    assert len(active_cells) == 0


def test_refine_octree_respects_max_depth() -> None:
    """Refinement should stop at the specified maximum depth."""
    cells = create_top_level_cells((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2)

    for cell in cells:
        cell["contributor_begin"] = 0
        cell["contributor_end"] = 1

    refined_cells, contributors = refine_octree(
        cells,
        positions=[(0.5, 0.5, 0.5)],
        smoothing_lengths=[0.6],
        isovalue=0.5,
        max_depth=1,
    )

    max_depth_found = max(
        (c.get("depth", 0) for c in refined_cells), default=0
    )
    assert max_depth_found <= 1


def test_refine_octree_empty_initial_cells() -> None:
    """Empty initial cells should return empty results."""
    result = refine_octree(
        [],
        positions=[(0.5, 0.5, 0.5)],
        smoothing_lengths=[0.6],
        isovalue=0.5,
        max_depth=3,
    )

    assert result == ([], [])
