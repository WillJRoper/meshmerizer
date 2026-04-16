"""Tests for the adaptive C++ core scaffold and core utilities."""

from meshmerizer.adaptive_core import (
    adaptive_status,
    bounding_box_contains,
    bounding_box_overlaps,
    morton_decode_3d,
    morton_encode_3d,
    particle_fields,
    query_cell_contributors,
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
