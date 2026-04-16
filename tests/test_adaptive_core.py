"""Tests for the adaptive C++ core scaffold and core utilities."""

from meshmerizer.adaptive_core import (
    adaptive_status,
    bounding_box_contains,
    bounding_box_overlaps,
    morton_decode_3d,
    morton_encode_3d,
    particle_fields,
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
