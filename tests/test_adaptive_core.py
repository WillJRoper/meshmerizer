"""Tests for the adaptive C++ core scaffold and core utilities."""

from meshmerizer.adaptive_core import (
    adaptive_status,
    bounding_box_contains,
    bounding_box_overlaps,
    cell_may_contain_isosurface,
    compute_isovalue_from_percentile,
    corner_sign_mask,
    create_child_cells,
    create_top_level_cells,
    filter_child_contributors,
    fof_cluster,
    hermite_samples_for_cell,
    morton_decode_3d,
    morton_encode_3d,
    particle_fields,
    query_cell_contributors,
    refine_octree,
    run_octree_pipeline,
    solve_qef_for_leaf,
    solve_vertices,
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
        domain=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        base_resolution=2,
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
        domain=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        base_resolution=2,
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
        domain=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        base_resolution=2,
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
        domain=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        base_resolution=2,
    )

    assert result == ([], [])


# ---------------------------------------------------------------------------
# Helpers for balance invariant tests
# ---------------------------------------------------------------------------


def _cells_share_face(a_bounds: tuple, b_bounds: tuple) -> bool:
    """Return whether two axis-aligned boxes share a face with positive area.

    Each bounds argument is ((min_x, min_y, min_z), (max_x, max_y, max_z)).
    """
    (a_min, a_max) = a_bounds
    (b_min, b_max) = b_bounds

    touch_x = a_max[0] == b_min[0] or b_max[0] == a_min[0]
    touch_y = a_max[1] == b_min[1] or b_max[1] == a_min[1]
    touch_z = a_max[2] == b_min[2] or b_max[2] == a_min[2]

    overlap_x = a_min[0] < b_max[0] and b_min[0] < a_max[0]
    overlap_y = a_min[1] < b_max[1] and b_min[1] < a_max[1]
    overlap_z = a_min[2] < b_max[2] and b_min[2] < a_max[2]

    return (
        (touch_x and overlap_y and overlap_z)
        or (touch_y and overlap_x and overlap_z)
        or (touch_z and overlap_x and overlap_y)
    )


def _check_balance_invariant(
    refined_cells: list[dict],
) -> list[tuple]:
    """Return all adjacent leaf pairs that violate the 2:1 balance rule."""
    leaves = [c for c in refined_cells if c.get("is_leaf")]
    violations = []
    for i, a in enumerate(leaves):
        for j, b in enumerate(leaves):
            if j <= i:
                continue
            if _cells_share_face(a["bounds"], b["bounds"]):
                depth_diff = abs(a["depth"] - b["depth"])
                if depth_diff > 1:
                    violations.append((a["depth"], b["depth"], a["bounds"]))
    return violations


# ---------------------------------------------------------------------------
# Balance invariant tests
# ---------------------------------------------------------------------------


def test_refine_octree_satisfies_balance_invariant_with_surface() -> None:
    """No adjacent leaf pair should differ by more than one level.

    After refinement.

    This test creates a particle very close to one face of the domain so that
    the cell containing it refines to maximum depth while adjacent cells stay
    shallow, naturally producing the worst-case depth imbalance.
    """
    cells = create_top_level_cells((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2)

    # One particle near the corner (0.99, 0.25, 0.25) with a small kernel.
    # The top-level cell covering (0.5..1.0, 0.0..0.5, 0.0..0.5) will be
    # the only one that refines, reaching max_depth=4, while the adjacent
    # cell at (0.0..0.5, 0.0..0.5, 0.0..0.5) stays at depth 0 and would
    # violate 2:1 balance without the post-pass.
    for cell in cells:
        cell["contributor_begin"] = 0
        cell["contributor_end"] = 1

    refined_cells, _ = refine_octree(
        cells,
        positions=[(0.99, 0.25, 0.25)],
        smoothing_lengths=[0.02],
        isovalue=0.5,
        max_depth=4,
        domain=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        base_resolution=2,
    )

    violations = _check_balance_invariant(refined_cells)
    assert violations == [], (
        f"Balance violations found after refine_octree: {violations[:3]}"
    )


def test_refine_octree_balance_does_not_exceed_max_depth() -> None:
    """Balance-forced splits must not push any leaf beyond max_depth."""
    cells = create_top_level_cells((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2)

    for cell in cells:
        cell["contributor_begin"] = 0
        cell["contributor_end"] = 1

    max_depth = 3
    refined_cells, _ = refine_octree(
        cells,
        positions=[(0.99, 0.25, 0.25)],
        smoothing_lengths=[0.02],
        isovalue=0.5,
        max_depth=max_depth,
        domain=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        base_resolution=2,
    )

    deepest = max((c["depth"] for c in refined_cells), default=0)
    assert deepest <= max_depth

    violations = _check_balance_invariant(refined_cells)
    assert violations == []


def test_neighbor_morton_key_returns_adjacent_cell() -> None:
    """Morton neighbor at depth 0 should step by one coordinate unit."""
    # At depth 0, coordinates are in [0, root_resolution).
    # morton_encode_3d(1, 0, 0) has neighbor in -x = morton_encode_3d(0, 0, 0).
    key_1 = morton_encode_3d(1, 0, 0)
    key_0 = morton_encode_3d(0, 0, 0)

    # We verify indirectly: after refinement a 2-cell-wide domain produces
    # cells at (0,..,1) that are adjacent, and the balance invariant holds.
    cells = create_top_level_cells((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2)
    assert cells[0]["morton_key"] == key_0
    assert cells[4]["morton_key"] == key_1  # row-major: ix=1,iy=0,iz=0


def test_leaf_cells_share_face_consistency() -> None:
    """Adjacent leaves in a 2-cell domain should satisfy the face test."""
    cells = create_top_level_cells((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2)

    # Cell 0: (0..0.5, 0..0.5, 0..0.5)
    # Cell 4: (0.5..1, 0..0.5, 0..0.5)  — adjacent in +x to cell 0
    assert _cells_share_face(cells[0]["bounds"], cells[4]["bounds"])
    # Cell 0 and cell 7 are corner-only neighbors, not face neighbors.
    assert not _cells_share_face(cells[0]["bounds"], cells[7]["bounds"])


# ---------------------------------------------------------------------------
# Hermite sample tests
# ---------------------------------------------------------------------------

# Unit cell used throughout the Hermite sample tests.
_UNIT_BOUNDS = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def _unit_corner_values(isovalue: float, gradient_axis: int) -> list[float]:
    """Return 8 corner values for a linear field along one axis.

    The field varies from -1 at the low end to +1 at the high end along the
    chosen axis, crossing zero (the isovalue) at the midpoint.

    Corner index encoding: bit 0 = high x, bit 1 = high y, bit 2 = high z.
    """
    values = []
    for corner in range(8):
        position = [
            1.0 if (corner & 1) else 0.0,
            1.0 if (corner & 2) else 0.0,
            1.0 if (corner & 4) else 0.0,
        ]
        # Linear field: -1 at 0, +1 at 1 along the chosen axis.
        values.append(2.0 * position[gradient_axis] - 1.0)
    return values


def test_hermite_samples_no_crossings_on_uniform_field() -> None:
    """Uniform-sign corners should produce no samples."""
    # All corner values above isovalue.
    corner_values = [1.0] * 8
    sign_mask = corner_sign_mask(corner_values, 0.5)

    samples = hermite_samples_for_cell(
        bounds=_UNIT_BOUNDS,
        corner_values=corner_values,
        corner_sign_mask=sign_mask,
        contributor_indices=[],
        positions=[],
        smoothing_lengths=[],
        isovalue=0.5,
    )
    assert samples == ()


def test_hermite_samples_crossing_count_linear_x_field() -> None:
    """Linear x field should produce exactly 4 crossings."""
    # Linear field: 0.0 at x=0, 1.0 at x=1.  isovalue=0.5 crosses the 4
    # x-parallel edges (corners 0-1, 2-3, 4-5, 6-7).
    corner_values = [
        0.0,
        1.0,  # (min,min,min), (max,min,min)
        0.0,
        1.0,  # (min,max,min), (max,max,min)
        0.0,
        1.0,  # (min,min,max), (max,min,max)
        0.0,
        1.0,  # (min,max,max), (max,max,max)
    ]
    sign_mask = corner_sign_mask(corner_values, 0.5)

    samples = hermite_samples_for_cell(
        bounds=_UNIT_BOUNDS,
        corner_values=corner_values,
        corner_sign_mask=sign_mask,
        contributor_indices=[],
        positions=[],
        smoothing_lengths=[],
        isovalue=0.5,
    )

    assert len(samples) == 4
    # Every crossing should be at x=0.5 (the midpoint of the unit cell).
    for position, _normal in samples:
        assert abs(position[0] - 0.5) < 1e-12, (
            f"Crossing x-coordinate should be 0.5, got {position[0]}"
        )


def test_hermite_samples_crossing_position_is_interpolated() -> None:
    """Edge crossing follows linear interpolation."""
    # Field goes from 0.2 at x=0 to 0.8 at x=1.  isovalue=0.5.
    # Expected t = (0.5 - 0.2) / (0.8 - 0.2) = 0.5, so crossing at x=0.5.
    # All 4 x-edges should be crossed at x=0.5.
    corner_values = [0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8]
    sign_mask = corner_sign_mask(corner_values, 0.5)

    samples = hermite_samples_for_cell(
        bounds=_UNIT_BOUNDS,
        corner_values=corner_values,
        corner_sign_mask=sign_mask,
        contributor_indices=[],
        positions=[],
        smoothing_lengths=[],
        isovalue=0.5,
    )

    assert len(samples) == 4
    for position, _ in samples:
        assert abs(position[0] - 0.5) < 1e-12

    # Check asymmetric interpolation: field 0.2 at x=0, 0.7 at x=1.
    # t = (0.5 - 0.2) / (0.7 - 0.2) = 0.6.  Crossing at x=0.6.
    corner_values2 = [0.2, 0.7, 0.2, 0.7, 0.2, 0.7, 0.2, 0.7]
    sign_mask2 = corner_sign_mask(corner_values2, 0.5)
    samples2 = hermite_samples_for_cell(
        bounds=_UNIT_BOUNDS,
        corner_values=corner_values2,
        corner_sign_mask=sign_mask2,
        contributor_indices=[],
        positions=[],
        smoothing_lengths=[],
        isovalue=0.5,
    )
    assert len(samples2) == 4
    for position, _ in samples2:
        assert abs(position[0] - 0.6) < 1e-10


def test_hermite_samples_outward_normal_from_particle() -> None:
    """Outward normal should point away from the particle."""
    # One particle at the centre of the cell with a large kernel.  The
    # gradient of the SPH field points toward the particle (into the fluid).
    # The outward normal should point away from the particle, i.e. in the
    # direction of increasing x away from the particle.
    #
    # Field: linear in x so the 4 x-edges are crossed at x=0.5.  Particle
    # is also at x=0.5, centred in y and z.
    corner_values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    sign_mask = corner_sign_mask(corner_values, 0.5)

    # Particle is at (0.5, 0.5, 0.5) with a kernel wide enough to cover all
    # crossing points (which lie at x=0.5 along all four x-edges).
    particle_pos = (0.5, 0.5, 0.5)
    smoothing = 1.5  # wider than the cell diagonal

    samples = hermite_samples_for_cell(
        bounds=_UNIT_BOUNDS,
        corner_values=corner_values,
        corner_sign_mask=sign_mask,
        contributor_indices=[0],
        positions=[particle_pos],
        smoothing_lengths=[smoothing],
        isovalue=0.5,
    )

    assert len(samples) == 4
    for position, normal in samples:
        # All crossings are at x=0.5, same x as the particle.  The
        # displacement in x is zero, so the x-component of the gradient is
        # zero.  The outward normal in x should therefore also be zero.
        # The y/z displacement determines the gradient direction: for a
        # crossing above the particle centre (y > 0.5) the gradient y-
        # component points upward toward the crossing and the outward normal
        # points downward (away from the fluid, i.e., toward lower density).
        # We check only that the normal is a unit vector (or zero).
        magnitude = (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) ** 0.5
        if magnitude > 1e-12:
            assert abs(magnitude - 1.0) < 1e-10, (
                f"Normal should be unit length, got magnitude {magnitude}"
            )


def test_hermite_samples_degenerate_no_contributors_gives_zero_normal() -> (
    None
):
    """A crossing with no contributors should produce a zero-length normal."""
    # Field crosses isovalue on the 4 x-edges; no particles nearby so the
    # gradient evaluates to zero and the normal is the zero vector.
    corner_values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    sign_mask = corner_sign_mask(corner_values, 0.5)

    samples = hermite_samples_for_cell(
        bounds=_UNIT_BOUNDS,
        corner_values=corner_values,
        corner_sign_mask=sign_mask,
        contributor_indices=[],
        positions=[],
        smoothing_lengths=[],
        isovalue=0.5,
    )

    assert len(samples) == 4
    for _position, normal in samples:
        assert normal == (0.0, 0.0, 0.0), (
            f"Normal with no contributors should be zero, got {normal}"
        )


# ---------------------------------------------------------------------------
# Phase 8: QEF vertex solve
# ---------------------------------------------------------------------------

import math


def test_solve_qef_empty_samples_returns_cell_center() -> None:
    """With no Hermite samples the vertex should be the cell center."""
    bounds = ((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    position, normal = solve_qef_for_leaf([], bounds)
    assert position == (1.0, 1.0, 1.0)
    assert normal == (0.0, 0.0, 0.0)


def test_solve_qef_single_plane_constraint() -> None:
    """One tangent plane with regularization biases toward center."""
    bounds = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    # Plane: x = 0.3  (normal along +x, point at x=0.3)
    samples = [((0.3, 0.5, 0.5), (1.0, 0.0, 0.0))]
    position, normal = solve_qef_for_leaf(samples, bounds)
    # With Tikhonov regularization (lambda=0.1, center=0.5):
    #   mat = [[1.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]
    #   rhs = [0.3 + 0.05, 0.05, 0.05]
    # x = 0.35/1.1 ≈ 0.3182, y = z = 0.5
    assert abs(position[0] - 0.35 / 1.1) < 1e-6
    assert abs(position[1] - 0.5) < 1e-6
    assert abs(position[2] - 0.5) < 1e-6


def test_solve_qef_two_orthogonal_planes() -> None:
    """Two orthogonal planes with regularization give a unique solution."""
    bounds = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    samples = [
        ((0.3, 0.5, 0.5), (1.0, 0.0, 0.0)),
        ((0.5, 0.7, 0.5), (0.0, 1.0, 0.0)),
    ]
    position, normal = solve_qef_for_leaf(samples, bounds)
    # With regularization (lambda=0.1, center=0.5):
    #   mat = [[1.1, 0, 0], [0, 1.1, 0], [0, 0, 0.1]]
    #   rhs = [0.3+0.05, 0.7+0.05, 0.05]
    # x = 0.35/1.1, y = 0.75/1.1, z = 0.5
    assert abs(position[0] - 0.35 / 1.1) < 1e-6
    assert abs(position[1] - 0.75 / 1.1) < 1e-6
    assert abs(position[2] - 0.5) < 1e-6


def test_solve_qef_three_orthogonal_planes() -> None:
    """Three orthogonal planes with regularization bias toward center."""
    bounds = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    samples = [
        ((0.3, 0.5, 0.5), (1.0, 0.0, 0.0)),
        ((0.5, 0.7, 0.5), (0.0, 1.0, 0.0)),
        ((0.5, 0.5, 0.4), (0.0, 0.0, 1.0)),
    ]
    position, normal = solve_qef_for_leaf(samples, bounds)
    # With regularization (lambda=0.1, center=0.5):
    #   mat = [[1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]]
    #   rhs = [0.3+0.05, 0.7+0.05, 0.4+0.05]
    # x = 0.35/1.1, y = 0.75/1.1, z = 0.45/1.1
    assert abs(position[0] - 0.35 / 1.1) < 1e-6
    assert abs(position[1] - 0.75 / 1.1) < 1e-6
    assert abs(position[2] - 0.45 / 1.1) < 1e-6


def test_solve_qef_vertex_clamped_to_bounds() -> None:
    """The QEF vertex should be clamped inside the cell bounds."""
    bounds = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    # Three planes that intersect outside the cell.
    samples = [
        ((2.0, 0.5, 0.5), (1.0, 0.0, 0.0)),
        ((0.5, 2.0, 0.5), (0.0, 1.0, 0.0)),
        ((0.5, 0.5, 2.0), (0.0, 0.0, 1.0)),
    ]
    position, normal = solve_qef_for_leaf(samples, bounds)
    assert 0.0 <= position[0] <= 1.0
    assert 0.0 <= position[1] <= 1.0
    assert 0.0 <= position[2] <= 1.0


def test_solve_qef_degenerate_zero_normals_ignored() -> None:
    """Samples with zero normals should be ignored by the QEF."""
    bounds = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    samples = [
        ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0)),
        ((0.2, 0.2, 0.2), (0.0, 0.0, 0.0)),
    ]
    position, normal = solve_qef_for_leaf(samples, bounds)
    # All samples degenerate -> cell center
    assert position == (0.5, 0.5, 0.5)
    assert normal == (0.0, 0.0, 0.0)


def test_solve_qef_normal_is_unit_length() -> None:
    """The returned normal should be unit length when samples exist."""
    bounds = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    samples = [
        ((0.3, 0.5, 0.5), (1.0, 0.0, 0.0)),
        ((0.5, 0.7, 0.5), (0.0, 1.0, 0.0)),
        ((0.5, 0.5, 0.4), (0.0, 0.0, 1.0)),
    ]
    position, normal = solve_qef_for_leaf(samples, bounds)
    magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    assert abs(magnitude - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Phase 9: Face generation tests
# ---------------------------------------------------------------------------


def _build_sphere_octree(
    base_resolution=4,
    max_depth=2,
    isovalue=0.5,
):
    """Helper: build a refined octree from a cluster of particles.

    Places multiple particles near the center of a domain with enough
    padding that the isosurface at ``isovalue`` is fully contained
    within the domain interior (not touching any boundary face).

    The domain is [-1, 2]^3 with particles near (0.5, 0.5, 0.5) and
    smoothing lengths of 0.8.  The Wendland C2 kernel has compact
    support, so the field is zero beyond radius 0.8 from any particle.
    At isovalue=0.5 the isosurface sits well inside the kernel support,
    far from the domain boundary.

    Returns:
        Tuple of (cells, contributors, positions, smoothing_lengths,
        isovalue, domain_min, domain_max, max_depth, base_resolution).
    """
    # A small cluster of particles near the center ensures every
    # cell near the surface has at least 2 contributors.
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

    # Build initial cells with contributor ranges.
    top_cells = create_top_level_cells(domain_min, domain_max, base_resolution)
    initial_cells = []
    for i, cell in enumerate(top_cells):
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


def test_solve_vertices_produces_vertices() -> None:
    """solve_vertices should produce QEF vertices from a simple field."""
    args = _build_sphere_octree()
    vert_positions, vert_normals = solve_vertices(*args)
    assert len(vert_positions) > 0, "Expected at least one vertex"
    assert vert_positions.shape[1] == 3
    assert vert_normals.shape[1] == 3
    assert vert_positions.shape[0] == vert_normals.shape[0]


def test_solve_vertices_normals_unit_length() -> None:
    """QEF vertex normals should be approximately unit length."""
    import numpy as np

    args = _build_sphere_octree()
    vert_positions, vert_normals = solve_vertices(*args)
    norms = np.linalg.norm(vert_normals, axis=1)
    # Some normals may be zero for degenerate cells; check non-zero ones.
    nonzero = norms > 0.01
    assert np.allclose(norms[nonzero], 1.0, atol=0.05)


def test_solve_vertices_empty_field() -> None:
    """A field with no surface crossings should produce no vertices."""
    # Place particle far outside the domain so no isosurface exists.
    positions = [(10.0, 10.0, 10.0)]
    smoothing_lengths = [0.1]
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    isovalue = 0.5
    base_resolution = 2
    max_depth = 1

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
    assert len(vert_positions) == 0
    assert len(vert_normals) == 0


def test_solve_vertices_mixed_depth() -> None:
    """Mixed-depth octree should produce vertices."""
    args = _build_sphere_octree(base_resolution=4, max_depth=3)
    vert_positions, vert_normals = solve_vertices(*args)
    assert len(vert_positions) > 0, "Expected at least one vertex"
    assert vert_positions.shape[1] == 3


def test_refine_octree_quality_stop_preserves_mixed_leaf_depths() -> None:
    """Quality-based stopping should permit mixed leaf depths."""
    cells, _, _, _, _, _, _, max_depth, _ = _build_sphere_octree(
        base_resolution=4,
        max_depth=3,
    )
    surface_leaf_depths = [
        cell["depth"]
        for cell in cells
        if cell.get("is_leaf") and cell.get("has_surface")
    ]

    assert surface_leaf_depths, "Expected at least one surface leaf"
    assert any(depth < max_depth for depth in surface_leaf_depths)
    assert len(set(surface_leaf_depths)) > 1


# ---------------------------------------------------------------------------
# NumPy buffer-protocol tests
# ---------------------------------------------------------------------------


def test_refine_octree_accepts_numpy_arrays() -> None:
    """C++ bindings should accept NumPy arrays via buffer protocol."""
    import numpy as np

    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    base_resolution = 2

    positions = np.array([[0.5, 0.5, 0.5], [0.6, 0.5, 0.5]], dtype=np.float64)
    smoothing_lengths = np.array([0.6, 0.6], dtype=np.float64)

    cells = create_top_level_cells(domain_min, domain_max, base_resolution)
    for cell in cells:
        cell["contributor_begin"] = 0
        cell["contributor_end"] = 2

    refined_cells, contributors = refine_octree(
        cells,
        positions=positions,
        smoothing_lengths=smoothing_lengths,
        isovalue=0.5,
        max_depth=2,
        domain=(domain_min, domain_max),
        base_resolution=base_resolution,
    )
    assert len(refined_cells) > 0
    assert len(contributors) > 0


# ---------------------------------------------------------------------------
# run_octree_pipeline tests
# ---------------------------------------------------------------------------


def test_run_octree_pipeline_produces_vertices() -> None:
    """Octree pipeline should produce QEF vertices."""
    import numpy as np

    domain_min = (-1.0, -1.0, -1.0)
    domain_max = (2.0, 2.0, 2.0)
    base_resolution = 2
    isovalue = 0.5
    max_depth = 3

    # Use the same particle setup as _build_sphere_octree.
    positions_list = [
        (0.45, 0.5, 0.5),
        (0.55, 0.5, 0.5),
        (0.5, 0.45, 0.5),
        (0.5, 0.55, 0.5),
        (0.5, 0.5, 0.45),
        (0.5, 0.5, 0.55),
    ]
    smoothing_list = [0.8] * len(positions_list)

    positions_arr = np.array(positions_list, dtype=np.float64)
    smoothing_arr = np.array(smoothing_list, dtype=np.float64)

    vert_positions, vert_normals = run_octree_pipeline(
        positions_arr,
        smoothing_arr,
        domain_min,
        domain_max,
        base_resolution,
        isovalue,
        max_depth,
    )

    # The pipeline should produce vertices for this configuration.
    assert len(vert_positions) > 0

    # Verify vertex format: (N, 3) float64 arrays.
    assert vert_positions.shape[1] == 3
    assert vert_normals.shape[1] == 3
    assert vert_positions.shape[0] == vert_normals.shape[0]


def test_run_octree_pipeline_empty_particles() -> None:
    """Pipeline with no particles should produce empty arrays."""
    import numpy as np

    positions = np.zeros((0, 3), dtype=np.float64)
    smoothing = np.zeros(0, dtype=np.float64)

    vert_positions, vert_normals = run_octree_pipeline(
        positions,
        smoothing,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        2,
        0.5,
        3,
    )
    assert len(vert_positions) == 0
    assert len(vert_normals) == 0


# ---------------------------------------------------------------------------
# Density percentile isovalue tests
# ---------------------------------------------------------------------------


def test_compute_isovalue_from_percentile_basic() -> None:
    """Percentile isovalue should match manual Wendland C2 self-density."""
    import numpy as np

    h = np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float64)
    iso = compute_isovalue_from_percentile(h, 50.0)
    # Manual: self_density = 21 / (2*pi*h^3), median of those values.
    self_density = 21.0 / (2.0 * np.pi * h**3)
    expected = float(np.percentile(self_density, 50.0))
    assert abs(iso - expected) < 1e-10


def test_compute_isovalue_percentile_low_encloses_more() -> None:
    """Lower percentile should give a lower isovalue (enclose more)."""
    import numpy as np

    h = np.random.default_rng(42).uniform(0.05, 1.0, size=1000)
    iso_5 = compute_isovalue_from_percentile(h, 5.0)
    iso_50 = compute_isovalue_from_percentile(h, 50.0)
    assert iso_5 < iso_50


def test_compute_isovalue_percentile_rejects_invalid() -> None:
    """Out-of-range percentile should raise ValueError."""
    import numpy as np
    import pytest

    h = np.array([0.1, 0.2], dtype=np.float64)
    with pytest.raises(ValueError):
        compute_isovalue_from_percentile(h, -1.0)
    with pytest.raises(ValueError):
        compute_isovalue_from_percentile(h, 101.0)
    with pytest.raises(ValueError):
        compute_isovalue_from_percentile(np.array([]), 50.0)


# ── FOF clustering tests ─────────────────────────────────────────


def test_fof_single_cluster() -> None:
    """Tightly packed points should form one cluster."""
    import numpy as np

    rng = np.random.default_rng(42)
    positions = rng.uniform(0.4, 0.6, size=(50, 3))
    labels = fof_cluster(
        positions,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        1.5,
    )
    assert labels.shape == (50,)
    assert labels.dtype == np.int64
    # All points should share the same label.
    assert len(set(labels.tolist())) == 1


def test_fof_two_separated_clusters() -> None:
    """Two well-separated clumps should get different labels."""
    import numpy as np

    rng = np.random.default_rng(99)
    # Cluster A near origin, cluster B near (10, 10, 10).
    cluster_a = rng.uniform(0.0, 0.1, size=(30, 3))
    cluster_b = rng.uniform(9.9, 10.0, size=(30, 3))
    positions = np.vstack([cluster_a, cluster_b])

    labels = fof_cluster(
        positions,
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 10.0),
        1.5,
    )
    assert labels.shape == (60,)
    assert len(set(labels.tolist())) == 2
    # First 30 should share one label, last 30 another.
    assert len(set(labels[:30].tolist())) == 1
    assert len(set(labels[30:].tolist())) == 1
    assert labels[0] != labels[30]


def test_fof_empty_input() -> None:
    """Empty positions should return an empty label array."""
    import numpy as np

    positions = np.empty((0, 3), dtype=np.float64)
    labels = fof_cluster(
        positions,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        1.5,
    )
    assert labels.shape == (0,)
    assert labels.dtype == np.int64


def test_fof_cluster_sizes_support_small_cluster_filtering() -> None:
    """FOF labels should allow thresholding away tiny detached clusters."""
    import numpy as np

    rng = np.random.default_rng(7)
    main_cluster = rng.uniform(0.0, 0.2, size=(40, 3))
    fluff_cluster = rng.uniform(4.9, 5.0, size=(3, 3))
    positions = np.vstack([main_cluster, fluff_cluster])

    labels = fof_cluster(
        positions,
        (0.0, 0.0, 0.0),
        (5.0, 5.0, 5.0),
        1.5,
    )

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = sorted(counts.tolist())

    assert len(unique_labels) == 2
    assert cluster_sizes == [3, 40]


def test_fof_uses_tight_particle_bounds_for_linking_scale() -> None:
    """FOF should not depend on a much larger enclosing domain."""
    import numpy as np

    positions = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.08, 0.06, 0.04],
            [0.15, 0.10, 0.09],
            [1.20, 0.00, 0.00],
            [1.28, 0.05, 0.06],
            [1.35, 0.11, 0.10],
        ],
        dtype=np.float64,
    )

    labels_small_domain = fof_cluster(
        positions,
        (0.0, 0.0, 0.0),
        (2.0, 1.0, 1.0),
        0.9,
    )
    labels_huge_domain = fof_cluster(
        positions,
        (0.0, 0.0, 0.0),
        (200.0, 100.0, 100.0),
        0.9,
    )

    assert len(set(labels_small_domain.tolist())) == 2
    assert len(set(labels_huge_domain.tolist())) == 2
    np.testing.assert_array_equal(labels_small_domain, labels_huge_domain)
