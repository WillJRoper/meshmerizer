"""Tests for Phase 20a: B-spline basis and DOF infrastructure."""

import pytest

from meshmerizer.adaptive_core import (
    assign_dof_indices,
    bspline1d_derivative,
    bspline1d_evaluate,
    bspline3d_evaluate,
    bspline3d_gradient,
    enumerate_stencils,
)

# ---- 1-D B-spline tests ------------------------------------------


class TestBSpline1D:
    """Tests for the 1-D degree-2 (quadratic) B-spline."""

    def test_peak_at_zero(self) -> None:
        """B(0) should be 3/4."""
        assert bspline1d_evaluate(0.0) == pytest.approx(0.75)

    def test_zero_at_boundary(self) -> None:
        """B(±3/2) should be 0 (end of support)."""
        assert bspline1d_evaluate(1.5) == 0.0
        assert bspline1d_evaluate(-1.5) == 0.0

    def test_zero_outside_support(self) -> None:
        """B(t) = 0 for |t| > 3/2."""
        assert bspline1d_evaluate(2.0) == 0.0
        assert bspline1d_evaluate(-2.0) == 0.0

    def test_midpoint_values(self) -> None:
        """B(0.5) = 0.5 and B(-0.5) = 0.5."""
        assert bspline1d_evaluate(0.5) == pytest.approx(0.5)
        assert bspline1d_evaluate(-0.5) == pytest.approx(0.5)

    def test_symmetry(self) -> None:
        """B(t) should be symmetric: B(t) = B(-t)."""
        for t in [0.1, 0.3, 0.7, 0.99]:
            assert bspline1d_evaluate(t) == pytest.approx(
                bspline1d_evaluate(-t)
            )


class TestBSpline1DDerivative:
    """Tests for the 1-D B-spline derivative."""

    def test_positive_side(self) -> None:
        """Derivative matches the quadratic B-spline pieces for t > 0."""
        assert bspline1d_derivative(0.5) == -1.0
        assert bspline1d_derivative(0.1) == pytest.approx(-0.2)

    def test_negative_side(self) -> None:
        """Derivative matches the quadratic B-spline pieces for t < 0."""
        assert bspline1d_derivative(-0.5) == 1.0
        assert bspline1d_derivative(-0.1) == pytest.approx(0.2)

    def test_zero_at_cusp(self) -> None:
        """B'(0) = 0 (average of left/right derivatives)."""
        assert bspline1d_derivative(0.0) == 0.0

    def test_zero_outside_support(self) -> None:
        """B'(t) = 0 for |t| >= 3/2."""
        assert bspline1d_derivative(1.5) == 0.0
        assert bspline1d_derivative(-1.5) == 0.0
        assert bspline1d_derivative(2.0) == 0.0


# ---- 3-D B-spline tests ------------------------------------------


class TestBSpline3D:
    """Tests for the 3-D degree-2 (quadratic) tensor-product B-spline."""

    def test_peak_at_center(self) -> None:
        """B_3d(center) = (3/4)^3 = 27/64."""
        c = (1.0, 2.0, 3.0)
        assert bspline3d_evaluate(c, c, 1.0) == pytest.approx(27.0 / 64.0)

    def test_zero_outside_support(self) -> None:
        """B_3d should be 0 when any axis is outside support."""
        c = (0.0, 0.0, 0.0)
        # End of support in x: |t| = 3/2.
        assert bspline3d_evaluate((1.5, 0.0, 0.0), c, 1.0) == 0.0
        # Far away
        assert bspline3d_evaluate((5.0, 5.0, 5.0), c, 1.0) == 0.0

    def test_partition_of_unity(self) -> None:
        """Sum of all overlapping B-splines at any point equals 1.

        For degree-2 B-splines on a uniform grid with cell width w,
        the point (x, y, z) overlaps with at most 3^3 = 27 cell
        centres. The sum of all overlapping basis values should be 1.
        """
        w = 0.5
        # Pick a point well inside the evaluated centre range.
        p = (2.37, 2.61, 2.83)

        # Find the 8 surrounding cell centres on a grid of
        # spacing w. Cell centres are at (i+0.5)*w for integer i.
        total = 0.0
        for ix in range(10):
            cx = (ix + 0.5) * w
            for iy in range(10):
                cy = (iy + 0.5) * w
                for iz in range(10):
                    cz = (iz + 0.5) * w
                    val = bspline3d_evaluate(p, (cx, cy, cz), w)
                    total += val

        assert total == pytest.approx(1.0, abs=1e-12)

    def test_partition_of_unity_offset_point(self) -> None:
        """Partition of unity holds for a different test point."""
        w = 1.0
        p = (2.3, -0.7, 1.1)
        total = 0.0
        for ix in range(-2, 6):
            cx = (ix + 0.5) * w
            for iy in range(-2, 6):
                cy = (iy + 0.5) * w
                for iz in range(-2, 6):
                    cz = (iz + 0.5) * w
                    total += bspline3d_evaluate(p, (cx, cy, cz), w)
        assert total == pytest.approx(1.0, abs=1e-12)


class TestBSpline3DGradient:
    """Tests for the 3-D B-spline gradient."""

    def test_gradient_finite_difference(self) -> None:
        """Analytic gradient should match finite differences."""
        c = (1.0, 2.0, 3.0)
        w = 0.5
        p = (1.1, 2.2, 2.8)
        eps = 1e-7

        gx, gy, gz = bspline3d_gradient(p, c, w)

        # Finite difference in x
        f_plus = bspline3d_evaluate((p[0] + eps, p[1], p[2]), c, w)
        f_minus = bspline3d_evaluate((p[0] - eps, p[1], p[2]), c, w)
        assert gx == pytest.approx((f_plus - f_minus) / (2 * eps), abs=1e-5)

        # Finite difference in y
        f_plus = bspline3d_evaluate((p[0], p[1] + eps, p[2]), c, w)
        f_minus = bspline3d_evaluate((p[0], p[1] - eps, p[2]), c, w)
        assert gy == pytest.approx((f_plus - f_minus) / (2 * eps), abs=1e-5)

        # Finite difference in z
        f_plus = bspline3d_evaluate((p[0], p[1], p[2] + eps), c, w)
        f_minus = bspline3d_evaluate((p[0], p[1], p[2] - eps), c, w)
        assert gz == pytest.approx((f_plus - f_minus) / (2 * eps), abs=1e-5)

    def test_gradient_zero_at_center(self) -> None:
        """Gradient at the cell centre should be (0, 0, 0).

        The quadratic B-spline is symmetric about the centre, so
        all three components vanish.
        """
        c = (0.0, 0.0, 0.0)
        gx, gy, gz = bspline3d_gradient(c, c, 1.0)
        assert gx == 0.0
        assert gy == 0.0
        assert gz == 0.0


# ---- DOF indexing tests -------------------------------------------


class TestDofIndexing:
    """Tests for DOF index assignment."""

    def test_all_leaves_get_indices(self) -> None:
        """Every leaf should get a contiguous DOF index."""
        cells = [
            {"is_leaf": True},
            {"is_leaf": False},
            {"is_leaf": True},
            {"is_leaf": True},
            {"is_leaf": False},
        ]
        c2d, d2c = assign_dof_indices(cells)

        # 3 leaves -> DOFs 0, 1, 2
        assert len(d2c) == 3
        assert c2d[0] == 0
        assert c2d[1] == -1  # not a leaf
        assert c2d[2] == 1
        assert c2d[3] == 2
        assert c2d[4] == -1

    def test_round_trip(self) -> None:
        """cell_to_dof and dof_to_cell should be inverses."""
        cells = [{"is_leaf": i % 2 == 0} for i in range(10)]
        c2d, d2c = assign_dof_indices(cells)

        for dof_idx, cell_idx in enumerate(d2c):
            assert c2d[cell_idx] == dof_idx

    def test_contiguous_indices(self) -> None:
        """DOF indices should be 0..N-1 with no gaps."""
        cells = [{"is_leaf": True}] * 7
        c2d, d2c = assign_dof_indices(cells)
        assert d2c == list(range(7))
        assert c2d == list(range(7))

    def test_no_leaves(self) -> None:
        """If no leaves, both mappings should be empty/all -1."""
        cells = [{"is_leaf": False}] * 3
        c2d, d2c = assign_dof_indices(cells)
        assert d2c == []
        assert all(v == -1 for v in c2d)


# ---- Stencil enumeration tests -----------------------------------


def _make_uniform_grid_cells(
    n: int,
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
) -> list[dict]:
    """Create a uniform n x n x n grid of leaf cells as dicts.

    All cells are at depth 0 with base_resolution = n.
    """
    dx = (domain_max[0] - domain_min[0]) / n
    dy = (domain_max[1] - domain_min[1]) / n
    dz = (domain_max[2] - domain_min[2]) / n
    cells = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                from meshmerizer.adaptive_core import (
                    morton_encode_3d,
                )

                cells.append(
                    {
                        "is_leaf": True,
                        "depth": 0,
                        "bounds_min": (
                            domain_min[0] + ix * dx,
                            domain_min[1] + iy * dy,
                            domain_min[2] + iz * dz,
                        ),
                        "bounds_max": (
                            domain_min[0] + (ix + 1) * dx,
                            domain_min[1] + (iy + 1) * dy,
                            domain_min[2] + (iz + 1) * dz,
                        ),
                        "morton_key": morton_encode_3d(ix, iy, iz),
                    }
                )
    return cells


class TestStencilEnumeration:
    """Tests for DOF stencil enumeration."""

    def test_single_cell(self) -> None:
        """A single cell should have itself as its only neighbor."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (1.0, 1.0, 1.0)
        cells = _make_uniform_grid_cells(1, dmin, dmax)
        offsets, neighbors = enumerate_stencils(cells, dmin, dmax, 1, 0)
        assert len(offsets) == 2  # 1 DOF + sentinel
        assert neighbors == [0]  # self only

    def test_2x2x2_interior_stencil_size(self) -> None:
        """In a 2x2x2 grid, every cell neighbors all 8 cells."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (2.0, 2.0, 2.0)
        cells = _make_uniform_grid_cells(2, dmin, dmax)
        offsets, neighbors = enumerate_stencils(cells, dmin, dmax, 2, 0)
        # 8 DOFs, each should see all 8 (including self)
        for i in range(8):
            stencil = neighbors[offsets[i] : offsets[i + 1]]
            assert len(stencil) == 8

    def test_stencil_symmetry(self) -> None:
        """If j is in stencil of i, then i must be in stencil of j."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (3.0, 3.0, 3.0)
        cells = _make_uniform_grid_cells(3, dmin, dmax)
        offsets, neighbors = enumerate_stencils(cells, dmin, dmax, 3, 0)
        n_dofs = len(offsets) - 1
        for i in range(n_dofs):
            stencil_i = set(neighbors[offsets[i] : offsets[i + 1]])
            for j in stencil_i:
                stencil_j = set(neighbors[offsets[j] : offsets[j + 1]])
                assert i in stencil_j, (
                    f"DOF {j} is in stencil of {i} but "
                    f"{i} is not in stencil of {j}"
                )

    def test_3x3x3_corner_stencil(self) -> None:
        """A corner cell in a 3x3x3 grid should have 27 neighbors."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (3.0, 3.0, 3.0)
        cells = _make_uniform_grid_cells(3, dmin, dmax)
        offsets, neighbors = enumerate_stencils(cells, dmin, dmax, 3, 0)
        # Cell (0,0,0) is DOF 0 (first in Morton order)
        stencil_0 = neighbors[offsets[0] : offsets[1]]
        # With a 5-wide stencil (offsets -2..+2), a 3^3 grid clips to
        # the full 3x3x3 set of cells in-range.
        assert len(stencil_0) == 27

    def test_3x3x3_center_stencil(self) -> None:
        """The centre cell in a 3x3x3 grid has 27 neighbors."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (3.0, 3.0, 3.0)
        cells = _make_uniform_grid_cells(3, dmin, dmax)
        offsets, neighbors = enumerate_stencils(cells, dmin, dmax, 3, 0)
        # Find DOF for cell (1,1,1) — Morton index for (1,1,1)
        from meshmerizer.adaptive_core import morton_encode_3d

        target_key = morton_encode_3d(1, 1, 1)
        target_dof = None
        for idx, c in enumerate(cells):
            if c["morton_key"] == target_key:
                # All cells are leaves, DOF = cell index
                target_dof = idx
                break
        assert target_dof is not None
        stencil = neighbors[offsets[target_dof] : offsets[target_dof + 1]]
        assert len(stencil) == 27

    def test_self_in_stencil(self) -> None:
        """Every DOF should include itself in its stencil."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (2.0, 2.0, 2.0)
        cells = _make_uniform_grid_cells(2, dmin, dmax)
        offsets, neighbors = enumerate_stencils(cells, dmin, dmax, 2, 0)
        n_dofs = len(offsets) - 1
        for i in range(n_dofs):
            stencil = neighbors[offsets[i] : offsets[i + 1]]
            assert i in stencil
