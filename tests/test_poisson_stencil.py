"""Tests for Phase 20c: stiffness stencil, screening, and operator."""

import numpy as np
import pytest

from meshmerizer.adaptive_core import (
    apply_poisson_operator,
    laplacian_stencil_weight,
    morton_encode_3d,
)


def _make_uniform_grid_cells(
    n: int,
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
) -> list[dict]:
    """Create a uniform n x n x n grid of leaf cells as dicts."""
    dx = (domain_max[0] - domain_min[0]) / n
    dy = (domain_max[1] - domain_min[1]) / n
    dz = (domain_max[2] - domain_min[2]) / n
    cells = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
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


class TestLaplacianStencil:
    """Tests for the 5^3 (125-point) Laplacian stencil weights."""

    def test_center_weight(self) -> None:
        """L(0,0,0) for h=1 should be 363/400."""
        w = laplacian_stencil_weight(0, 0, 0, 1.0)
        assert w == pytest.approx(363.0 / 400.0)

    def test_face_neighbor_weight(self) -> None:
        """L(1,0,0) for h=1 should be 11/80."""
        w = laplacian_stencil_weight(1, 0, 0, 1.0)
        assert w == pytest.approx(11.0 / 80.0)

    def test_edge_neighbor_weight(self) -> None:
        """L(1,1,0) for h=1 should be -13/400."""
        w = laplacian_stencil_weight(1, 1, 0, 1.0)
        assert w == pytest.approx(-13.0 / 400.0)

    def test_corner_neighbor_weight(self) -> None:
        """L(1,1,1) for h=1 should be -169/3600."""
        w = laplacian_stencil_weight(1, 1, 1, 1.0)
        assert w == pytest.approx(-169.0 / 3600.0)

    def test_symmetry(self) -> None:
        """Stencil should be symmetric: L(d) = L(-d)."""
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    a = laplacian_stencil_weight(dx, dy, dz, 1.0)
                    b = laplacian_stencil_weight(-dx, -dy, -dz, 1.0)
                    assert a == pytest.approx(b, abs=1e-15)

    def test_stencil_row_sum(self) -> None:
        """Sum of all 5^3 stencil weights should be zero
        (Laplacian of a constant is zero)."""
        h = 1.0
        total = 0.0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    total += laplacian_stencil_weight(dx, dy, dz, h)
        assert total == pytest.approx(0.0, abs=1e-14)

    def test_h_scaling(self) -> None:
        """Stencil weight should scale linearly with h."""
        w1 = laplacian_stencil_weight(0, 0, 0, 1.0)
        w2 = laplacian_stencil_weight(0, 0, 0, 2.0)
        assert w2 == pytest.approx(2.0 * w1)


class TestOperator:
    """Tests for the screened Poisson operator."""

    def test_operator_preserves_constants_when_alpha_zero(
        self,
    ) -> None:
        """A*ones should be zero when alpha=0 (pure Laplacian)
        for interior DOFs on a large enough grid."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (6.0, 6.0, 6.0)
        cells = _make_uniform_grid_cells(6, dmin, dmax)

        # Dummy positions (screening not used for alpha=0)
        pos = np.array([[3.0, 3.0, 3.0]])
        n_dofs = 216  # 6^3
        x = [1.0] * n_dofs

        result = apply_poisson_operator(pos, cells, dmin, dmax, 6, 0, 0.0, x)

        for ix in range(2, 4):
            for iy in range(2, 4):
                for iz in range(2, 4):
                    idx = ix * 36 + iy * 6 + iz
                    assert result[idx] == pytest.approx(0.0, abs=1e-10)

    def test_operator_preserves_linear_when_alpha_zero(
        self,
    ) -> None:
        """Laplacian of a linear function should be zero
        for interior DOFs."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (6.0, 6.0, 6.0)
        cells = _make_uniform_grid_cells(6, dmin, dmax)
        pos = np.array([[3.0, 3.0, 3.0]])

        # x = x-coordinate of cell centre.
        x = []
        for ix in range(6):
            for iy in range(6):
                for iz in range(6):
                    x.append(ix + 0.5)

        result = apply_poisson_operator(pos, cells, dmin, dmax, 6, 0, 0.0, x)

        for ix in range(2, 4):
            for iy in range(2, 4):
                for iz in range(2, 4):
                    idx = ix * 36 + iy * 6 + iz
                    assert result[idx] == pytest.approx(0.0, abs=1e-10)

    def test_operator_is_spd(self) -> None:
        """x^T A x > 0 for random nonzero x (SPD property)."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (3.0, 3.0, 3.0)
        cells = _make_uniform_grid_cells(3, dmin, dmax)

        # Place samples at cell centres for screening
        pos_list = []
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    pos_list.append([ix + 0.5, iy + 0.5, iz + 0.5])
        pos = np.array(pos_list)
        n_dofs = 27

        rng = np.random.RandomState(99)
        x = rng.randn(n_dofs).tolist()

        result = apply_poisson_operator(pos, cells, dmin, dmax, 3, 0, 4.0, x)

        xtAx = sum(xi * ri for xi, ri in zip(x, result))
        assert xtAx > 0.0

    def test_screening_adds_to_diagonal(self) -> None:
        """With alpha > 0, A*e_i should have larger diagonal
        component than pure Laplacian."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (3.0, 3.0, 3.0)
        cells = _make_uniform_grid_cells(3, dmin, dmax)

        # Place sample at centre of domain
        pos = np.array([[1.5, 1.5, 1.5]])
        n_dofs = 27

        # Unit vector at the centre DOF (index 13 = 1*9+1*3+1)
        x = [0.0] * n_dofs
        x[13] = 1.0

        result_no_screen = apply_poisson_operator(
            pos, cells, dmin, dmax, 3, 0, 0.0, x
        )
        result_with_screen = apply_poisson_operator(
            pos, cells, dmin, dmax, 3, 0, 4.0, x
        )

        # The centre DOF diagonal should be larger with screening
        assert result_with_screen[13] > result_no_screen[13]

    def test_screening_symmetry(self) -> None:
        """A should be symmetric: x^T A y = y^T A x."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (3.0, 3.0, 3.0)
        cells = _make_uniform_grid_cells(3, dmin, dmax)

        pos_list = []
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    pos_list.append([ix + 0.5, iy + 0.5, iz + 0.5])
        pos = np.array(pos_list)
        n_dofs = 27

        rng = np.random.RandomState(42)
        x = rng.randn(n_dofs).tolist()
        y = rng.randn(n_dofs).tolist()

        ax = apply_poisson_operator(pos, cells, dmin, dmax, 3, 0, 4.0, x)
        ay = apply_poisson_operator(pos, cells, dmin, dmax, 3, 0, 4.0, y)

        xtay = sum(xi * ayi for xi, ayi in zip(x, ay))
        ytax = sum(yi * axi for yi, axi in zip(y, ax))
        assert xtay == pytest.approx(ytax, abs=1e-10)
