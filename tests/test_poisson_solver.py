"""Tests for Phase 20d: PCG solver for screened Poisson."""

import numpy as np
import pytest

from meshmerizer.adaptive_core import (
    apply_poisson_operator,
    morton_encode_3d,
    solve_poisson,
)


def _make_uniform_grid_cells(
    n: int,
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
) -> list[dict]:
    """Create a uniform n x n x n grid of leaf cells."""
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


class TestPCGSolver:
    """Tests for the PCG Poisson solver."""

    def test_zero_rhs_gives_zero_solution(self) -> None:
        """If b = 0, the solution should be x = 0."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (3.0, 3.0, 3.0)
        cells = _make_uniform_grid_cells(3, dmin, dmax)
        pos = np.array([[1.5, 1.5, 1.5]])
        n_dofs = 27
        b = [0.0] * n_dofs

        x, iters, resid, converged = solve_poisson(
            pos,
            cells,
            dmin,
            dmax,
            3,
            0,
            4.0,
            b,
            max_iters=100,
            tol=1e-10,
        )

        assert converged
        for xi in x:
            assert xi == pytest.approx(0.0, abs=1e-12)

    def test_solve_recovers_known_solution(self) -> None:
        """Solve A x = b where b = A * x_true, and verify
        that the solver recovers x_true."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (3.0, 3.0, 3.0)
        cells = _make_uniform_grid_cells(3, dmin, dmax)

        # Place samples at all cell centres for screening
        pos_list = []
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    pos_list.append([ix + 0.5, iy + 0.5, iz + 0.5])
        pos = np.array(pos_list)
        n_dofs = 27
        alpha = 4.0

        # Known solution: random
        rng = np.random.RandomState(42)
        x_true = rng.randn(n_dofs).tolist()

        # Compute b = A * x_true
        b = apply_poisson_operator(pos, cells, dmin, dmax, 3, 0, alpha, x_true)

        # Solve
        x, iters, resid, converged = solve_poisson(
            pos,
            cells,
            dmin,
            dmax,
            3,
            0,
            alpha,
            b,
            max_iters=500,
            tol=1e-10,
        )

        assert converged
        for xi, xt in zip(x, x_true):
            assert xi == pytest.approx(xt, abs=1e-6)

    def test_convergence_rate(self) -> None:
        """Solver should converge within a reasonable number
        of iterations for a well-conditioned problem."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (4.0, 4.0, 4.0)
        cells = _make_uniform_grid_cells(4, dmin, dmax)

        pos_list = []
        for ix in range(4):
            for iy in range(4):
                for iz in range(4):
                    pos_list.append([ix + 0.5, iy + 0.5, iz + 0.5])
        pos = np.array(pos_list)
        n_dofs = 64
        alpha = 4.0

        # Random RHS
        rng = np.random.RandomState(123)
        b = rng.randn(n_dofs).tolist()

        x, iters, resid, converged = solve_poisson(
            pos,
            cells,
            dmin,
            dmax,
            4,
            0,
            alpha,
            b,
            max_iters=200,
            tol=1e-10,
        )

        assert converged
        # PCG should converge in at most n_dofs iterations
        # (typically much fewer with preconditioning)
        assert iters < n_dofs

    def test_pure_laplacian_solve(self) -> None:
        """Solve with alpha=0 (pure Laplacian, no screening).
        The Laplacian has a null space (constants), so we
        need to constrain the problem.  We use alpha > 0 to
        make it SPD, but here we test with a very small alpha
        to approximate the pure Laplacian."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (3.0, 3.0, 3.0)
        cells = _make_uniform_grid_cells(3, dmin, dmax)

        pos = np.array([[1.5, 1.5, 1.5]])
        n_dofs = 27
        # Small screening to make it SPD
        alpha = 0.01

        rng = np.random.RandomState(7)
        x_true = rng.randn(n_dofs).tolist()
        b = apply_poisson_operator(pos, cells, dmin, dmax, 3, 0, alpha, x_true)

        x, iters, resid, converged = solve_poisson(
            pos,
            cells,
            dmin,
            dmax,
            3,
            0,
            alpha,
            b,
            max_iters=500,
            tol=1e-10,
        )

        assert converged
        for xi, xt in zip(x, x_true):
            assert xi == pytest.approx(xt, abs=1e-5)

    def test_residual_below_tolerance(self) -> None:
        """After solving, the residual |b - Ax| / |b| should
        be below the requested tolerance."""
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
        alpha = 4.0

        rng = np.random.RandomState(55)
        b = rng.randn(n_dofs).tolist()

        x, iters, resid, converged = solve_poisson(
            pos,
            cells,
            dmin,
            dmax,
            3,
            0,
            alpha,
            b,
            max_iters=500,
            tol=1e-8,
        )

        assert converged
        # Verify residual independently
        ax = apply_poisson_operator(pos, cells, dmin, dmax, 3, 0, alpha, x)
        r = [bi - ai for bi, ai in zip(b, ax)]
        r_norm = sum(ri**2 for ri in r) ** 0.5
        b_norm = sum(bi**2 for bi in b) ** 0.5
        assert r_norm / b_norm < 1e-8
