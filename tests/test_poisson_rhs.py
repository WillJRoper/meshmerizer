"""Tests for Phase 20b: vector field splatting and RHS assembly."""

import numpy as np
import pytest

from meshmerizer.adaptive_core import (
    morton_encode_3d,
    pc_integrals_1d,
    pc_laplacian_weight,
    splat_and_compute_rhs,
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


class TestSplatNormals:
    """Tests for normal field splatting."""

    def test_single_point_at_center(self) -> None:
        """A single point at the cell centre should splat into
        that DOF with weight B_3d(center) (times area_weight = 1/h)."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (1.0, 1.0, 1.0)
        cells = _make_uniform_grid_cells(1, dmin, dmax)

        pos = np.array([[0.5, 0.5, 0.5]])
        nor = np.array([[0.0, 0.0, 1.0]])

        vx, vy, vz, rhs = splat_and_compute_rhs(
            pos, nor, cells, dmin, dmax, 1, 0
        )

        # Single DOF, single sample, area_weight = 1/h = 1.0.
        # For degree-2 B-splines, B_3d(center) = (3/4)^3 = 27/64.
        assert len(vx) == 1
        assert vx[0] == pytest.approx(0.0)
        assert vy[0] == pytest.approx(0.0)
        assert vz[0] == pytest.approx(27.0 / 64.0)

    def test_partition_of_unity_preserves_total(self) -> None:
        """Total splatted normal magnitude should equal area weight
        times normal magnitude (partition of unity) for interior
        points that have all overlapping DOFs."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (3.0, 3.0, 3.0)
        cells = _make_uniform_grid_cells(3, dmin, dmax)

        # Single point well in the interior (all 8 neighbors exist)
        pos = np.array([[1.3, 1.7, 1.4]])
        nor = np.array([[0.0, 1.0, 0.0]])

        vx, vy, vz, _ = splat_and_compute_rhs(
            pos, nor, cells, dmin, dmax, 3, 0
        )

        # Sum of all V_j.y should equal area_weight * n.y = 1.0
        total_y = sum(vy)
        assert total_y == pytest.approx(1.0, abs=1e-10)
        assert sum(vx) == pytest.approx(0.0, abs=1e-10)
        assert sum(vz) == pytest.approx(0.0, abs=1e-10)

    def test_multiple_samples_accumulate(self) -> None:
        """Multiple samples should accumulate additively.
        Points must be in the interior (not near domain boundary)
        for partition-of-unity to hold exactly."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (4.0, 4.0, 4.0)
        cells = _make_uniform_grid_cells(4, dmin, dmax)

        n = 10
        # Place points well inside the domain so all 8 DOFs exist
        pos = np.random.RandomState(42).uniform(1.1, 2.9, (n, 3))
        nor = np.zeros((n, 3))
        nor[:, 2] = 1.0  # All normals point in z

        vx, vy, vz, _ = splat_and_compute_rhs(
            pos, nor, cells, dmin, dmax, 4, 0
        )

        # Total z-component should sum to area_weight * N * 1.0.
        # The implementation uses area_weight = 1/h; here h=1.
        assert sum(vz) == pytest.approx(float(n), abs=1e-10)

    def test_symmetry(self) -> None:
        """Symmetric sample placement should produce symmetric V."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (2.0, 2.0, 2.0)
        cells = _make_uniform_grid_cells(2, dmin, dmax)

        # Two points symmetric about centre, both with z-normal
        pos = np.array([[0.5, 1.0, 1.0], [1.5, 1.0, 1.0]])
        nor = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        vx, vy, vz, _ = splat_and_compute_rhs(
            pos, nor, cells, dmin, dmax, 2, 0
        )

        # Due to x-symmetry, the sum of vx should be 0
        assert sum(vx) == pytest.approx(0.0, abs=1e-10)


class TestRHSAssembly:
    """Tests for RHS vector computation."""

    def test_constant_normal_field_rhs_zero(self) -> None:
        """A constant normal field has zero divergence, so the
        RHS should be zero for interior DOFs.

        With a constant V field, V_j = const for all j. Then
        b_i = sum_j const . G_ij = const . sum_j G_ij.
        By the gradient theorem, sum_j G_ij = 0 for interior
        DOFs (the divergence of a partition-of-unity field is
        zero away from boundaries).
        """
        dmin = (0.0, 0.0, 0.0)
        dmax = (10.0, 10.0, 10.0)
        cells = _make_uniform_grid_cells(10, dmin, dmax)

        # Place samples at every cell centre with uniform z-normal
        n = 10
        pos_list = []
        nor_list = []
        for ix in range(n):
            for iy in range(n):
                for iz in range(n):
                    pos_list.append(
                        [
                            (ix + 0.5),
                            (iy + 0.5),
                            (iz + 0.5),
                        ]
                    )
                    nor_list.append([0.0, 0.0, 1.0])
        pos = np.array(pos_list)
        nor = np.array(nor_list)

        _, _, _, rhs = splat_and_compute_rhs(
            pos, nor, cells, dmin, dmax, 10, 0
        )

        # Interior DOFs (cells not touching the boundary) should
        # have RHS very close to zero. Boundary DOFs may be
        # nonzero due to truncated stencils.
        # With a 5-wide stencil (offsets -2..+2), interior cells must be
        # far enough from the boundary that all stencil neighbors also
        # have full support. A 10x10x10 grid has a clean interior at
        # indices 4 and 5.
        interior_rhs = []
        for ix in range(4, 6):
            for iy in range(4, 6):
                for iz in range(4, 6):
                    # Find the cell index (in our row-major order)
                    idx = ix * 100 + iy * 10 + iz
                    interior_rhs.append(rhs[idx])

        for val in interior_rhs:
            assert val == pytest.approx(0.0, abs=1e-10)

    def test_rhs_nonzero_for_varying_field(self) -> None:
        """A spatially varying normal field should give nonzero
        RHS (at least for some DOFs)."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (3.0, 3.0, 3.0)
        cells = _make_uniform_grid_cells(3, dmin, dmax)

        # Place a cluster of z-normals in one octant
        rng = np.random.RandomState(123)
        pos = rng.uniform(0.1, 1.4, (20, 3))
        nor = np.zeros((20, 3))
        nor[:, 2] = 1.0

        _, _, _, rhs = splat_and_compute_rhs(pos, nor, cells, dmin, dmax, 3, 0)

        # At least some RHS values should be nonzero
        assert any(abs(r) > 1e-12 for r in rhs)

    def test_rhs_antisymmetric_sign(self) -> None:
        """For a point source normal, RHS should have both
        positive and negative values (divergence changes sign
        across the source)."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (3.0, 3.0, 3.0)
        cells = _make_uniform_grid_cells(3, dmin, dmax)

        # Single point at centre with z-normal
        pos = np.array([[1.5, 1.5, 1.5]])
        nor = np.array([[0.0, 0.0, 1.0]])

        _, _, _, rhs = splat_and_compute_rhs(pos, nor, cells, dmin, dmax, 3, 0)

        nonzero = [r for r in rhs if abs(r) > 1e-14]
        has_pos = any(r > 0 for r in nonzero)
        has_neg = any(r < 0 for r in nonzero)
        assert has_pos and has_neg, (
            "RHS should have both positive and negative values "
            "for a point source normal field"
        )


class TestParentChildIntegrals:
    """Tests for Phase 21a: parent-child cross-depth 1-D integrals."""

    # Expected values: (j, mass, stiffness, grad_value)
    EXPECTED = [
        (-4, 1 / 15360, -1 / 96, 1 / 768),
        (-3, 1 / 64, -1 / 4, 13 / 128),
        (-2, 599 / 3840, -11 / 24, 191 / 384),
        (-1, 31 / 64, 1 / 4, 89 / 128),
        (0, 1761 / 2560, 15 / 16, 0.0),
        (1, 31 / 64, 1 / 4, -89 / 128),
        (2, 599 / 3840, -11 / 24, -191 / 384),
        (3, 1 / 64, -1 / 4, -13 / 128),
        (4, 1 / 15360, -1 / 96, -1 / 768),
    ]

    @pytest.mark.parametrize(
        "j,exp_m,exp_k,exp_s",
        EXPECTED,
        ids=[f"j={j}" for j, _, _, _ in EXPECTED],
    )
    def test_1d_values(
        self, j: int, exp_m: float, exp_k: float, exp_s: float
    ) -> None:
        """Each 1-D integral matches the known rational value."""
        m, k, s = pc_integrals_1d(j)
        assert m == pytest.approx(exp_m, rel=1e-14)
        assert k == pytest.approx(exp_k, rel=1e-14)
        assert s == pytest.approx(exp_s, rel=1e-14)

    def test_mass_sum(self) -> None:
        """Sum of mass integrals = 2 (partition of unity)."""
        total = sum(pc_integrals_1d(j)[0] for j in range(-4, 5))
        assert total == pytest.approx(2.0, rel=1e-14)

    def test_stiffness_sum(self) -> None:
        """Sum of stiffness integrals = 0."""
        total = sum(pc_integrals_1d(j)[1] for j in range(-4, 5))
        assert total == pytest.approx(0.0, abs=1e-14)

    def test_grad_value_antisymmetry(self) -> None:
        """S_pc(-j) = -S_pc(j) (antisymmetry)."""
        for j in range(-4, 5):
            sp = pc_integrals_1d(j)[2]
            sm = pc_integrals_1d(-j)[2]
            assert sp == pytest.approx(-sm, abs=1e-15)

    def test_mass_symmetry(self) -> None:
        """M_pc(j) = M_pc(-j) (symmetry)."""
        for j in range(-4, 5):
            mp = pc_integrals_1d(j)[0]
            mm = pc_integrals_1d(-j)[0]
            assert mp == pytest.approx(mm, abs=1e-15)

    def test_stiffness_symmetry(self) -> None:
        """K_pc(j) = K_pc(-j) (symmetry)."""
        for j in range(-4, 5):
            kp = pc_integrals_1d(j)[1]
            km = pc_integrals_1d(-j)[1]
            assert kp == pytest.approx(km, abs=1e-15)

    def test_out_of_range_zero(self) -> None:
        """Offsets |j| > 4 return zero."""
        for j in [-5, -10, 5, 10]:
            m, k, s = pc_integrals_1d(j)
            assert m == 0.0
            assert k == 0.0
            assert s == 0.0

    def test_3d_laplacian_constant_is_zero(self) -> None:
        """Sum of 3-D parent-child Laplacian weights = 0.

        If all child DOFs have x=1, the Laplacian contribution
        from the parent should vanish (Laplacian of constant = 0).
        """
        total = 0.0
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                for dz in range(-4, 5):
                    total += pc_laplacian_weight(dx, dy, dz)
        assert total == pytest.approx(0.0, abs=1e-12)
