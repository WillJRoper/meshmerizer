"""Tests for Phase 20b: vector field splatting and RHS assembly."""

import numpy as np
import pytest

from meshmerizer.adaptive_core import (
    morton_encode_3d,
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
        that DOF with weight 1.0 (times area_weight = 1/1)."""
        dmin = (0.0, 0.0, 0.0)
        dmax = (1.0, 1.0, 1.0)
        cells = _make_uniform_grid_cells(1, dmin, dmax)

        pos = np.array([[0.5, 0.5, 0.5]])
        nor = np.array([[0.0, 0.0, 1.0]])

        vx, vy, vz, rhs = splat_and_compute_rhs(
            pos, nor, cells, dmin, dmax, 1, 0
        )

        # Single DOF, single sample, area_weight = 1/1 = 1.0
        # B(center) = 1.0, so V = 1.0 * (0, 0, 1) * 1.0
        assert len(vx) == 1
        assert vx[0] == pytest.approx(0.0)
        assert vy[0] == pytest.approx(0.0)
        assert vz[0] == pytest.approx(1.0)

    def test_partition_of_unity_preserves_total(self) -> None:
        """Total splatted normal magnitude should equal area weight
        times normal magnitude (partition of unity) for interior
        points that have all 8 overlapping DOFs."""
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

        # Total z-component should sum to area_weight * N * 1.0
        # = (1/N) * N * 1.0 = 1.0
        assert sum(vz) == pytest.approx(1.0, abs=1e-10)

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
        dmax = (4.0, 4.0, 4.0)
        cells = _make_uniform_grid_cells(4, dmin, dmax)

        # Place samples at every cell centre with uniform z-normal
        n = 4
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

        _, _, _, rhs = splat_and_compute_rhs(pos, nor, cells, dmin, dmax, 4, 0)

        # Interior DOFs (cells not touching the boundary) should
        # have RHS very close to zero. Boundary DOFs may be
        # nonzero due to truncated stencils.
        # In a 4x4x4 grid, interior cells are (1,1,1) to (2,2,2)
        interior_rhs = []
        for ix in range(1, 3):
            for iy in range(1, 3):
                for iz in range(1, 3):
                    # Find the cell index (in our row-major order)
                    idx = ix * 16 + iy * 4 + iz
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
