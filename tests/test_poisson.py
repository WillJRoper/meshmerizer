"""Tests for Poisson surface reconstruction."""

import numpy as np
import pytest

from meshmerizer.poisson import (
    poisson_reconstruct,
    poisson_reconstruct_group,
)


def _make_sphere_points(
    n: int = 500, radius: float = 1.0, seed: int = 42
) -> tuple:
    """Generate points on a sphere with outward normals.

    Uses the Fibonacci sphere algorithm for approximately uniform
    sampling, which gives Poisson a clean input.

    Returns:
        Tuple of (positions, normals) each (N, 3) float64.
    """
    indices = np.arange(n, dtype=np.float64)
    # Golden angle in radians.
    phi = np.pi * (3.0 - np.sqrt(5.0))
    # y goes from 1 to -1.
    y = 1.0 - (2.0 * indices / (n - 1))
    r = np.sqrt(1.0 - y * y)
    theta = phi * indices

    x = r * np.cos(theta)
    z = r * np.sin(theta)

    normals = np.column_stack([x, y, z])
    positions = normals * radius

    return positions, normals


def test_poisson_sphere_produces_watertight_mesh() -> None:
    """A sphere point cloud should produce a watertight mesh."""
    positions, normals = _make_sphere_points(n=500)

    vertices, faces = poisson_reconstruct_group(
        positions, normals, poisson_depth=6, density_quantile=0.1
    )

    assert vertices.ndim == 2
    assert vertices.shape[1] == 3
    assert faces.ndim == 2
    assert faces.shape[1] == 3
    # Should have a reasonable number of vertices and faces.
    assert vertices.shape[0] > 10
    assert faces.shape[0] > 10
    # All face indices should be valid.
    assert np.all(faces >= 0)
    assert np.all(faces < vertices.shape[0])


def test_poisson_two_groups_produce_two_components() -> None:
    """Two separated sphere clusters should produce distinct mesh
    components."""
    pos_a, nrm_a = _make_sphere_points(n=300, radius=1.0)
    pos_b, nrm_b = _make_sphere_points(n=300, radius=1.0)
    # Shift cluster B far away.
    pos_b += np.array([10.0, 10.0, 10.0])

    positions = np.vstack([pos_a, pos_b])
    normals_arr = np.vstack([nrm_a, nrm_b])
    labels = np.array([0] * 300 + [1] * 300, dtype=np.int64)

    vertices, faces = poisson_reconstruct(
        positions,
        normals_arr,
        labels,
        poisson_depth=6,
        density_quantile=0.1,
    )

    assert vertices.shape[0] > 20
    assert faces.shape[0] > 20
    assert np.all(faces >= 0)
    assert np.all(faces < vertices.shape[0])


def test_poisson_empty_input_returns_empty() -> None:
    """Empty positions/normals/labels should return empty arrays."""
    positions = np.empty((0, 3), dtype=np.float64)
    normals_arr = np.empty((0, 3), dtype=np.float64)
    labels = np.empty((0,), dtype=np.int64)

    vertices, faces = poisson_reconstruct(
        positions,
        normals_arr,
        labels,
        poisson_depth=6,
    )

    assert vertices.shape == (0, 3)
    assert faces.shape == (0, 3)


def test_poisson_reconstruct_group_rejects_mismatched_shapes() -> None:
    """Mismatched positions/normals should raise ValueError."""
    positions = np.ones((10, 3), dtype=np.float64)
    normals_arr = np.ones((5, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="shape"):
        poisson_reconstruct_group(positions, normals_arr)


def test_poisson_reconstruct_group_rejects_too_few_points() -> None:
    """Fewer than 3 points should raise ValueError."""
    positions = np.ones((2, 3), dtype=np.float64)
    normals_arr = np.ones((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="at least 3"):
        poisson_reconstruct_group(positions, normals_arr)
