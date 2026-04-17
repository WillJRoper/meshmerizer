"""Tests for Poisson surface reconstruction (C++ backend).

These tests exercise the new C++ full pipeline that replaces the
Open3D-based Poisson reconstruction.  The pipeline takes raw
particles (positions + smoothing lengths) and returns a triangle
mesh.
"""

import numpy as np
import pytest

from meshmerizer.adaptive_core import run_full_pipeline
from meshmerizer.poisson import (
    poisson_reconstruct,
    poisson_reconstruct_group,
)


def _make_sphere_particles(
    n: int = 2000,
    radius: float = 1.0,
    seed: int = 42,
) -> tuple:
    """Generate particles in a spherical shell with smoothing lengths.

    Places particles in a thin shell around a sphere so that the
    density field has a clear isosurface.

    Returns:
        Tuple of (positions, smoothing_lengths) where positions is
        (N, 3) float64 and smoothing_lengths is (N,) float64.
    """
    rng = np.random.default_rng(seed)

    # Fibonacci sphere for uniform distribution.
    indices = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * indices / (n - 1))
    r = np.sqrt(1.0 - y * y)
    theta = phi * indices

    x = r * np.cos(theta)
    z = r * np.sin(theta)

    # Place particles on the sphere surface with small jitter.
    normals = np.column_stack([x, y, z])
    jitter = rng.normal(0, 0.02, size=(n,))
    positions = normals * (radius + jitter[:, np.newaxis])

    # Center at (2, 2, 2) so domain is well away from origin.
    positions += np.array([2.0, 2.0, 2.0])

    # Smoothing lengths: ~0.2 for decent kernel overlap.
    smoothing_lengths = np.full(n, 0.2, dtype=np.float64)

    return positions, smoothing_lengths


def test_run_full_pipeline_sphere() -> None:
    """Full pipeline on a sphere should produce a non-empty mesh."""
    positions, sml = _make_sphere_particles(n=1000)

    result = run_full_pipeline(
        positions,
        sml,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(4.0, 4.0, 4.0),
        base_resolution=4,
        isovalue=0.01,
        max_depth=3,
        screening_weight=4.0,
        max_iters=200,
        tol=1e-4,
    )

    assert "vertices" in result
    assert "faces" in result
    assert result["vertices"].ndim == 2
    assert result["vertices"].shape[1] == 3
    assert result["faces"].ndim == 2
    assert result["faces"].shape[1] == 3
    # Should produce some geometry.
    assert result["vertices"].shape[0] > 0
    assert result["faces"].shape[0] > 0
    # Face indices must be valid.
    assert np.all(result["faces"] >= 0)
    assert np.all(result["faces"] < result["vertices"].shape[0])


def test_run_full_pipeline_metadata() -> None:
    """Pipeline result should contain solver metadata."""
    positions, sml = _make_sphere_particles(n=500)

    result = run_full_pipeline(
        positions,
        sml,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(4.0, 4.0, 4.0),
        base_resolution=4,
        isovalue=0.01,
        max_depth=2,
        screening_weight=4.0,
        max_iters=100,
        tol=1e-4,
    )

    assert "isovalue" in result
    assert "n_qef_vertices" in result
    assert "solver_converged" in result
    assert "solver_iterations" in result
    assert "solver_residual" in result
    assert isinstance(result["solver_converged"], bool)
    assert result["solver_iterations"] >= 0
    assert result["solver_residual"] >= 0.0


def test_poisson_reconstruct_group_sphere() -> None:
    """poisson_reconstruct_group should produce a mesh."""
    positions, sml = _make_sphere_particles(n=1000)

    verts, faces = poisson_reconstruct_group(
        positions,
        None,
        sml,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(4.0, 4.0, 4.0),
        base_resolution=4,
        isovalue=0.01,
        max_depth=3,
        screening_weight=4.0,
    )

    assert verts.ndim == 2
    assert verts.shape[1] == 3
    assert faces.ndim == 2
    assert faces.shape[1] == 3
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0
    assert np.all(faces >= 0)
    assert np.all(faces < verts.shape[0])


def test_poisson_reconstruct_group_rejects_too_few() -> None:
    """Fewer than 3 points should raise ValueError."""
    positions = np.ones((2, 3), dtype=np.float64)
    sml = np.ones(2, dtype=np.float64)

    with pytest.raises(ValueError, match="at least 3"):
        poisson_reconstruct_group(
            positions,
            None,
            sml,
            domain_min=(0.0, 0.0, 0.0),
            domain_max=(4.0, 4.0, 4.0),
            base_resolution=4,
            isovalue=0.01,
            max_depth=2,
        )


def test_poisson_reconstruct_empty_returns_empty() -> None:
    """Empty input should return empty arrays."""
    positions = np.empty((0, 3), dtype=np.float64)
    sml = np.empty(0, dtype=np.float64)

    verts, faces = poisson_reconstruct(
        positions,
        None,
        sml,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(4.0, 4.0, 4.0),
        base_resolution=4,
        isovalue=0.01,
        max_depth=2,
    )

    assert verts.shape == (0, 3)
    assert faces.shape == (0, 3)
