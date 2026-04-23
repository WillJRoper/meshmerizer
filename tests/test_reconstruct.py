"""Tests for adaptive mesh reconstruction wrappers.

These tests exercise the wrapper layer around the C++ adaptive dual
contouring pipeline. The pipeline takes raw particles (positions +
smoothing lengths) and returns a triangle mesh.
"""

import numpy as np
import pytest

from meshmerizer.adaptive_core import run_full_pipeline
from meshmerizer.reconstruct import reconstruct_group, reconstruct_mesh


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

    indices = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * indices / (n - 1))
    r = np.sqrt(1.0 - y * y)
    theta = phi * indices

    x = r * np.cos(theta)
    z = r * np.sin(theta)

    normals = np.column_stack([x, y, z])
    jitter = rng.normal(0, 0.02, size=(n,))
    positions = normals * (radius + jitter[:, np.newaxis])
    positions += np.array([2.0, 2.0, 2.0])

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
    )

    assert "vertices" in result
    assert "faces" in result
    assert result["vertices"].ndim == 2
    assert result["vertices"].shape[1] == 3
    assert result["faces"].ndim == 2
    assert result["faces"].shape[1] == 3
    assert result["vertices"].shape[0] > 0
    assert result["faces"].shape[0] > 0
    assert np.all(result["faces"] >= 0)
    assert np.all(result["faces"] < result["vertices"].shape[0])


def test_run_full_pipeline_metadata() -> None:
    """Pipeline result should contain mesh and basic pipeline metadata."""
    positions, sml = _make_sphere_particles(n=500)

    result = run_full_pipeline(
        positions,
        sml,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(4.0, 4.0, 4.0),
        base_resolution=4,
        isovalue=0.01,
        max_depth=2,
    )

    assert "isovalue" in result
    assert "n_qef_vertices" in result


def test_reconstruct_group_sphere() -> None:
    """reconstruct_group should produce a mesh."""
    positions, sml = _make_sphere_particles(n=1000)

    verts, faces = reconstruct_group(
        positions,
        sml,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(4.0, 4.0, 4.0),
        base_resolution=4,
        isovalue=0.01,
        max_depth=3,
    )

    assert verts.ndim == 2
    assert verts.shape[1] == 3
    assert faces.ndim == 2
    assert faces.shape[1] == 3
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0
    assert np.all(faces >= 0)
    assert np.all(faces < verts.shape[0])


def test_reconstruct_group_rejects_too_few() -> None:
    """Fewer than 3 points should raise ValueError."""
    positions = np.ones((2, 3), dtype=np.float64)
    sml = np.ones(2, dtype=np.float64)

    with pytest.raises(ValueError, match="at least 3"):
        reconstruct_group(
            positions,
            sml,
            domain_min=(0.0, 0.0, 0.0),
            domain_max=(4.0, 4.0, 4.0),
            base_resolution=4,
            isovalue=0.01,
            max_depth=2,
        )


def test_reconstruct_mesh_empty_returns_empty() -> None:
    """Empty input should return empty arrays."""
    positions = np.empty((0, 3), dtype=np.float64)
    sml = np.empty(0, dtype=np.float64)

    verts, faces = reconstruct_mesh(
        positions,
        sml,
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(4.0, 4.0, 4.0),
        base_resolution=4,
        isovalue=0.01,
        max_depth=2,
    )

    assert verts.shape == (0, 3)
    assert faces.shape == (0, 3)


def test_reconstruct_group_propagates_keyboard_interrupt(monkeypatch) -> None:
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    sml = np.ones(3, dtype=np.float64)

    monkeypatch.setattr(
        "meshmerizer.reconstruct.run_full_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        reconstruct_group(
            positions,
            sml,
            domain_min=(0.0, 0.0, 0.0),
            domain_max=(4.0, 4.0, 4.0),
            base_resolution=4,
            isovalue=0.01,
            max_depth=2,
        )
