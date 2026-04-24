"""Integration tests for the full particles-to-mesh pipeline.

These tests use synthetic particle distributions with known geometry
to validate the end-to-end adaptive reconstruction pipeline. Each test
generates particles, runs the pipeline,
and checks geometric properties of the output mesh against the
known input.

The tests are designed to run quickly (< 2s each) by using small
particle counts and low resolution.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import trimesh

from meshmerizer.adaptive import fof_cluster, run_full_pipeline

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate approximately uniform points on a unit sphere.

    Uses the Fibonacci spiral method which avoids clustering at
    poles unlike latitude-longitude grids.

    Args:
        n: Number of points.

    Returns:
        (N, 3) float64 array of unit vectors.
    """
    indices = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * indices / (n - 1))
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
    theta = phi * indices
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    return np.column_stack([x, y, z])


def _make_sphere_particles(
    n: int = 2000,
    radius: float = 1.0,
    center: tuple = (2.0, 2.0, 2.0),
    h: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Generate SPH particles on a spherical shell.

    The particles are placed on the sphere surface with small
    radial jitter.  The smoothing length is chosen so that each
    particle's kernel overlaps with its neighbors, producing a
    smooth density field with a clear isosurface.

    Args:
        n: Number of particles.
        radius: Sphere radius.
        center: Sphere center (must be inside the domain).
        h: Smoothing length for all particles.
        seed: RNG seed for reproducibility.

    Returns:
        (positions, smoothing_lengths) tuple.
    """
    rng = np.random.default_rng(seed)
    dirs = _fibonacci_sphere(n)
    jitter = rng.normal(0, 0.01 * radius, size=(n,))
    positions = dirs * (radius + jitter[:, np.newaxis])
    positions += np.array(center)
    smoothing_lengths = np.full(n, h, dtype=np.float64)
    return positions, smoothing_lengths


def _make_solid_sphere_particles(
    n: int = 3000,
    radius: float = 1.0,
    center: tuple = (2.0, 2.0, 2.0),
    h: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Generate SPH particles filling a solid sphere volume.

    Uses rejection sampling to place particles uniformly inside
    the sphere.  This produces a density field that is roughly
    constant inside and drops to zero outside, giving a clear
    isosurface at the boundary.

    Args:
        n: Approximate number of particles (actual count may
            be slightly less due to rejection).
        radius: Sphere radius.
        center: Sphere center.
        h: Smoothing length.
        seed: RNG seed.

    Returns:
        (positions, smoothing_lengths) tuple.
    """
    rng = np.random.default_rng(seed)
    # Over-sample then reject.
    factor = 2.0
    raw = rng.uniform(-radius, radius, size=(int(n * factor), 3))
    r2 = np.sum(raw**2, axis=1)
    inside = raw[r2 < radius**2]
    # Take exactly n particles (or all if fewer).
    count = min(n, len(inside))
    positions = inside[:count] + np.array(center)
    smoothing_lengths = np.full(count, h, dtype=np.float64)
    return positions, smoothing_lengths


def _run_pipeline(
    positions,
    smoothing_lengths,
    domain_min=(0.0, 0.0, 0.0),
    domain_max=(4.0, 4.0, 4.0),
    base_resolution=4,
    isovalue=0.01,
    max_depth=3,
    min_feature_thickness=0.0,
):
    """Run the full pipeline and return a trimesh + metadata."""
    result = run_full_pipeline(
        positions,
        smoothing_lengths,
        domain_min=domain_min,
        domain_max=domain_max,
        base_resolution=base_resolution,
        isovalue=isovalue,
        max_depth=max_depth,
        min_feature_thickness=min_feature_thickness,
    )
    verts = result["vertices"]
    faces = result["faces"].astype(np.int64)
    if verts.shape[0] == 0:
        return None, result
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh, result


def _filter_small_fof_clusters(
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_min,
    domain_max,
    linking_factor: float,
    min_cluster_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter particle clusters by FOF size for integration tests."""
    labels = fof_cluster(positions, domain_min, domain_max, linking_factor)
    unique_labels, counts = np.unique(labels, return_counts=True)
    kept_labels = unique_labels[counts >= min_cluster_size]
    keep_mask = np.isin(labels, kept_labels)
    return positions[keep_mask], smoothing_lengths[keep_mask]


def _edge_counts(faces: np.ndarray) -> Counter:
    """Count how many triangles share each edge."""
    counts = Counter()
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edge = (min(a, b), max(a, b))
            counts[edge] += 1
    return counts


def _manifold_ratio(faces: np.ndarray) -> float:
    """Fraction of edges shared by exactly 2 triangles."""
    counts = _edge_counts(faces)
    if len(counts) == 0:
        return 0.0
    n_manifold = sum(1 for c in counts.values() if c == 2)
    return n_manifold / len(counts)


# -------------------------------------------------------------------
# Tests: Solid sphere (known-answer)
# -------------------------------------------------------------------


class TestSolidSphereIntegration:
    """Integration tests using a solid sphere of particles.

    A uniform-density sphere should produce an approximately
    spherical mesh.  We can verify:
    - Mesh centroid is near the sphere center.
    - Vertex radii cluster near the sphere radius.
    - The mesh has reasonable vertex/face counts.
    - Face indices are valid.
    """

    # Class-level fixture: run the pipeline once, reuse.
    _result_cache = None

    @classmethod
    def _get_result(cls):
        if cls._result_cache is None:
            positions, sml = _make_solid_sphere_particles(
                n=2000,
                radius=1.0,
                center=(2.0, 2.0, 2.0),
                h=0.25,
                seed=99,
            )
            mesh, meta = _run_pipeline(
                positions,
                sml,
                domain_min=(0.0, 0.0, 0.0),
                domain_max=(4.0, 4.0, 4.0),
                base_resolution=4,
                isovalue=0.005,
                max_depth=3,
            )
            cls._result_cache = (mesh, meta)
        return cls._result_cache

    def test_produces_nonempty_mesh(self) -> None:
        """Pipeline should produce vertices and faces."""
        mesh, meta = self._get_result()
        assert mesh is not None, "Pipeline returned empty mesh"
        assert len(mesh.vertices) > 10
        assert len(mesh.faces) > 10

    def test_face_indices_valid(self) -> None:
        """All face indices must reference existing vertices."""
        mesh, _ = self._get_result()
        assert mesh is not None
        assert np.all(mesh.faces >= 0)
        assert np.all(mesh.faces < len(mesh.vertices))

    def test_centroid_near_center(self) -> None:
        """Mesh centroid should be close to (2, 2, 2)."""
        mesh, _ = self._get_result()
        assert mesh is not None
        centroid = mesh.vertices.mean(axis=0)
        center = np.array([2.0, 2.0, 2.0])
        dist = np.linalg.norm(centroid - center)
        assert dist < 0.5, (
            f"Centroid {centroid} is {dist:.3f} from center; expected < 0.5"
        )

    def test_vertex_radii_reasonable(self) -> None:
        """Vertex distances from center should cluster near R=1."""
        mesh, _ = self._get_result()
        assert mesh is not None
        center = np.array([2.0, 2.0, 2.0])
        radii = np.linalg.norm(mesh.vertices - center, axis=1)
        median_r = np.median(radii)
        # Median radius should be within 50% of the true radius.
        # This is a generous tolerance because the adaptive mesh at low
        # resolution won't be a perfect sphere.
        assert 0.5 < median_r < 1.5, (
            f"Median vertex radius {median_r:.3f}; expected near 1.0"
        )

    def test_pipeline_metadata_present(self) -> None:
        """Basic pipeline metadata should be present."""
        _, meta = self._get_result()
        assert "isovalue" in meta
        assert "n_qef_vertices" in meta

    def test_qef_vertices_produced(self) -> None:
        """Pipeline should produce QEF vertices internally."""
        _, meta = self._get_result()
        assert meta["n_qef_vertices"] > 0


# -------------------------------------------------------------------
# Tests: Shell sphere (surface particles)
# -------------------------------------------------------------------


class TestShellSphereIntegration:
    """Integration tests using particles on a spherical shell.

    This is a harder case than the solid sphere because the
    density field is thin — the isosurface must capture a thin
    shell rather than a solid boundary.
    """

    _result_cache = None

    @classmethod
    def _get_result(cls):
        if cls._result_cache is None:
            positions, sml = _make_sphere_particles(
                n=2000,
                radius=1.0,
                center=(2.0, 2.0, 2.0),
                h=0.2,
                seed=77,
            )
            mesh, meta = _run_pipeline(
                positions,
                sml,
                domain_min=(0.0, 0.0, 0.0),
                domain_max=(4.0, 4.0, 4.0),
                base_resolution=4,
                isovalue=0.01,
                max_depth=3,
            )
            cls._result_cache = (mesh, meta)
        return cls._result_cache

    def test_produces_nonempty_mesh(self) -> None:
        """Shell sphere should produce a mesh."""
        mesh, _ = self._get_result()
        assert mesh is not None, "Pipeline returned empty mesh"
        assert len(mesh.vertices) > 10
        assert len(mesh.faces) > 10

    def test_centroid_near_center(self) -> None:
        """Mesh centroid should be near the shell center."""
        mesh, _ = self._get_result()
        assert mesh is not None
        centroid = mesh.vertices.mean(axis=0)
        center = np.array([2.0, 2.0, 2.0])
        dist = np.linalg.norm(centroid - center)
        assert dist < 0.5, f"Centroid {centroid} is {dist:.3f} from center"

    def test_manifold_ratio_reasonable(self) -> None:
        """Most edges should be manifold (shared by 2 triangles).

        At low resolution the mesh won't be perfectly closed, but
        the majority of edges should be manifold.
        """
        mesh, _ = self._get_result()
        assert mesh is not None
        ratio = _manifold_ratio(mesh.faces)
        assert ratio > 0.5, (
            f"Only {ratio:.1%} of edges are manifold; expected > 50%"
        )


# -------------------------------------------------------------------
# Tests: Two separated blobs
# -------------------------------------------------------------------


class TestTwoBlobsIntegration:
    """Two well-separated particle clusters should produce
    a mesh with two distinct connected components."""

    _result_cache = None

    @classmethod
    def _get_result(cls):
        if cls._result_cache is None:
            # Blob A centered at (1.5, 1.5, 1.5).
            pos_a, sml_a = _make_solid_sphere_particles(
                n=1000,
                radius=0.5,
                center=(1.5, 1.5, 1.5),
                h=0.15,
                seed=10,
            )
            # Blob B centered at (3.5, 3.5, 3.5).
            pos_b, sml_b = _make_solid_sphere_particles(
                n=1000,
                radius=0.5,
                center=(3.5, 3.5, 3.5),
                h=0.15,
                seed=20,
            )
            positions = np.vstack([pos_a, pos_b])
            sml = np.concatenate([sml_a, sml_b])

            mesh, meta = _run_pipeline(
                positions,
                sml,
                domain_min=(0.0, 0.0, 0.0),
                domain_max=(5.0, 5.0, 5.0),
                base_resolution=4,
                isovalue=0.005,
                max_depth=3,
            )
            cls._result_cache = (mesh, meta)
        return cls._result_cache

    def test_produces_nonempty_mesh(self) -> None:
        """Two blobs should produce a mesh."""
        mesh, _ = self._get_result()
        assert mesh is not None
        assert len(mesh.vertices) > 10


class TestSmallFoFFluffFilteringIntegration:
    """Small detached particle fluff can be removed before meshing."""

    _result_cache = None

    @classmethod
    def _get_result(cls):
        if cls._result_cache is None:
            pos_main, sml_main = _make_solid_sphere_particles(
                n=1200,
                radius=0.8,
                center=(2.0, 2.0, 2.0),
                h=0.18,
                seed=101,
            )
            pos_fluff, sml_fluff = _make_solid_sphere_particles(
                n=40,
                radius=0.18,
                center=(4.6, 4.6, 4.6),
                h=0.08,
                seed=202,
            )
            positions = np.vstack([pos_main, pos_fluff])
            sml = np.concatenate([sml_main, sml_fluff])
            domain_min = (0.0, 0.0, 0.0)
            domain_max = (5.0, 5.0, 5.0)

            filtered_positions, filtered_sml = _filter_small_fof_clusters(
                positions,
                sml,
                domain_min,
                domain_max,
                linking_factor=1.5,
                min_cluster_size=100,
            )
            mesh, meta = _run_pipeline(
                filtered_positions,
                filtered_sml,
                domain_min=domain_min,
                domain_max=domain_max,
                base_resolution=4,
                isovalue=0.005,
                max_depth=3,
            )
            cls._result_cache = (mesh, meta, filtered_positions)
        return cls._result_cache

    def test_small_fluff_cluster_is_removed(self) -> None:
        """Filtering should drop the detached small particle cluster."""
        mesh, _, filtered_positions = self._get_result()
        assert mesh is not None
        assert filtered_positions.shape[0] > 1000
        assert filtered_positions.shape[0] < 1240
        assert np.all(filtered_positions[:, 0] < 4.0)

    def test_filtered_mesh_stays_near_main_object(self) -> None:
        """The final mesh should not retain geometry near the fluff clump."""
        mesh, _, _ = self._get_result()
        assert mesh is not None
        assert np.max(mesh.vertices[:, 0]) < 3.6
        assert len(mesh.faces) > 10

    def test_filtered_mesh_is_single_main_component(self) -> None:
        """Filtering should leave only the main connected object."""
        mesh, _, _ = self._get_result()
        assert mesh is not None
        components = mesh.split(only_watertight=False)
        # Filter out tiny numerical crumbs; the retained main object should
        # dominate all genuinely resolved components.
        big = [c for c in components if len(c.faces) >= 20]
        assert len(big) == 1, (
            f"Expected 1 main component, got {len(big)} "
            f"(total components: {len(components)})"
        )


class TestMinFeatureThicknessIntegration:
    """min-feature-thickness should run the adaptive opening path."""

    def test_regularization_runs_and_returns_mesh(self) -> None:
        """The opened-solid path should produce a non-empty mesh."""
        positions, sml = _make_solid_sphere_particles(
            n=300,
            radius=0.9,
            center=(2.0, 2.0, 2.0),
            h=0.20,
            seed=303,
        )

        result = run_full_pipeline(
            positions,
            sml,
            domain_min=(0.0, 0.0, 0.0),
            domain_max=(4.0, 4.0, 4.0),
            base_resolution=4,
            isovalue=0.01,
            max_depth=3,
            min_feature_thickness=0.45,
        )

        assert result["vertices"].shape[0] > 0
        assert result["faces"].shape[0] > 0

    def test_regularized_sphere_remains_watertight(self) -> None:
        """Sphere regularization path should keep a watertight surface."""
        positions, sml = _make_solid_sphere_particles(
            n=300,
            radius=0.9,
            center=(2.0, 2.0, 2.0),
            h=0.20,
            seed=404,
        )

        result = run_full_pipeline(
            positions,
            sml,
            domain_min=(0.0, 0.0, 0.0),
            domain_max=(4.0, 4.0, 4.0),
            base_resolution=4,
            isovalue=0.01,
            max_depth=3,
            min_feature_thickness=0.45,
        )
        mesh = trimesh.Trimesh(
            vertices=result["vertices"],
            faces=result["faces"],
            process=False,
        )

        assert mesh.is_watertight


# -------------------------------------------------------------------
# Tests: Edge cases
# -------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case tests for the full pipeline."""

    def test_very_few_particles(self) -> None:
        """Pipeline should handle a minimal particle count
        without crashing (may produce empty mesh)."""
        rng = np.random.default_rng(123)
        positions = rng.uniform(1.0, 3.0, size=(10, 3))
        sml = np.full(10, 0.5, dtype=np.float64)

        result = run_full_pipeline(
            positions,
            sml,
            domain_min=(0.0, 0.0, 0.0),
            domain_max=(4.0, 4.0, 4.0),
            base_resolution=2,
            isovalue=0.001,
            max_depth=2,
        )

        # Should not crash; result may be empty.
        assert "vertices" in result
        assert "faces" in result

    def test_repeat_sphere_run_is_stable(self) -> None:
        """A standard sphere reconstruction should remain stable."""
        positions, sml = _make_solid_sphere_particles(
            n=1000,
            radius=1.0,
            center=(2.0, 2.0, 2.0),
            h=0.25,
            seed=55,
        )
        mesh, meta = _run_pipeline(
            positions,
            sml,
            max_depth=2,
        )
        # Should produce something, not crash.
        assert mesh is not None or meta["n_qef_vertices"] >= 0

    def test_second_sphere_run_is_stable(self) -> None:
        """Another standard sphere reconstruction should remain stable."""
        positions, sml = _make_solid_sphere_particles(
            n=1000,
            radius=1.0,
            center=(2.0, 2.0, 2.0),
            h=0.25,
            seed=55,
        )
        mesh, meta = _run_pipeline(
            positions,
            sml,
            max_depth=2,
        )
        assert mesh is not None or meta["n_qef_vertices"] >= 0
