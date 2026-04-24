"""Tree-building and meshing wrappers around the native adaptive core.

This module groups together the native entry points that operate on adaptive
octree state directly. The public API and CLI use these wrappers when they need
to build a resumable tree, inspect intermediate cell data, or extract a mesh
from pre-built octree state rather than running the one-shot full pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._native import _adaptive

if TYPE_CHECKING:
    import numpy


def create_top_level_cells_with_contributors(
    positions: list[tuple[float, float, float]],
    smoothing_lengths: list[float],
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    base_resolution: int,
) -> tuple[dict, ...]:
    """Create top-level cells and query contributors in one pass.

    Args:
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths with shape ``(N,)``.
        domain_minimum: Lower corner of the working domain.
        domain_maximum: Upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.

    Returns:
        Native tuple containing the top-level cell dictionaries and aligned
        contributor payload needed by later refinement stages.
    """
    # Delegate the combined construction/query step to the native helper so
    # Python callers can reuse the documented historical bridge format.
    return _adaptive.create_top_level_cells_with_contributors(
        positions,
        smoothing_lengths,
        domain_minimum,
        domain_maximum,
        base_resolution,
    )


def create_top_level_cells(
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    base_resolution: int,
) -> tuple[dict[str, object], ...]:
    """Create the documented top-level octree cells.

    Args:
        domain_minimum: Lower corner of the working domain.
        domain_maximum: Upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.

    Returns:
        Tuple of top-level cell dictionaries in the historical bridge format.
    """
    # This path is useful for tests and diagnostics that want the explicit
    # top-level cell layout before any contributor lookup or refinement.
    return _adaptive.create_top_level_cells(
        domain_minimum,
        domain_maximum,
        base_resolution,
    )


def create_child_cells(
    morton_key: int,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    depth: int,
) -> tuple[dict[str, object], ...]:
    """Create the eight children of one parent octree cell.

    Args:
        morton_key: Parent Morton key.
        bounds: Parent cell bounds as ``(minimum, maximum)``.
        depth: Parent cell depth.

    Returns:
        Tuple of eight child cell dictionaries.
    """
    # Child-cell generation is kept native so Morton bookkeeping and geometric
    # subdivision stay perfectly aligned with the refinement implementation.
    return _adaptive.create_child_cells(morton_key, bounds, depth)


def filter_child_contributors(
    parent_contributors: list[int],
    positions: list[tuple[float, float, float]],
    smoothing_lengths: list[float],
    parent_bounds: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ],
) -> tuple[tuple[int, ...], ...]:
    """Filter parent contributors into each child cell.

    Args:
        parent_contributors: Contributor indices attached to the parent cell.
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths.
        parent_bounds: Parent cell bounds as ``(minimum, maximum)``.

    Returns:
        Tuple of eight contributor-index tuples, one per child cell.
    """
    # Restricting the search to parent contributors keeps refinement cheaper
    # than re-querying the full particle set for every child.
    return _adaptive.filter_child_contributors(
        parent_contributors,
        positions,
        smoothing_lengths,
        parent_bounds,
    )


def hermite_samples_for_cell(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    corner_values: list[float],
    corner_sign_mask: int,
    contributor_indices: list[int],
    positions: list[tuple[float, float, float]],
    smoothing_lengths: list[float],
    isovalue: float,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    """Compute Hermite samples for one leaf cell.

    Args:
        bounds: Leaf-cell bounds as ``(minimum, maximum)``.
        corner_values: Scalar field values at the eight cell corners.
        corner_sign_mask: Bit mask describing which corners lie above the
            isovalue.
        contributor_indices: Candidate particle indices for the cell.
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths.
        isovalue: Scalar field threshold.

    Returns:
        Tuple of Hermite samples, each as ``(position, normal)``.
    """
    # Hermite samples are the geometry constraints used by QEF vertex solving,
    # so exposing them helps tests and diagnostics inspect reconstruction
    # quality at the cell level.
    return _adaptive.hermite_samples_for_cell(
        bounds,
        corner_values,
        corner_sign_mask,
        contributor_indices,
        positions,
        smoothing_lengths,
        isovalue,
    )


def refine_octree(
    initial_cells: tuple[dict[str, object], ...],
    positions: list[tuple[float, float, float]],
    smoothing_lengths: list[float],
    isovalue: float,
    max_depth: int,
    domain: tuple[tuple[float, float, float], tuple[float, float, float]],
    base_resolution: int,
    minimum_usable_hermite_samples: int = 3,
    max_qef_rms_residual_ratio: float = 0.1,
    min_normal_alignment_threshold: float = 0.97,
) -> tuple[tuple[dict[str, object], ...], tuple[int, ...]]:
    """Refine the octree using breadth-first refinement.

    Args:
        initial_cells: Starting cell dictionaries, usually top-level cells.
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths.
        isovalue: Scalar field threshold used for split decisions.
        max_depth: Maximum permitted refinement depth.
        domain: Full working-domain bounds.
        base_resolution: Number of top-level cells per axis.
        minimum_usable_hermite_samples: Minimum usable Hermite sample count.
        max_qef_rms_residual_ratio: Maximum acceptable RMS QEF residual ratio.
        min_normal_alignment_threshold: Minimum acceptable normal alignment.

    Returns:
        Tuple ``(cells, contributors)`` describing the refined octree.
    """
    # The native routine owns the real refinement queue, Hermite sampling, and
    # split heuristics; this wrapper just preserves the Python-facing contract.
    return _adaptive.refine_octree(
        initial_cells,
        positions,
        smoothing_lengths,
        isovalue,
        max_depth,
        domain,
        base_resolution,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
    )


def solve_qef_for_leaf(
    samples: list[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ],
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Solve the QEF for one leaf cell and return its mesh vertex.

    Args:
        samples: Hermite samples as ``(position, normal)`` pairs.
        bounds: Leaf-cell bounds as ``(minimum, maximum)``.

    Returns:
        Tuple ``(position, normal)`` for the solved vertex.
    """
    # Keep the solve at the native boundary so the Python helper remains a thin
    # inspection/debug hook rather than a second implementation.
    return _adaptive.solve_qef_for_leaf(samples, bounds)


def generate_mesh(
    cells: list[dict[str, object]],
    contributors: list[int],
    positions: list[tuple[float, float, float]],
    smoothing_lengths: list[float],
    isovalue: float,
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    max_depth: int,
    base_resolution: int,
) -> tuple["numpy.ndarray", "numpy.ndarray", "numpy.ndarray"]:
    """Generate a dual-contour mesh from pre-built octree cells.

    Args:
        cells: Refined octree cell dictionaries.
        contributors: Flat contributor index array.
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths.
        isovalue: Scalar field threshold for extraction.
        domain_minimum: Lower corner of the working domain.
        domain_maximum: Upper corner of the working domain.
        max_depth: Maximum permitted refinement depth.
        base_resolution: Number of top-level cells per axis.

    Returns:
        Native tuple of ``(vertices, normals, faces)`` arrays.
    """
    # This wrapper is used when callers already have a refined tree and want to
    # skip rebuilding it before meshing.
    return _adaptive.generate_mesh(
        cells,
        contributors,
        positions,
        smoothing_lengths,
        isovalue,
        domain_minimum,
        domain_maximum,
        max_depth,
        base_resolution,
    )


def solve_vertices(
    cells: list[dict[str, object]],
    contributors: list[int],
    positions: list[tuple[float, float, float]],
    smoothing_lengths: list[float],
    isovalue: float,
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    max_depth: int,
    base_resolution: int,
) -> tuple["numpy.ndarray", "numpy.ndarray"]:
    """Solve QEF vertices for all active leaf cells in a refined octree.

    Args:
        cells: Refined octree cell dictionaries.
        contributors: Flat contributor index array.
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths.
        isovalue: Scalar field threshold for Hermite sampling.
        domain_minimum: Lower corner of the working domain.
        domain_maximum: Upper corner of the working domain.
        max_depth: Maximum permitted refinement depth.
        base_resolution: Number of top-level cells per axis.

    Returns:
        Tuple of solved vertex positions and associated normals.
    """
    # Vertex-only solving powers CLI diagnostics without forcing full face
    # generation, which is useful when debugging octree quality.
    return _adaptive.solve_vertices(
        cells,
        contributors,
        positions,
        smoothing_lengths,
        isovalue,
        domain_minimum,
        domain_maximum,
        max_depth,
        base_resolution,
    )


def run_octree_pipeline(
    positions: "numpy.ndarray",
    smoothing_lengths: "numpy.ndarray",
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    base_resolution: int,
    isovalue: float,
    max_depth: int,
    minimum_usable_hermite_samples: int = 3,
    max_qef_rms_residual_ratio: float = 0.1,
    min_normal_alignment_threshold: float = 0.97,
) -> tuple["numpy.ndarray", "numpy.ndarray"]:
    """Run the octree pipeline in C++ and return QEF vertices.

    Args:
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths with shape ``(N,)``.
        domain_minimum: Lower corner of the working domain.
        domain_maximum: Upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.
        isovalue: Scalar field threshold.
        max_depth: Maximum permitted refinement depth.
        minimum_usable_hermite_samples: Minimum usable Hermite sample count.
        max_qef_rms_residual_ratio: Maximum acceptable RMS QEF residual ratio.
        min_normal_alignment_threshold: Minimum acceptable normal alignment.

    Returns:
        Tuple of QEF vertex positions and normals.
    """
    # This path is a compact diagnostic entry point: it runs tree build and QEF
    # solving but stops short of face generation.
    return _adaptive.run_octree_pipeline(
        positions,
        smoothing_lengths,
        domain_minimum,
        domain_maximum,
        base_resolution,
        isovalue,
        max_depth,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
    )


def build_refined_tree(
    positions: "numpy.ndarray",
    smoothing_lengths: "numpy.ndarray",
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    base_resolution: int,
    isovalue: float,
    max_depth: int,
    minimum_usable_hermite_samples: int = 3,
    max_qef_rms_residual_ratio: float = 0.1,
    min_normal_alignment_threshold: float = 0.97,
) -> tuple[tuple[dict[str, object], ...], "numpy.ndarray"]:
    """Build and refine the adaptive tree in C++ and return resumable state.

    Args:
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths with shape ``(N,)``.
        domain_minimum: Lower corner of the working domain.
        domain_maximum: Upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.
        isovalue: Scalar field threshold used for split decisions.
        max_depth: Maximum permitted refinement depth.
        minimum_usable_hermite_samples: Minimum usable Hermite sample count.
        max_qef_rms_residual_ratio: Maximum acceptable RMS QEF residual ratio.
        min_normal_alignment_threshold: Minimum acceptable normal alignment.

    Returns:
        Tuple ``(cells, contributors)`` suitable for later extraction or
        serialization.
    """
    # Normalize to contiguous float64 buffers before crossing into C++ so the
    # native side can assume stable memory layout.
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    sml = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
    # Return the contributors as a NumPy array because downstream state objects
    # and serialization logic operate on array-like integer buffers.
    cells, contributors = _adaptive.build_refined_tree(
        pos,
        sml,
        domain_minimum,
        domain_maximum,
        base_resolution,
        isovalue,
        max_depth,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
    )
    return cells, np.asarray(contributors, dtype=np.int64)


__all__ = [
    "build_refined_tree",
    "create_child_cells",
    "create_top_level_cells",
    "create_top_level_cells_with_contributors",
    "filter_child_contributors",
    "generate_mesh",
    "hermite_samples_for_cell",
    "refine_octree",
    "run_octree_pipeline",
    "solve_qef_for_leaf",
    "solve_vertices",
]
