"""Tree-building and meshing wrappers around the native adaptive core."""

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
    """Create top-level cells and query contributors in one pass."""
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
    """Create the documented top-level octree cells."""
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
    """Create the eight children of one parent octree cell."""
    return _adaptive.create_child_cells(morton_key, bounds, depth)


def filter_child_contributors(
    parent_contributors: list[int],
    positions: list[tuple[float, float, float]],
    smoothing_lengths: list[float],
    parent_bounds: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ],
) -> tuple[tuple[int, ...], ...]:
    """Filter parent contributors into each child cell."""
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
    """Compute Hermite samples for one leaf cell."""
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
    """Refine the octree using breadth-first refinement."""
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
    """Solve the QEF for one leaf cell and return its mesh vertex."""
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
    """Generate a dual-contour mesh from pre-built octree cells."""
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
    """Solve QEF vertices for all active leaf cells in a refined octree."""
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
    """Run the octree pipeline in C++ and return QEF vertices."""
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
    """Build and refine the adaptive tree in C++ and return resumable state."""
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    sml = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
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
