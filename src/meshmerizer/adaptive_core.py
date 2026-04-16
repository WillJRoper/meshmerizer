"""Python wrapper for the adaptive meshing C++ core.

This module is the stable Python entry point for the new adaptive rewrite. The
initial implementation only exposes scaffold-level functionality while the C++
core types and algorithms are introduced in later phases.
"""

from __future__ import annotations

from importlib import import_module

_adaptive = import_module("meshmerizer._adaptive")


def adaptive_status() -> str:
    """Return the current adaptive core scaffold status string.

    Returns:
        Short human-readable status from the C++ adaptive extension.
    """
    # Delegate to the compiled module so tests verify the C++ extension is the
    # real implementation entry point.
    return _adaptive.adaptive_status()


def morton_encode_3d(x: int, y: int, z: int) -> int:
    """Encode three integer coordinates into a 3-D Morton key.

    Args:
        x: X-axis integer coordinate.
        y: Y-axis integer coordinate.
        z: Z-axis integer coordinate.

    Returns:
        Bit-interleaved Morton key.
    """
    return _adaptive.morton_encode_3d(x, y, z)


def morton_decode_3d(key: int) -> tuple[int, int, int]:
    """Decode a 3-D Morton key back into integer coordinates.

    Args:
        key: Bit-interleaved Morton key.

    Returns:
        Tuple containing the decoded `(x, y, z)` coordinates.
    """
    return _adaptive.morton_decode_3d(key)


def bounding_box_contains(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    point: tuple[float, float, float],
) -> bool:
    """Return whether a point lies within a half-open bounding box.

    Args:
        minimum: Inclusive lower corner of the bounding box.
        maximum: Exclusive upper corner of the bounding box.
        point: Point to test.

    Returns:
        `True` if the point lies inside the bounding box.
    """
    return _adaptive.bounding_box_contains(minimum, maximum, point)


def bounding_box_overlaps(
    left_minimum: tuple[float, float, float],
    left_maximum: tuple[float, float, float],
    right_minimum: tuple[float, float, float],
    right_maximum: tuple[float, float, float],
) -> bool:
    """Return whether two half-open bounding boxes overlap.

    Args:
        left_minimum: Inclusive lower corner of the left bounding box.
        left_maximum: Exclusive upper corner of the left bounding box.
        right_minimum: Inclusive lower corner of the right bounding box.
        right_maximum: Exclusive upper corner of the right bounding box.

    Returns:
        `True` when the boxes overlap with positive volume.
    """
    return _adaptive.bounding_box_overlaps(
        left_minimum,
        left_maximum,
        right_minimum,
        right_maximum,
    )


def particle_fields() -> tuple[str, str, str, str]:
    """Return the documented field names of the adaptive particle payload.

    Returns:
        Tuple containing the current adaptive `Particle` member names.
    """
    return _adaptive.particle_fields()


def wendland_c2_value(
    radius: float,
    smoothing_length: float,
    normalize: bool = False,
) -> float:
    """Evaluate the Wendland C2 kernel at one radius.

    Args:
        radius: Distance from the particle center.
        smoothing_length: Particle kernel support radius.
        normalize: Whether to apply the 3-D normalization constant.

    Returns:
        Kernel value at the requested radius.
    """
    return _adaptive.wendland_c2_value(radius, smoothing_length, normalize)


def wendland_c2_gradient(
    displacement: tuple[float, float, float],
    smoothing_length: float,
    normalize: bool = False,
) -> tuple[float, float, float]:
    """Evaluate the Wendland C2 kernel gradient for one displacement.

    Args:
        displacement: Vector from particle center to query position.
        smoothing_length: Particle kernel support radius.
        normalize: Whether to apply the 3-D normalization constant.

    Returns:
        Gradient vector of the kernel at the requested displacement.
    """
    return _adaptive.wendland_c2_gradient(
        displacement,
        smoothing_length,
        normalize,
    )


def top_level_bin_counts(
    positions: list[tuple[float, float, float]],
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    resolution: int,
) -> tuple[int, ...]:
    """Count particles in the flattened top-level bins.

    Args:
        positions: Particle positions in world space.
        domain_minimum: Inclusive lower corner of the working domain.
        domain_maximum: Exclusive upper corner of the working domain.
        resolution: Number of bins per axis.

    Returns:
        Flattened row-major tuple of particle counts per top-level bin.
    """
    return _adaptive.top_level_bin_counts(
        positions,
        domain_minimum,
        domain_maximum,
        resolution,
    )


def query_cell_contributors(
    positions: list[tuple[float, float, float]],
    smoothing_lengths: list[float],
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    resolution: int,
    cell_minimum: tuple[float, float, float],
    cell_maximum: tuple[float, float, float],
) -> tuple[int, ...]:
    """Return candidate contributor indices for one query cell.

    Args:
        positions: Particle positions in world space.
        smoothing_lengths: Per-particle support radii.
        domain_minimum: Inclusive lower corner of the working domain.
        domain_maximum: Exclusive upper corner of the working domain.
        resolution: Number of top-level bins per axis.
        cell_minimum: Inclusive lower corner of the query cell.
        cell_maximum: Exclusive upper corner of the query cell.

    Returns:
        Tuple of particle indices whose support overlaps the query cell.
    """
    return _adaptive.query_cell_contributors(
        positions,
        smoothing_lengths,
        domain_minimum,
        domain_maximum,
        resolution,
        cell_minimum,
        cell_maximum,
    )


def cell_may_contain_isosurface(
    corner_values: list[float],
    isovalue: float,
) -> bool:
    """Return whether eight corner values can contain the isosurface.

    Args:
        corner_values: Scalar field samples at the eight cell corners.
        isovalue: Requested surface level.

    Returns:
        `True` when the sampled value range straddles the isovalue.
    """
    return _adaptive.cell_may_contain_isosurface(corner_values, isovalue)


def corner_sign_mask(corner_values: list[float], isovalue: float) -> int:
    """Return the bit mask of corner values relative to the isovalue.

    Args:
        corner_values: Scalar field samples at the eight cell corners.
        isovalue: Requested surface level.

    Returns:
        Integer bit mask with one bit per corner for `value >= isovalue`.
    """
    return _adaptive.corner_sign_mask(corner_values, isovalue)


def create_top_level_cells(
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    base_resolution: int,
) -> tuple[dict[str, object], ...]:
    """Create the documented top-level octree cells.

    Args:
        domain_minimum: Inclusive lower corner of the working domain.
        domain_maximum: Exclusive upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.

    Returns:
        Tuple of dictionaries describing the top-level cells.
    """
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
        morton_key: Parent cell Morton key.
        bounds: Parent cell bounds as `(minimum, maximum)`.
        depth: Parent cell depth.

    Returns:
        Tuple of dictionaries describing the child cells.
    """
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
        parent_contributors: Particle indices attached to the parent cell.
        positions: Particle positions in world space.
        smoothing_lengths: Per-particle support radii.
        parent_bounds: Parent cell bounds as `(minimum, maximum)`.

    Returns:
        Tuple containing one contributor-index tuple per child cell.
    """
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

    For each sign-changing edge of the cell, linearly interpolates the
    isosurface crossing point and evaluates the outward SPH gradient normal.

    Args:
        bounds: Cell bounding box as ``(minimum, maximum)``.
        corner_values: Scalar field samples at the eight cell corners.
        corner_sign_mask: Precomputed sign mask from ``corner_sign_mask()``.
        contributor_indices: Particle indices contributing to this cell.
        positions: Particle positions in world space.
        smoothing_lengths: Per-particle support radii.
        isovalue: The target surface level.

    Returns:
        Tuple of ``((px, py, pz), (nx, ny, nz))`` sample pairs, one per
        sign-changing edge.  The normal is the outward surface normal
        (pointing away from the fluid).  A zero-length normal indicates a
        degenerate sample where the SPH gradient vanished at the crossing.
    """
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
) -> tuple[tuple[dict[str, object], ...], tuple[int, ...]]:
    """Refine the octree using breadth-first refinement.

    Args:
        initial_cells: Top-level cells with attached contributors.
        positions: Particle positions in world space.
        smoothing_lengths: Per-particle support radii.
        isovalue: The target surface level.
        max_depth: Maximum refinement depth.

    Returns:
        Tuple of (all_cells, all_contributors) where cells store indices
        into the contributors vector.
    """
    return _adaptive.refine_octree(
        initial_cells,
        positions,
        smoothing_lengths,
        isovalue,
        max_depth,
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

    The QEF minimizer finds the point inside the cell that best
    satisfies all tangent-plane constraints from the provided Hermite
    samples.  If the system is rank-deficient the solver falls back
    to the sample centroid.  The result is always clamped to the cell
    bounding box.

    Args:
        samples: Hermite samples as ``((px, py, pz), (nx, ny, nz))``
            pairs.  Zero-length normals are treated as degenerate and
            ignored.
        bounds: Cell bounding box as ``(minimum, maximum)``.

    Returns:
        Tuple ``((px, py, pz), (nx, ny, nz))`` giving the vertex
        position and the normalized mean sample normal.
    """
    return _adaptive.solve_qef_for_leaf(samples, bounds)
