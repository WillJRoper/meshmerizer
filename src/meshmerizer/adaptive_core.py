"""Python wrapper for the adaptive meshing C++ core.

This module is the stable Python entry point for the new adaptive rewrite. The
initial implementation only exposes scaffold-level functionality while the C++
core types and algorithms are introduced in later phases.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy

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


def create_top_level_cells_with_contributors(
    positions: list[tuple[float, float, float]],
    smoothing_lengths: list[float],
    domain_minimum: tuple[float, float, float],
    domain_maximum: tuple[float, float, float],
    base_resolution: int,
) -> tuple[dict, ...]:
    """Create top-level cells and query contributors in one pass.

    Builds the particle grid once and reuses it for every top-level
    cell, avoiding repeated O(n_particles) binning.

    Args:
        positions: Particle positions in world space.
        smoothing_lengths: Per-particle support radii.
        domain_minimum: Inclusive lower corner of the working domain.
        domain_maximum: Exclusive upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.

    Returns:
        Tuple of cell dicts, each with a ``"contributors"`` key
        holding a tuple of particle indices.
    """
    return _adaptive.create_top_level_cells_with_contributors(
        positions,
        smoothing_lengths,
        domain_minimum,
        domain_maximum,
        base_resolution,
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
    domain: tuple[tuple[float, float, float], tuple[float, float, float]],
    base_resolution: int,
) -> tuple[tuple[dict[str, object], ...], tuple[int, ...]]:
    """Refine the octree using breadth-first refinement.

    Args:
        initial_cells: Top-level cells with attached contributors.
        positions: Particle positions in world space.
        smoothing_lengths: Per-particle support radii.
        isovalue: The target surface level.
        max_depth: Maximum refinement depth.
        domain: Bounding box of the simulation domain as
            ``((min_x, min_y, min_z), (max_x, max_y, max_z))``.
        base_resolution: Number of top-level cells per axis.

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
        domain,
        base_resolution,
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

    This is the legacy dual contouring path kept as a fallback
    alternative to the Poisson reconstruction pipeline.  It
    solves QEF vertices AND generates triangle faces entirely
    in C++.

    Args:
        cells: Octree cell dictionaries (from ``refine_octree``).
        contributors: Flat contributor index array.
        positions: Particle positions in world space.
        smoothing_lengths: Per-particle support radii.
        isovalue: Target surface level.
        domain_minimum: Inclusive lower corner of the working domain.
        domain_maximum: Exclusive upper corner of the working domain.
        max_depth: Maximum octree depth.
        base_resolution: Number of top-level cells per axis.

    Returns:
        Tuple of ``(vert_positions, vert_normals, triangles)`` where
        ``vert_positions`` is an (N, 3) float64 array of vertex
        positions, ``vert_normals`` is an (N, 3) float64 array of
        normals, and ``triangles`` is an (M, 3) int64 array of
        triangle indices.
    """
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

    This is the stepwise counterpart to ``run_octree_pipeline``:
    it takes pre-built octree cells (e.g. from a saved HDF5 file)
    and solves QEF vertex positions and normals without repeating
    the tree construction.

    Args:
        cells: Octree cell dictionaries (from ``refine_octree``).
        contributors: Flat contributor index array.
        positions: Particle positions in world space.
        smoothing_lengths: Per-particle support radii.
        isovalue: Target surface level.
        domain_minimum: Inclusive lower corner of the working domain.
        domain_maximum: Exclusive upper corner of the working domain.
        max_depth: Maximum octree depth.
        base_resolution: Number of top-level cells per axis.

    Returns:
        Tuple of ``(vert_positions, vert_normals)`` where
        ``vert_positions`` is an (N, 3) float64 array of QEF vertex
        positions and ``vert_normals`` is an (N, 3) float64 array
        of outward-facing unit normals.
    """
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
) -> tuple["numpy.ndarray", "numpy.ndarray"]:
    """Run the octree pipeline in C++ and return QEF vertices.

    This combines top-level cell creation, contributor queries,
    octree refinement, and QEF vertex solving into a single C++
    call.  Triangle face generation is NOT included — that is
    handled in Python via FOF clustering + Poisson reconstruction.

    Args:
        positions: Nx3 float64 array of particle positions.
        smoothing_lengths: N float64 array of support radii.
        domain_minimum: Simulation domain lower corner.
        domain_maximum: Simulation domain upper corner.
        base_resolution: Number of top-level cells per axis.
        isovalue: Target surface level.
        max_depth: Maximum octree refinement depth.

    Returns:
        Tuple of ``(vert_positions, vert_normals)`` where
        ``vert_positions`` is an (N, 3) float64 array of QEF
        vertex positions and ``vert_normals`` is an (N, 3) float64
        array of outward-facing unit normals.
    """
    return _adaptive.run_octree_pipeline(
        positions,
        smoothing_lengths,
        domain_minimum,
        domain_maximum,
        base_resolution,
        isovalue,
        max_depth,
    )


def compute_isovalue_from_percentile(
    smoothing_lengths: "numpy.ndarray",
    percentile: float,
) -> float:
    """Compute an isovalue from a density percentile of the particles.

    In SPH the smoothing length ``h`` adapts to the local particle
    density: denser regions have smaller ``h``.  The Wendland C2 kernel
    self-contribution at a particle's own position is:

        W(0, h) = 21 / (2 * pi * h^3)

    This is proportional to the local number density and serves as a
    fast, parameter-free proxy for the full SPH density field that the
    octree evaluates at cell corners.

    The ``percentile`` parameter controls where the isosurface sits
    relative to the distribution of these self-density values:

    - ``percentile=5`` places the surface at the 5th percentile,
      enclosing ~95% of the particle mass.  Good for capturing the
      full extent of a galaxy or halo.
    - ``percentile=50`` places the surface at the median density,
      showing only the denser half of the distribution.

    Args:
        smoothing_lengths: (N,) float64 array of per-particle
            support radii.
        percentile: Percentile of the self-density distribution
            at which to place the isosurface.  Must be in [0, 100].

    Returns:
        Isovalue suitable for passing to the adaptive pipeline.

    Raises:
        ValueError: If ``percentile`` is outside [0, 100] or the
            input array is empty.
    """
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    h = np.asarray(smoothing_lengths, dtype=np.float64)
    if h.size == 0:
        raise ValueError("smoothing_lengths array is empty")

    # Wendland C2 3-D normalization: 21 / (2 * pi * h^3).
    self_density = 21.0 / (2.0 * np.pi * h**3)

    isovalue = float(np.percentile(self_density, percentile))
    return isovalue


def fof_cluster(
    positions: "numpy.ndarray",
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    linking_factor: float = 1.5,
) -> "numpy.ndarray":
    """Cluster points using a friends-of-friends algorithm.

    Groups 3-D positions into connected components where any two
    points within a linking length of each other belong to the same
    group.  The linking length is derived from the mean inter-point
    separation scaled by ``linking_factor``.

    This is used before Poisson surface reconstruction to identify
    distinct objects (e.g. separate galaxies or halos) so that each
    object can be reconstructed independently, avoiding thin bridges
    between unrelated structures.

    Args:
        positions: (N, 3) float64 array of point positions.
        domain_min: Lower corner of the spatial domain as
            ``(x_min, y_min, z_min)``.
        domain_max: Upper corner of the spatial domain as
            ``(x_max, y_max, z_max)``.
        linking_factor: Multiplicative factor applied to the mean
            inter-point separation to obtain the linking length.
            Default is 1.5.

    Returns:
        (N,) int64 array of group labels starting from 0.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    labels = _adaptive.fof_cluster(pos, domain_min, domain_max, linking_factor)
    return labels


# ---- Poisson basis functions (Phase 20a) --------------------------


def bspline1d_evaluate(t: float) -> float:
    """Evaluate the 1-D degree-1 (hat) B-spline at normalised t.

    Args:
        t: Normalised coordinate (distance from centre in units
            of cell width).

    Returns:
        B-spline value in [0, 1].
    """
    return _adaptive.bspline1d_evaluate(t)


def bspline1d_derivative(t: float) -> float:
    """Evaluate the derivative of the 1-D degree-1 B-spline at t.

    Args:
        t: Normalised coordinate.

    Returns:
        Derivative value in {-1, 0, 1}.
    """
    return _adaptive.bspline1d_derivative(t)


def bspline3d_evaluate(
    point: tuple[float, float, float],
    center: tuple[float, float, float],
    width: float,
) -> float:
    """Evaluate the 3-D trilinear B-spline.

    Args:
        point: Evaluation point (x, y, z).
        center: Cell centre (x, y, z).
        width: Cell width.

    Returns:
        Basis function value in [0, 1].
    """
    return _adaptive.bspline3d_evaluate(point, center, width)


def bspline3d_gradient(
    point: tuple[float, float, float],
    center: tuple[float, float, float],
    width: float,
) -> tuple[float, float, float]:
    """Evaluate the gradient of the 3-D trilinear B-spline.

    Args:
        point: Evaluation point (x, y, z).
        center: Cell centre (x, y, z).
        width: Cell width.

    Returns:
        Gradient tuple (dB/dx, dB/dy, dB/dz).
    """
    return _adaptive.bspline3d_gradient(point, center, width)


def assign_dof_indices(
    cells: list[dict],
) -> tuple[list[int], list[int]]:
    """Assign contiguous DOF indices to leaf cells.

    Args:
        cells: List of cell dicts, each with an ``"is_leaf"`` key.

    Returns:
        Tuple of (cell_to_dof, dof_to_cell) lists.
    """
    return _adaptive.assign_dof_indices(cells)


def enumerate_stencils(
    cells: list[dict],
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    base_resolution: int,
    max_depth: int,
) -> tuple[list[int], list[int]]:
    """Enumerate DOF stencils (neighbor DOFs) for each DOF.

    Args:
        cells: List of cell dicts with keys ``is_leaf``,
            ``depth``, ``bounds_min``, ``bounds_max``,
            ``morton_key``.
        domain_min: Lower corner of the domain.
        domain_max: Upper corner of the domain.
        base_resolution: Top-level cells per axis.
        max_depth: Maximum octree depth.

    Returns:
        Tuple of (stencil_offsets, stencil_neighbors).
    """
    return _adaptive.enumerate_stencils(
        cells,
        domain_min,
        domain_max,
        base_resolution,
        max_depth,
    )


def splat_and_compute_rhs(
    positions: "numpy.ndarray",
    normals: "numpy.ndarray",
    cells: list[dict],
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    base_resolution: int,
    max_depth: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Splat oriented normals and compute the Poisson RHS.

    Performs two steps of the screened Poisson pipeline:

    1. **Normal splatting** (SGP06 Sec 3): distributes each
       sample's unit normal into the overlapping B-spline DOFs,
       weighted by the trilinear basis value and a uniform area
       weight (1/N).
    2. **RHS assembly** (SGP06 Sec 3): computes
       ``b_i = sum_j V_j . G_ij`` where ``G_ij`` is the
       precomputed gradient inner product between B-spline
       basis functions.

    Args:
        positions: (N, 3) float64 array of sample positions.
        normals: (N, 3) float64 array of sample unit normals.
        cells: List of cell dicts with keys ``is_leaf``,
            ``depth``, ``bounds_min``, ``bounds_max``,
            ``morton_key``.
        domain_min: Lower corner of the domain.
        domain_max: Upper corner of the domain.
        base_resolution: Top-level cells per axis.
        max_depth: Maximum octree depth.

    Returns:
        Tuple of (v_field_x, v_field_y, v_field_z, rhs) where
        each is a list of floats with one entry per DOF.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    nor = np.ascontiguousarray(normals, dtype=np.float64)
    return _adaptive.splat_and_compute_rhs(
        pos,
        nor,
        cells,
        domain_min,
        domain_max,
        base_resolution,
        max_depth,
    )


def laplacian_stencil_weight(
    dx: int,
    dy: int,
    dz: int,
    h: float,
) -> float:
    """Compute the 3-D Laplacian stencil weight for a given offset.

    For degree-1 B-splines, the Laplacian stencil weight at
    offset (dx, dy, dz) with cell width h is:

        L = h * [K(dx)*M(dy)*M(dz) + M(dx)*K(dy)*M(dz)
              + M(dx)*M(dy)*K(dz)]

    where K is the 1-D stiffness integral and M is the 1-D mass
    integral.

    Args:
        dx: Offset in x (-1, 0, +1).
        dy: Offset in y (-1, 0, +1).
        dz: Offset in z (-1, 0, +1).
        h: Cell width.

    Returns:
        Stiffness integral value.
    """
    return _adaptive.laplacian_stencil_weight(dx, dy, dz, h)


def apply_poisson_operator(
    positions: "numpy.ndarray",
    cells: list[dict],
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    base_resolution: int,
    max_depth: int,
    alpha: float,
    x: list[float],
) -> list[float]:
    """Apply the screened Poisson operator A*x.

    Builds the Laplacian stencil and screening term, then
    computes A*x = (L + S)*x where L is the Laplacian and
    S is the point-sampled screening matrix.

    Args:
        positions: (N, 3) float64 array of sample positions
            (used for screening accumulation).
        cells: List of cell dicts.
        domain_min: Lower corner of the domain.
        domain_max: Upper corner of the domain.
        base_resolution: Top-level cells per axis.
        max_depth: Maximum octree depth.
        alpha: Screening weight (0.0 for pure Laplacian).
        x: Input vector (length = n_dofs).

    Returns:
        List of floats: the result A*x.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    return _adaptive.apply_poisson_operator(
        pos,
        cells,
        domain_min,
        domain_max,
        base_resolution,
        max_depth,
        alpha,
        x,
    )


def solve_poisson(
    positions: "numpy.ndarray",
    cells: list[dict],
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    base_resolution: int,
    max_depth: int,
    alpha: float,
    b: list[float],
    max_iters: int = 1000,
    tol: float = 1e-10,
) -> tuple[list[float], int, float, bool]:
    """Solve the screened Poisson system Ax = b with PCG.

    Builds the operator (Laplacian + screening) and solves
    using preconditioned Conjugate Gradients with Jacobi
    preconditioning.

    Args:
        positions: (N, 3) float64 array of sample positions
            (used for screening accumulation).
        cells: List of cell dicts.
        domain_min: Lower corner of the domain.
        domain_max: Upper corner of the domain.
        base_resolution: Top-level cells per axis.
        max_depth: Maximum octree depth.
        alpha: Screening weight (0.0 for pure Laplacian).
        b: Right-hand side vector (length = n_dofs).
        max_iters: Maximum CG iterations.
        tol: Relative residual tolerance.

    Returns:
        Tuple of (solution, iterations, residual_norm,
        converged) where solution is a list of floats.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    return _adaptive.solve_poisson(
        pos,
        cells,
        domain_min,
        domain_max,
        base_resolution,
        max_depth,
        alpha,
        b,
        max_iters,
        tol,
    )


def extract_poisson_mesh(
    positions: np.ndarray,
    cells,
    domain_min,
    domain_max,
    base_resolution: int,
    max_depth: int,
    solution,
):
    """Extract an isosurface from a Poisson solution via Marching Cubes.

    Evaluates the indicator function chi at leaf-cell corners, computes
    the isovalue as the mean chi at sample positions, and runs classic
    Marching Cubes (Lorensen & Cline 1987) to extract the isosurface.

    Args:
        positions: (N, 3) float64 array of sample positions (used to
            compute the isovalue).
        cells: List of cell dicts with keys ``is_leaf``, ``depth``,
            ``bounds_min``, ``bounds_max``, ``morton_key``.
        domain_min: (3,) lower corner of the domain bounding box.
        domain_max: (3,) upper corner of the domain bounding box.
        base_resolution: Number of top-level cells per axis.
        max_depth: Maximum octree depth.
        solution: Poisson solution vector (list or array of floats,
            length = n_dofs).

    Returns:
        Tuple of (vertices, triangles, isovalue) where vertices is a
        (V, 3) float64 ndarray, triangles is a (F, 3) uint32 ndarray,
        and isovalue is a float.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    sol = list(solution)
    return _adaptive.extract_poisson_mesh(
        pos,
        cells,
        domain_min,
        domain_max,
        base_resolution,
        max_depth,
        sol,
    )


def run_full_pipeline(
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_min: tuple,
    domain_max: tuple,
    base_resolution: int,
    isovalue: float,
    max_depth: int,
    screening_weight: float = 4.0,
    max_iters: int = 1000,
    tol: float = 1e-6,
) -> dict:
    """Run the full particles-to-mesh pipeline in C++.

    Combines octree construction, QEF vertex solving, Poisson
    surface reconstruction, and Marching Cubes isosurface
    extraction into a single C++ call.  No intermediate data
    is returned to Python.

    Args:
        positions: (N, 3) float64 array of particle positions.
        smoothing_lengths: (N,) float64 array of smoothing lengths.
        domain_min: (x, y, z) lower corner of the domain.
        domain_max: (x, y, z) upper corner of the domain.
        base_resolution: Number of top-level cells per axis.
        isovalue: Density isovalue for octree refinement.
        max_depth: Maximum octree refinement depth.
        screening_weight: Poisson screening weight alpha.
            Higher values produce tighter fit to data points.
        max_iters: Maximum PCG iterations.
        tol: PCG relative residual tolerance.

    Returns:
        Dict with keys ``vertices`` (V, 3) float64 ndarray,
        ``faces`` (F, 3) uint32 ndarray, ``isovalue`` float,
        ``n_qef_vertices`` int, ``solver_converged`` bool,
        ``solver_iterations`` int, ``solver_residual`` float.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    sml = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
    return _adaptive.run_full_pipeline(
        pos,
        sml,
        tuple(domain_min),
        tuple(domain_max),
        base_resolution,
        isovalue,
        max_depth,
        screening_weight,
        max_iters,
        tol,
    )
