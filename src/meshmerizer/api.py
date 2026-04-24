"""High-level public API for adaptive mesh generation.

The public Python surface is intentionally organized around two workflows:

- ``generate_mesh`` for the common particles-to-mesh path.
- ``build_tree`` -> ``regularize`` -> ``extract_mesh`` for staged use cases.

This keeps the intended usage small and consistent while still supporting
inspection and editing of intermediate states when needed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from meshmerizer.adaptive import (
    build_refined_tree,
    classify_occupied_solid,
    compute_isovalue_from_percentile,
    extract_opened_surface_mesh,
    fof_cluster,
    run_full_pipeline,
)
from meshmerizer.adaptive import (
    generate_mesh as generate_native_mesh,
)
from meshmerizer.mesh import Mesh
from meshmerizer.mesh.operations import remove_islands as remove_small_islands
from meshmerizer.state import MeshResult, TopologyState, TreeState, Vec3


def _as_positions(positions: Sequence[Sequence[float]]) -> np.ndarray:
    """Normalize particle positions to a contiguous ``(N, 3)`` array.

    Args:
        positions: Sequence of XYZ coordinate triplets.

    Returns:
        Contiguous float64 array with shape ``(N, 3)``.

    Raises:
        ValueError: If the input cannot be interpreted as ``(N, 3)``.
    """
    arr = np.ascontiguousarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    return arr


def _as_smoothing_lengths(smoothing_lengths: Sequence[float]) -> np.ndarray:
    """Normalize smoothing lengths to a contiguous ``(N,)`` array.

    Args:
        smoothing_lengths: Sequence of per-particle support radii.

    Returns:
        Contiguous float64 array with shape ``(N,)``.

    Raises:
        ValueError: If the input is not one-dimensional.
    """
    arr = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("smoothing_lengths must have shape (N,)")
    return arr


def _validate_particle_arrays(
    positions: Sequence[Sequence[float]],
    smoothing_lengths: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize particle arrays used by the public API.

    Args:
        positions: Sequence of particle positions.
        smoothing_lengths: Sequence of per-particle smoothing lengths.

    Returns:
        Tuple of validated ``(positions, smoothing_lengths)`` arrays.

    Raises:
        ValueError: If the arrays have incompatible shapes or lengths.
    """
    pos = _as_positions(positions)
    sml = _as_smoothing_lengths(smoothing_lengths)
    if pos.shape[0] != sml.shape[0]:
        raise ValueError(
            "positions and smoothing_lengths must contain the same number "
            "of particles"
        )
    return pos, sml


def _mesh_result_from_pipeline_dict(result: dict) -> MeshResult:
    """Convert a native pipeline result dictionary into ``MeshResult``.

    Args:
        result: Mapping returned by ``run_full_pipeline``.

    Returns:
        Public ``MeshResult`` wrapper.
    """
    return MeshResult(
        mesh=Mesh(vertices=result["vertices"], faces=result["faces"]),
        isovalue=float(result["isovalue"]),
        n_qef_vertices=int(result["n_qef_vertices"]),
    )


def build_tree(
    positions: Sequence[Sequence[float]],
    smoothing_lengths: Sequence[float],
    domain_min: Vec3,
    domain_max: Vec3,
    base_resolution: int,
    isovalue: float,
    max_depth: int,
    minimum_usable_hermite_samples: int = 3,
    max_qef_rms_residual_ratio: float = 0.1,
    min_normal_alignment_threshold: float = 0.97,
) -> TreeState:
    """Build and refine the adaptive tree without extracting a final mesh.

    Args:
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle support radii with shape ``(N,)``.
        domain_min: Inclusive lower corner of the working domain.
        domain_max: Exclusive upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.
        isovalue: Scalar field threshold used for refinement decisions.
        max_depth: Maximum octree refinement depth.
        minimum_usable_hermite_samples: Minimum usable Hermite sample count
            required before a corner-crossing cell may stop refining.
        max_qef_rms_residual_ratio: Maximum RMS QEF residual as a fraction of
            the local cell radius.
        min_normal_alignment_threshold: Minimum allowed alignment between
            usable Hermite normals and their mean direction.

    Returns:
        ``TreeState`` containing the refined tree and validated inputs.
    """
    pos, sml = _validate_particle_arrays(positions, smoothing_lengths)
    cells, contributors = build_refined_tree(
        pos,
        sml,
        tuple(domain_min),
        tuple(domain_max),
        base_resolution,
        isovalue,
        max_depth,
        minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio,
        min_normal_alignment_threshold,
    )
    return TreeState(
        cells=cells,
        contributors=tuple(int(v) for v in contributors.tolist()),
        positions=pos,
        smoothing_lengths=sml,
        domain_min=tuple(domain_min),
        domain_max=tuple(domain_max),
        base_resolution=base_resolution,
        max_depth=max_depth,
        isovalue=float(isovalue),
        minimum_usable_hermite_samples=minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio=max_qef_rms_residual_ratio,
        min_normal_alignment_threshold=min_normal_alignment_threshold,
    )


def regularize(
    tree: TreeState,
    min_feature_thickness: float,
    *,
    pre_thickening_radius: float = 0.0,
) -> TopologyState:
    """Build the opened-solid topology used by the regularized pipeline.

    Args:
        tree: Previously built tree state.
        min_feature_thickness: Minimum feature thickness to preserve.
        pre_thickening_radius: Optional outward thickening radius applied
            before the opening stage.

    Returns:
        ``TopologyState`` representing the regularized opened solid.
    """
    erosion_radius = 0.5 * float(min_feature_thickness)
    result = classify_occupied_solid(
        tree.positions,
        tree.smoothing_lengths,
        tree.domain_min,
        tree.domain_max,
        tree.base_resolution,
        tree.isovalue,
        tree.max_depth,
        tree.minimum_usable_hermite_samples,
        tree.max_qef_rms_residual_ratio,
        tree.min_normal_alignment_threshold,
        max_surface_leaf_size=erosion_radius,
        erosion_radius=erosion_radius,
        pre_thickening_radius=pre_thickening_radius,
    )
    return TopologyState(
        tree=tree,
        occupancy=result["occupancy"],
        depths=result["depths"],
        centers=result["center_values"],
        sizes=result["cell_sizes"],
        clearance=result["clearance"],
        thickening_distance=result["thickening_distance"],
        thickened_inside=result["thickened_inside"],
        eroded_inside=result["eroded_inside"],
        dilation_distance=result["dilation_distance"],
        opened_inside=result["opened_inside"],
        sample_positions=result["opened_boundary_positions"],
        sample_normals=result["opened_boundary_normals"],
        mesh_vertices=result["opened_surface_vertices"],
        mesh_faces=result["opened_surface_faces"],
        min_feature_thickness=float(min_feature_thickness),
    )


def _extract_mesh_from_tree(
    tree: TreeState,
    *,
    smoothing_iterations: int = 0,
    smoothing_strength: float = 0.5,
    max_edge_ratio: float = 1.5,
    min_feature_thickness: float = 0.0,
    pre_thickening_radius: float = 0.0,
    remove_islands_fraction: Optional[float] = None,
) -> MeshResult:
    """Extract a mesh from a tree state.

    Args:
        tree: Refined tree state to extract from.
        smoothing_iterations: Number of smoothing iterations for the full
            pipeline fallback.
        smoothing_strength: Laplacian smoothing strength for the fallback path.
        max_edge_ratio: Maximum permitted edge length relative to local cell
            size on the fallback path.
        min_feature_thickness: Minimum feature thickness to preserve.
        pre_thickening_radius: Optional outward pre-thickening radius.
        remove_islands_fraction: Optional connected-component filtering
            threshold.

    Returns:
        ``MeshResult`` extracted from the provided tree state.
    """
    # Use the lighter direct mesh-generation path when the caller already has a
    # refined tree and does not request regularization.
    if (
        tree.cells
        and tree.contributors
        and min_feature_thickness <= 0.0
        and pre_thickening_radius <= 0.0
    ):
        vertices, _, faces = generate_native_mesh(
            tree.cells,
            tree.contributors,
            tree.positions,
            tree.smoothing_lengths,
            tree.isovalue,
            tree.domain_min,
            tree.domain_max,
            tree.max_depth,
            tree.base_resolution,
        )
        # Wrap the direct native arrays immediately so the rest of the function
        # can treat both branches uniformly.
        mesh_result = MeshResult(
            mesh=Mesh(vertices=vertices, faces=faces.astype(np.uint32)),
            isovalue=tree.isovalue,
            n_qef_vertices=int(vertices.shape[0]),
        )
    else:
        # Fall back to the whole-pipeline entry point when regularization
        # or smoothing requests mean the direct shortcut is insufficient.
        result = run_full_pipeline(
            tree.positions,
            tree.smoothing_lengths,
            tree.domain_min,
            tree.domain_max,
            tree.base_resolution,
            tree.isovalue,
            tree.max_depth,
            smoothing_iterations=smoothing_iterations,
            smoothing_strength=smoothing_strength,
            max_edge_ratio=max_edge_ratio,
            minimum_usable_hermite_samples=tree.minimum_usable_hermite_samples,
            max_qef_rms_residual_ratio=tree.max_qef_rms_residual_ratio,
            min_normal_alignment_threshold=tree.min_normal_alignment_threshold,
            min_feature_thickness=min_feature_thickness,
            pre_thickening_radius=pre_thickening_radius,
        )
        mesh_result = _mesh_result_from_pipeline_dict(result)

    # Apply optional island filtering last so both extraction branches
    # share the same cleanup semantics.
    if remove_islands_fraction is not None:
        mesh_result.mesh = remove_islands(
            mesh_result.mesh,
            remove_islands_fraction=remove_islands_fraction,
        )
    return mesh_result


def _extract_mesh_from_topology(
    topology: TopologyState,
    *,
    remove_islands_fraction: Optional[float] = None,
) -> MeshResult:
    """Extract a mesh from a topology state.

    Args:
        topology: Regularized topology state to extract from.
        remove_islands_fraction: Optional connected-component filtering
            threshold.

    Returns:
        ``MeshResult`` extracted from the opened-solid topology.
    """
    # Topology extraction always uses the opened-solid mesh path because the
    # caller has already committed to the regularized occupancy state.
    vertices, faces = extract_opened_surface_mesh(
        topology.tree.positions,
        topology.tree.smoothing_lengths,
        topology.tree.domain_min,
        topology.tree.domain_max,
        topology.tree.base_resolution,
        topology.tree.isovalue,
        topology.tree.max_depth,
        topology.opened_inside,
        topology.tree.minimum_usable_hermite_samples,
        topology.tree.max_qef_rms_residual_ratio,
        topology.tree.min_normal_alignment_threshold,
    )
    # The opened-surface extractor does not solve a fresh QEF vertex cloud in
    # the same sense as the full tree pipeline, so report zero here.
    mesh_result = MeshResult(
        mesh=Mesh(vertices=vertices, faces=faces),
        isovalue=topology.tree.isovalue,
        n_qef_vertices=0,
    )
    # Apply the same optional cleanup hook as the tree-based extraction path.
    if remove_islands_fraction is not None:
        mesh_result.mesh = remove_islands(
            mesh_result.mesh,
            remove_islands_fraction=remove_islands_fraction,
        )
    return mesh_result


def extract_mesh(
    state: TreeState | TopologyState,
    *,
    smoothing_iterations: int = 0,
    smoothing_strength: float = 0.5,
    max_edge_ratio: float = 1.5,
    min_feature_thickness: float = 0.0,
    pre_thickening_radius: float = 0.0,
    remove_islands_fraction: Optional[float] = None,
) -> MeshResult:
    """Extract a mesh from either staged public state object.

    Args:
        state: ``TreeState`` or ``TopologyState`` to extract from.
        smoothing_iterations: Number of smoothing iterations to apply when
            extracting from a tree state.
        smoothing_strength: Laplacian smoothing strength in ``(0, 1]`` when
            extracting from a tree state.
        max_edge_ratio: Maximum edge length as a multiple of local cell size
            when extracting from a tree state.
        min_feature_thickness: Minimum feature thickness to preserve when
            extracting from a tree state.
        pre_thickening_radius: Optional outward thickening radius when
            extracting from a tree state.
        remove_islands_fraction: Optional connected-component filtering
            threshold.

    Returns:
        ``MeshResult`` containing the extracted mesh.

    Raises:
        TypeError: If ``state`` is not a supported staged public state.
    """
    # Dispatch on the staged state type so callers can use one public
    # extraction entry point regardless of where they stopped.
    if isinstance(state, TreeState):
        return _extract_mesh_from_tree(
            state,
            smoothing_iterations=smoothing_iterations,
            smoothing_strength=smoothing_strength,
            max_edge_ratio=max_edge_ratio,
            min_feature_thickness=min_feature_thickness,
            pre_thickening_radius=pre_thickening_radius,
            remove_islands_fraction=remove_islands_fraction,
        )
    if isinstance(state, TopologyState):
        return _extract_mesh_from_topology(
            state,
            remove_islands_fraction=remove_islands_fraction,
        )
    raise TypeError(
        "state must be a TreeState or TopologyState, "
        f"got {type(state).__name__}"
    )


def generate_mesh(
    positions: Sequence[Sequence[float]],
    smoothing_lengths: Sequence[float],
    domain_min: Vec3,
    domain_max: Vec3,
    base_resolution: int,
    max_depth: int,
    isovalue: float,
    smoothing_iterations: int = 0,
    smoothing_strength: float = 0.5,
    max_edge_ratio: float = 1.5,
    minimum_usable_hermite_samples: int = 3,
    max_qef_rms_residual_ratio: float = 0.1,
    min_normal_alignment_threshold: float = 0.97,
    min_feature_thickness: float = 0.0,
    pre_thickening_radius: float = 0.0,
    remove_islands_fraction: Optional[float] = None,
) -> MeshResult:
    """Run the full public particles-to-mesh workflow.

    Args:
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle support radii with shape ``(N,)``.
        domain_min: Inclusive lower corner of the working domain.
        domain_max: Exclusive upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.
        max_depth: Maximum octree refinement depth.
        isovalue: Scalar field threshold for surface extraction.
        smoothing_iterations: Number of smoothing iterations to apply.
        smoothing_strength: Laplacian smoothing strength in ``(0, 1]``.
        max_edge_ratio: Maximum edge length as a multiple of local cell size.
        minimum_usable_hermite_samples: Minimum usable Hermite sample count.
        max_qef_rms_residual_ratio: Maximum RMS QEF residual ratio.
        min_normal_alignment_threshold: Minimum usable-normal alignment.
        min_feature_thickness: Minimum feature thickness to preserve.
        pre_thickening_radius: Optional outward thickening radius.
        remove_islands_fraction: Optional connected-component filtering
            threshold.

    Returns:
        ``MeshResult`` containing the final extracted mesh.
    """
    # Validate the particle arrays once at the public boundary so the rest of
    # the workflow can assume matched, contiguous inputs.
    pos, sml = _validate_particle_arrays(positions, smoothing_lengths)
    # The one-shot API is intentionally a thin wrapper over the native full
    # pipeline followed by optional mesh cleanup.
    result = run_full_pipeline(
        pos,
        sml,
        tuple(domain_min),
        tuple(domain_max),
        base_resolution,
        isovalue,
        max_depth,
        smoothing_iterations=smoothing_iterations,
        smoothing_strength=smoothing_strength,
        max_edge_ratio=max_edge_ratio,
        minimum_usable_hermite_samples=minimum_usable_hermite_samples,
        max_qef_rms_residual_ratio=max_qef_rms_residual_ratio,
        min_normal_alignment_threshold=min_normal_alignment_threshold,
        min_feature_thickness=min_feature_thickness,
        pre_thickening_radius=pre_thickening_radius,
    )
    mesh_result = _mesh_result_from_pipeline_dict(result)
    # Apply cleanup after wrapping the result so the public return type stays
    # consistent regardless of whether filtering is requested.
    if remove_islands_fraction is not None:
        mesh_result.mesh = remove_islands(
            mesh_result.mesh,
            remove_islands_fraction=remove_islands_fraction,
        )
    return mesh_result


def cluster_particles(
    positions: Sequence[Sequence[float]],
    domain_min: Vec3,
    domain_max: Vec3,
    linking_factor: float = 1.5,
) -> np.ndarray:
    """Cluster particle positions with friends-of-friends.

    Args:
        positions: Particle positions with shape ``(N, 3)``.
        domain_min: Inclusive lower corner of the working domain.
        domain_max: Exclusive upper corner of the working domain.
        linking_factor: Multiplicative FOF linking factor.

    Returns:
        ``(N,)`` int64 array of cluster labels.
    """
    # Reuse the public position normalizer so clustering follows the same input
    # contract as the rest of the API.
    pos = _as_positions(positions)
    return fof_cluster(
        pos, tuple(domain_min), tuple(domain_max), linking_factor
    )


def smooth_mesh(
    mesh: Mesh,
    iterations: int = 10,
    *,
    inplace: bool = False,
) -> Mesh:
    """Apply the existing mesh-repair smoothing workflow to a mesh.

    Args:
        mesh: Mesh to smooth.
        iterations: Number of repair/smoothing iterations.
        inplace: Whether to modify the input mesh directly.

    Returns:
        Smoothed mesh instance.
    """
    # Copy by default so smoothing behaves like a pure helper unless the caller
    # explicitly opts into in-place modification.
    target = mesh if inplace else Mesh(mesh=mesh.mesh.copy())
    target.repair(smoothing_iters=iterations)
    return target


def remove_islands(
    mesh: Mesh,
    remove_islands_fraction: Optional[float],
) -> Mesh:
    """Remove connected components below a fraction of the largest volume.

    Args:
        mesh: Mesh to filter.
        remove_islands_fraction: Optional threshold relative to the largest
            connected component volume.

    Returns:
        Filtered mesh.
    """
    # Delegate to the shared mesh-operations helper so CLI and public API keep
    # identical island-filtering semantics.
    return remove_small_islands(mesh, remove_islands_fraction)


def subdivide_long_edges(mesh: Mesh, iterations: int = 1) -> Mesh:
    """Subdivide mesh edges using the existing trimesh-based subdivision.

    Args:
        mesh: Mesh to subdivide.
        iterations: Number of subdivision passes.

    Returns:
        Subdivided mesh copy.
    """
    # Operate on a copy so subdivision does not mutate the caller's mesh unless
    # they choose to keep the returned object.
    result = Mesh(mesh=mesh.mesh.copy())
    result.subdivide(iterations=iterations)
    return result


__all__ = [
    "MeshResult",
    "TopologyState",
    "TreeState",
    "build_tree",
    "cluster_particles",
    "compute_isovalue_from_percentile",
    "extract_mesh",
    "generate_mesh",
    "remove_islands",
    "regularize",
    "smooth_mesh",
    "subdivide_long_edges",
]
