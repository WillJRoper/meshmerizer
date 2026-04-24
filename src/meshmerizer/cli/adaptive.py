"""Adaptive CLI execution flow.

This module is the application-layer entry point for the adaptive command-line
workflow. It owns the sequencing of:

- input loading from either SWIFT snapshots or saved octree files,
- conversion of user-facing print-space controls into native units,
- dispatch into the native adaptive reconstruction pipeline,
- optional octree persistence and diagnostic output, and
- final mesh post-processing and STL export.

The CLI deliberately stays thin relative to the library layer: all heavy
geometry and topology work is delegated to native bindings or reusable library
helpers, while this module focuses on user-facing orchestration and progress
reporting.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from meshmerizer.adaptive_core import (
    compute_isovalue_from_percentile,
    create_top_level_cells_with_contributors,
    fof_cluster,
    refine_octree,
    solve_vertices,
)
from meshmerizer.cli.diagnostics import (
    emit_tree_structure_summary,
    visualize_vertices,
)
from meshmerizer.cli.particles import (
    filter_small_particle_fof_clusters,
    load_particles_for_adaptive,
)
from meshmerizer.cli.units import convert_print_length_to_native_units
from meshmerizer.io import export_octree, import_octree, save_mesh_output
from meshmerizer.logging import (
    abort_with_error,
    log_status,
    log_summary_status,
    record_elapsed,
)
from meshmerizer.mesh import Mesh
from meshmerizer.mesh.operations import remove_islands, simplify_mesh
from meshmerizer.printing import scale_mesh_to_print
from meshmerizer.reconstruct import reconstruct_mesh


def _configure_threads(args) -> None:
    """Configure OpenMP threads for the native extension when requested.

    Args:
        args: Parsed CLI namespace.
    """
    if args.nthreads is None:
        return

    try:
        from meshmerizer._adaptive import set_num_threads
    except Exception as exc:
        abort_with_error(
            "Config",
            "Could not configure OpenMP threads because the adaptive "
            "extension failed to import. Reinstall with `pip install -e .` "
            f"to build `_adaptive` ({exc}).",
        )

    set_num_threads(args.nthreads)
    log_status("Config", f"OpenMP threads set to {args.nthreads}.")


def _resolve_output_path(args) -> Path:
    """Resolve the final mesh output path from CLI arguments.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Final mesh output path, either explicit or derived from the input.
    """
    if args.output is not None:
        return Path(args.output)
    if args.filename is not None:
        return args.filename.with_suffix(".stl")
    return Path(args.load_octree).with_suffix(".stl")


def _convert_regularization_lengths(
    args,
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    min_feature_thickness: float,
    pre_thickening_radius: float,
) -> tuple[float, float]:
    """Convert print-space regularization controls back to native units.

    Args:
        args: Parsed CLI namespace.
        domain_min: Lower corner of the working domain.
        domain_max: Upper corner of the working domain.
        min_feature_thickness: Requested minimum feature thickness.
        pre_thickening_radius: Requested outward pre-thickening radius.

    Returns:
        Tuple of ``(min_feature_thickness, pre_thickening_radius)`` in native
        meshing units.
    """
    if args.target_size is None:
        return min_feature_thickness, pre_thickening_radius

    effective_min_feature_thickness = min_feature_thickness
    effective_pre_thickening_radius = pre_thickening_radius

    if effective_min_feature_thickness > 0.0:
        effective_min_feature_thickness = convert_print_length_to_native_units(
            effective_min_feature_thickness,
            domain_min,
            domain_max,
            args.target_size,
        )
        log_status(
            "Config",
            "Interpreting --min-feature-thickness in print units: "
            f"{args.min_feature_thickness} cm -> "
            f"{effective_min_feature_thickness:.6g} native units",
        )

    if effective_pre_thickening_radius > 0.0:
        effective_pre_thickening_radius = convert_print_length_to_native_units(
            effective_pre_thickening_radius,
            domain_min,
            domain_max,
            args.target_size,
        )
        log_status(
            "Config",
            "Interpreting --pre-thickening-radius in print units: "
            f"{args.pre_thickening_radius} cm -> "
            f"{effective_pre_thickening_radius:.6g} native units",
        )

    return effective_min_feature_thickness, effective_pre_thickening_radius


def _postprocess_mesh(mesh: Mesh, args) -> Mesh:
    """Apply cleanup and print-scaling steps to a reconstructed mesh.

    Args:
        mesh: Reconstructed mesh in native units.
        args: Parsed CLI namespace.

    Returns:
        Post-processed mesh ready for export.
    """
    cleanup_start = time.perf_counter()
    mesh = _remove_islands(mesh, args.remove_islands_fraction)
    record_elapsed("Island removal", cleanup_start, operation="Cleaning")

    simplify_start = time.perf_counter()
    mesh = _simplify_mesh(mesh, args.simplify_factor)
    record_elapsed(
        "Mesh simplification",
        simplify_start,
        operation="Cleaning",
    )

    if args.target_size is not None:
        log_status("Cleaning", f"Scaling mesh to {args.target_size} cm...")
        scale_start = time.perf_counter()
        mesh = scale_mesh_to_print(mesh, args.target_size)
        record_elapsed("Print scaling", scale_start, operation="Cleaning")

    return mesh


def _build_mesh(mesh_verts, mesh_faces, origin: np.ndarray) -> Mesh:
    """Translate native vertices back to world space and wrap them.

    Args:
        mesh_verts: Vertex positions returned by native reconstruction.
        mesh_faces: Triangle indices returned by native reconstruction.
        origin: Offset that restores snapshot/world coordinates.

    Returns:
        Mesh wrapper in world-space coordinates.
    """
    mesh_verts += origin
    return Mesh(vertices=mesh_verts, faces=mesh_faces)


def _save_final_mesh(mesh: Mesh, output_path: Path, *, summary: bool) -> None:
    """Write the final mesh and emit the appropriate CLI status message.

    Args:
        mesh: Mesh to export.
        output_path: Destination path.
        summary: Whether to emit summary-style rather than standard status
            logging.
    """
    if summary:
        log_summary_status("Saving", f"Writing STL to {output_path}...")
    else:
        log_status("Saving", f"Writing STL to {output_path}...")

    save_start = time.perf_counter()
    save_mesh_output(mesh, output_path)
    record_elapsed("STL export", save_start, operation="Saving")


def _run_full_pipeline_path(
    args,
    *,
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    origin: np.ndarray,
    base_resolution: int,
    max_depth: int,
    isovalue: float,
    min_feature_thickness: float,
    pre_thickening_radius: float,
    total_start: float,
) -> None:
    """Run the direct particles-to-mesh path without octree reuse.

    This is the fast path used when the user does not request octree save/load
    behavior. It keeps the whole reconstruction in the native pipeline and only
    returns to Python for post-processing and export.
    """
    if getattr(args, "fof", False):
        log_status(
            "Clustering",
            "Running FOF clustering "
            f"(linking_factor={args.linking_factor})...",
        )
        cluster_start = time.perf_counter()
        group_labels = fof_cluster(
            positions,
            domain_min,
            domain_max,
            args.linking_factor,
        )
        n_groups = int(np.unique(group_labels).size)
        record_elapsed("FOF clustering", cluster_start, operation="Clustering")
        log_status("Clustering", f"Found {n_groups} group(s).")
    else:
        group_labels = None

    log_status(
        "Pipeline",
        f"Running C++ full pipeline: base_resolution={base_resolution}, "
        f"max_depth={max_depth}, isovalue={isovalue}",
    )
    pipeline_start = time.perf_counter()
    reconstruction_start = time.perf_counter()
    mesh_verts, mesh_faces = reconstruct_mesh(
        positions,
        smoothing_lengths,
        domain_min,
        domain_max,
        base_resolution,
        isovalue,
        max_depth,
        group_labels=group_labels,
        smoothing_iterations=getattr(args, "smoothing_iterations", 0),
        smoothing_strength=getattr(args, "smoothing_strength", 0.5),
        max_edge_ratio=getattr(args, "max_edge_ratio", 1.5),
        minimum_usable_hermite_samples=getattr(
            args, "min_usable_hermite_samples", 3
        ),
        max_qef_rms_residual_ratio=getattr(
            args, "max_qef_rms_residual_ratio", 0.1
        ),
        min_normal_alignment_threshold=getattr(
            args, "min_normal_alignment_threshold", 0.97
        ),
        min_feature_thickness=min_feature_thickness,
        pre_thickening_radius=pre_thickening_radius,
    )
    record_elapsed(
        "Mesh reconstruction core",
        reconstruction_start,
        operation="Meshing",
    )
    record_elapsed("Full pipeline", pipeline_start, operation="Pipeline")

    n_tris = len(mesh_faces)
    log_status(
        "Meshing",
        f"Mesh: {len(mesh_verts)} vertices, {n_tris} triangles.",
    )
    if n_tris == 0:
        abort_with_error(
            "Meshing",
            "Pipeline produced no triangles. Check isovalue and domain "
            "selection.",
        )

    mesh = _build_mesh(mesh_verts, mesh_faces, origin)
    mesh = _postprocess_mesh(mesh, args)
    output_path = _resolve_output_path(args)
    _save_final_mesh(mesh, output_path, summary=False)

    record_elapsed("Total pipeline", total_start, operation="Done")
    log_summary_status("Done", f"Adaptive mesh saved to {output_path}")


def _load_or_prepare_inputs(args):
    """Load inputs either from a saved octree or from a SWIFT snapshot.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Dictionary containing normalized particle/domain state plus optional
        pre-built octree state.
    """
    if args.load_octree is not None:
        log_status("Loading", f"Loading octree from {args.load_octree}")
        load_start = time.perf_counter()
        state = import_octree(str(args.load_octree))
        record_elapsed("Octree load", load_start, operation="Loading")

        positions = state["positions"]
        smoothing_lengths = state["smoothing_lengths"]
        domain_min = state["domain_minimum"]
        domain_max = state["domain_maximum"]
        cells = state["cells"]
        contributors = state["contributors"]
        isovalue = state["isovalue"]
        max_depth = state["max_depth"]
        base_resolution = state["base_resolution"]
        origin = np.zeros(3, dtype=np.float64)

        log_status(
            "Loading",
            f"Loaded {len(cells)} cells, {len(positions)} particles "
            "from HDF5.",
        )
        return {
            "positions": positions,
            "smoothing_lengths": smoothing_lengths,
            "domain_min": domain_min,
            "domain_max": domain_max,
            "cells": cells,
            "contributors": contributors,
            "isovalue": isovalue,
            "max_depth": max_depth,
            "base_resolution": base_resolution,
            "origin": origin,
        }

    positions, smoothing_lengths, domain_min, domain_max, origin = (
        _load_particles_for_adaptive(args)
    )
    min_fof_cluster_size = getattr(args, "min_fof_cluster_size", None)
    if min_fof_cluster_size is not None:
        log_status(
            "Clustering",
            "Running particle FOF filtering "
            f"(linking_factor={args.linking_factor}, "
            f"min_cluster_size={min_fof_cluster_size})...",
        )
        cluster_start = time.perf_counter()
        positions, smoothing_lengths = filter_small_particle_fof_clusters(
            positions,
            smoothing_lengths,
            domain_min,
            domain_max,
            args.linking_factor,
            min_fof_cluster_size,
        )
        record_elapsed(
            "Particle FOF filtering",
            cluster_start,
            operation="Clustering",
        )
        if len(positions) == 0:
            abort_with_error(
                "Clustering",
                "Particle FOF filtering removed all particles. Lower "
                "--min-fof-cluster-size or adjust --linking-factor.",
            )

    if args.isovalue is not None:
        isovalue = args.isovalue
    else:
        isovalue = compute_isovalue_from_percentile(
            smoothing_lengths,
            args.surface_percentile,
        )
        log_status(
            "Config",
            f"Isovalue from {args.surface_percentile}th percentile: "
            f"{isovalue:.6g}",
        )

    return {
        "positions": positions,
        "smoothing_lengths": smoothing_lengths,
        "domain_min": domain_min,
        "domain_max": domain_max,
        "cells": None,
        "contributors": None,
        "isovalue": isovalue,
        "max_depth": args.max_depth,
        "base_resolution": args.base_resolution,
        "origin": origin,
    }


def _build_and_optionally_save_octree(
    args,
    *,
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    base_resolution: int,
    max_depth: int,
    isovalue: float,
    min_feature_thickness: float,
    pre_thickening_radius: float,
):
    """Build and optionally persist an octree from particle inputs.

    Args:
        args: Parsed CLI namespace.
        positions: Particle positions.
        smoothing_lengths: Per-particle smoothing lengths.
        domain_min: Lower corner of the working domain.
        domain_max: Upper corner of the working domain.
        base_resolution: Number of top-level cells per axis.
        max_depth: Maximum octree refinement depth.
        isovalue: Scalar field threshold used during refinement.
        min_feature_thickness: Regularization control in native units.
        pre_thickening_radius: Optional pre-thickening control in native units.

    Returns:
        Tuple ``(cells, contributors)`` describing the refined octree.
    """
    log_status(
        "Building",
        f"Building octree: base_resolution={base_resolution}, "
        f"max_depth={max_depth}, isovalue={isovalue}",
    )
    tree_start = time.perf_counter()

    top_cells = create_top_level_cells_with_contributors(
        positions,
        smoothing_lengths,
        domain_min,
        domain_max,
        base_resolution,
    )
    initial_cells = []
    for cell in top_cells:
        cell_dict = dict(cell)
        contributors = cell_dict.pop("contributors")
        cell_dict["contributor_begin"] = 0
        cell_dict["contributor_end"] = len(contributors)
        cell_dict["contributors"] = contributors
        initial_cells.append(cell_dict)

    cells, contributors = refine_octree(
        initial_cells,
        positions,
        smoothing_lengths,
        isovalue,
        max_depth,
        domain=(domain_min, domain_max),
        base_resolution=base_resolution,
        minimum_usable_hermite_samples=getattr(
            args, "min_usable_hermite_samples", 3
        ),
        max_qef_rms_residual_ratio=getattr(
            args, "max_qef_rms_residual_ratio", 0.1
        ),
        min_normal_alignment_threshold=getattr(
            args, "min_normal_alignment_threshold", 0.97
        ),
        min_feature_thickness=min_feature_thickness,
        pre_thickening_radius=pre_thickening_radius,
    )
    record_elapsed("Octree construction", tree_start, operation="Building")

    log_status(
        "Building",
        f"Octree built: {len(cells)} cells, "
        f"{sum(1 for cell in cells if cell['is_leaf'])} leaves.",
    )

    if args.save_octree is not None:
        log_status("Saving", f"Saving octree to {args.save_octree}")
        save_start = time.perf_counter()
        export_octree(
            str(args.save_octree),
            isovalue=isovalue,
            base_resolution=base_resolution,
            max_depth=max_depth,
            domain_minimum=domain_min,
            domain_maximum=domain_max,
            positions=positions,
            smoothing_lengths=smoothing_lengths,
            cells=cells,
            contributors=contributors,
        )
        record_elapsed("Octree save", save_start, operation="Saving")

    return cells, contributors


def _run_octree_backed_pipeline(
    args,
    *,
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    cells,
    contributors,
    base_resolution: int,
    max_depth: int,
    isovalue: float,
    origin: np.ndarray,
    min_feature_thickness: float,
    pre_thickening_radius: float,
    total_start: float,
) -> None:
    """Run the octree-backed reconstruction path used after save/load.

    This path is used when the CLI either loads an octree from disk or builds
    one explicitly for persistence or diagnostics before reconstructing the
    final mesh.
    """
    if getattr(args, "visualise_verts", None):
        log_status("Meshing", "Solving QEF vertices for visualization...")
        mesh_start = time.perf_counter()
        vert_positions, _ = solve_vertices(
            cells,
            contributors,
            positions,
            smoothing_lengths,
            isovalue,
            domain_min,
            domain_max,
            max_depth,
            base_resolution,
        )
        record_elapsed("Vertex solve", mesh_start, operation="Meshing")
        log_status("Meshing", f"Solved {len(vert_positions)} QEF vertices.")
        if len(vert_positions) == 0:
            abort_with_error(
                "Meshing",
                "No QEF vertices produced for visualization. Check isovalue "
                "and domain selection.",
            )
        visualize_vertices(vert_positions, args.visualise_verts)

    if getattr(args, "fof", False):
        log_status(
            "Clustering",
            "Running FOF clustering "
            f"(linking_factor={args.linking_factor})...",
        )
        cluster_start = time.perf_counter()
        group_labels = fof_cluster(
            positions,
            domain_min,
            domain_max,
            args.linking_factor,
        )
        n_groups = int(np.unique(group_labels).size)
        record_elapsed("FOF clustering", cluster_start, operation="Clustering")
        log_status("Clustering", f"Found {n_groups} group(s).")
    else:
        group_labels = np.zeros(len(positions), dtype=np.int64)

    reconstruction_depth = max_depth
    log_status(
        "Meshing",
        "Running adaptive mesh reconstruction "
        f"(depth={reconstruction_depth})...",
    )
    reconstruction_start = time.perf_counter()
    mesh_verts, mesh_faces = reconstruct_mesh(
        positions,
        smoothing_lengths,
        domain_min,
        domain_max,
        base_resolution,
        isovalue,
        reconstruction_depth,
        group_labels=group_labels,
        smoothing_iterations=getattr(args, "smoothing_iterations", 0),
        smoothing_strength=getattr(args, "smoothing_strength", 0.5),
        max_edge_ratio=getattr(args, "max_edge_ratio", 1.5),
        minimum_usable_hermite_samples=getattr(
            args, "min_usable_hermite_samples", 3
        ),
        max_qef_rms_residual_ratio=getattr(
            args, "max_qef_rms_residual_ratio", 0.1
        ),
        min_normal_alignment_threshold=getattr(
            args, "min_normal_alignment_threshold", 0.97
        ),
        min_feature_thickness=min_feature_thickness,
        pre_thickening_radius=pre_thickening_radius,
    )
    record_elapsed(
        "Mesh reconstruction core",
        reconstruction_start,
        operation="Meshing",
    )
    record_elapsed(
        "Mesh reconstruction",
        reconstruction_start,
        operation="Meshing",
    )

    n_tris = len(mesh_faces)
    log_status(
        "Meshing",
        f"Reconstructed mesh: {len(mesh_verts)} vertices, {n_tris} triangles.",
    )
    if n_tris == 0:
        abort_with_error(
            "Meshing",
            "Reconstruction produced no triangles. Check isovalue and domain "
            "selection.",
        )

    mesh = _build_mesh(mesh_verts, mesh_faces, origin)
    mesh = _postprocess_mesh(mesh, args)
    output_path = _resolve_output_path(args)
    _save_final_mesh(mesh, output_path, summary=True)

    emit_tree_structure_summary(cells)
    record_elapsed("Total pipeline", total_start, operation="Done")
    log_summary_status("Done", f"Adaptive mesh saved to {output_path}")


def run_adaptive(args) -> None:
    """Run the adaptive meshing pipeline from CLI arguments.

    Args:
        args: Parsed CLI namespace for the adaptive command.
    """
    total_start = time.perf_counter()

    center = getattr(args, "center", None)
    extent = getattr(args, "extent", None)
    if center is not None and extent is None:
        abort_with_error("Config", "--center requires --extent.")
    if extent is not None and center is None:
        abort_with_error("Config", "--extent requires --center.")
    if args.filename is None and args.load_octree is None:
        abort_with_error(
            "Config",
            "Provide a snapshot filename or use --load-octree.",
        )

    _configure_threads(args)

    state = _load_or_prepare_inputs(args)
    effective_min_feature_thickness, effective_pre_thickening_radius = (
        _convert_regularization_lengths(
            args,
            state["domain_min"],
            state["domain_max"],
            getattr(args, "min_feature_thickness", 0.0),
            getattr(args, "pre_thickening_radius", 0.0),
        )
    )

    if args.load_octree is None and args.save_octree is None:
        _run_full_pipeline_path(
            args,
            positions=state["positions"],
            smoothing_lengths=state["smoothing_lengths"],
            domain_min=state["domain_min"],
            domain_max=state["domain_max"],
            origin=state["origin"],
            base_resolution=state["base_resolution"],
            max_depth=state["max_depth"],
            isovalue=state["isovalue"],
            min_feature_thickness=effective_min_feature_thickness,
            pre_thickening_radius=effective_pre_thickening_radius,
            total_start=total_start,
        )
        return

    cells = state["cells"]
    contributors = state["contributors"]
    if args.load_octree is None:
        cells, contributors = _build_and_optionally_save_octree(
            args,
            positions=state["positions"],
            smoothing_lengths=state["smoothing_lengths"],
            domain_min=state["domain_min"],
            domain_max=state["domain_max"],
            base_resolution=state["base_resolution"],
            max_depth=state["max_depth"],
            isovalue=state["isovalue"],
            min_feature_thickness=effective_min_feature_thickness,
            pre_thickening_radius=effective_pre_thickening_radius,
        )
    elif args.save_octree is not None:
        log_status("Saving", f"Saving octree to {args.save_octree}")
        save_start = time.perf_counter()
        export_octree(
            str(args.save_octree),
            isovalue=state["isovalue"],
            base_resolution=state["base_resolution"],
            max_depth=state["max_depth"],
            domain_minimum=state["domain_min"],
            domain_maximum=state["domain_max"],
            positions=state["positions"],
            smoothing_lengths=state["smoothing_lengths"],
            cells=cells,
            contributors=contributors,
        )
        record_elapsed("Octree save", save_start, operation="Saving")

    _run_octree_backed_pipeline(
        args,
        positions=state["positions"],
        smoothing_lengths=state["smoothing_lengths"],
        domain_min=state["domain_min"],
        domain_max=state["domain_max"],
        cells=cells,
        contributors=contributors,
        base_resolution=state["base_resolution"],
        max_depth=state["max_depth"],
        isovalue=state["isovalue"],
        origin=state["origin"],
        min_feature_thickness=effective_min_feature_thickness,
        pre_thickening_radius=effective_pre_thickening_radius,
        total_start=total_start,
    )


__all__ = [
    "_convert_print_length_to_native_units",
    "_load_particles_for_adaptive",
    "_remove_islands",
    "_save_mesh_output",
    "_simplify_mesh",
    "run_adaptive",
]

_convert_print_length_to_native_units = convert_print_length_to_native_units
_remove_islands = remove_islands
_simplify_mesh = simplify_mesh
_load_particles_for_adaptive = load_particles_for_adaptive
_save_mesh_output = save_mesh_output
