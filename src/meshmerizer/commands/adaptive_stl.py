"""Adaptive meshing CLI command.

This legacy module still hosts the adaptive CLI execution path while the code
base is being moved into the dedicated ``meshmerizer.cli`` and
``meshmerizer.io`` packages.
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

_convert_print_length_to_native_units = convert_print_length_to_native_units
_remove_islands = remove_islands
_simplify_mesh = simplify_mesh
_load_particles_for_adaptive = load_particles_for_adaptive
_filter_small_particle_fof_clusters = filter_small_particle_fof_clusters
_visualize_vertices = visualize_vertices
_emit_tree_structure_summary = emit_tree_structure_summary
_save_mesh_output = save_mesh_output


def run_adaptive(args) -> None:
    """Run the adaptive meshing pipeline from CLI arguments.

    Args:
        args: Parsed CLI namespace containing all adaptive flags.

    Returns:
        ``None``. The output STL is written to disk.
    """
    total_start = time.perf_counter()

    effective_min_feature_thickness = getattr(
        args, "min_feature_thickness", 0.0
    )
    effective_pre_thickening_radius = getattr(
        args, "pre_thickening_radius", 0.0
    )

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

    # Set OpenMP thread count if requested.
    if args.nthreads is not None:
        try:
            from meshmerizer._adaptive import set_num_threads
        except Exception as exc:
            abort_with_error(
                "Config",
                "Could not configure OpenMP threads because the adaptive "
                "extension failed to import. Reinstall with "
                "`pip install -e .` "
                f"to build `_adaptive` ({exc}).",
            )

        set_num_threads(args.nthreads)
        log_status(
            "Config",
            f"OpenMP threads set to {args.nthreads}.",
        )

    # ------------------------------------------------------------------
    # Step 1: Load from HDF5 octree or from a SWIFT snapshot.
    # ------------------------------------------------------------------
    if args.load_octree is not None:
        log_status("Loading", f"Loading octree from {args.load_octree}")
        load_start = time.perf_counter()
        state = import_octree(str(args.load_octree))
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
        record_elapsed("Octree load", load_start, operation="Loading")

        log_status(
            "Loading",
            f"Loaded {len(cells)} cells, "
            f"{len(positions)} particles from HDF5.",
        )

        if (
            args.target_size is not None
            and effective_min_feature_thickness > 0.0
        ):
            effective_min_feature_thickness = (
                _convert_print_length_to_native_units(
                    effective_min_feature_thickness,
                    domain_min,
                    domain_max,
                    args.target_size,
                )
            )
            log_status(
                "Config",
                "Interpreting --min-feature-thickness in print units: "
                f"{args.min_feature_thickness} cm -> "
                f"{effective_min_feature_thickness:.6g} native units",
            )

        if (
            args.target_size is not None
            and effective_pre_thickening_radius > 0.0
        ):
            effective_pre_thickening_radius = (
                _convert_print_length_to_native_units(
                    effective_pre_thickening_radius,
                    domain_min,
                    domain_max,
                    args.target_size,
                )
            )
            log_status(
                "Config",
                "Interpreting --pre-thickening-radius in print units: "
                f"{args.pre_thickening_radius} cm -> "
                f"{effective_pre_thickening_radius:.6g} native units",
            )
    else:
        # Load particles from the SWIFT snapshot.
        positions, smoothing_lengths, domain_min, domain_max, origin = (
            _load_particles_for_adaptive(args)
        )

        min_fof_cluster_size = getattr(args, "min_fof_cluster_size", None)
        if min_fof_cluster_size is not None:
            log_status(
                "Clustering",
                (
                    "Running particle FOF filtering "
                    f"(linking_factor={args.linking_factor}, "
                    f"min_cluster_size={min_fof_cluster_size})..."
                ),
            )
            cluster_start = time.perf_counter()
            positions, smoothing_lengths = _filter_small_particle_fof_clusters(
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

        if (
            args.target_size is not None
            and effective_min_feature_thickness > 0.0
        ):
            effective_min_feature_thickness = (
                _convert_print_length_to_native_units(
                    effective_min_feature_thickness,
                    domain_min,
                    domain_max,
                    args.target_size,
                )
            )
            log_status(
                "Config",
                "Interpreting --min-feature-thickness in print units: "
                f"{args.min_feature_thickness} cm -> "
                f"{effective_min_feature_thickness:.6g} native units",
            )

        if (
            args.target_size is not None
            and effective_pre_thickening_radius > 0.0
        ):
            effective_pre_thickening_radius = (
                _convert_print_length_to_native_units(
                    effective_pre_thickening_radius,
                    domain_min,
                    domain_max,
                    args.target_size,
                )
            )
            log_status(
                "Config",
                "Interpreting --pre-thickening-radius in print units: "
                f"{args.pre_thickening_radius} cm -> "
                f"{effective_pre_thickening_radius:.6g} native units",
            )

        max_depth = args.max_depth
        base_resolution = args.base_resolution

        # Determine isovalue: explicit --isovalue takes priority. Otherwise we
        # use a fast self-density proxy derived from the smoothing lengths.
        if args.isovalue is not None:
            isovalue = args.isovalue
        else:
            isovalue = compute_isovalue_from_percentile(
                smoothing_lengths, args.surface_percentile
            )
            log_status(
                "Config",
                f"Isovalue from {args.surface_percentile}th "
                f"percentile: {isovalue:.6g}",
            )

        if args.save_octree is not None:
            # We need intermediate cells/contributors for serialization, so use
            # the explicit octree-build path through Python.
            log_status(
                "Building",
                f"Building octree: base_resolution="
                f"{base_resolution}, "
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
                contribs = cell_dict.pop("contributors")
                cell_dict["contributor_begin"] = 0
                cell_dict["contributor_end"] = len(contribs)
                cell_dict["contributors"] = contribs
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
                min_feature_thickness=effective_min_feature_thickness,
                pre_thickening_radius=effective_pre_thickening_radius,
            )
            record_elapsed(
                "Octree construction",
                tree_start,
                operation="Building",
            )

            log_status(
                "Building",
                f"Octree built: {len(cells)} cells, "
                f"{sum(1 for c in cells if c['is_leaf'])}"
                f" leaves.",
            )

            # Save the octree state.
            log_status(
                "Saving",
                f"Saving octree to {args.save_octree}",
            )
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

        else:
            # Fast path: run the full particles-to-mesh pipeline
            # in C++.
            # If FOF is enabled, cluster particles first and
            # reconstruct each group independently.
            if getattr(args, "fof", False):
                log_status(
                    "Clustering",
                    f"Running FOF clustering "
                    f"(linking_factor={args.linking_factor})"
                    f"...",
                )
                cluster_start = time.perf_counter()
                group_labels = fof_cluster(
                    positions,
                    domain_min,
                    domain_max,
                    args.linking_factor,
                )
                n_groups = int(np.unique(group_labels).size)
                record_elapsed(
                    "FOF clustering",
                    cluster_start,
                    operation="Clustering",
                )
                log_status(
                    "Clustering",
                    f"Found {n_groups} group(s).",
                )
            else:
                group_labels = None

            log_status(
                "Pipeline",
                f"Running C++ full pipeline: "
                f"base_resolution={base_resolution}, "
                f"max_depth={max_depth}, "
                f"isovalue={isovalue}",
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
                min_feature_thickness=effective_min_feature_thickness,
                pre_thickening_radius=effective_pre_thickening_radius,
            )
            record_elapsed(
                "Mesh reconstruction core",
                reconstruction_start,
                operation="Meshing",
            )
            record_elapsed(
                "Full pipeline",
                pipeline_start,
                operation="Pipeline",
            )

            n_mesh_verts = len(mesh_verts)
            n_tris = len(mesh_faces)
            log_status(
                "Meshing",
                f"Mesh: {n_mesh_verts} vertices, {n_tris} triangles.",
            )

            if n_tris == 0:
                abort_with_error(
                    "Meshing",
                    "Pipeline produced no triangles. Check isovalue and "
                    "domain selection.",
                )

            # Translate back to world-space and build Mesh.
            mesh_verts += origin

            mesh = Mesh(
                vertices=mesh_verts,
                faces=mesh_faces,
            )

            # Remove small disconnected islands if requested.
            cleanup_start = time.perf_counter()
            mesh = _remove_islands(mesh, args.remove_islands_fraction)
            record_elapsed(
                "Island removal",
                cleanup_start,
                operation="Cleaning",
            )

            # Simplify the final mesh if requested.
            simplify_start = time.perf_counter()
            mesh = _simplify_mesh(mesh, args.simplify_factor)
            record_elapsed(
                "Mesh simplification",
                simplify_start,
                operation="Cleaning",
            )

            # Scale the mesh to physical print size.
            if args.target_size is not None:
                log_status(
                    "Cleaning",
                    f"Scaling mesh to {args.target_size} cm...",
                )
                scale_start = time.perf_counter()
                mesh = scale_mesh_to_print(mesh, args.target_size)
                record_elapsed(
                    "Print scaling",
                    scale_start,
                    operation="Cleaning",
                )

            # Save the STL.
            output_path = args.output
            if output_path is None:
                if args.filename is not None:
                    output_path = args.filename.with_suffix(".stl")
                else:
                    output_path = Path(args.load_octree).with_suffix(".stl")

            log_status(
                "Saving",
                f"Writing STL to {output_path}...",
            )
            save_start = time.perf_counter()
            _save_mesh_output(mesh, Path(output_path))
            record_elapsed("STL export", save_start, operation="Saving")

            record_elapsed(
                "Total pipeline",
                total_start,
                operation="Done",
            )
            log_summary_status(
                "Done",
                f"Adaptive mesh saved to {output_path}",
            )
            return

    # ------------------------------------------------------------------
    # Step 3: Optionally save the octree state (load-octree path).
    # ------------------------------------------------------------------
    if args.load_octree is not None and args.save_octree is not None:
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

    # ------------------------------------------------------------------
    # Step 4: Optional diagnostic QEF-vertex solve for octree-backed runs.
    # ------------------------------------------------------------------
    if args.load_octree is not None or args.save_octree is not None:
        if getattr(args, "visualise_verts", None):
            log_status(
                "Meshing",
                "Solving QEF vertices for visualization...",
            )
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
            log_status(
                "Meshing",
                f"Solved {len(vert_positions)} QEF vertices.",
            )
            if len(vert_positions) == 0:
                abort_with_error(
                    "Meshing",
                    "No QEF vertices produced for visualization. Check "
                    "isovalue and domain selection.",
                )
            _visualize_vertices(vert_positions, args.visualise_verts)

    # ------------------------------------------------------------------
    # Step 5: Optional particle FOF clustering + mesh reconstruction.
    # ------------------------------------------------------------------
    if getattr(args, "fof", False):
        log_status(
            "Clustering",
            f"Running FOF clustering "
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
        log_status(
            "Clustering",
            f"Found {n_groups} group(s).",
        )
    else:
        group_labels = np.zeros(len(positions), dtype=np.int64)

    # Reconstruct each requested group and merge the meshes.
    reconstruction_depth = max_depth
    log_status(
        "Meshing",
        f"Running adaptive mesh reconstruction "
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
        min_feature_thickness=effective_min_feature_thickness,
        pre_thickening_radius=effective_pre_thickening_radius,
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

    n_mesh_verts = len(mesh_verts)
    n_tris = len(mesh_faces)
    log_status(
        "Meshing",
        f"Reconstructed mesh: {n_mesh_verts} vertices, {n_tris} triangles.",
    )

    if n_tris == 0:
        abort_with_error(
            "Meshing",
            "Reconstruction produced no triangles. Check isovalue and domain "
            "selection.",
        )

    # ------------------------------------------------------------------
    # Step 6: Convert to a Mesh object and apply post-processing.
    # ------------------------------------------------------------------

    # Translate vertex positions back to world-space coordinates.
    mesh_verts += origin

    mesh = Mesh(
        vertices=mesh_verts,
        faces=mesh_faces,
    )

    # Remove small disconnected islands if requested.
    cleanup_start = time.perf_counter()
    mesh = _remove_islands(mesh, args.remove_islands_fraction)
    record_elapsed(
        "Island removal",
        cleanup_start,
        operation="Cleaning",
    )

    # Simplify the final mesh if requested.
    simplify_start = time.perf_counter()
    mesh = _simplify_mesh(mesh, args.simplify_factor)
    record_elapsed(
        "Mesh simplification",
        simplify_start,
        operation="Cleaning",
    )

    # Scale the mesh to a physical print size if requested.
    if args.target_size is not None:
        log_status(
            "Cleaning",
            f"Scaling mesh to {args.target_size} cm...",
        )
        scale_start = time.perf_counter()
        mesh = scale_mesh_to_print(mesh, args.target_size)
        record_elapsed(
            "Print scaling",
            scale_start,
            operation="Cleaning",
        )

    # ------------------------------------------------------------------
    # Step 6: Save the STL.
    # ------------------------------------------------------------------
    output_path = args.output
    if output_path is None:
        if args.filename is not None:
            output_path = args.filename.with_suffix(".stl")
        else:
            output_path = Path(args.load_octree).with_suffix(".stl")

    log_summary_status("Saving", f"Writing STL to {output_path}...")
    save_start = time.perf_counter()
    _save_mesh_output(mesh, Path(output_path))
    record_elapsed("STL export", save_start, operation="Saving")

    if args.load_octree is not None or args.save_octree is not None:
        _emit_tree_structure_summary(cells)

    record_elapsed("Total pipeline", total_start, operation="Done")
    log_summary_status("Done", f"Adaptive mesh saved to {output_path}")
