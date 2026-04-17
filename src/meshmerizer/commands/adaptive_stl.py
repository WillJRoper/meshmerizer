"""Adaptive meshing CLI command.

This module implements the ``adaptive`` subcommand which runs the new
adaptive octree pipeline from a SWIFT snapshot to an STL file.  QEF
vertices are solved in C++, then clustered via FOF and reconstructed
into a watertight mesh via Poisson surface reconstruction.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import numpy as np

from meshmerizer.adaptive_core import (
    compute_isovalue_from_percentile,
    create_top_level_cells_with_contributors,
    fof_cluster,
    refine_octree,
    solve_vertices,
)
from meshmerizer.logging import log_status, record_elapsed
from meshmerizer.mesh.core import Mesh
from meshmerizer.poisson import poisson_reconstruct
from meshmerizer.printing import scale_mesh_to_print
from meshmerizer.serialize import export_octree, import_octree


def _remove_islands(
    mesh: Mesh,
    remove_islands_fraction: Optional[float],
) -> Mesh:
    """Remove small disconnected components from a mesh.

    Args:
        mesh: Input mesh.
        remove_islands_fraction: Fraction of the total volume below
            which a connected component is discarded.  ``0.0`` keeps
            only the largest component.  ``None`` disables island
            removal entirely.

    Returns:
        A new ``Mesh`` with small islands removed, or the original
        mesh unchanged when removal is disabled.
    """
    if remove_islands_fraction is None:
        return mesh

    # Split into connected components via trimesh.
    tm = mesh.mesh
    components = tm.split(only_watertight=False)
    if len(components) <= 1:
        return mesh

    if remove_islands_fraction == 0.0:
        # Keep only the single largest component by volume.
        largest = max(
            components,
            key=lambda c: abs(c.volume) if c.is_watertight else 0.0,
        )
        log_status(
            "Cleaning",
            f"Kept largest of {len(components)} components.",
        )
        return Mesh(mesh=largest)

    # Keep components whose volume fraction exceeds the threshold.
    volumes = []
    for comp in components:
        vol = abs(comp.volume) if comp.is_watertight else 0.0
        volumes.append(vol)
    total_volume = sum(volumes)
    if total_volume == 0.0:
        return mesh

    kept = [
        c
        for c, v in zip(components, volumes)
        if v / total_volume >= remove_islands_fraction
    ]
    if not kept:
        # Keep at least the largest if nothing passes the threshold.
        kept = [
            max(
                components,
                key=lambda c: abs(c.volume) if c.is_watertight else 0.0,
            )
        ]
    log_status(
        "Cleaning",
        f"Kept {len(kept)} of {len(components)} components "
        f"(fraction >= {remove_islands_fraction}).",
    )
    import trimesh

    combined = trimesh.util.concatenate(kept)
    return Mesh(mesh=combined)


def _load_particles_for_adaptive(args):
    """Load and prepare particles for the adaptive pipeline.

    This reuses the existing ``load_swift_particles`` function from
    the loading module, then converts the outputs into the format
    expected by the adaptive C++ core.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Tuple of ``(positions, smoothing_lengths, domain_min,
        domain_max, origin)`` where positions is a list of 3-tuples,
        smoothing_lengths is a list of floats, domain_min and
        domain_max are 3-tuples, and origin is a numpy array.
    """
    from meshmerizer.commands.loading import load_swift_particles

    field_data, coords, h, effective_box_size, origin = load_swift_particles(
        filename=args.filename,
        particle_type=args.particle_type,
        field=args.field,
        smoothing_factor=args.smoothing_factor,
        box_size=args.box_size,
        shift=list(args.shift),
        wrap_shift=args.wrap_shift,
        center=args.center,
        extent=args.extent,
        periodic=args.periodic,
        tight_bounds=args.tight_bounds,
    )

    if h is None:
        print(
            "Error: Smoothing lengths are required for the "
            "adaptive pipeline but could not be determined.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_particles = coords.shape[0]
    if n_particles == 0:
        print(
            "Error: No particles selected. Check domain selection flags.",
            file=sys.stderr,
        )
        sys.exit(1)

    log_status(
        "Loading",
        f"Prepared {n_particles} particles for adaptive meshing.",
    )

    # Ensure positions are a contiguous float64 (N, 3) array and
    # smoothing lengths are a contiguous float64 (N,) array so the
    # C++ extension can read them via the buffer protocol in a single
    # memcpy instead of per-element PyFloat_AsDouble calls.
    positions = np.ascontiguousarray(coords, dtype=np.float64)
    smoothing_lengths = np.ascontiguousarray(h, dtype=np.float64)

    # The adaptive pipeline uses an explicit domain bounding box.
    # After loading, coordinates live in [0, effective_box_size)^3.
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (
        float(effective_box_size),
        float(effective_box_size),
        float(effective_box_size),
    )

    return positions, smoothing_lengths, domain_min, domain_max, origin


def _visualize_vertices(vert_positions, output_path: str) -> None:
    """Save a 6-panel figure showing QEF vertices from each face.

    Each panel is a 2D scatter plot of the vertex positions
    projected along one of the six axis-aligned directions
    (+X, -X, +Y, -Y, +Z, -Z), giving a complete view of the
    vertex distribution from every side of the bounding box.
    All panels use equal aspect ratio so the geometry is not
    distorted.

    Args:
        vert_positions: (N, 3) float64 array of vertex positions.
        output_path: File path to save the figure to (e.g.
            ``"qef_vertices.png"``).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Error: matplotlib is required for --visualise-verts. "
            "Install it with: pip install matplotlib",
            file=sys.stderr,
        )
        return

    # Define the six face views.  Each entry is:
    #   (title, horizontal_axis_index, vertical_axis_index,
    #    horizontal_label, vertical_label)
    # The axis indices select columns from the Nx3 positions array.
    views = [
        ("+X face (Y-Z)", 1, 2, "Y", "Z"),
        ("-X face (Y-Z)", 1, 2, "Y", "Z"),
        ("+Y face (X-Z)", 0, 2, "X", "Z"),
        ("-Y face (X-Z)", 0, 2, "X", "Z"),
        ("+Z face (X-Y)", 0, 1, "X", "Y"),
        ("-Z face (X-Y)", 0, 1, "X", "Y"),
    ]

    n_pts = len(vert_positions)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"QEF Vertices ({n_pts:,} points)",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (title, hi, vi, hlabel, vlabel) in zip(axes.flat, views):
        ax.scatter(
            vert_positions[:, hi],
            vert_positions[:, vi],
            s=0.5,
            alpha=0.4,
            color="C0",
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlabel(hlabel)
        ax.set_ylabel(vlabel)
        ax.set_title(title)
        ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved vertex visualisation to {output_path}")


def run_adaptive(args) -> None:
    """Run the adaptive meshing pipeline from CLI arguments.

    Args:
        args: Parsed CLI namespace containing all adaptive flags.

    Returns:
        ``None``. The output STL is written to disk.
    """
    total_start = time.perf_counter()

    # Set OpenMP thread count if requested.
    if args.nthreads is not None:
        from meshmerizer._adaptive import set_num_threads

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
    else:
        # Load particles from the SWIFT snapshot.
        positions, smoothing_lengths, domain_min, domain_max, origin = (
            _load_particles_for_adaptive(args)
        )
        max_depth = args.max_depth
        base_resolution = args.base_resolution

        # Determine isovalue: explicit --isovalue takes priority,
        # otherwise compute from the density percentile.
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
            # We need intermediate cells/contributors for saving,
            # so use the step-by-step path through Python.
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

            # Generate QEF vertices from the saved octree data.
            log_status(
                "Meshing",
                "Solving QEF vertices...",
            )
            mesh_start = time.perf_counter()
            vert_positions, vert_normals = solve_vertices(
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
            record_elapsed(
                "Vertex solve",
                mesh_start,
                operation="Meshing",
            )
        else:
            # Fast path: run the full particles-to-mesh pipeline
            # in C++ (octree + QEF + Poisson + Marching Cubes).
            # If FOF is enabled, cluster particles first and
            # reconstruct each group independently.
            screening_weight = getattr(args, "screening_weight", 4.0)
            poisson_depth = (
                args.poisson_depth if args.poisson_depth is not None else 9
            )

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
                n_groups = len(set(group_labels.tolist()))
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
                f"max_depth={poisson_depth}, "
                f"isovalue={isovalue}, "
                f"screening={screening_weight}",
            )
            pipeline_start = time.perf_counter()
            mesh_verts, mesh_faces = poisson_reconstruct(
                positions,
                None,
                smoothing_lengths,
                domain_min,
                domain_max,
                base_resolution,
                isovalue,
                poisson_depth,
                group_labels=group_labels,
                screening_weight=screening_weight,
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
                print(
                    "Warning: Pipeline produced no "
                    "triangles. Check isovalue and "
                    "domain selection.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Translate back to world-space and build Mesh.
            mesh_verts += origin

            mesh = Mesh(
                vertices=mesh_verts,
                faces=mesh_faces,
            )

            # Remove small disconnected islands if requested.
            mesh = _remove_islands(mesh, args.remove_islands_fraction)

            # Scale the mesh to physical print size.
            if args.target_size is not None:
                log_status(
                    "Cleaning",
                    f"Scaling mesh to {args.target_size} cm...",
                )
                mesh = scale_mesh_to_print(mesh, args.target_size)

            # Save the STL.
            output_path = args.output
            if output_path is None:
                output_path = args.filename.with_suffix(".stl")

            log_status(
                "Saving",
                f"Writing STL to {output_path}...",
            )
            mesh.save(str(output_path))

            record_elapsed(
                "Total pipeline",
                total_start,
                operation="Done",
            )
            log_status(
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
    # Step 4: Generate mesh for load-octree path.
    # ------------------------------------------------------------------
    if args.load_octree is not None:
        # If the loaded HDF5 already contains QEF vertices, reuse them
        # instead of re-solving (saves significant compute time).
        if state.get("vertices") is not None:
            log_status(
                "Meshing",
                "Using QEF vertices from loaded octree.",
            )
            vert_positions = state["vertices"]
            vert_normals = state["normals"]
        else:
            log_status(
                "Meshing",
                "Solving QEF vertices from loaded octree...",
            )
            mesh_start = time.perf_counter()
            vert_positions, vert_normals = solve_vertices(
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

    n_verts = len(vert_positions)
    log_status(
        "Meshing",
        f"Solved {n_verts} QEF vertices.",
    )

    # Optionally save a 6-panel vertex visualisation figure.
    if getattr(args, "visualise_verts", None):
        _visualize_vertices(vert_positions, args.visualise_verts)

    if n_verts == 0:
        print(
            "Warning: No QEF vertices produced. "
            "Check isovalue and domain selection.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Optional FOF clustering + Poisson surface reconstruction.
    # ------------------------------------------------------------------
    poisson_depth = args.poisson_depth if args.poisson_depth is not None else 9

    if getattr(args, "fof", False):
        # Cluster QEF vertices into distinct objects.
        log_status(
            "Clustering",
            f"Running FOF clustering "
            f"(linking_factor={args.linking_factor})...",
        )
        cluster_start = time.perf_counter()
        group_labels = fof_cluster(
            vert_positions,
            domain_min,
            domain_max,
            args.linking_factor,
        )
        n_groups = len(set(group_labels.tolist()))
        record_elapsed("FOF clustering", cluster_start, operation="Clustering")
        log_status(
            "Clustering",
            f"Found {n_groups} group(s).",
        )
    else:
        # Single group — treat all vertices as one object.
        group_labels = np.zeros(len(vert_positions), dtype=np.int64)

    # Run Poisson reconstruction per group and merge.
    screening_weight = getattr(args, "screening_weight", 4.0)
    poisson_depth = args.poisson_depth if args.poisson_depth is not None else 9
    log_status(
        "Meshing",
        f"Running Poisson reconstruction "
        f"(depth={poisson_depth}, "
        f"screening={screening_weight})...",
    )
    poisson_start = time.perf_counter()
    mesh_verts, mesh_faces = poisson_reconstruct(
        positions,
        vert_normals,
        smoothing_lengths,
        domain_min,
        domain_max,
        base_resolution,
        isovalue,
        poisson_depth,
        group_labels=group_labels,
        screening_weight=screening_weight,
    )
    record_elapsed(
        "Poisson reconstruction",
        poisson_start,
        operation="Meshing",
    )

    n_mesh_verts = len(mesh_verts)
    n_tris = len(mesh_faces)
    log_status(
        "Meshing",
        f"Poisson mesh: {n_mesh_verts} vertices, {n_tris} triangles.",
    )

    if n_tris == 0:
        print(
            "Warning: Poisson reconstruction produced no "
            "triangles. Check isovalue and domain selection.",
            file=sys.stderr,
        )
        sys.exit(1)

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
    mesh = _remove_islands(mesh, args.remove_islands_fraction)

    # Scale the mesh to a physical print size if requested.
    if args.target_size is not None:
        log_status(
            "Cleaning",
            f"Scaling mesh to {args.target_size} cm...",
        )
        mesh = scale_mesh_to_print(mesh, args.target_size)

    # ------------------------------------------------------------------
    # Step 6: Save the STL.
    # ------------------------------------------------------------------
    output_path = args.output
    if output_path is None:
        output_path = args.filename.with_suffix(".stl")

    log_status("Saving", f"Writing STL to {output_path}...")
    mesh.save(str(output_path))

    record_elapsed("Total pipeline", total_start, operation="Done")
    log_status("Done", f"Adaptive mesh saved to {output_path}")
