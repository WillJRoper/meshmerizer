"""Adaptive meshing CLI command.

This module implements the ``adaptive`` subcommand which runs the new
adaptive octree pipeline from a SWIFT snapshot to an STL file.  It can
optionally use particle-level FOF clustering either to reconstruct
distinct objects independently or to discard small detached fluff
clusters before meshing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from meshmerizer.adaptive_core import (
    compute_isovalue_from_percentile,
    create_top_level_cells_with_contributors,
    fof_cluster,
    refine_octree,
    solve_vertices,
)
from meshmerizer.logging import (
    abort_with_error,
    log_status,
    log_summary_status,
    record_elapsed,
)
from meshmerizer.mesh.core import Mesh
from meshmerizer.printing import scale_mesh_to_print
from meshmerizer.reconstruct import reconstruct_mesh
from meshmerizer.serialize import export_octree, import_octree


def _compute_print_scale_factor_cm(
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    target_size_cm: float,
) -> float:
    """Return scale factor from native units to print millimetres.

    The adaptive pipeline runs in native snapshot units. When a target print
    size is requested, we convert user-facing print-length controls back into
    native units using the same longest-dimension scale factor that will later
    be applied to the final mesh.

    Args:
        domain_min: Lower corner of the working domain.
        domain_max: Upper corner of the working domain.
        target_size_cm: Requested longest print dimension in centimetres.

    Returns:
        Uniform scale factor mapping native units to output millimetres.

    Raises:
        ValueError: If the working domain has zero extent.
    """
    extents = np.asarray(domain_max, dtype=np.float64) - np.asarray(
        domain_min, dtype=np.float64
    )
    max_dimension = float(np.max(extents))
    if max_dimension <= 0.0:
        raise ValueError("working domain has zero extent")
    target_size_mm = float(target_size_cm) * 10.0
    return target_size_mm / max_dimension


def _convert_print_length_to_native_units(
    length_cm: float,
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    target_size_cm: float,
) -> float:
    """Convert a print-space centimetre length to native meshing units."""
    scale_factor = _compute_print_scale_factor_cm(
        domain_min, domain_max, target_size_cm
    )
    return (float(length_cm) * 10.0) / scale_factor


def _remove_islands(
    mesh: Mesh,
    remove_islands_fraction: Optional[float],
) -> Mesh:
    """Remove small disconnected components from a mesh.

    Args:
        mesh: Input mesh.
        remove_islands_fraction: Fraction of the largest component
            volume below which a connected component is discarded.
            ``0.0`` keeps only the largest component. ``None``
            disables island removal entirely.

    Returns:
        A new ``Mesh`` with small islands removed, or the original
        mesh unchanged when removal is disabled.
    """
    if remove_islands_fraction is None:
        return mesh

    def _component_reference_volume(component) -> float:
        """Return a robust size estimate for island filtering.

        Prefer the true enclosed volume for watertight meshes. When a
        component is not watertight, fall back to its convex-hull volume so
        that a large but imperfect main body is not incorrectly treated as
        having zero size.
        """
        if component.is_watertight:
            volume = abs(float(component.volume))
            if np.isfinite(volume):
                return volume

        try:
            hull_volume = abs(float(component.convex_hull.volume))
        except Exception:
            hull_volume = 0.0
        return hull_volume if np.isfinite(hull_volume) else 0.0

    # Split into connected components via trimesh.
    tm = mesh.mesh
    components = tm.split(only_watertight=False)
    if len(components) <= 1:
        return mesh

    if remove_islands_fraction == 0.0:
        # Keep only the single largest component by volume.
        largest = max(
            components,
            key=_component_reference_volume,
        )
        log_status(
            "Cleaning",
            f"Kept largest of {len(components)} components.",
        )
        return Mesh(mesh=largest)

    # Keep components whose volume fraction exceeds the threshold relative to
    # the largest resolved object, not the sum of all objects.
    volumes = [_component_reference_volume(comp) for comp in components]
    largest_volume = max(volumes, default=0.0)
    if largest_volume == 0.0:
        return mesh

    kept = [
        c
        for c, v in zip(components, volumes)
        if v / largest_volume >= remove_islands_fraction
    ]
    if not kept:
        # Keep at least the largest if nothing passes the threshold.
        kept = [
            max(
                components,
                key=_component_reference_volume,
            )
        ]
    log_status(
        "Cleaning",
        f"Kept {len(kept)} of {len(components)} components "
        f"(fraction >= {remove_islands_fraction} of largest volume).",
    )
    import trimesh

    combined = trimesh.util.concatenate(kept)
    return Mesh(mesh=combined)


def _simplify_mesh(mesh: Mesh, simplify_factor: float) -> Mesh:
    """Optionally simplify the mesh after extraction and cleanup."""
    if simplify_factor == 1.0:
        return mesh

    log_status(
        "Cleaning",
        f"Simplifying mesh to retain factor {simplify_factor:.6g}...",
    )
    before_faces = len(mesh.faces)
    mesh.simplify(factor=simplify_factor)
    after_faces = len(mesh.faces)
    log_status(
        "Cleaning",
        f"Simplified mesh faces: {before_faces} -> {after_faces}",
    )
    return mesh


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

    coords, h, effective_box_size, origin = load_swift_particles(
        filename=args.filename,
        particle_type=args.particle_type,
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
        abort_with_error(
            "Loading",
            "Smoothing lengths are required for the adaptive pipeline but "
            "could not be determined.",
        )

    n_particles = coords.shape[0]
    if n_particles == 0:
        abort_with_error(
            "Loading",
            "No particles selected. Check domain selection flags.",
        )

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


def _filter_small_particle_fof_clusters(
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    linking_factor: float,
    min_cluster_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Discard particle FOF clusters smaller than ``min_cluster_size``.

    Args:
        positions: Particle positions as an ``(N, 3)`` float64 array.
        smoothing_lengths: Per-particle smoothing lengths as ``(N,)``.
        domain_min: Lower corner of the working domain.
        domain_max: Upper corner of the working domain.
        linking_factor: FOF linking factor.
        min_cluster_size: Minimum number of particles required for a
            cluster to be retained.

    Returns:
        Tuple of filtered ``(positions, smoothing_lengths)`` arrays.

    Raises:
        ValueError: If ``min_cluster_size`` is not positive.
    """
    if min_cluster_size <= 0:
        raise ValueError(
            f"min_cluster_size must be positive, got {min_cluster_size}"
        )

    if len(positions) == 0:
        return positions, smoothing_lengths

    labels = fof_cluster(positions, domain_min, domain_max, linking_factor)
    unique_labels, counts = np.unique(labels, return_counts=True)
    kept_labels = unique_labels[counts >= min_cluster_size]

    n_groups = int(unique_labels.size)
    n_kept_groups = int(kept_labels.size)
    n_removed_groups = n_groups - n_kept_groups
    kept_mask = np.isin(labels, kept_labels)
    n_kept_particles = int(np.count_nonzero(kept_mask))
    n_removed_particles = int(len(positions) - n_kept_particles)

    log_status(
        "Clustering",
        (
            f"FOF particle filtering: kept {n_kept_groups}/{n_groups} groups "
            f"and {n_kept_particles}/{len(positions)} particles "
            f"(removed {n_removed_groups} groups, {n_removed_particles} "
            f"particles; min size = {min_cluster_size})."
        ),
    )

    if n_kept_particles == 0:
        return positions[:0], smoothing_lengths[:0]

    return positions[kept_mask], smoothing_lengths[kept_mask]


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
        abort_with_error(
            "Meshing",
            "matplotlib is required for --visualise-verts. Install it with: "
            "pip install matplotlib",
        )

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
    log_summary_status(
        "Saving", f"Saved vertex visualisation to {output_path}"
    )


def _emit_tree_structure_summary(cells) -> None:
    """Log a concise per-depth summary of the octree structure.

    Args:
        cells: Iterable of octree cell dictionaries.
    """
    if not cells:
        log_status(
            "Tree",
            "Summary:\n"
            "total=0 leaf=0 internal=0 active=0 inactive=0 surface=0",
        )
        return

    max_depth = max(int(cell.get("depth", 0)) for cell in cells)
    per_depth = [
        {
            "total": 0,
            "leaf": 0,
            "active": 0,
            "surface": 0,
        }
        for _ in range(max_depth + 1)
    ]

    total_leaf = 0
    total_active = 0
    total_surface = 0

    for cell in cells:
        depth = int(cell.get("depth", 0))
        summary = per_depth[depth]
        summary["total"] += 1
        is_leaf = bool(cell.get("is_leaf", False))
        is_active = bool(cell.get("is_active", False))
        has_surface = bool(cell.get("has_surface", False))
        if is_leaf:
            summary["leaf"] += 1
            total_leaf += 1
        if is_active:
            summary["active"] += 1
            total_active += 1
        if has_surface:
            summary["surface"] += 1
            total_surface += 1

    lines = [
        "Summary:",
        (
            f"total={len(cells)} leaf={total_leaf} "
            f"internal={len(cells) - total_leaf} "
            f"active={total_active} inactive={len(cells) - total_active} "
            f"surface={total_surface}"
        ),
    ]
    for depth, summary in enumerate(per_depth):
        lines.append(
            (
                f"depth {depth}: total={summary['total']} "
                f"leaf={summary['leaf']} "
                f"internal={summary['total'] - summary['leaf']} "
                f"active={summary['active']} "
                f"inactive={summary['total'] - summary['active']} "
                f"surface={summary['surface']}"
            )
        )

    log_summary_status("Tree", "\n".join(lines))


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
            mesh.save(str(output_path))
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
    mesh.save(str(output_path))
    record_elapsed("STL export", save_start, operation="Saving")

    if args.load_octree is not None or args.save_octree is not None:
        _emit_tree_structure_summary(cells)

    record_elapsed("Total pipeline", total_start, operation="Done")
    log_summary_status("Done", f"Adaptive mesh saved to {output_path}")
