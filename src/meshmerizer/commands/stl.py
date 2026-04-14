"""STL command implementation.

This module contains the end-to-end CLI workflow for converting particle data
into STL files. It coordinates snapshot loading, voxel preprocessing, dense or
chunked meshing, mesh post-processing, and final file output.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import trimesh

from meshmerizer.chunks import (
    VirtualGrid,
    generate_hard_chunk_meshes,
    keep_largest_mesh_component,
    union_hard_chunk_meshes,
)
from meshmerizer.logging_utils import log_status
from meshmerizer.mesh import Mesh, voxels_to_stl_via_sdf
from meshmerizer.printing import scale_mesh_to_print

from .loading import (
    apply_preprocess,
    load_swift_particles,
    load_swift_volume,
    print_elapsed,
)


def run_stl(args: argparse.Namespace) -> None:
    """Execute the ``meshmerizer stl`` command.

    Args:
        args: Parsed CLI arguments for the STL subcommand.

    Returns:
        ``None``. The command writes STL output or exits with an error.
    """

    def fail_for_unsupported_separate_chunk_options() -> None:
        """Reject post-processing options unsupported by separate chunks.

        Returns:
            ``None``. This helper exits the program on invalid option usage.
        """
        # Separate chunk output writes each emitted mesh independently, so
        # whole-mesh operations that rely on one final combined surface must be
        # rejected explicitly rather than ignored silently.
        if args.target_size is not None:
            log_status(
                "Cleaning",
                "Error: --target-size is not supported with "
                "--chunk-output separate",
            )
            sys.exit(1)
        if args.subdivide_iters > 0:
            log_status(
                "Cleaning",
                "Error: --subdivide-iters is not supported with "
                "--chunk-output separate",
            )
            sys.exit(1)
        if args.smooth_iters > 0:
            log_status(
                "Cleaning",
                "Error: --smooth-iters is not supported with "
                "--chunk-output separate",
            )
            sys.exit(1)
        if args.remove_islands is not None:
            log_status(
                "Cleaning",
                "Error: --remove-islands is not supported with "
                "--chunk-output separate",
            )
            sys.exit(1)

    run_start = time.perf_counter()

    # Reject unsupported flag combinations before any expensive snapshot load
    # or chunk meshing work begins.
    if args.nchunks > 1 and args.chunk_output == "separate":
        fail_for_unsupported_separate_chunk_options()

    # Validate the high-level CLI controls before doing any expensive I/O.
    if args.nchunks < 1:
        log_status("Meshing", "Error: --nchunks must be >= 1")
        sys.exit(1)
    if not (0.0 < args.simplify_factor <= 1.0):
        log_status(
            "Cleaning",
            "Error: --simplify-factor must satisfy 0 < factor <= 1",
        )
        sys.exit(1)
    # Dispatch to the chunked pipeline when the user requests more than one
    # chunk per axis. This path loads particles directly and avoids building a
    # full dense grid.
    if args.nchunks > 1:
        try:
            field_data, coords, h, effective_box_size, origin = (
                load_swift_particles(
                    filename=args.filename,
                    particle_type=args.particle_type,
                    field=args.field,
                    smoothing_factor=args.smoothing_factor,
                    box_size=args.box_size,
                    shift=args.shift,
                    wrap_shift=args.wrap_shift,
                    center=args.center,
                    extent=args.extent,
                    periodic=args.periodic,
                    tight_bounds=args.tight_bounds,
                )
            )
        except (RuntimeError, ValueError) as exc:
            log_status("Loading", str(exc))
            sys.exit(1)

        # Represent the global cube virtually so chunk geometry can be computed
        # without allocating the full dense field.
        final_threshold = args.threshold
        virtual_grid = VirtualGrid(
            origin=origin,
            box_size=effective_box_size,
            resolution=args.resolution,
            nchunks=args.nchunks,
        )

        log_status(
            "Meshing",
            "Chunked meshing setup:\n"
            f"  threshold:    {final_threshold:.4e}\n"
            f"  chunks/axis:  {args.nchunks}\n"
            f"  resolution:   {args.resolution}\n"
            f"  chunk output: {args.chunk_output}",
        )
        if args.chunk_output == "separate":
            log_status("Meshing", "Chunk extraction pass:")
        else:
            log_status("Meshing", "Chunk union pass:")

        # Generate all chunk-local meshes first. The later save step either
        # writes them separately or unions them into one final mesh.
        try:
            mesh_start = time.perf_counter()
            chunk_meshes = generate_hard_chunk_meshes(
                field_data,
                coords,
                h,
                virtual_grid,
                threshold=final_threshold,
                preprocess=args.preprocess,
                clip_halos=args.clip_halos,
                gaussian_sigma=args.gaussian_sigma,
                nthreads=args.nthreads,
                overlap_voxels=1,
                clip_to_bounds=args.chunk_output != "unioned",
            )
            print_elapsed("Hard chunk mesh generation", mesh_start)
        except (ValueError, RuntimeError) as exc:
            log_status("Meshing", f"Error generating chunked mesh: {exc}")
            sys.exit(1)

        if not chunk_meshes:
            log_status("Meshing", "Error: No chunk meshes generated.")
            sys.exit(1)

        output_path = args.output or args.filename.with_suffix(".stl")
        # The ``separate`` mode writes one file per emitted mesh without any
        # global union step.
        if args.chunk_output == "separate":
            output_dir = output_path.with_suffix("")
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = output_path.stem
            chunk_counter = 0
            log_status("Saving", f"Writing separate chunks to {output_dir}/")
            for _bounds, meshes in chunk_meshes:
                for mesh in meshes:
                    chunk_counter += 1
                    chunk_path = output_dir / f"{stem}_{chunk_counter}.stl"
                    if args.simplify_factor < 1.0:
                        log_status(
                            "Cleaning",
                            "Simplifying chunk mesh "
                            f"to factor {args.simplify_factor:.3f}...",
                        )
                        mesh.simplify(args.simplify_factor)
                    log_status("Saving", f"Saving chunk to {chunk_path}...")
                    mesh.save(str(chunk_path))
            print_elapsed("Total STL pipeline", run_start)
            log_status("Saving", "Done.")
            return

        # The ``unioned`` mode assigns seam ownership and assembles the chunk
        # meshes into one final watertight surface.
        log_status("Meshing", "Union assembly pass:")
        final_mesh = union_hard_chunk_meshes(chunk_meshes)

        # Apply connected-component filtering after unioning so the decision is
        # based on the final assembled surface rather than per-chunk fragments.
        if args.remove_islands is not None:
            if args.remove_islands == 0:
                log_status(
                    "Cleaning",
                    "Island filtering: keeping only the largest component",
                )
                final_mesh = keep_largest_mesh_component(final_mesh)
            else:
                log_status(
                    "Cleaning",
                    "Island filtering: removing components smaller than "
                    f"{args.remove_islands} voxels",
                )
                # Convert the requested voxel-count threshold into an
                # approximate physical volume threshold so connected
                # components can be filtered after the final surface has been
                # assembled.
                min_volume = args.remove_islands * (virtual_grid.voxel_size**3)
                components = final_mesh.to_trimesh().split(
                    only_watertight=False
                )
                kept = []
                for component in components:
                    if component.is_volume:
                        component_volume = abs(component.volume)
                    else:
                        component_volume = float(np.prod(component.extents))
                    if component_volume >= min_volume:
                        kept.append(component)
                if not kept:
                    log_status(
                        "Cleaning",
                        "Error: Island removal removed all chunked mesh "
                        "components.",
                    )
                    sys.exit(1)
                if len(kept) == 1:
                    final_mesh = Mesh(mesh=kept[0].copy())
                else:
                    final_mesh = Mesh(mesh=trimesh.util.concatenate(kept))

        # Run the optional post-extraction operations only after the final mesh
        # topology is in place.
        if args.simplify_factor < 1.0:
            log_status(
                "Cleaning",
                "Simplifying final mesh to factor "
                f"{args.simplify_factor:.3f}...",
            )
            final_mesh.simplify(args.simplify_factor)
        if args.target_size:
            log_status(
                "Cleaning",
                f"Scaling mesh to target size: {args.target_size} cm...",
            )
            scale_mesh_to_print(final_mesh, args.target_size)

        if args.subdivide_iters < 0:
            log_status("Cleaning", "Error: --subdivide-iters must be >= 0")
            sys.exit(1)
        if args.subdivide_iters > 0:
            log_status(
                "Cleaning",
                "Applying Loop subdivision to final mesh "
                f"({args.subdivide_iters} iterations)...",
            )
            final_mesh.subdivide(iterations=args.subdivide_iters)

        # Repair is always run at the end so the saved mesh sees the final
        # subdivision and smoothing state.
        if args.smooth_iters < 0:
            log_status("Cleaning", "Error: --smooth-iters must be >= 0")
            sys.exit(1)
        if args.smooth_iters > 0:
            log_status(
                "Cleaning",
                "Applying Taubin smoothing to final mesh "
                f"({args.smooth_iters} iterations)...",
            )
            final_mesh.repair(smoothing_iters=args.smooth_iters)
        else:
            final_mesh.repair(smoothing_iters=0)

        if not final_mesh.to_trimesh().is_watertight:
            log_status(
                "Saving",
                "⚠️ Warning: final mesh is not watertight. "
                "The STL will still be written.",
            )
        log_status("Saving", f"Saving final chunked mesh to {output_path}...")
        save_start = time.perf_counter()
        final_mesh.save(str(output_path))
        print_elapsed("Mesh save", save_start)
        print_elapsed("Total STL pipeline", run_start)
        log_status("Saving", "Done.")
        return

    # The dense path loads and voxelizes the full field before one global SDF
    # extraction pass.
    try:
        grid, voxel_size, origin = load_swift_volume(
            filename=args.filename,
            particle_type=args.particle_type,
            field=args.field,
            resolution=args.resolution,
            nthreads=args.nthreads,
            smoothing_factor=args.smoothing_factor,
            box_size=args.box_size,
            shift=args.shift,
            wrap_shift=args.wrap_shift,
            center=args.center,
            extent=args.extent,
            periodic=args.periodic,
            tight_bounds=args.tight_bounds,
        )
    except (RuntimeError, ValueError) as exc:
        log_status("Loading", str(exc))
        sys.exit(1)

    # Apply grid preprocessing only after the full dense field has been built.
    try:
        preprocess_start = time.perf_counter()
        grid = apply_preprocess(
            grid,
            args.preprocess,
            args.clip_halos,
            args.gaussian_sigma,
        )
    except ValueError as exc:
        log_status("Cleaning", str(exc))
        sys.exit(1)
    print_elapsed("Grid preprocessing", preprocess_start)

    final_threshold = args.threshold

    log_status(
        "Meshing",
        "Generating watertight mesh using SDF "
        f"(threshold={final_threshold:.4e})...",
    )

    # The dense path always uses SDF extraction because watertightness is the
    # primary requirement for the final STL.
    try:
        mesh_start = time.perf_counter()
        meshes = voxels_to_stl_via_sdf(
            grid,
            threshold=final_threshold,
            remove_islands=args.remove_islands,
            voxel_size=voxel_size,
        )
        print_elapsed("Dense mesh generation", mesh_start)
    except ValueError as exc:
        log_status("Meshing", f"Error generating mesh: {exc}")
        sys.exit(1)

    if not meshes:
        log_status("Meshing", "Error: No mesh generated.")
        sys.exit(1)

    # Merge multiple extracted islands into one wrapper object before the final
    # translate, simplify, and repair steps.
    final_mesh: Mesh = meshes[0]
    if len(meshes) > 1:
        log_status("Meshing", f"Merging {len(meshes)} mesh components...")
        combined = trimesh.util.concatenate([m.to_trimesh() for m in meshes])
        final_mesh = Mesh(mesh=combined)

    final_mesh.translate(origin)

    if args.simplify_factor < 1.0:
        log_status(
            "Cleaning",
            f"Simplifying final mesh to factor {args.simplify_factor:.3f}...",
        )
        final_mesh.simplify(args.simplify_factor)

    if args.target_size:
        log_status(
            "Cleaning",
            f"Scaling mesh to target size: {args.target_size} cm...",
        )
        scale_mesh_to_print(final_mesh, args.target_size)

    if args.subdivide_iters < 0:
        log_status("Cleaning", "Error: --subdivide-iters must be >= 0")
        sys.exit(1)
    if args.subdivide_iters > 0:
        log_status(
            "Cleaning",
            "Applying Loop subdivision to final mesh "
            f"({args.subdivide_iters} iterations)...",
        )
        final_mesh.subdivide(iterations=args.subdivide_iters)

    if args.smooth_iters < 0:
        log_status("Cleaning", "Error: --smooth-iters must be >= 0")
        sys.exit(1)
    if args.smooth_iters > 0:
        log_status(
            "Cleaning",
            "Applying Taubin smoothing to final mesh "
            f"({args.smooth_iters} iterations)...",
        )
        final_mesh.repair(smoothing_iters=args.smooth_iters)
    else:
        final_mesh.repair(smoothing_iters=0)

    output_path = args.output or args.filename.with_suffix(".stl")
    if not final_mesh.to_trimesh().is_watertight:
        log_status(
            "Saving",
            "⚠️ Warning: final mesh is not watertight. "
            "The STL will still be written.",
        )
    log_status("Saving", f"Saving to {output_path}...")
    save_start = time.perf_counter()
    final_mesh.save(str(output_path))
    print_elapsed("Mesh save", save_start)
    print_elapsed("Total STL pipeline", run_start)
    log_status("Saving", "Done.")
