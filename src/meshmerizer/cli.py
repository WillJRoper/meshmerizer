"""Command Line Interface for Meshmerizer."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from meshmerizer.chunking import (
    VirtualGrid,
    generate_hard_chunk_meshes,
    union_hard_chunk_meshes,
)
from meshmerizer.debug_plots import save_histogram_png, save_z_projection_png
from meshmerizer.mesh import Mesh, voxels_to_stl_via_sdf
from meshmerizer.printing import scale_mesh_to_print
from meshmerizer.volume_io import compute_grid_stats, write_volume_h5
from meshmerizer.voxels import (
    generate_voxel_grid,
    process_filament_filter,
    process_gaussian_smoothing,
    process_log_scale,
    process_remove_halos,
)


def _boxsize_to_float(boxsize: object) -> float:
    box = boxsize
    if hasattr(box, "value"):
        box = box.value
    arr = np.asarray(box)
    if arr.ndim == 0:
        return float(arr)
    return float(np.max(arr))


def _print_elapsed(label: str, start: float) -> None:
    """Print a simple stage timing line."""
    print(f"{label} took {time.perf_counter() - start:.3f} s")


def _add_common_voxel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shift",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help=(
            "Apply a coordinate shift (dx dy dz) in simulation units before "
            "any cropping/voxelization. Default: 0 0 0"
        ),
    )

    wrap_shift_group = parser.add_mutually_exclusive_group()
    wrap_shift_group.add_argument(
        "--wrap-shift",
        dest="wrap_shift",
        action="store_true",
        help=(
            "After applying --shift, wrap coordinates into [0, box_size) "
            "assuming a periodic box. Default for SWIFT snapshots."
        ),
    )
    wrap_shift_group.add_argument(
        "--no-wrap-shift",
        dest="wrap_shift",
        action="store_false",
        help=(
            "Do not wrap coordinates after applying --shift (still shifts). "
            "Useful for non-periodic inputs."
        ),
    )
    parser.set_defaults(wrap_shift=True)

    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        default=128,
        help="Voxel grid resolution (N x N x N). Default: 128",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=1,
        help=(
            "Number of threads to use for C-accelerated smoothing "
            "deposition. Default: 1"
        ),
    )
    parser.add_argument(
        "--box-size",
        "-b",
        type=float,
        default=None,
        help=(
            "Physical size of the simulation box (in simulation units). "
            "Used to define the voxel grid boundaries."
        ),
    )
    parser.add_argument(
        "--particle-type",
        "-p",
        type=str,
        default="gas",
        choices=["gas", "dark_matter", "stars", "black_holes"],
        help="Particle type to extract. Default: 'gas'",
    )
    parser.add_argument(
        "--field",
        "-f",
        type=str,
        default="masses",
        help=(
            "Particle field to project (e.g., 'masses', 'densities'). "
            "Default: 'masses'"
        ),
    )
    parser.add_argument(
        "--smoothing-factor",
        type=float,
        default=1.0,
        help=(
            "Multiplier for particle smoothing lengths. Increase to make the "
            "fluid look more connected. Default: 1.0"
        ),
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="none",
        choices=["none", "log", "filaments"],
        help=(
            "Preprocessing to apply to the grid. 'log': log scaling. "
            "'filaments': Hessian-based filament enhancement."
        ),
    )
    parser.add_argument(
        "--clip-halos",
        type=float,
        default=None,
        help=(
            "Percentile (0-100) above which to clip density values. "
            "Useful for suppressing massive halos to reveal filaments."
        ),
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.0,
        help=(
            "Gaussian smoothing sigma in voxel units to apply to the voxel "
            "grid before thresholding. Default: 0"
        ),
    )

    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Center of an extracted subregion (x y z) in simulation units. "
            "Requires --extent."
        ),
    )
    parser.add_argument(
        "--extent",
        type=float,
        default=None,
        help=(
            "Side length of an extracted cubic subregion in simulation units. "
            "Requires --center."
        ),
    )
    parser.add_argument(
        "--tight-bounds",
        action="store_true",
        help=(
            "Shrink the voxelization cube to the occupied particle bounds "
            "after any shift/crop, reducing wasted resolution."
        ),
    )
    parser.add_argument(
        "--no-periodic",
        dest="periodic",
        action="store_false",
        help=(
            "Disable periodic wrapping for subregion selection. By default, "
            "SWIFT snapshots are treated as periodic."
        ),
    )
    parser.set_defaults(periodic=True)


def _apply_coordinate_shift(
    coords: np.ndarray,
    *,
    shift: np.ndarray,
    wrap_shift: bool,
    box_size: Optional[float],
) -> np.ndarray:
    """Apply a translation (and optional periodic wrap) to coordinates."""
    coords_arr = np.asarray(coords, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")

    shift_arr = np.asarray(shift, dtype=np.float64)
    if shift_arr.shape != (3,):
        raise ValueError("shift must have shape (3,)")

    shifted = coords_arr + shift_arr
    if not wrap_shift:
        return shifted

    if box_size is None:
        raise ValueError(
            "--wrap-shift requested but box_size is not known. "
            "Pass --box-size (or ensure snapshot metadata includes boxsize), "
            "or use --no-wrap-shift."
        )
    box = float(box_size)
    if box <= 0.0:
        raise ValueError("box_size must be > 0 when using --wrap-shift")

    return np.mod(shifted, box)


def _crop_particles_to_region(
    coords: np.ndarray,
    values: np.ndarray,
    smoothing_lengths: Optional[np.ndarray],
    *,
    center: np.ndarray,
    extent: float,
    box_size: float,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Crop particle arrays to an axis-aligned cubic region.

    Returns region-local coordinates in [0, extent) and the world-space origin
    (min corner) used for the translation.
    """
    if extent <= 0:
        raise ValueError("extent must be > 0")
    if box_size <= 0:
        raise ValueError("box_size must be > 0")

    c = np.asarray(center, dtype=np.float64)
    if c.shape != (3,):
        raise ValueError("center must have shape (3,)")

    coords_arr = np.asarray(coords, dtype=np.float64)
    values_arr = np.asarray(values)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if values_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError("values must have shape (N,)")

    half = 0.5 * float(extent)
    mins = c - half
    maxs = c + half

    if periodic:
        # Center-wrapped selection using the minimum image convention.
        # This avoids missing particles when the structure straddles periodic
        # boundaries (e.g. appears in multiple corners in projection).
        delta = coords_arr - c
        delta = (delta + 0.5 * box_size) % box_size - 0.5 * box_size
        mask = np.all(np.abs(delta) <= half, axis=1)

        local = delta + half
        origin = np.mod(mins, box_size)
    else:
        if np.any(mins < 0.0) or np.any(maxs > box_size):
            raise ValueError(
                "Non-periodic region must lie within [0, box_size]"
            )
        origin = mins
        mask = np.all((coords_arr >= mins) & (coords_arr < maxs), axis=1)
        local = coords_arr - origin

    cropped_coords = local[mask]
    cropped_values = values_arr[mask]
    if smoothing_lengths is None:
        cropped_h = None
    else:
        h_arr = np.asarray(smoothing_lengths)
        cropped_h = h_arr[mask]

    return cropped_coords, cropped_values, cropped_h, origin


def _tighten_voxelization_bounds(
    coords: np.ndarray,
    smoothing_lengths: Optional[np.ndarray],
    *,
    box_size: float,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, float]:
    """Shrink the voxelization cube to the occupied particle bounds.

    The returned coordinates are shifted so the tightened min corner becomes
    the new local origin. The new cube side length is the maximum occupied span
    across x/y/z so the downstream voxel grid remains cubic.
    """
    if box_size <= 0:
        raise ValueError("box_size must be > 0")

    coords_arr = np.asarray(coords, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if coords_arr.shape[0] == 0:
        raise ValueError("coords must not be empty")

    mins = np.min(coords_arr, axis=0)
    maxs = np.max(coords_arr, axis=0)

    if smoothing_lengths is None:
        h_arr = None
    else:
        h_arr = np.asarray(smoothing_lengths, dtype=np.float64)
        if h_arr.shape != (coords_arr.shape[0],):
            raise ValueError("smoothing_lengths must have shape (N,)")
        mins = np.min(coords_arr - h_arr[:, None], axis=0)
        maxs = np.max(coords_arr + h_arr[:, None], axis=0)

    mins = np.clip(mins, 0.0, box_size)
    maxs = np.clip(maxs, 0.0, box_size)
    spans = maxs - mins
    tight_box_size = float(np.max(spans))

    if tight_box_size <= 0.0:
        return coords_arr, h_arr, np.zeros(3, dtype=np.float64), box_size

    return coords_arr - mins, h_arr, mins, tight_box_size


def _raise_if_empty_subregion_selection(
    n_selected: int,
    *,
    particle_type: str,
    field: str,
    center: np.ndarray,
    extent: float,
    periodic: bool,
) -> None:
    """Raise a helpful error if a subregion selection is empty."""
    if n_selected != 0:
        return

    c = np.asarray(center, dtype=np.float64)
    msg = (
        "Subregion selection contains no particles: "
        f"particle_type={particle_type} field={field} "
        f"center=({c[0]:.6g},{c[1]:.6g},{c[2]:.6g}) "
        f"extent={float(extent):.6g} periodic={periodic} "
        f"selected_particles={n_selected}. "
        "Try increasing --extent or choosing a different --center "
        "(for example, the particle center-of-mass). "
        "If periodic wrapping is causing confusion, try --no-periodic."
    )
    raise RuntimeError(msg)


def _apply_preprocess(
    grid: np.ndarray,
    preprocess: str,
    clip_halos: Optional[float],
    gaussian_sigma: float,
) -> np.ndarray:
    if clip_halos is not None:
        grid = process_remove_halos(grid, threshold_percentile=clip_halos)

    if preprocess != "none":
        print(f"Applying preprocessing: {preprocess}")
        if preprocess == "log":
            grid = process_log_scale(grid)
        elif preprocess == "filaments":
            grid = process_filament_filter(grid)

    if gaussian_sigma < 0:
        raise ValueError("--gaussian-sigma must be >= 0")
    grid = process_gaussian_smoothing(grid, sigma=gaussian_sigma)

    print(f"Processed grid range: [{grid.min():.4e}, {grid.max():.4e}]")
    return grid


def _load_swift_volume(
    filename: Path,
    particle_type: str,
    field: str,
    resolution: int,
    nthreads: int,
    smoothing_factor: float,
    box_size: Optional[float],
    shift: list[float],
    wrap_shift: bool,
    center: Optional[list[float]],
    extent: Optional[float],
    periodic: bool,
    tight_bounds: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    voxelize_start = time.perf_counter()
    field_data, coords, h, effective_box_size, origin = _load_swift_particles(
        filename=filename,
        particle_type=particle_type,
        field=field,
        smoothing_factor=smoothing_factor,
        box_size=box_size,
        shift=shift,
        wrap_shift=wrap_shift,
        center=center,
        extent=extent,
        periodic=periodic,
        tight_bounds=tight_bounds,
    )

    print(f"Voxelizing to {resolution}^3 grid...")
    grid, voxel_size = generate_voxel_grid(
        data=field_data,
        coordinates=coords,
        resolution=resolution,
        smoothing_lengths=h,
        box_size=effective_box_size,
        nthreads=nthreads,
    )
    print(f"Voxel size: {voxel_size:.4e} (sim units)")
    _print_elapsed("Voxelization", voxelize_start)
    return grid, voxel_size, origin


def _load_swift_particles(
    filename: Path,
    particle_type: str,
    field: str,
    smoothing_factor: float,
    box_size: Optional[float],
    shift: list[float],
    wrap_shift: bool,
    center: Optional[list[float]],
    extent: Optional[float],
    periodic: bool,
    tight_bounds: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    float,
    np.ndarray,
]:
    import swiftsimio as sw
    from swiftsimio.visualisation import generate_smoothing_lengths

    total_start = time.perf_counter()

    print(f"Loading SWIFT data from {filename}...")
    load_start = time.perf_counter()
    try:
        data = sw.load(str(filename))
    except Exception as exc:
        raise RuntimeError(f"Error loading file: {exc}") from exc
    _print_elapsed("Snapshot load", load_start)

    if (center is None) != (extent is None):
        raise ValueError("--center and --extent must be provided together")

    full_box_size_source: str
    if box_size is not None:
        box_size_source = "--box-size"
    else:
        meta_box = None
        if hasattr(data, "metadata"):
            meta_box = getattr(data.metadata, "boxsize", None)

        if meta_box is None:
            raise RuntimeError(
                "Snapshot metadata does not include boxsize. "
                "Pass --box-size to define the physical volume."
            )

        box_size = _boxsize_to_float(meta_box)
        box_size_source = "snapshot metadata"

    full_box_size = float(box_size)
    full_box_size_source = box_size_source

    if particle_type == "gas":
        part_data = data.gas
    elif particle_type == "dark_matter":
        part_data = data.dark_matter
    elif particle_type == "stars":
        part_data = data.stars
    elif particle_type == "black_holes":
        part_data = data.black_holes
    else:
        raise ValueError(f"Unknown particle type '{particle_type}'")

    print(f"Extracting {particle_type} particles...")
    extract_start = time.perf_counter()
    coords_cosmo = part_data.coordinates
    coords = coords_cosmo.value

    if not hasattr(part_data, field):
        available_fields = [
            k for k in part_data.__dict__.keys() if not k.startswith("_")
        ]
        raise ValueError(
            f"Field '{field}' not found in {particle_type} particles. "
            f"Available fields: {available_fields}"
        )

    field_data = getattr(part_data, field).value
    _print_elapsed("Particle field extraction", extract_start)

    shift_arr = np.asarray(shift, dtype=np.float64)
    if shift_arr.shape != (3,):
        raise ValueError("--shift must provide exactly 3 values: dx dy dz")
    print(
        "Coordinate shift: "
        f"({shift_arr[0]:.6g},{shift_arr[1]:.6g},{shift_arr[2]:.6g}) "
        f"wrap_shift={wrap_shift}"
    )
    shift_start = time.perf_counter()
    coords = _apply_coordinate_shift(
        coords,
        shift=shift_arr,
        wrap_shift=wrap_shift,
        box_size=full_box_size,
    )
    _print_elapsed("Coordinate shifting", shift_start)

    if hasattr(part_data, "smoothing_lengths"):
        smoothing_start = time.perf_counter()
        h = part_data.smoothing_lengths.value * smoothing_factor
        _print_elapsed("Smoothing-length extraction", smoothing_start)
    else:
        print("Smoothing lengths not found. Generating...")
        boxsize = data.metadata.boxsize
        smoothing_start = time.perf_counter()
        try:
            h_cosmo = generate_smoothing_lengths(
                coords_cosmo,
                boxsize,
                kernel_gamma=1.8,
            )
            h = h_cosmo.value * smoothing_factor
            print("Smoothing lengths generated.")
            _print_elapsed("Smoothing-length generation", smoothing_start)
        except Exception as exc:
            print(f"Error generating smoothing lengths: {exc}")
            print("Falling back to point deposition.")
            h = None
            _print_elapsed("Smoothing-length generation", smoothing_start)

    print(
        f"Snapshot box size: {full_box_size:.6g} "
        f"(sim units; from {full_box_size_source})"
    )

    origin = np.zeros(3, dtype=np.float64)
    effective_box_size = full_box_size

    if center is not None:
        crop_start = time.perf_counter()
        assert extent is not None
        center_arr = np.asarray(center, dtype=np.float64)
        extent_f = float(extent)
        coords, field_data, h, origin = _crop_particles_to_region(
            coords,
            field_data,
            h,
            center=center_arr,
            extent=extent_f,
            box_size=full_box_size,
            periodic=periodic,
        )
        effective_box_size = extent_f
        print(
            "Subregion: "
            f"center=({center_arr[0]:.6g},{center_arr[1]:.6g},"
            f"{center_arr[2]:.6g}) "
            f"extent={extent_f:.6g} periodic={periodic}"
        )
        print(
            "Subregion origin (world-space min corner): "
            f"({origin[0]:.6g},{origin[1]:.6g},{origin[2]:.6g})"
        )
        n_selected = int(np.asarray(coords).shape[0])
        _raise_if_empty_subregion_selection(
            n_selected,
            particle_type=particle_type,
            field=field,
            center=center_arr,
            extent=extent_f,
            periodic=periodic,
        )
        print(
            f"Selected {n_selected} {particle_type} particles "
            f"for field '{field}'."
        )
        _print_elapsed("Subregion crop", crop_start)

    if tight_bounds:
        tighten_start = time.perf_counter()
        coords, h, origin_offset, effective_box_size = (
            _tighten_voxelization_bounds(
                coords,
                h,
                box_size=effective_box_size,
            )
        )
        origin = origin + origin_offset
        if center is not None and periodic:
            origin = np.mod(origin, full_box_size)
        print(
            "Tightened voxelization bounds: "
            f"origin=({origin[0]:.6g},{origin[1]:.6g},{origin[2]:.6g}) "
            f"box_size={effective_box_size:.6g}"
        )
        _print_elapsed("Tight bounds", tighten_start)

    _print_elapsed("Particle preparation", total_start)
    return field_data, coords, h, effective_box_size, origin


def _run_stl(args: argparse.Namespace) -> None:
    run_start = time.perf_counter()
    if args.nchunks < 1:
        print("Error: --nchunks must be >= 1")
        sys.exit(1)
    if args.nchunks > 1:
        try:
            field_data, coords, h, effective_box_size, origin = (
                _load_swift_particles(
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
            print(str(exc))
            sys.exit(1)

        final_threshold = args.threshold
        virtual_grid = VirtualGrid(
            origin=origin,
            box_size=effective_box_size,
            resolution=args.resolution,
            nchunks=args.nchunks,
        )

        print(
            "Generating hard chunk watertight mesh(es) "
            f"(threshold={final_threshold:.4e}, "
            f"nchunks={args.nchunks})..."
        )

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
            _print_elapsed("Hard chunk mesh generation", mesh_start)
        except ValueError as exc:
            print(f"Error generating chunked mesh: {exc}")
            sys.exit(1)

        if not chunk_meshes:
            print("Error: No chunk meshes generated (result was empty).")
            sys.exit(1)

        output_path = args.output or args.filename.with_suffix(".stl")
        if args.chunk_output == "separate":
            output_dir = output_path.with_suffix("")
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = output_path.stem
            chunk_counter = 0
            for _bounds, meshes in chunk_meshes:
                for mesh in meshes:
                    chunk_counter += 1
                    chunk_path = output_dir / f"{stem}_{chunk_counter}.stl"
                    print(f"Saving chunk to {chunk_path}...")
                    mesh.save(str(chunk_path))
            _print_elapsed("Total STL pipeline", run_start)
            print("Done.")
            return

        print("Regenerating chunk meshes with overlap for watertight union...")
        union_start = time.perf_counter()
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
        )
        _print_elapsed("Overlapped hard chunk generation", union_start)
        final_mesh = union_hard_chunk_meshes(chunk_meshes)

        if args.target_size:
            print(f"Scaling mesh to target size: {args.target_size} cm...")
            scale_mesh_to_print(final_mesh, args.target_size)

        if args.subdivide_iters < 0:
            print("Error: --subdivide-iters must be >= 0")
            sys.exit(1)
        if args.subdivide_iters > 0:
            print(
                "Applying Loop subdivision to final mesh "
                f"({args.subdivide_iters} iterations)..."
            )
            final_mesh.subdivide(iterations=args.subdivide_iters)

        if args.smooth_iters < 0:
            print("Error: --smooth-iters must be >= 0")
            sys.exit(1)
        if args.smooth_iters > 0:
            print(
                "Applying Taubin smoothing to final mesh "
                f"({args.smooth_iters} iterations)..."
            )
            final_mesh.repair(smoothing_iters=args.smooth_iters)

        print(f"Saving to {output_path}...")
        save_start = time.perf_counter()
        final_mesh.save(str(output_path))
        _print_elapsed("Mesh save", save_start)
        _print_elapsed("Total STL pipeline", run_start)
        print("Done.")
        return

    try:
        grid, voxel_size, origin = _load_swift_volume(
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
        print(str(exc))
        sys.exit(1)

    try:
        preprocess_start = time.perf_counter()
        grid = _apply_preprocess(
            grid,
            args.preprocess,
            args.clip_halos,
            args.gaussian_sigma,
        )
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)
    _print_elapsed("Grid preprocessing", preprocess_start)

    final_threshold = args.threshold

    print(
        "Generating watertight mesh using SDF "
        f"(threshold={final_threshold:.4e})..."
    )

    try:
        mesh_start = time.perf_counter()
        meshes = voxels_to_stl_via_sdf(
            grid,
            threshold=final_threshold,
            remove_islands=args.remove_islands,
            voxel_size=voxel_size,
        )
        _print_elapsed("Dense mesh generation", mesh_start)
    except ValueError as exc:
        print(f"Error generating mesh: {exc}")
        sys.exit(1)

    if not meshes:
        print("Error: No mesh generated (result was empty).")
        sys.exit(1)

    final_mesh: Mesh = meshes[0]
    if len(meshes) > 1:
        print(f"Merging {len(meshes)} mesh components...")
        combined = trimesh.util.concatenate([m.to_trimesh() for m in meshes])
        final_mesh = Mesh(mesh=combined)

    final_mesh.translate(origin)

    if args.target_size:
        print(f"Scaling mesh to target size: {args.target_size} cm...")
        scale_mesh_to_print(final_mesh, args.target_size)

    if args.subdivide_iters < 0:
        print("Error: --subdivide-iters must be >= 0")
        sys.exit(1)
    if args.subdivide_iters > 0:
        print(
            "Applying Loop subdivision to final mesh "
            f"({args.subdivide_iters} iterations)..."
        )
        final_mesh.subdivide(iterations=args.subdivide_iters)

    if args.smooth_iters < 0:
        print("Error: --smooth-iters must be >= 0")
        sys.exit(1)
    if args.smooth_iters > 0:
        print(
            "Applying Taubin smoothing to final mesh "
            f"({args.smooth_iters} iterations)..."
        )
        final_mesh.repair(smoothing_iters=args.smooth_iters)

    output_path = args.output or args.filename.with_suffix(".stl")
    print(f"Saving to {output_path}...")
    save_start = time.perf_counter()
    final_mesh.save(str(output_path))
    _print_elapsed("Mesh save", save_start)
    _print_elapsed("Total STL pipeline", run_start)
    print("Done.")


def _synthetic_volume(name: str, resolution: int) -> np.ndarray:
    if name != "sphere":
        raise ValueError(f"Unknown synthetic volume '{name}'")

    c = (resolution - 1) / 2.0
    x, y, z = np.ogrid[:resolution, :resolution, :resolution]
    r = 0.35 * resolution
    vol = ((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) <= r**2
    return vol.astype(np.float32)


def _run_export_vdb(args: argparse.Namespace) -> None:
    if args.synthetic is None and args.filename is None:
        print("Error: provide a snapshot filename or --synthetic")
        sys.exit(1)

    if args.synthetic is not None:
        if args.center is not None or args.extent is not None:
            print(
                "Error: --center/--extent are only supported for snapshot "
                "inputs"
            )
            sys.exit(1)
        grid = _synthetic_volume(args.synthetic, args.resolution)
        voxel_size = args.box_size / args.resolution if args.box_size else 1.0
        origin = np.zeros(3, dtype=np.float64)
        if args.box_size is not None:
            print(f"Box size: {args.box_size:.6g} (sim units)")
        print(f"Voxel size: {voxel_size:.4e} (sim units)")
    else:
        try:
            grid, voxel_size, origin = _load_swift_volume(
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
            print(str(exc))
            sys.exit(1)

    try:
        grid = _apply_preprocess(
            grid,
            args.preprocess,
            args.clip_halos,
            args.gaussian_sigma,
        )
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    stats = compute_grid_stats(grid)
    print(
        "Preprocessed grid stats: "
        f"min={stats['min']:.6e} max={stats['max']:.6e} "
        f"p50={stats['p50']:.6e} p90={stats['p90']:.6e} "
        f"p99={stats['p99']:.6e} p99.9={stats['p999']:.6e}"
    )
    if stats["p99"] > 0.0 and np.isfinite(stats["p99"]):
        print(
            "Recommended Blender density multiply (so p99~1): "
            f"{(1.0 / stats['p99']):.6g}"
        )

    if args.save_projection is not None:
        try:
            out = save_z_projection_png(
                args.save_projection, grid, method="sum"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error saving projection: {exc}")
            sys.exit(1)
        print(f"Wrote projection PNG: {out}")

    if args.save_histogram is not None:
        try:
            out = save_histogram_png(args.save_histogram, grid)
        except Exception as exc:  # noqa: BLE001
            print(f"Error saving histogram: {exc}")
            sys.exit(1)
        print(f"Wrote histogram PNG: {out}")

    out_base: Path
    if args.filename is not None:
        out_base = args.filename.with_suffix("")
    else:
        out_base = Path("synthetic")

    out_h5 = args.out_h5 or out_base.with_suffix(".h5")
    out_vdb = args.out_vdb or out_base.with_suffix(".vdb")

    write_volume_h5(
        out_h5,
        grid,
        voxel_size=voxel_size,
        origin=origin,
        grid_name=args.grid_name,
    )
    print(f"Wrote HDF5 intermediate: {out_h5}")

    dataset_path = f"/grids/{args.grid_name}"
    cmd = [
        args.vdb_writer,
        "--in",
        str(out_h5),
        "--dataset",
        dataset_path,
        "--out",
        str(out_vdb),
        "--grid-name",
        args.grid_name,
    ]
    print("Running VDB writer:")
    print("  " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        print(
            f"Error: VDB writer not found: '{args.vdb_writer}'. "
            "Build tools/vdb_writer and/or put meshmerizer-vdb-writer on PATH."
        )
        raise SystemExit(1) from exc
    except subprocess.CalledProcessError as exc:
        print(f"Error: VDB writer failed with exit code {exc.returncode}")
        raise SystemExit(1) from exc

    print(f"Wrote VDB: {out_vdb}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SWIFT simulation snapshots to 3D-printable STL meshes "
            "and volume formats."
        )
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    stl = subparsers.add_parser(
        "stl",
        help="Convert a SWIFT snapshot to an STL mesh",
    )
    stl.add_argument(
        "filename",
        type=Path,
        help="Path to the SWIFT snapshot file (HDF5).",
    )
    stl.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output STL filename. Defaults to <input_name>.stl",
    )
    _add_common_voxel_args(stl)
    stl.add_argument(
        "--nchunks",
        type=int,
        default=1,
        help=("Number of chunks per axis for chunked meshing. Default: 1"),
    )
    stl.add_argument(
        "--chunk-output",
        type=str,
        choices=["separate", "unioned"],
        default="unioned",
        help=(
            "When chunking is enabled, either write one STL per chunk or a "
            "single watertight unioned STL. Default: unioned"
        ),
    )
    stl.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Iso-surface threshold for mesh generation. Default: 0.5",
    )
    stl.add_argument(
        "--remove-islands",
        action="store_true",
        help=(
            "Keep only the largest connected component and discard any "
            "disconnected islands."
        ),
    )
    stl.add_argument(
        "--subdivide-iters",
        type=int,
        default=0,
        help=(
            "Number of Loop subdivision iterations to apply to the final "
            "mesh before smoothing. Default: 0"
        ),
    )
    stl.add_argument(
        "--smooth-iters",
        type=int,
        default=0,
        help=(
            "Number of Taubin smoothing iterations to apply to the final "
            "mesh surface. Default: 0"
        ),
    )
    stl.add_argument(
        "--target-size",
        "-s",
        type=float,
        default=None,
        help=(
            "Target size for the longest dimension of the final print (cm). "
            "If provided, the mesh is scaled to this size."
        ),
    )
    stl.set_defaults(func=_run_stl)

    export_vdb = subparsers.add_parser(
        "export-vdb",
        help="Export a dense volume to HDF5 then write OpenVDB",
    )
    export_vdb.add_argument(
        "filename",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the SWIFT snapshot file (HDF5).",
    )
    export_vdb.add_argument(
        "--synthetic",
        type=str,
        default=None,
        choices=["sphere"],
        help="Generate a synthetic test volume instead of reading a snapshot.",
    )
    _add_common_voxel_args(export_vdb)
    export_vdb.add_argument(
        "--save-projection",
        type=Path,
        default=None,
        help=(
            "Save a z-axis projection PNG (sum over z) of the preprocessed "
            "grid. Useful to verify non-uniformity in Blender."
        ),
    )
    export_vdb.add_argument(
        "--save-histogram",
        type=Path,
        default=None,
        help=(
            "Save a histogram PNG of preprocessed grid values (log10 x-axis "
            "when values>0)."
        ),
    )
    export_vdb.add_argument(
        "--grid-name",
        type=str,
        default="density",
        help="Grid/dataset name to write (stored at /grids/<grid-name>).",
    )
    export_vdb.add_argument(
        "--out-h5",
        type=Path,
        default=None,
        help="Output HDF5 intermediate path. Defaults to <base>.h5",
    )
    export_vdb.add_argument(
        "--out-vdb",
        type=Path,
        default=None,
        help="Output VDB path. Defaults to <base>.vdb",
    )
    export_vdb.add_argument(
        "--vdb-writer",
        type=str,
        default="meshmerizer-vdb-writer",
        help=(
            "Path to the VDB writer helper. Default: meshmerizer-vdb-writer "
            "(found on PATH)."
        ),
    )
    export_vdb.set_defaults(func=_run_export_vdb)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the `meshmerizer` CLI."""
    if argv is None:
        argv = sys.argv[1:]

    # Backwards-compatible mode: `meshmerizer snapshot.hdf5 ...`.
    if argv and argv[0] not in {"stl", "export-vdb"}:
        argv = ["stl", *argv]

    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
