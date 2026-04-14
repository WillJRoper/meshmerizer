"""Argument parsing for the Meshmerizer CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from .stl import run_stl


def add_common_voxel_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments shared by voxelization-based commands.

    Args:
        parser: Parser or subparser to extend in place.

    Returns:
        ``None``. The parser is modified in place.
    """
    # Group the shared voxelization options here so dense and chunked STL paths
    # stay in sync when flags or help text change.
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
            "Number of worker threads to use. In chunked runs this controls "
            "chunk-level parallelism; in unchunked runs it is passed to the "
            "C-accelerated smoothing deposition. Default: 1"
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
        "--simplify-factor",
        type=float,
        default=1.0,
        help=(
            "Fraction of mesh faces to keep after extraction. Use 1.0 to "
            "disable simplification. Default: 1.0"
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


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the package CLI.

    Returns:
        Configured top-level parser.
    """
    # Keep the top-level parser minimal.
    # Almost all behaviour is delegated to the ``stl`` subcommand, which is
    # currently the only exposed workflow.
    parser = argparse.ArgumentParser(
        description=(
            "Convert SWIFT simulation snapshots to 3D-printable STL meshes."
        )
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Define the STL subcommand explicitly even though the main entrypoint
    # still supports the historical shorthand form.
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
    add_common_voxel_args(stl)
    stl.add_argument(
        "--nchunks",
        type=int,
        default=1,
        help="Number of chunks per axis for chunked meshing. Default: 1",
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
        "--chunk-overlap-percent",
        type=float,
        default=10.0,
        help=(
            "Percent of each chunk width to overlap on interior faces during "
            "chunked meshing. Rounded to the nearest whole voxel and applied "
            "on both sides of interior chunk boundaries. Default: 10.0"
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
        nargs="?",
        const=0,
        default=None,
        type=int,
        help=(
            "Remove disconnected islands before meshing. Use the flag alone "
            "to keep only the largest island, or pass an integer to discard "
            "islands smaller than that many voxels."
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
    stl.set_defaults(func=run_stl)

    return parser
