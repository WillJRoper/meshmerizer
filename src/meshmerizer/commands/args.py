"""Argument parsing for the Meshmerizer CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from .adaptive_stl import run_adaptive


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the package CLI.

    Returns:
        Configured top-level parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert SWIFT simulation snapshots to 3D-printable "
            "STL meshes using adaptive octree dual contouring."
        )
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -----------------------------------------------------------------
    # ``adaptive`` subcommand — adaptive octree pipeline.
    # -----------------------------------------------------------------
    adaptive = subparsers.add_parser(
        "adaptive",
        help=(
            "Convert a SWIFT snapshot to an STL mesh using the "
            "adaptive octree + dual contouring pipeline"
        ),
    )
    adaptive.add_argument(
        "filename",
        type=Path,
        help="Path to the SWIFT snapshot file (HDF5).",
    )
    adaptive.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output STL filename. Defaults to <input_name>.stl",
    )

    # Domain selection flags.
    adaptive.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Center of an extracted subregion (x y z) in simulation "
            "units. Requires --extent."
        ),
    )
    adaptive.add_argument(
        "--extent",
        type=float,
        default=None,
        help=(
            "Side length of an extracted cubic subregion in "
            "simulation units. Requires --center."
        ),
    )
    adaptive.add_argument(
        "--tight-bounds",
        action="store_true",
        help=(
            "Shrink the working domain to the occupied particle "
            "bounds after any shift/crop."
        ),
    )
    adaptive.add_argument(
        "--shift",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help=(
            "Coordinate shift (dx dy dz) in simulation units "
            "before cropping. Default: 0 0 0"
        ),
    )
    wrap_group = adaptive.add_mutually_exclusive_group()
    wrap_group.add_argument(
        "--wrap-shift",
        dest="wrap_shift",
        action="store_true",
        help=(
            "Wrap coordinates into [0, box_size) after shifting. "
            "Default for SWIFT snapshots."
        ),
    )
    wrap_group.add_argument(
        "--no-wrap-shift",
        dest="wrap_shift",
        action="store_false",
        help="Do not wrap coordinates after shifting.",
    )
    adaptive.set_defaults(wrap_shift=True)
    adaptive.add_argument(
        "--no-periodic",
        dest="periodic",
        action="store_false",
        help="Disable periodic wrapping for subregion selection.",
    )
    adaptive.set_defaults(periodic=True)

    # Particle / field selection.
    adaptive.add_argument(
        "--particle-type",
        "-p",
        type=str,
        default="gas",
        choices=["gas", "dark_matter", "stars", "black_holes"],
        help="Particle type to extract. Default: 'gas'",
    )
    adaptive.add_argument(
        "--field",
        "-f",
        type=str,
        default="masses",
        help="Particle field to project. Default: 'masses'",
    )
    adaptive.add_argument(
        "--box-size",
        "-b",
        type=float,
        default=None,
        help="Physical size of the simulation box override.",
    )
    adaptive.add_argument(
        "--smoothing-factor",
        type=float,
        default=1.0,
        help=("Multiplier for particle smoothing lengths. Default: 1.0"),
    )

    # Adaptive pipeline flags.
    adaptive.add_argument(
        "--base-resolution",
        type=int,
        default=4,
        help=("Number of top-level octree cells per axis. Default: 4"),
    )
    adaptive.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum octree refinement depth. Default: 4",
    )
    adaptive.add_argument(
        "--isovalue",
        "-t",
        type=float,
        default=None,
        help=(
            "Isosurface threshold for mesh extraction. "
            "Overrides --surface-percentile when set."
        ),
    )
    adaptive.add_argument(
        "--surface-percentile",
        type=float,
        default=5.0,
        help=(
            "Automatically compute the isovalue from the Nth "
            "percentile of the particle self-density distribution. "
            "Lower values enclose more mass (e.g. 5 captures ~95%% "
            "of particles). Ignored when --isovalue is set. "
            "Default: 5.0"
        ),
    )

    # Post-processing.
    adaptive.add_argument(
        "--poisson-depth",
        type=int,
        default=None,
        help=(
            "Octree depth for the Poisson surface reconstruction. "
            "Higher values produce finer detail. Defaults to 9."
        ),
    )
    adaptive.add_argument(
        "--density-quantile",
        type=float,
        default=0.02,
        help=(
            "Fraction of lowest-density vertices to trim after "
            "Poisson reconstruction. Removes spurious membranes. "
            "Default: 0.02"
        ),
    )
    adaptive.add_argument(
        "--linking-factor",
        type=float,
        default=1.5,
        help=(
            "FOF linking factor: multiplier on mean inter-point "
            "separation for clustering vertices into distinct "
            "objects. Default: 1.5"
        ),
    )
    adaptive.add_argument(
        "--remove-islands-fraction",
        type=float,
        default=None,
        help=(
            "Remove connected components whose volume fraction is "
            "below this threshold. Use 0.0 to keep only the "
            "largest component."
        ),
    )
    adaptive.add_argument(
        "--target-size",
        "-s",
        type=float,
        default=None,
        help=(
            "Target size for the longest mesh dimension (cm). "
            "Scales the mesh for 3D printing."
        ),
    )

    # Serialization.
    adaptive.add_argument(
        "--save-octree",
        type=Path,
        default=None,
        help=(
            "Save the octree state to an HDF5 file after "
            "construction. Allows resuming meshing later."
        ),
    )
    adaptive.add_argument(
        "--load-octree",
        type=Path,
        default=None,
        help=(
            "Load a previously saved octree from HDF5 instead of "
            "building from a snapshot. The positional filename is "
            "still required for the output path default."
        ),
    )
    adaptive.add_argument(
        "--visualise-verts",
        action="store_true",
        help=(
            "Open a 3D scatter plot of QEF vertices after meshing "
            "(requires matplotlib)."
        ),
    )
    adaptive.add_argument(
        "--nthreads",
        type=int,
        default=None,
        help=(
            "Number of OpenMP threads. Defaults to all available "
            "cores. Only effective when built with WITH_OPENMP."
        ),
    )

    adaptive.set_defaults(func=run_adaptive)

    return parser
