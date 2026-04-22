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

    # Particle selection.
    adaptive.add_argument(
        "--particle-type",
        "-p",
        type=str,
        default="gas",
        choices=["gas", "dark_matter", "stars", "black_holes"],
        help="Particle type to extract. Default: 'gas'",
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
    adaptive.add_argument(
        "--min-usable-hermite-samples",
        type=int,
        default=3,
        help=(
            "Minimum number of usable Hermite samples required before "
            "a corner-crossing cell is allowed to stop refining. Cells "
            "with fewer usable samples keep refining until this support "
            "improves or --max-depth is reached. Default: 3"
        ),
    )
    adaptive.add_argument(
        "--max-qef-rms-residual-ratio",
        type=float,
        default=0.1,
        help=(
            "Maximum RMS QEF plane residual as a fraction of the local "
            "cell radius before a corner-crossing cell is forced to "
            "refine further. Lower values refine more aggressively. "
            "Default: 0.1"
        ),
    )
    adaptive.add_argument(
        "--min-normal-alignment-threshold",
        type=float,
        default=0.97,
        help=(
            "Minimum alignment between usable Hermite normals and their "
            "mean direction before a corner-crossing cell is considered "
            "well represented. Lower values tolerate more curvature; "
            "higher values refine more aggressively. Default: 0.97"
        ),
    )

    # Post-processing.
    adaptive.add_argument(
        "--smoothing-iterations",
        type=int,
        default=0,
        help=(
            "Number of Laplacian smoothing iterations applied "
            "to the extracted mesh vertices. 0 disables "
            "smoothing. Default: 0"
        ),
    )
    adaptive.add_argument(
        "--smoothing-strength",
        type=float,
        default=0.5,
        help=(
            "Laplacian smoothing strength lambda in (0, 1]. "
            "0.0 = no movement, 1.0 = snap to neighbor centroid. "
            "Only effective when --smoothing-iterations > 0. "
            "Default: 0.5"
        ),
    )
    adaptive.add_argument(
        "--min-feature-thickness",
        type=float,
        default=0.0,
        help=(
            "Minimum physical feature thickness to preserve via adaptive "
            "solid opening. Features thinner than this may be removed. "
            "When --target-size is provided, this value is interpreted in "
            "print centimetres and converted back to native meshing units. "
            "0 disables the regularizer. Default: 0.0"
        ),
    )
    adaptive.add_argument(
        "--pre-thickening-radius",
        type=float,
        default=0.0,
        help=(
            "Optional outward pre-thickening radius applied to the leaf-wise "
            "occupied solid before the minimum-thickness opening stage. This "
            "can puff up fragile features so they survive regularization. "
            "When --target-size is provided, this value is interpreted in "
            "print centimetres and converted back to native meshing units. "
            "0 disables pre-thickening. Default: 0.0"
        ),
    )
    adaptive.add_argument(
        "--max-edge-ratio",
        type=float,
        default=1.5,
        help=(
            "Maximum triangle edge length as a multiple of the "
            "local octree cell size. Edges longer than this are "
            "subdivided to fill gaps. Default: 1.5"
        ),
    )
    adaptive.add_argument(
        "--fof",
        action="store_true",
        help=(
            "Run Friends-of-Friends clustering before reconstruction "
            "and mesh each cluster independently. Off by default. "
            "Prefer --min-fof-cluster-size when the goal is simply "
            "to discard small fluff populations before meshing."
        ),
    )
    adaptive.add_argument(
        "--min-fof-cluster-size",
        type=int,
        default=None,
        help=(
            "Discard particle FOF clusters smaller than this many "
            "particles before octree construction and meshing. "
            "This removes small detached fluff populations without "
            "splitting the remaining scene into separate meshes. "
            "Disabled by default."
        ),
    )
    adaptive.add_argument(
        "--linking-factor",
        type=float,
        default=0.2,
        help=(
            "FOF linking factor: multiplier on mean inter-point "
            "separation for clustering particles into FOF groups. "
            "Used by --fof and --min-fof-cluster-size. Default: 0.2"
        ),
    )
    adaptive.add_argument(
        "--remove-islands-fraction",
        type=float,
        default=None,
        help=(
            "Remove connected components whose volume is below this "
            "fraction of the largest component volume. Use 0.0 to keep "
            "only the largest component."
        ),
    )
    adaptive.add_argument(
        "--simplify-factor",
        type=float,
        default=1.0,
        help=(
            "Fraction of faces to keep during final mesh simplification. "
            "Use values in (0, 1]; smaller values simplify more aggressively. "
            "1.0 disables simplification. Default: 1.0"
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
            "construction so later runs can reuse the adaptive tree and "
            "particle data."
        ),
    )
    adaptive.add_argument(
        "--load-octree",
        type=Path,
        default=None,
        help=(
            "Load a previously saved octree from HDF5 instead of "
            "building from a snapshot. The saved file's isovalue, domain, "
            "base resolution, max depth, particles, cells, and contributors "
            "are reused. Snapshot-loading flags are ignored in this mode. "
            "The positional filename is still required for the output path "
            "default."
        ),
    )
    adaptive.add_argument(
        "--visualise-verts",
        nargs="?",
        const="qef_vertices.png",
        default=None,
        metavar="PATH",
        help=(
            "Save a 6-panel figure of QEF vertex projections "
            "(one per face of the bounding box). When used with "
            "--load-octree, QEF vertices are solved only for this diagnostic "
            "visualization. Optionally provide an output path; defaults to "
            "'qef_vertices.png'."
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
