"""Argument parsing for the Meshmerizer CLI."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from meshmerizer.cli.adaptive import run_adaptive


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI value.

    Args:
        value: Raw CLI string value.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not strictly positive.
    """
    # Parse first, then validate the sign so argparse can surface a targeted
    # error message for this specific constraint.
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    """Parse a non-negative integer CLI value.

    Args:
        value: Raw CLI string value.

    Returns:
        Parsed non-negative integer.

    Raises:
        argparse.ArgumentTypeError: If the value is negative.
    """
    # Non-negative integer parsing is used for iteration counts and similar
    # knobs that legitimately allow zero.
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative integer"
        )
    return parsed


def _finite_float(value: str) -> float:
    """Parse a finite floating-point CLI value.

    Args:
        value: Raw CLI string value.

    Returns:
        Parsed finite float.

    Raises:
        argparse.ArgumentTypeError: If the parsed float is not finite.
    """
    # Parse once centrally so the other float validators can layer additional
    # range checks on top of a shared finite-number rule.
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def _positive_float(value: str) -> float:
    """Parse a strictly positive floating-point CLI value.

    Args:
        value: Raw CLI string value.

    Returns:
        Parsed positive float.

    Raises:
        argparse.ArgumentTypeError: If the value is not strictly positive.
    """
    # Reuse the finite-float parser so positivity checks never need to handle
    # NaN or infinity explicitly.
    parsed = _finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _nonnegative_float(value: str) -> float:
    """Parse a non-negative floating-point CLI value.

    Args:
        value: Raw CLI string value.

    Returns:
        Parsed non-negative float.

    Raises:
        argparse.ArgumentTypeError: If the value is negative.
    """
    # This variant is used for thresholds and radii where zero is a meaningful
    # "disabled" value.
    parsed = _finite_float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _fraction_or_zero(value: str) -> float:
    """Parse a value constrained to the closed unit interval.

    Args:
        value: Raw CLI string value.

    Returns:
        Parsed float in ``[0, 1]``.

    Raises:
        argparse.ArgumentTypeError: If the value lies outside ``[0, 1]``.
    """
    # Island-removal fractions use the closed interval so ``0`` can mean
    # "keep only the largest component".
    parsed = _nonnegative_float(value)
    if parsed > 1.0:
        raise argparse.ArgumentTypeError("value must lie in [0, 1]")
    return parsed


def _unit_interval_open_closed(value: str) -> float:
    """Parse a value constrained to ``(0, 1]``.

    Args:
        value: Raw CLI string value.

    Returns:
        Parsed float in ``(0, 1]``.

    Raises:
        argparse.ArgumentTypeError: If the value lies outside ``(0, 1]``.
    """
    # This validator is used for strengths and fractions where zero would make
    # the option semantically meaningless.
    parsed = _finite_float(value)
    if parsed <= 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("value must lie in (0, 1]")
    return parsed


def _percentile(value: str) -> float:
    """Parse a percentile value constrained to ``[0, 100]``.

    Args:
        value: Raw CLI string value.

    Returns:
        Parsed percentile.

    Raises:
        argparse.ArgumentTypeError: If the value lies outside ``[0, 100]``.
    """
    # Percentile parsing is kept separate from generic fraction parsing because
    # the CLI exposes percent values in the user-facing convention.
    parsed = _finite_float(value)
    if parsed < 0.0 or parsed > 100.0:
        raise argparse.ArgumentTypeError("value must lie in [0, 100]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the package CLI.

    Returns:
        Configured top-level parser.
    """
    # Build the parser in one place so the CLI entrypoint and any tests share
    # the exact same option definitions.
    parser = argparse.ArgumentParser(
        description=(
            "Convert SWIFT simulation snapshots to 3D-printable "
            "STL meshes using adaptive octree dual contouring."
        )
    )

    # Positional and output-path arguments define the overall data source and
    # default export location.
    parser.add_argument(
        "filename",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the SWIFT snapshot file (HDF5).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output STL filename. Defaults to <input_name>.stl",
    )

    parser.add_argument(
        "--center",
        type=_finite_float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Center of an extracted subregion (x y z) in simulation "
            "units. Requires --extent."
        ),
    )
    parser.add_argument(
        "--extent",
        type=_positive_float,
        default=None,
        help=(
            "Side length of an extracted cubic subregion in "
            "simulation units. Requires --center."
        ),
    )
    parser.add_argument(
        "--tight-bounds",
        action="store_true",
        help=(
            "Shrink the working domain to the occupied particle "
            "bounds after any shift/crop."
        ),
    )
    parser.add_argument(
        "--shift",
        type=_finite_float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help=(
            "Coordinate shift (dx dy dz) in simulation units "
            "before cropping. Default: 0 0 0"
        ),
    )
    wrap_group = parser.add_mutually_exclusive_group()
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
    parser.set_defaults(wrap_shift=True)
    parser.add_argument(
        "--no-periodic",
        dest="periodic",
        action="store_false",
        help="Disable periodic wrapping for subregion selection.",
    )
    parser.set_defaults(periodic=True)

    parser.add_argument(
        "--particle-type",
        "-p",
        type=str,
        default="gas",
        choices=["gas", "dark_matter", "stars", "black_holes"],
        help="Particle type to extract. Default: 'gas'",
    )
    parser.add_argument(
        "--box-size",
        "-b",
        type=_positive_float,
        default=None,
        help="Physical size of the simulation box override.",
    )
    parser.add_argument(
        "--smoothing-factor",
        type=_positive_float,
        default=1.0,
        help="Multiplier for particle smoothing lengths. Default: 1.0",
    )

    parser.add_argument(
        "--base-resolution",
        type=_positive_int,
        default=4,
        help="Number of top-level octree cells per axis. Default: 4",
    )
    parser.add_argument(
        "--max-depth",
        type=_positive_int,
        default=4,
        help="Maximum octree refinement depth. Default: 4",
    )
    parser.add_argument(
        "--isovalue",
        "-t",
        type=_finite_float,
        default=None,
        help=(
            "Isosurface threshold for mesh extraction. "
            "Overrides --surface-percentile when set."
        ),
    )
    parser.add_argument(
        "--surface-percentile",
        type=_percentile,
        default=5.0,
        help=(
            "Automatically compute the isovalue from the Nth "
            "percentile of the particle self-density distribution. "
            "Lower values enclose more mass (e.g. 5 captures ~95%% "
            "of particles). Ignored when --isovalue is set. "
            "Default: 5.0"
        ),
    )
    parser.add_argument(
        "--min-usable-hermite-samples",
        type=_positive_int,
        default=3,
        help=(
            "Minimum number of usable Hermite samples required before "
            "a corner-crossing cell is allowed to stop refining. Cells "
            "with fewer usable samples keep refining until this support "
            "improves or --max-depth is reached. Default: 3"
        ),
    )
    parser.add_argument(
        "--max-qef-rms-residual-ratio",
        type=_nonnegative_float,
        default=0.1,
        help=(
            "Maximum RMS QEF plane residual as a fraction of the local "
            "cell radius before a corner-crossing cell is forced to "
            "refine further. Lower values refine more aggressively. "
            "Default: 0.1"
        ),
    )
    parser.add_argument(
        "--min-normal-alignment-threshold",
        type=_unit_interval_open_closed,
        default=0.97,
        help=(
            "Minimum alignment between usable Hermite normals and their "
            "mean direction before a corner-crossing cell is considered "
            "well represented. Lower values tolerate more curvature; "
            "higher values refine more aggressively. Default: 0.97"
        ),
    )

    parser.add_argument(
        "--smoothing-iterations",
        type=_nonnegative_int,
        default=0,
        help=(
            "Number of Laplacian smoothing iterations applied "
            "to the extracted mesh vertices. 0 disables "
            "smoothing. Default: 0"
        ),
    )
    parser.add_argument(
        "--smoothing-strength",
        type=_unit_interval_open_closed,
        default=0.5,
        help=(
            "Laplacian smoothing strength lambda in (0, 1]. "
            "0.0 = no movement, 1.0 = snap to neighbor centroid. "
            "Only effective when --smoothing-iterations > 0. "
            "Default: 0.5"
        ),
    )
    parser.add_argument(
        "--min-feature-thickness",
        type=_nonnegative_float,
        default=0.0,
        help=(
            "Minimum physical feature thickness to preserve via adaptive "
            "solid opening. Features thinner than this may be removed. "
            "When --target-size is provided, this value is interpreted in "
            "print centimetres and converted back to native meshing units. "
            "0 disables the regularizer. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--pre-thickening-radius",
        type=_nonnegative_float,
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
    parser.add_argument(
        "--max-edge-ratio",
        type=_positive_float,
        default=1.5,
        help=(
            "Maximum triangle edge length as a multiple of the "
            "local octree cell size. Edges longer than this are "
            "subdivided to fill gaps. Default: 1.5"
        ),
    )
    parser.add_argument(
        "--fof",
        action="store_true",
        help=(
            "Run Friends-of-Friends clustering before reconstruction "
            "and mesh each cluster independently. Off by default. "
            "Prefer --min-fof-cluster-size when the goal is simply "
            "to discard small fluff populations before meshing."
        ),
    )
    parser.add_argument(
        "--min-fof-cluster-size",
        type=_positive_int,
        default=None,
        help=(
            "Discard particle FOF clusters smaller than this many "
            "particles before octree construction and meshing. "
            "This removes small detached fluff populations without "
            "splitting the remaining scene into separate meshes. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--linking-factor",
        type=_positive_float,
        default=0.2,
        help=(
            "FOF linking factor: multiplier on mean inter-point "
            "separation for clustering particles into FOF groups. "
            "Used by --fof and --min-fof-cluster-size. Default: 0.2"
        ),
    )
    parser.add_argument(
        "--remove-islands-fraction",
        type=_fraction_or_zero,
        default=None,
        help=(
            "Remove connected components whose volume is below this "
            "fraction of the largest component volume. Use 0.0 to keep "
            "only the largest component."
        ),
    )
    parser.add_argument(
        "--simplify-factor",
        type=_unit_interval_open_closed,
        default=1.0,
        help=(
            "Fraction of faces to keep during final mesh simplification. "
            "Use values in (0, 1]; smaller values simplify more aggressively. "
            "1.0 disables simplification. Default: 1.0"
        ),
    )
    parser.add_argument(
        "--target-size",
        "-s",
        type=_positive_float,
        default=None,
        help=(
            "Target size for the longest mesh dimension (cm). "
            "Scales the mesh for 3D printing."
        ),
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help=(
            "Suppress per-update progress rendering while keeping Python "
            "status logs on stdout."
        ),
    )

    parser.add_argument(
        "--save-octree",
        type=Path,
        default=None,
        help=(
            "Save the octree state to an HDF5 file after "
            "construction so later runs can reuse the adaptive tree and "
            "particle data."
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
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
    parser.add_argument(
        "--nthreads",
        type=_positive_int,
        default=None,
        help=(
            "Number of OpenMP threads. Defaults to all available "
            "cores. Only effective when built with WITH_OPENMP."
        ),
    )

    parser.set_defaults(func=run_adaptive)
    return parser
