"""Helper CLI for plotting threshold-variable histograms."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the histogram helper.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot a histogram for an HDF5 per-particle scalar so you can "
            "choose useful --low-thresh/--up-thresh values."
        )
    )
    parser.add_argument("filename", type=Path, help="Input HDF5 snapshot.")
    parser.add_argument(
        "--threshold-key",
        required=True,
        help="Full HDF5 dataset path to analyse.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Output image path. Defaults to "
            "<snapshot_stem>_<dataset_name>_hist.png"
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=200,
        help="Histogram bin count. Default: 200",
    )
    parser.add_argument(
        "--low-thresh",
        type=float,
        default=None,
        help="Optional lower threshold to draw as a guide line.",
    )
    parser.add_argument(
        "--up-thresh",
        type=float,
        default=None,
        help="Optional upper threshold to draw as a guide line.",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a log-scaled x-axis when all selected values are positive.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a log-scaled y-axis.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively as well as saving it.",
    )
    return parser


def sanitize_dataset_name(threshold_key: str) -> str:
    """Convert an HDF5 dataset path into a filename-safe suffix.

    Args:
        threshold_key: Full HDF5 dataset path.

    Returns:
        Sanitized dataset identifier.
    """
    stripped = threshold_key.strip("/")
    if not stripped:
        return "dataset"
    return stripped.replace("/", "_")


def default_output_path(filename: Path, threshold_key: str) -> Path:
    """Build the default output path for the histogram image.

    Args:
        filename: Input snapshot path.
        threshold_key: Analysed HDF5 dataset path.

    Returns:
        Default PNG output path beside the input snapshot.
    """
    suffix = sanitize_dataset_name(threshold_key)
    return filename.with_name(f"{filename.stem}_{suffix}_hist.png")


def read_threshold_values(filename: Path, threshold_key: str) -> np.ndarray:
    """Load a threshold dataset from an HDF5 file as a flat float array.

    Args:
        filename: Input snapshot path.
        threshold_key: Full HDF5 dataset path.

    Returns:
        Flattened numeric array of dataset values.

    Raises:
        RuntimeError: If the dataset cannot be read or is non-numeric.
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required for threshold histogram plotting. Install it "
            "with: pip install h5py"
        ) from exc

    try:
        with h5py.File(filename, "r") as handle:
            values = np.asarray(handle[threshold_key])
    except Exception as exc:
        raise RuntimeError(
            f"Error reading threshold dataset '{threshold_key}': {exc}"
        ) from exc

    if not np.issubdtype(values.dtype, np.number):
        raise RuntimeError(
            f"Threshold dataset '{threshold_key}' must be numeric, got "
            f"{values.dtype}"
        )

    flat_values = np.ravel(values).astype(np.float64, copy=False)
    finite_values = flat_values[np.isfinite(flat_values)]
    if finite_values.size == 0:
        raise RuntimeError(
            f"Threshold dataset '{threshold_key}' contains no finite values"
        )
    return finite_values


def percentile_summary(
    values: np.ndarray,
    percentiles: Sequence[float] = (1.0, 5.0, 50.0, 95.0, 99.0),
) -> list[tuple[float, float]]:
    """Compute a compact percentile summary for a value distribution.

    Args:
        values: Flat numeric array.
        percentiles: Percentiles to compute.

    Returns:
        List of ``(percentile, value)`` pairs.
    """
    computed = np.percentile(values, list(percentiles))
    return [
        (float(percentile), float(value))
        for percentile, value in zip(percentiles, computed)
    ]


def format_summary_lines(
    values: np.ndarray,
    *,
    threshold_key: str,
    low_thresh: Optional[float],
    up_thresh: Optional[float],
) -> list[str]:
    """Create human-readable summary lines for the histogram input.

    Args:
        values: Flat numeric array.
        threshold_key: Analysed HDF5 dataset path.
        low_thresh: Optional lower threshold.
        up_thresh: Optional upper threshold.

    Returns:
        Summary lines suitable for stdout.
    """
    lines = [
        f"Dataset: {threshold_key}",
        f"Count: {values.size:,}",
        f"Min/Max: {values.min():.6g} / {values.max():.6g}",
        f"Mean/Median: {values.mean():.6g} / {np.median(values):.6g}",
    ]
    for percentile, value in percentile_summary(values):
        lines.append(f"P{percentile:.0f}: {value:.6g}")

    if low_thresh is not None or up_thresh is not None:
        keep_mask = np.ones(values.shape[0], dtype=bool)
        if low_thresh is not None:
            keep_mask &= values >= low_thresh
        if up_thresh is not None:
            keep_mask &= values <= up_thresh
        kept = int(np.count_nonzero(keep_mask))
        lines.append(f"Kept by thresholds: {kept:,}/{values.size:,}")

    return lines


def _build_histogram_bins(
    values: np.ndarray, bins: int, log_x: bool
) -> Iterable[float]:
    """Build histogram bins in either linear or log space.

    Args:
        values: Flat numeric array.
        bins: Number of bins.
        log_x: Whether log-spaced bins are requested.

    Returns:
        Histogram bin edges.

    Raises:
        ValueError: If the requested binning mode is invalid.
    """
    if bins <= 0:
        raise ValueError("--bins must be positive")

    if not log_x:
        return bins

    positive_values = values[values > 0.0]
    if positive_values.size != values.size:
        raise ValueError("--log-x requires all finite values to be > 0")
    vmin = float(positive_values.min())
    vmax = float(positive_values.max())
    if vmin == vmax:
        return np.array([vmin / 1.01, vmax * 1.01], dtype=np.float64)
    return np.logspace(np.log10(vmin), np.log10(vmax), bins + 1)


def plot_threshold_histogram(
    values: np.ndarray,
    *,
    threshold_key: str,
    output: Path,
    bins: int,
    low_thresh: Optional[float],
    up_thresh: Optional[float],
    log_x: bool,
    log_y: bool,
    show: bool,
) -> None:
    """Plot and save a threshold-variable histogram.

    Args:
        values: Flat numeric array.
        threshold_key: Analysed HDF5 dataset path.
        output: Output image path.
        bins: Histogram bin count.
        low_thresh: Optional lower threshold guide.
        up_thresh: Optional upper threshold guide.
        log_x: Whether to log-scale the x-axis.
        log_y: Whether to log-scale the y-axis.
        show: Whether to also display the figure interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for threshold histogram plotting. "
            "Install it with: pip install matplotlib"
        ) from exc

    fig, axis = plt.subplots(figsize=(10, 6))
    hist_bins = _build_histogram_bins(values, bins, log_x)
    axis.hist(values, bins=hist_bins, color="C0", alpha=0.8)
    axis.set_title(f"Histogram: {threshold_key}")
    axis.set_xlabel("Value")
    axis.set_ylabel("Particle count")
    axis.grid(True, alpha=0.25)

    if log_x:
        axis.set_xscale("log")
    if log_y:
        axis.set_yscale("log")

    for percentile, value in percentile_summary(values, (5.0, 50.0, 95.0)):
        axis.axvline(
            value,
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=f"P{percentile:.0f} = {value:.3g}",
        )

    if low_thresh is not None:
        axis.axvline(
            low_thresh,
            color="C3",
            linestyle="-",
            linewidth=1.5,
            label=f"low = {low_thresh:.3g}",
        )
    if up_thresh is not None:
        axis.axvline(
            up_thresh,
            color="C2",
            linestyle="-",
            linewidth=1.5,
            label=f"high = {up_thresh:.3g}",
        )

    handles, labels = axis.get_legend_handles_labels()
    if labels:
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys())

    summary_text = "\n".join(
        format_summary_lines(
            values,
            threshold_key=threshold_key,
            low_thresh=low_thresh,
            up_thresh=up_thresh,
        )[1:]
    )
    axis.text(
        0.98,
        0.98,
        summary_text,
        transform=axis.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the histogram helper CLI.

    Args:
        argv: Optional argument vector.

    Returns:
        Shell-style status code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    values = read_threshold_values(args.filename, args.threshold_key)
    output = args.output or default_output_path(
        args.filename, args.threshold_key
    )

    for line in format_summary_lines(
        values,
        threshold_key=args.threshold_key,
        low_thresh=args.low_thresh,
        up_thresh=args.up_thresh,
    ):
        print(line)

    plot_threshold_histogram(
        values,
        threshold_key=args.threshold_key,
        output=output,
        bins=args.bins,
        low_thresh=args.low_thresh,
        up_thresh=args.up_thresh,
        log_x=args.log_x,
        log_y=args.log_y,
        show=args.show,
    )
    print(f"Saved histogram to {output}")
    return 0


__all__ = [
    "build_parser",
    "default_output_path",
    "format_summary_lines",
    "main",
    "percentile_summary",
    "plot_threshold_histogram",
    "read_threshold_values",
    "sanitize_dataset_name",
]
