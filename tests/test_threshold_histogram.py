"""Tests for the threshold histogram helper."""

from pathlib import Path

import numpy as np
import pytest

from meshmerizer.cli.threshold_histogram import (
    _build_histogram_bins,
    build_parser,
    default_output_path,
    format_summary_lines,
    read_threshold_values,
    sanitize_dataset_name,
)


def test_build_parser_accepts_threshold_histogram_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "snapshot.hdf5",
            "--threshold-key",
            "/PartType0/Densities",
            "--bins",
            "128",
            "--low-thresh",
            "1.0",
            "--up-thresh",
            "10.0",
            "--log-x",
            "--log-y",
        ]
    )

    assert args.filename == Path("snapshot.hdf5")
    assert args.threshold_key == "/PartType0/Densities"
    assert args.bins == 128
    assert args.low_thresh == pytest.approx(1.0)
    assert args.up_thresh == pytest.approx(10.0)
    assert args.log_x is True
    assert args.log_y is True


def test_sanitize_dataset_name_and_default_output_path() -> None:
    snapshot = Path("/tmp/snapshot.hdf5")

    assert sanitize_dataset_name("/PartType0/Densities") == (
        "PartType0_Densities"
    )
    assert default_output_path(snapshot, "/PartType0/Densities") == Path(
        "/tmp/snapshot_PartType0_Densities_hist.png"
    )


def test_read_threshold_values_flattens_and_filters_nonfinite(
    tmp_path,
) -> None:
    h5py = pytest.importorskip("h5py")
    snapshot = tmp_path / "snapshot.hdf5"
    with h5py.File(snapshot, "w") as handle:
        handle.create_dataset(
            "/PartType0/Densities",
            data=np.array([[1.0, np.nan], [2.0, np.inf], [3.0, 4.0]]),
        )

    values = read_threshold_values(snapshot, "/PartType0/Densities")

    assert np.array_equal(values, np.array([1.0, 2.0, 3.0, 4.0]))


def test_format_summary_lines_reports_threshold_keep_fraction() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    lines = format_summary_lines(
        values,
        threshold_key="/PartType0/Densities",
        low_thresh=2.0,
        up_thresh=3.0,
    )

    assert "Dataset: /PartType0/Densities" in lines
    assert "Count: 4" in lines
    assert "Kept by thresholds: 2/4" in lines


def test_build_histogram_bins_supports_log_and_rejects_nonpositive() -> None:
    log_bins = _build_histogram_bins(
        np.array([1.0, 10.0, 100.0], dtype=np.float64),
        bins=3,
        log_x=True,
    )

    assert len(log_bins) == 4
    with pytest.raises(ValueError, match="log-x"):
        _build_histogram_bins(
            np.array([0.0, 1.0], dtype=np.float64),
            bins=3,
            log_x=True,
        )
