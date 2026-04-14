"""Tests for CLI logging helpers."""

from __future__ import annotations

from pathlib import Path

from meshmerizer.logging import cli_logging_context, record_timing


def test_cli_logging_context_writes_detailed_log(
    tmp_path, monkeypatch
) -> None:
    """Ensure the CLI logging context creates the detailed log file.

    Args:
        tmp_path: Pytest-managed temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        ``None``. Assertions verify the log file contents.
    """
    # Redirect the working directory so the test can inspect the generated log
    # file without touching the repository root.
    monkeypatch.chdir(tmp_path)

    with cli_logging_context():
        record_timing("Synthetic stage", 1.25, operation="Timing")

    log_path = Path(tmp_path) / "meshmerizer.log"
    assert log_path.exists()
    assert "Synthetic stage took 1.250 s" in log_path.read_text()
