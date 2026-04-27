"""Tests for CLI logging helpers."""

from __future__ import annotations

from meshmerizer.logging import (
    cli_logging_context,
    log_debug_status,
    record_timing,
)


def test_cli_logging_context_prints_debug_logs_to_stdout(capsys) -> None:
    """Ensure Python debug logs are printed to stdout."""
    with cli_logging_context():
        log_debug_status("Logging", "debug message")

    captured = capsys.readouterr()
    assert "debug message" in captured.out


def test_silent_mode_only_hides_progress_bars(capsys) -> None:
    """Ensure silent mode still prints Python logs to stdout."""
    with cli_logging_context(silent=True):
        record_timing("Synthetic stage", 1.25, operation="Timing")

    captured = capsys.readouterr()
    assert "Synthetic stage took 1.250 s" in captured.out
