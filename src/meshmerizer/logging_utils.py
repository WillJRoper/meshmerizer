"""Terminal logging helpers for consistent status output.

This module centralizes the short status prefixes used throughout the package's
command-line output. It keeps terminal messages visually consistent across the
dense and chunked workflows, and can include worker identifiers for threaded
chunk processing.
"""

from __future__ import annotations

import threading


def format_status_prefix(operation: str, thread: int | None = None) -> str:
    """Return a standardized terminal status prefix.

    Args:
        operation: Short operation label such as ``"Loading"`` or
            ``"Meshing"``.
        thread: Optional 1-based worker identifier.

    Returns:
        Formatted prefix like ``"[Loading]"`` or ``"[Meshing][2]"``.
    """
    if thread is None:
        return f"[{operation}]"
    return f"[{operation}][{thread}]"


def log_status(
    operation: str,
    message: str,
    *,
    thread: int | None = None,
) -> None:
    """Print a terminal status line with a standardized prefix.

    Args:
        operation: Short operation label.
        message: Human-readable status message.
        thread: Optional 1-based worker identifier.
    """
    print(f"{format_status_prefix(operation, thread=thread)} {message}")


def current_thread_number() -> int | None:
    """Return a best-effort 1-based worker index for thread-pool workers.

    Returns:
        Parsed worker number for ``ThreadPoolExecutor`` threads, or ``None``
        when the current thread does not expose a pool-style suffix.
    """
    name = threading.current_thread().name
    if "_" not in name:
        return None
    suffix = name.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        return None
    return int(suffix) + 1
