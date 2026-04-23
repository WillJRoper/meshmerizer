"""CLI logging, timing, and progress helpers.

This module centralizes user-facing CLI output, detailed per-run log files,
progress bars, and timing aggregation used for end-of-run summaries. Library
callers can import the package without seeing CLI noise; runtime handlers are
only attached during an active CLI command.
"""

from __future__ import annotations

import inspect
import logging as std_logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Iterator, NoReturn, Optional

from tqdm.auto import tqdm

LOGGER_NAME = "meshmerizer"


def _ensure_null_handler() -> None:
    """Attach a null handler so library imports stay quiet by default.

    Returns:
        ``None``. The package logger is updated in place when needed.
    """
    logger = std_logging.getLogger(LOGGER_NAME)
    if not any(
        isinstance(handler, std_logging.NullHandler)
        for handler in logger.handlers
    ):
        logger.addHandler(std_logging.NullHandler())


_ensure_null_handler()


@dataclass
class TimingStat:
    """Aggregate timing data for one labeled stage.

    Attributes:
        operation: Broad operation category associated with the label.
        total: Total accumulated time in seconds.
        count: Number of recorded samples.
    """

    operation: str
    total: float = 0.0
    count: int = 0


@dataclass
class LoggingState:
    """Mutable process-wide CLI logging state.

    Attributes:
        active: Whether CLI logging is currently enabled.
        depth: Nesting depth for re-entrant CLI contexts.
        timings: Aggregated timing samples recorded during the run.
        log_path: Path to the current detailed log file, when active.
        logger: Root logger used by the package.
        console_handler: Active console handler, if configured.
        file_handler: Active file handler, if configured.
        warnings: Deferred warning messages collected during the run.
        silent: Whether console progress output should be suppressed.
        cpp_log_path: Path to the C++ status log file, when active.
        lock: Re-entrant lock protecting shared logging state.
    """

    active: bool = False
    depth: int = 0
    timings: "OrderedDict[str, TimingStat]" = field(
        default_factory=OrderedDict
    )
    log_path: Optional[Path] = None
    logger: std_logging.Logger = field(
        default_factory=lambda: std_logging.getLogger(LOGGER_NAME)
    )
    console_handler: Optional[std_logging.Handler] = None
    file_handler: Optional[std_logging.Handler] = None
    warnings: list[str] = field(default_factory=list)
    silent: bool = False
    cpp_log_path: Optional[Path] = None
    lock: threading.RLock = field(default_factory=threading.RLock)


_STATE = LoggingState()


CPP_STATUS_LOG_NAME = "meshmerizer_cpp.log"


class TqdmConsoleHandler(std_logging.Handler):
    """Console logging handler that plays nicely with tqdm progress bars."""

    def emit(self, record: std_logging.LogRecord) -> None:
        """Write one formatted record via ``tqdm.write``.

        Args:
            record: Logging record to emit.

        Returns:
            ``None``. The message is written to the terminal.
        """
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


class ConsoleVisibilityFilter(std_logging.Filter):
    """Only show errors or explicitly surfaced summaries."""

    def filter(self, record: std_logging.LogRecord) -> bool:
        """Return whether a record should be shown on stdout."""
        return record.levelno >= std_logging.ERROR or bool(
            getattr(record, "console_visible", False)
        )


def _caller_function_name(stack_offset: int = 2) -> str:
    """Return the name of the calling function.

    Args:
        stack_offset: Number of frames to skip.  The default of 2
            skips ``_caller_function_name`` itself and the direct
            caller (typically ``log_status``), returning the name of
            the function that invoked the logging helper.

    Returns:
        Name of the calling function, or ``"<unknown>"`` if the
        frame cannot be inspected.
    """
    frame = inspect.currentframe()
    try:
        for _ in range(stack_offset):
            if frame is not None:
                frame = frame.f_back
        if frame is not None:
            return frame.f_code.co_name
    finally:
        del frame
    return "<unknown>"


def current_thread_label() -> str:
    """Return a stable thread label for the standardized prefix.

    Returns:
        ``"main"`` for the main thread, a 1-based pool worker index when the
        thread name ends in ``_<n>``, or the raw thread name otherwise.
    """
    thread = threading.current_thread()
    if thread.name == "MainThread":
        return "main"
    if current_thread_number() is not None:
        return "worker"
    return thread.name


def format_status_prefix(
    operation: str,
    thread: str | int | None = None,
    func: str | None = None,
) -> str:
    """Return a standardized terminal status prefix.

    Args:
        operation: Short operation label such as ``"Loading"`` or
            ``"Meshing"``.
        thread: Optional 1-based worker identifier.
        func: Optional function name to include in the prefix.

    Returns:
        Formatted prefix like ``"[Loading][my_func][main]"`` or
        ``"[Meshing][my_func][worker]"``.
    """
    resolved_func = func if func is not None else "<unknown>"
    resolved_thread = current_thread_label() if thread is None else str(thread)
    return f"[{operation}][{resolved_func}][{resolved_thread}]"


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


def log_warning_status(
    operation: str,
    message: str,
    *,
    thread: str | int | None = None,
) -> None:
    """Log a warning status line with the standardized prefix."""
    func = _caller_function_name(stack_offset=3)
    formatted = (
        f"{format_status_prefix(operation, thread=thread, func=func)} "
        f"{message}"
    )
    with _STATE.lock:
        if _STATE.active:
            _STATE.warnings.append(formatted)
    log_status(
        operation,
        message,
        thread=thread,
        level=std_logging.WARNING,
        _stack_offset=3,
    )


def log_error_status(
    operation: str,
    message: str,
    *,
    thread: str | int | None = None,
) -> None:
    """Log an error status line with the standardized prefix."""
    log_status(
        operation,
        message,
        thread=thread,
        level=std_logging.ERROR,
        _stack_offset=3,
    )


def abort_with_error(
    operation: str,
    message: str,
    *,
    exit_code: int = 1,
) -> "NoReturn":
    """Log an error line and terminate the process.

    Args:
        operation: Short operation label.
        message: Human-readable error message.
        exit_code: Process exit code.
    """
    log_error_status(operation, message)
    raise SystemExit(exit_code)


def get_logger(name: str | None = None) -> std_logging.Logger:
    """Return a meshmerizer logger.

    Args:
        name: Optional child logger suffix.

    Returns:
        Logger rooted at ``meshmerizer``.
    """
    if name is None:
        return std_logging.getLogger(LOGGER_NAME)
    return std_logging.getLogger(f"{LOGGER_NAME}.{name}")


def _infer_operation(label: str) -> str:
    """Infer a broad operation category from a timing label.

    Args:
        label: Timing label describing one stage.

    Returns:
        Broad operation category used for status prefixes.
    """
    label_lower = label.lower()
    if any(token in label_lower for token in ["load", "extract"]):
        return "Loading"
    if any(token in label_lower for token in ["crop", "bound", "clean"]):
        return "Cleaning"
    if any(token in label_lower for token in ["voxel", "deposit"]):
        return "Voxelising"
    if any(
        token in label_lower
        for token in ["mesh", "sdf", "stitch", "chunk", "pipeline"]
    ):
        return "Meshing"
    if "save" in label_lower:
        return "Saving"
    return "Timing"


def log_status(
    operation: str,
    message: str,
    *,
    thread: int | None = None,
    level: int = std_logging.INFO,
    console: bool = False,
    _stack_offset: int = 2,
) -> None:
    """Log a terminal status line with a standardized prefix.

    Args:
        operation: Short operation label.
        message: Human-readable status message.
        thread: Optional 1-based worker identifier.
        level: Standard-library logging level.
        console: Whether to also surface the message on stdout.
        _stack_offset: Internal parameter controlling how many stack
            frames to skip when resolving the caller function name.
    """
    func = _caller_function_name(stack_offset=_stack_offset)
    get_logger().log(
        level,
        f"{format_status_prefix(operation, thread=thread, func=func)}"
        f" {message}",
        extra={"console_visible": console},
    )


def log_summary_status(
    operation: str,
    message: str,
    *,
    thread: int | None = None,
) -> None:
    """Log a final or summary message to both stdout and the file log."""
    log_status(
        operation,
        message,
        thread=thread,
        console=True,
        _stack_offset=3,
    )


def record_timing(
    label: str,
    elapsed: float,
    *,
    operation: str | None = None,
    _stack_offset: int = 3,
) -> float:
    """Store one timing sample and write it to the detailed log.

    Args:
        label: Human-readable stage label.
        elapsed: Stage duration in seconds.
        operation: Optional explicit operation category.
        _stack_offset: Internal parameter for caller resolution.

    Returns:
        The recorded elapsed time.
    """
    resolved_operation = operation or _infer_operation(label)
    with _STATE.lock:
        stat = _STATE.timings.get(label)
        if stat is None:
            stat = TimingStat(operation=resolved_operation)
            _STATE.timings[label] = stat
        stat.total += elapsed
        stat.count += 1
    log_status(
        resolved_operation,
        f"{label} took {elapsed:.3f} s",
        level=std_logging.DEBUG,
        console=False,
        _stack_offset=_stack_offset,
    )
    return elapsed


def record_elapsed(
    label: str,
    start: float,
    *,
    operation: str | None = None,
) -> float:
    """Measure elapsed time from a ``perf_counter`` start value.

    Args:
        label: Human-readable stage label.
        start: Start timestamp captured with ``time.perf_counter()``.
        operation: Optional explicit operation category.

    Returns:
        Recorded elapsed time in seconds.
    """
    return record_timing(
        label,
        time.perf_counter() - start,
        operation=operation,
        _stack_offset=4,
    )


def emit_timing_summary() -> None:
    """Print a concise summary of the recorded stage timings.

    Returns:
        ``None``. A timing summary is emitted when timing samples exist.
    """
    with _STATE.lock:
        if not _STATE.timings:
            return
        entries = list(_STATE.timings.items())

    ranked = sorted(entries, key=lambda item: item[1].total, reverse=True)
    total_time = sum(stat.total for _, stat in ranked)
    lines = [f"total instrumented time: {total_time:.3f} s"]
    for label, stat in ranked:
        count_suffix = ""
        if stat.count > 1:
            count_suffix = f" ({stat.count}x)"
        lines.append(f"  {label}: {stat.total:.3f} s{count_suffix}")
    log_summary_status("Timing", "Summary:\n" + "\n".join(lines))


def emit_warning_summary() -> None:
    """Print a concise end-of-run summary of deferred warnings."""
    with _STATE.lock:
        if not _STATE.warnings:
            return
        warnings = list(_STATE.warnings)

    lines = [f"{len(warnings)} warning(s) recorded:"]
    lines.extend(f"  - {warning}" for warning in warnings)
    log_summary_status("Warning", "Summary:\n" + "\n".join(lines))


def _remove_handler(handler: Optional[std_logging.Handler]) -> None:
    """Detach and close one logger handler if present.

    Args:
        handler: Handler to remove from the package logger.

    Returns:
        ``None``. The handler is detached and closed when present.
    """
    if handler is None:
        return
    _STATE.logger.removeHandler(handler)
    handler.close()


def _set_cpp_status_log_path(path: Optional[Path]) -> None:
    """Tell the adaptive C++ extension where to write status lines."""
    try:
        adaptive = import_module("meshmerizer._adaptive")
    except Exception:
        return

    setter = getattr(adaptive, "set_status_log_path", None)
    if setter is None:
        return

    setter(None if path is None else str(path))


def _set_cpp_silent_mode(silent: bool) -> None:
    """Tell the adaptive C++ extension whether to suppress progress output."""
    try:
        adaptive = import_module("meshmerizer._adaptive")
    except Exception:
        return

    setter = getattr(adaptive, "set_silent_mode", None)
    if setter is None:
        return

    setter(bool(silent))


def _configure_cli_logging() -> None:
    """Attach tqdm-safe console and file logging handlers.

    Returns:
        ``None``. The process-wide package logger is configured in place.
    """
    logger = _STATE.logger
    logger.setLevel(std_logging.DEBUG)
    logger.propagate = False

    console_handler = TqdmConsoleHandler()
    console_handler.setLevel(std_logging.INFO)
    console_handler.setFormatter(std_logging.Formatter("%(message)s"))
    console_handler.addFilter(ConsoleVisibilityFilter())

    log_path = Path.cwd() / "meshmerizer.log"
    cpp_log_path = Path.cwd() / CPP_STATUS_LOG_NAME
    file_handler = std_logging.FileHandler(
        log_path, mode="w", encoding="utf-8"
    )
    file_handler.setLevel(std_logging.DEBUG)
    file_handler.setFormatter(
        std_logging.Formatter(
            "%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s"
        )
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    _STATE.console_handler = console_handler
    _STATE.file_handler = file_handler
    _STATE.log_path = log_path
    _STATE.cpp_log_path = cpp_log_path
    _STATE.timings.clear()
    _STATE.warnings.clear()
    _set_cpp_status_log_path(cpp_log_path)
    _set_cpp_silent_mode(_STATE.silent)

    log_debug_status("Logging", f"Detailed log: {log_path}")


def _teardown_cli_logging() -> None:
    """Flush the timing summary and close CLI logging handlers.

    Returns:
        ``None``. Logging handlers are removed and shared state is reset.
    """
    try:
        emit_warning_summary()
        emit_timing_summary()
        if _STATE.log_path is not None:
            log_summary_status(
                "Logging",
                "Detailed logs saved to "
                f"{_STATE.log_path} and {_STATE.cpp_log_path}",
            )
    finally:
        _set_cpp_status_log_path(None)
        _set_cpp_silent_mode(False)
        _remove_handler(_STATE.console_handler)
        _remove_handler(_STATE.file_handler)
        _STATE.console_handler = None
        _STATE.file_handler = None
        _STATE.log_path = None
        _STATE.cpp_log_path = None
        _STATE.timings.clear()
        _STATE.warnings.clear()
        _STATE.silent = False


@contextmanager
def cli_logging_context(*, silent: bool = False) -> Iterator[None]:
    """Run code with CLI logging, progress, and timing enabled.

    Yields:
        ``None`` while the CLI logging configuration is active.
    """
    should_configure = False
    with _STATE.lock:
        if _STATE.active:
            _STATE.depth += 1
        else:
            _STATE.active = True
            _STATE.depth = 1
            _STATE.silent = silent
            should_configure = True

    if should_configure:
        _configure_cli_logging()

    try:
        yield
    finally:
        should_teardown = False
        with _STATE.lock:
            _STATE.depth -= 1
            if _STATE.depth == 0:
                _STATE.active = False
                should_teardown = True
        if should_teardown:
            _teardown_cli_logging()


@contextmanager
def progress_bar(
    total: int,
    *,
    desc: str,
    unit: str,
    enabled: bool = True,
) -> Iterator[tqdm]:
    """Create a tqdm progress bar that stays quiet outside the CLI runtime.

    Args:
        total: Number of expected steps.
        desc: Short bar description.
        unit: Item unit label shown by tqdm.
        enabled: Whether the caller wants a visible progress bar.

    Yields:
        Progress-bar object.
    """
    show_bar = enabled and _STATE.active and (not _STATE.silent) and total > 1
    bar = tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=False,
        dynamic_ncols=True,
        disable=not show_bar,
    )
    try:
        yield bar
    finally:
        bar.close()


def log_debug_status(
    operation: str,
    message: str,
    *,
    thread: int | None = None,
) -> None:
    """Log a detailed status line to the file log without console noise.

    Args:
        operation: Short operation label.
        message: Human-readable diagnostic message.
        thread: Optional 1-based worker identifier.

    Returns:
        ``None``. The message is emitted at debug level.
    """
    log_status(
        operation,
        message,
        thread=thread,
        level=std_logging.DEBUG,
        _stack_offset=3,
    )
