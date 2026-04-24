"""Top-level CLI entrypoint."""

from __future__ import annotations

from typing import Optional

from meshmerizer.logging import cli_logging_context, log_summary_status

from .args import build_parser


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the ``meshmerizer`` CLI.

    Args:
        argv: Optional argument vector. When omitted, ``sys.argv[1:]`` is used.

    Returns:
        ``None``. This function dispatches to the selected subcommand.
    """
    # Build and parse the CLI once here so subcommands receive a fully
    # validated namespace.
    parser = build_parser()
    args = parser.parse_args(argv)

    # Run the dispatched CLI command inside the shared logging context so
    # progress bars, terminal messages, and the per-run log file stay aligned.
    with cli_logging_context(silent=getattr(args, "silent", False)):
        try:
            args.func(args)
        except KeyboardInterrupt as exc:
            # Translate Ctrl-C into the conventional shell exit code
            # while still emitting a clean summary line.
            log_summary_status(
                "Cancelled",
                "Interrupted by user; cancelled without writing output.",
            )
            raise SystemExit(130) from exc
