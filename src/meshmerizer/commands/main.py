"""Top-level CLI entrypoint."""

from __future__ import annotations

import sys
from typing import Optional

from meshmerizer.logging import cli_logging_context

from .args import build_parser


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the ``meshmerizer`` CLI.

    Args:
        argv: Optional argument vector. When omitted, ``sys.argv[1:]`` is used.

    Returns:
        ``None``. This function dispatches to the selected subcommand.
    """
    # Default to the real shell argument vector so tests can still inject a
    # synthetic argv while normal CLI use remains unchanged.
    if argv is None:
        argv = sys.argv[1:]

    # Preserve the shorthand ``meshmerizer snapshot.hdf5`` by rewriting it
    # to the explicit ``adaptive`` subcommand form.
    if argv and argv[0] not in ("adaptive",):
        argv = ["adaptive", *argv]

    # Build the parser only after the argv normalization step so the parser
    # always sees the canonical command layout.
    parser = build_parser()
    args = parser.parse_args(argv)

    # Run the dispatched CLI command inside the shared logging context so
    # progress bars, terminal messages, and the per-run log file stay aligned.
    with cli_logging_context(silent=getattr(args, "silent", False)):
        args.func(args)
