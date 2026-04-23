"""CLI adaptive command entrypoint.

This module currently delegates to the legacy implementation while the CLI is
being migrated out of ``meshmerizer.commands``.
"""

from meshmerizer.commands.adaptive_stl import run_adaptive

__all__ = ["run_adaptive"]
