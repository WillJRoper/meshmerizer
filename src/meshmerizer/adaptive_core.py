"""Python wrapper for the adaptive meshing C++ core.

This module is the stable Python entry point for the new adaptive rewrite. The
initial implementation only exposes scaffold-level functionality while the C++
core types and algorithms are introduced in later phases.
"""

from __future__ import annotations

from importlib import import_module

_adaptive = import_module("meshmerizer._adaptive")


def adaptive_status() -> str:
    """Return the current adaptive core scaffold status string.

    Returns:
        Short human-readable status from the C++ adaptive extension.
    """
    # Delegate to the compiled module so tests verify the C++ extension is the
    # real implementation entry point.
    return _adaptive.adaptive_status()
