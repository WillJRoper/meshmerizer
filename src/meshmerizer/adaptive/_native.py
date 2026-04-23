"""Shared import of the compiled adaptive extension."""

from importlib import import_module

_adaptive = import_module("meshmerizer._adaptive")

__all__ = ["_adaptive"]
