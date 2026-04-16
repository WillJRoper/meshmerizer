"""CLI command package for Meshmerizer.

This package contains argument parsing, SWIFT-loading helpers, and the
adaptive meshing command entrypoint used by the ``meshmerizer`` CLI.
"""

from .adaptive_stl import run_adaptive
from .args import build_parser
from .loading import load_swift_particles
from .main import main

__all__ = [
    "build_parser",
    "load_swift_particles",
    "main",
    "run_adaptive",
]
