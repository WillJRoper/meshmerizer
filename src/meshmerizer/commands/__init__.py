"""CLI command package for Meshmerizer.

This package contains argument parsing, SWIFT-loading helpers, and the STL
command entrypoint used by the top-level ``meshmerizer`` executable.
"""

from .args import build_parser
from .loading import load_swift_particles, load_swift_volume
from .main import main
from .stl import run_stl

__all__ = [
    "build_parser",
    "load_swift_particles",
    "load_swift_volume",
    "main",
    "run_stl",
]
