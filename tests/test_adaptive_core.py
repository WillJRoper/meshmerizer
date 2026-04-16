"""Tests for the adaptive C++ core scaffold."""

from meshmerizer.adaptive_core import adaptive_status


def test_adaptive_extension_scaffold_imports() -> None:
    """The adaptive C++ extension should import through its Python wrapper."""
    assert adaptive_status() == "adaptive core scaffold ready"
