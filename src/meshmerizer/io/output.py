"""Output helpers for writing generated assets safely.

The CLI writes mesh outputs through this module so atomic-save policy lives in
one place instead of being reimplemented across command handlers.
"""

from __future__ import annotations

from pathlib import Path

from meshmerizer.mesh import Mesh


def save_mesh_output(mesh: Mesh, output_path: Path) -> None:
    """Write mesh output atomically to avoid partial files on cancel.

    Args:
        mesh: Mesh to serialize.
        output_path: Final desired output path.

    Returns:
        ``None``. The mesh is written to disk.
    """
    temp_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    try:
        mesh.save(str(temp_path))
        if not temp_path.exists() and output_path.exists():
            return
        if not temp_path.exists():
            output_path.touch()
            return
        temp_path.replace(output_path)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


__all__ = ["save_mesh_output"]
