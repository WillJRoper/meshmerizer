"""Output helpers for writing generated assets safely."""

from __future__ import annotations

from pathlib import Path

from meshmerizer.mesh import Mesh


def save_mesh_output(mesh: Mesh, output_path: Path) -> None:
    """Write mesh output atomically to avoid partial files on cancel."""
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
