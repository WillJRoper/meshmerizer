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
    # Write to a sibling temporary path that preserves the real mesh suffix so
    # trimesh still selects the correct exporter.
    temp_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    try:
        # Serialize to the temporary location first so interrupted writes never
        # corrupt the final output path.
        mesh.save(str(temp_path))
        # Some writers may write directly to the final output path; if that has
        # already happened there is nothing left for this helper to replace.
        if not temp_path.exists() and output_path.exists():
            return
        # If no file was produced at all, still touch the target so callers do
        # not fail later on a missing path assumption.
        if not temp_path.exists():
            output_path.touch()
            return
        # Replace atomically once the temporary write succeeded.
        temp_path.replace(output_path)
    except BaseException:
        try:
            # Best-effort cleanup of the temporary file keeps repeated retries
            # from accumulating stale artifacts.
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


__all__ = ["save_mesh_output"]
