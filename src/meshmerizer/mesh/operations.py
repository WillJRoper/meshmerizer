"""Mesh post-processing operations shared by the CLI and API."""

from __future__ import annotations

from typing import Optional

import numpy as np

from meshmerizer.logging import log_status

from .core import Mesh


def remove_islands(
    mesh: Mesh,
    remove_islands_fraction: Optional[float],
) -> Mesh:
    """Remove small disconnected components from a mesh.

    Args:
        mesh: Input mesh.
        remove_islands_fraction: Fraction of the largest component volume below
            which a connected component is discarded. ``0.0`` keeps only the
            largest component. ``None`` disables island removal entirely.

    Returns:
        A new ``Mesh`` with small islands removed, or the original mesh
        unchanged when removal is disabled.
    """
    if remove_islands_fraction is None:
        return mesh

    def _component_reference_volume(component) -> float:
        """Return a robust size estimate for island filtering."""
        if component.is_watertight:
            volume = abs(float(component.volume))
            if np.isfinite(volume):
                return volume

        try:
            hull_volume = abs(float(component.convex_hull.volume))
        except Exception:
            hull_volume = 0.0
        return hull_volume if np.isfinite(hull_volume) else 0.0

    components = mesh.mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh

    if remove_islands_fraction == 0.0:
        largest = max(components, key=_component_reference_volume)
        log_status(
            "Cleaning",
            f"Kept largest of {len(components)} components.",
        )
        return Mesh(mesh=largest)

    volumes = [_component_reference_volume(comp) for comp in components]
    largest_volume = max(volumes, default=0.0)
    if largest_volume == 0.0:
        return mesh

    kept = [
        c
        for c, volume in zip(components, volumes)
        if volume / largest_volume >= remove_islands_fraction
    ]
    if not kept:
        kept = [max(components, key=_component_reference_volume)]

    log_status(
        "Cleaning",
        f"Kept {len(kept)} of {len(components)} components "
        f"(fraction >= {remove_islands_fraction} of largest volume).",
    )

    import trimesh

    return Mesh(mesh=trimesh.util.concatenate(kept))


def simplify_mesh(mesh: Mesh, simplify_factor: float) -> Mesh:
    """Optionally simplify the mesh after extraction and cleanup."""
    if simplify_factor == 1.0:
        return mesh

    log_status(
        "Cleaning",
        f"Simplifying mesh to retain factor {simplify_factor:.6g}...",
    )
    before_faces = len(mesh.faces)
    mesh.simplify(factor=simplify_factor)
    after_faces = len(mesh.faces)
    log_status(
        "Cleaning",
        f"Simplified mesh faces: {before_faces} -> {after_faces}",
    )
    return mesh


__all__ = ["remove_islands", "simplify_mesh"]
