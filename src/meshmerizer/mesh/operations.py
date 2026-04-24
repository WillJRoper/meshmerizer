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
    # Skip all connected-component work when the feature is disabled so callers
    # do not pay for an unnecessary mesh split.
    if remove_islands_fraction is None:
        return mesh

    def _component_reference_volume(component) -> float:
        """Return a robust size estimate for island filtering.

        Args:
            component: One connected ``trimesh`` component.

        Returns:
            Non-negative reference volume used for island ranking.
        """
        # Prefer true volume for watertight components because it best reflects
        # printable size.
        if component.is_watertight:
            volume = abs(float(component.volume))
            if np.isfinite(volume):
                return volume

        # Fall back to convex-hull volume when the component is open or broken,
        # which still provides a stable relative size estimate for filtering.
        try:
            hull_volume = abs(float(component.convex_hull.volume))
        except Exception:
            hull_volume = 0.0
        return hull_volume if np.isfinite(hull_volume) else 0.0

    # Split first so every later branch works with the same connected-component
    # representation.
    components = mesh.mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh

    # Treat zero as the special "keep only the largest component" mode.
    if remove_islands_fraction == 0.0:
        largest = max(components, key=_component_reference_volume)
        log_status(
            "Cleaning",
            f"Kept largest of {len(components)} components.",
        )
        return Mesh(mesh=largest)

    # Compute the per-component reference sizes once so they can drive both the
    # keep mask and the status message.
    volumes = [_component_reference_volume(comp) for comp in components]
    largest_volume = max(volumes, default=0.0)
    if largest_volume == 0.0:
        return mesh

    # Keep every component whose reference volume is large enough relative to
    # the largest observed component.
    kept = [
        c
        for c, volume in zip(components, volumes)
        if volume / largest_volume >= remove_islands_fraction
    ]
    if not kept:
        # Always keep at least one component so filtering never returns
        # an empty mesh purely because the threshold was too aggressive.
        kept = [max(components, key=_component_reference_volume)]

    log_status(
        "Cleaning",
        f"Kept {len(kept)} of {len(components)} components "
        f"(fraction >= {remove_islands_fraction} of largest volume).",
    )

    # Concatenate the kept components back into one mesh wrapper for the
    # rest of the pipeline.
    import trimesh

    return Mesh(mesh=trimesh.util.concatenate(kept))


def simplify_mesh(mesh: Mesh, simplify_factor: float) -> Mesh:
    """Optionally simplify the mesh after extraction and cleanup.

    Args:
        mesh: Mesh to simplify in place.
        simplify_factor: Fraction of faces to retain in ``(0, 1]``.

    Returns:
        Simplified mesh instance.
    """
    # Treat a factor of 1.0 as an explicit no-op so callers can pass
    # through CLI input directly without branching.
    if simplify_factor == 1.0:
        return mesh

    # Log before and after face counts so simplification strength is visible in
    # CLI output and tests.
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
