"""Chunk assembly and union helpers.

This module combines chunk-local meshes into one final surface using a robust
boolean-union backend. To improve stability on large chunk sets, unions are
scheduled pairwise over meshes whose actual bounding boxes overlap.
"""

from __future__ import annotations

import trimesh
from trimesh import repair

from meshmerizer.logging import log_status
from meshmerizer.mesh import Mesh

from .geometry import HardChunkBounds


def _bounds_overlap(
    left_bounds,
    right_bounds,
    *,
    tol: float = 1e-8,
) -> bool:
    """Return whether two axis-aligned boxes overlap or touch."""
    return bool(
        (left_bounds[0][0] <= right_bounds[1][0] + tol)
        and (right_bounds[0][0] <= left_bounds[1][0] + tol)
        and (left_bounds[0][1] <= right_bounds[1][1] + tol)
        and (right_bounds[0][1] <= left_bounds[1][1] + tol)
        and (left_bounds[0][2] <= right_bounds[1][2] + tol)
        and (right_bounds[0][2] <= left_bounds[1][2] + tol)
    )


def _prepare_mesh_for_union(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize one mesh before robust boolean union."""
    candidate = mesh.copy()
    candidate.process()
    repair.fix_winding(candidate)
    repair.fix_inversion(candidate, multibody=True)
    repair.fix_normals(candidate, multibody=True)
    if not candidate.is_volume:
        candidate.invert()
        repair.fix_winding(candidate)
        repair.fix_inversion(candidate, multibody=True)
        repair.fix_normals(candidate, multibody=True)
    return candidate


def _boolean_union_group(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Union one connected mesh group with the manifold backend."""
    if not meshes:
        raise ValueError("No meshes to union")
    if len(meshes) == 1:
        result = meshes[0].copy()
        result.process()
        result.fix_normals()
        return result

    prepared = []
    for mesh in meshes:
        prepared.append(_prepare_mesh_for_union(mesh))

    unioned = trimesh.boolean.union(
        prepared,
        engine="manifold",
        check_volume=False,
    )
    if isinstance(unioned, trimesh.Scene):
        geometries = list(unioned.geometry.values())
        if not geometries:
            raise ValueError("Boolean union produced an empty scene")
        unioned = trimesh.util.concatenate(geometries)
    if unioned is None or len(unioned.faces) == 0:
        raise ValueError("Boolean union produced no faces")

    unioned.process()
    unioned.fix_normals()
    return unioned


def _pairwise_union_meshes(
    meshes: list[trimesh.Trimesh],
) -> tuple[list[trimesh.Trimesh], int]:
    """Iteratively union overlapping mesh pairs until convergence.

    Args:
        meshes: Source chunk meshes.

    Returns:
        Tuple of the reduced mesh list and the number of pairwise unions run.
    """
    pending = [mesh.copy() for mesh in meshes]
    unions_run = 0
    changed = True
    while changed:
        changed = False
        next_pending: list[trimesh.Trimesh] = []
        used = [False] * len(pending)
        for i, left in enumerate(pending):
            if used[i]:
                continue
            left_bounds = left.bounds
            partner = None
            for j in range(i + 1, len(pending)):
                if used[j]:
                    continue
                right = pending[j]
                if _bounds_overlap(left_bounds, right.bounds):
                    partner = j
                    break
            if partner is None:
                next_pending.append(left)
                used[i] = True
                continue

            merged = _boolean_union_group([left, pending[partner]])
            next_pending.append(merged)
            used[i] = True
            used[partner] = True
            unions_run += 1
            changed = True
        pending = next_pending
    return pending, unions_run


def _cleanup_union_components(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    """Repair union result component-wise and drop obvious artifacts."""
    components = mesh.split(only_watertight=False)
    if not components:
        raise ValueError("Union result contains no connected components")

    cleaned: list[trimesh.Trimesh] = []
    kept_faces: list[int] = []
    dropped = 0
    watertight_components = 0
    open_components = 0
    for component in components:
        candidate = component.copy()
        candidate.process()
        repair.fix_winding(candidate)
        repair.fix_inversion(candidate, multibody=True)
        repair.fix_normals(candidate, multibody=True)
        repair.fill_holes(candidate)
        candidate.process()
        repair.fix_winding(candidate)
        repair.fix_inversion(candidate, multibody=True)
        repair.fix_normals(candidate, multibody=True)

        if len(candidate.faces) == 0:
            dropped += 1
            continue
        if (not candidate.is_watertight) and len(candidate.faces) < 100:
            dropped += 1
            continue

        if candidate.is_watertight:
            watertight_components += 1
        else:
            open_components += 1
        kept_faces.append(len(candidate.faces))
        cleaned.append(candidate)

    if not cleaned:
        raise ValueError("All union result components were discarded")

    if len(cleaned) == 1:
        merged = cleaned[0]
    else:
        merged = trimesh.util.concatenate(cleaned)
        # Do not run a global topology rewrite across multiple already-
        # repaired watertight bodies. Treating distinct closed components as
        # one mesh can perturb cached volume/orientation bookkeeping and make
        # a valid multibody result appear non-volumetric.
        merged.fix_normals()

    diagnostics = {
        "input_components": len(components),
        "kept_components": len(cleaned),
        "dropped_components": dropped,
        "watertight_components": watertight_components,
        "open_components": open_components,
        "largest_faces": sorted(kept_faces, reverse=True)[:5],
    }
    return merged, diagnostics


def union_hard_chunk_meshes(
    chunk_meshes: list[tuple[HardChunkBounds, list[Mesh]]],
) -> Mesh:
    """Union overlapped hard chunk meshes into one solid surface.

    Args:
        chunk_meshes: Per-chunk meshes generated on overlapped chunk domains.

    Returns:
        Final unioned mesh.

    Raises:
        ValueError: If there is no geometry to assemble.
    """
    if not chunk_meshes:
        raise ValueError("No chunk meshes to union")

    source_mesh_count = sum(len(meshes) for _bounds, meshes in chunk_meshes)
    source_watertight = sum(
        1
        for _bounds, meshes in chunk_meshes
        for mesh in meshes
        if mesh.to_trimesh().is_watertight
    )
    source_volumes = sum(
        1
        for _bounds, meshes in chunk_meshes
        for mesh in meshes
        if mesh.to_trimesh().is_volume
    )
    source_meshes = [
        mesh.to_trimesh().copy()
        for _bounds, meshes in chunk_meshes
        for mesh in meshes
    ]
    prepared_volumes = sum(
        1 for mesh in source_meshes if _prepare_mesh_for_union(mesh).is_volume
    )
    log_status(
        "Meshing",
        "Union assembly input:\n"
        f"  chunk groups:   {len(chunk_meshes)}\n"
        f"  source meshes:  {source_mesh_count}\n"
        f"  watertight:     {source_watertight}/{source_mesh_count} "
        f"source meshes\n"
        f"  volumes:        {source_volumes}/{source_mesh_count} "
        f"source meshes\n"
        f"  prep volumes:   {prepared_volumes}/{source_mesh_count} "
        f"source meshes",
    )

    if not source_meshes:
        raise ValueError("No chunk geometry remained for union")

    unioned_meshes, unions_run = _pairwise_union_meshes(source_meshes)
    if len(unioned_meshes) == 1:
        final = unioned_meshes[0]
    else:
        final = trimesh.util.concatenate(unioned_meshes)
        final.process()
        repair.fix_winding(final)
        repair.fix_inversion(final, multibody=True)
        repair.fix_normals(final, multibody=True)

    final, component_diag = _cleanup_union_components(final)

    log_status(
        "Meshing",
        "Pairwise robust union result:\n"
        f"  pair unions:    {unions_run}\n"
        f"  remaining:      {len(unioned_meshes)}\n"
        f"  components:     {component_diag['input_components']} -> "
        f"{component_diag['kept_components']} kept, "
        f"{component_diag['dropped_components']} dropped\n"
        f"  component wt:   {component_diag['watertight_components']} closed, "
        f"{component_diag['open_components']} open\n"
        f"  largest faces:  {component_diag['largest_faces']}\n"
        f"  faces:          {len(final.faces)}\n"
        f"  components:     {len(final.split(only_watertight=False))}\n"
        f"  watertight:     {final.is_watertight}",
    )
    return Mesh(mesh=final)


def keep_largest_mesh_component(mesh: Mesh) -> Mesh:
    """Keep only the largest connected component of a stitched mesh.

    Args:
        mesh: Mesh whose connected components should be filtered.

    Returns:
        Mesh containing only the largest component.

    Raises:
        ValueError: If no connected components exist.
    """
    trimesh_mesh = mesh.to_trimesh()
    components = trimesh_mesh.split(only_watertight=False)
    if not components:
        raise ValueError("No connected mesh components found")
    if len(components) == 1:
        return mesh

    def component_size(component: trimesh.Trimesh) -> float:
        if component.is_volume:
            try:
                return float(abs(component.volume))
            except Exception:  # noqa: BLE001
                pass
        return float(len(component.faces))

    largest = max(components, key=component_size)
    largest.process()
    largest.fix_normals()
    return Mesh(mesh=largest)
