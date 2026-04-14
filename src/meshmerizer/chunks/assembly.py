"""Chunk assembly and union helpers.

This module contains the seam-ownership and assembly logic used to combine
overlapped hard-chunk meshes into one final watertight surface. It also holds
connected-component filtering helpers and the small planar seam-capping
utilities used after chunk assembly.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import trimesh
from trimesh import repair

from meshmerizer.mesh import Mesh

from .geometry import HardChunkBounds


def union_hard_chunk_meshes(
    chunk_meshes: list[tuple[HardChunkBounds, list[Mesh]]],
) -> Mesh:
    """Assemble overlapped hard chunk meshes into one watertight solid.

    Args:
        chunk_meshes: Per-chunk meshes generated on overlapped chunk domains.

    Returns:
        Watertight assembled mesh.

    Raises:
        ValueError: If there is no geometry to assemble.
    """
    # The union step works by assigning each boundary face to exactly one chunk
    # rather than running a heavyweight geometric boolean across all parts.
    if not chunk_meshes:
        raise ValueError("No chunk meshes to union")

    # Keep only the triangles owned by each chunk's hard box so overlap regions
    # do not duplicate surfaces in the assembled mesh.
    assembled_parts: list[trimesh.Trimesh] = []
    for bounds, meshes in chunk_meshes:
        for mesh in meshes:
            trimesh_mesh = mesh.to_trimesh().copy()
            centroids = trimesh_mesh.triangles_center
            keep = np.ones(len(trimesh_mesh.faces), dtype=bool)

            # Assign ownership of triangles on every seam to exactly one chunk.
            # Interior chunks use a half-open interval on their upper faces,
            # while the last chunk on an axis keeps the closed upper boundary.
            # This avoids duplicate faces in overlap regions without requiring
            # a global boolean union.
            for axis in range(3):
                lower = bounds.hard_world_start[axis]
                upper = bounds.hard_world_stop[axis]
                if bounds.index[axis] == bounds.nchunks - 1:
                    axis_keep = (centroids[:, axis] >= lower - 1e-8) & (
                        centroids[:, axis] <= upper + 1e-8
                    )
                else:
                    axis_keep = (centroids[:, axis] >= lower - 1e-8) & (
                        centroids[:, axis] < upper - 1e-8
                    )
                keep &= axis_keep

            if not np.any(keep):
                continue
            trimesh_mesh.update_faces(keep)
            trimesh_mesh.remove_unreferenced_vertices()
            assembled_parts.append(trimesh_mesh)

    if not assembled_parts:
        raise ValueError("No chunk geometry remained after seam ownership")

    # After ownership filtering, clean and cap the combined seam graph so the
    # final mesh is watertight.
    unioned = trimesh.util.concatenate(assembled_parts)
    unioned.merge_vertices()
    unioned.update_faces(unioned.unique_faces())
    unioned.update_faces(unioned.nondegenerate_faces())
    unioned.remove_unreferenced_vertices()
    repair.fill_holes(unioned)
    unioned = cap_planar_boundary_loops(unioned)
    unioned.fix_normals()
    return Mesh(mesh=unioned)


def keep_largest_mesh_component(mesh: Mesh) -> Mesh:
    """Keep only the largest connected component of a stitched mesh.

    Args:
        mesh: Mesh whose connected components should be filtered.

    Returns:
        Mesh containing only the largest component.

    Raises:
        ValueError: If no connected components exist.
    """
    # Split on connected components after assembly so tiny debris can be
    # removed using a simple and robust post-process.
    trimesh_mesh = mesh.to_trimesh()
    components = trimesh_mesh.split(only_watertight=False)
    if not components:
        raise ValueError("No connected mesh components found")
    if len(components) == 1:
        return mesh

    def component_size(component: trimesh.Trimesh) -> float:
        """Score one component for largest-component selection.

        Args:
            component: Candidate connected component.

        Returns:
            Volume when available, otherwise face count.
        """
        # Prefer geometric volume when it is meaningful, but fall back to face
        # count for open or otherwise non-volumetric components.
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


def mesh_boundary_loops(mesh: trimesh.Trimesh) -> list[list[int]]:
    """Return boundary vertex loops for a mesh with open boundaries.

    Args:
        mesh: Mesh whose open boundary loops should be extracted.

    Returns:
        Boundary loops as lists of vertex indices.

    Raises:
        ValueError: If the boundary graph is not a collection of simple loops.
    """
    # Identify boundary edges as edges referenced by exactly one face.
    edge_counts: Counter[tuple[int, int]] = Counter()
    for tri in mesh.faces:
        a, b, c = tri
        edge_counts[tuple(sorted((a, b)))] += 1
        edge_counts[tuple(sorted((b, c)))] += 1
        edge_counts[tuple(sorted((a, c)))] += 1

    boundary_edges = [
        edge for edge, count in edge_counts.items() if count == 1
    ]
    if not boundary_edges:
        return []

    # Convert the boundary-edge set into a vertex adjacency graph so each open
    # boundary can be walked as a loop.
    adjacency: defaultdict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    # Walk each connected boundary component and ensure it forms a simple loop
    # rather than an open chain or branching graph.
    loops: list[list[int]] = []
    visited_vertices: set[int] = set()
    for start in list(adjacency):
        if start in visited_vertices:
            continue

        loop = [start]
        visited_vertices.add(start)
        prev = None
        cur = start
        while True:
            next_vertices = [v for v in adjacency[cur] if v != prev]
            if not next_vertices:
                raise ValueError("Encountered open boundary chain")
            nxt = next_vertices[0]
            if nxt == start:
                break
            if nxt in visited_vertices:
                raise ValueError("Boundary graph is not a simple loop")
            loop.append(nxt)
            visited_vertices.add(nxt)
            prev, cur = cur, nxt
        loops.append(loop)
    return loops


def polygon_area_2d(points: np.ndarray) -> float:
    """Return the signed area of a 2D polygon.

    Args:
        points: Polygon vertices in 2D.

    Returns:
        Signed polygon area.
    """
    # Use the shoelace formula because the loops have already been projected to
    # a 2D plane.
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def point_in_triangle_2d(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> bool:
    """Return whether a 2D point lies inside or on a triangle.

    Args:
        point: Query point in 2D.
        a: First triangle vertex.
        b: Second triangle vertex.
        c: Third triangle vertex.

    Returns:
        ``True`` if the point lies inside or on the triangle.
    """

    def sign(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Return the oriented area sign for three 2D points.

        Args:
            p1: First 2D point.
            p2: Second 2D point.
            p3: Third 2D point.

        Returns:
            Signed orientation value.
        """
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (
            p1[1] - p3[1]
        )

    d1 = sign(point, a, b)
    d2 = sign(point, b, c)
    d3 = sign(point, c, a)
    has_neg = (d1 < -1e-12) or (d2 < -1e-12) or (d3 < -1e-12)
    has_pos = (d1 > 1e-12) or (d2 > 1e-12) or (d3 > 1e-12)
    return not (has_neg and has_pos)


def triangulate_loop(loop: list[int], vertices: np.ndarray) -> list[list[int]]:
    """Triangulate one small planar boundary loop with ear clipping.

    Args:
        loop: Boundary loop as vertex indices into ``vertices``.
        vertices: Full vertex array.

    Returns:
        Triangles expressed as vertex-index triplets.

    Raises:
        ValueError: If ear clipping cannot triangulate the loop.
    """
    # Project the loop to the flattest 2D plane before running ear clipping.
    points = vertices[loop]
    spans = points.max(axis=0) - points.min(axis=0)
    flat_axis = int(np.argmin(spans))
    planar = points[:, [axis for axis in range(3) if axis != flat_axis]]

    # Clip one valid ear at a time until only one triangle remains.
    indices = list(range(len(loop)))
    ccw = polygon_area_2d(planar) > 0
    faces: list[list[int]] = []
    guard = 0
    while len(indices) > 3 and guard < 1000:
        guard += 1
        clipped = False
        n = len(indices)
        for i in range(n):
            ia = indices[(i - 1) % n]
            ib = indices[i]
            ic = indices[(i + 1) % n]
            a = planar[ia]
            b = planar[ib]
            c = planar[ic]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (
                c[0] - a[0]
            )
            if (ccw and cross <= 1e-12) or ((not ccw) and cross >= -1e-12):
                continue
            if any(
                point_in_triangle_2d(planar[j], a, b, c)
                for j in indices
                if j not in (ia, ib, ic)
            ):
                continue

            if ccw:
                faces.append([loop[ia], loop[ib], loop[ic]])
            else:
                faces.append([loop[ia], loop[ic], loop[ib]])
            indices.pop(i)
            clipped = True
            break
        if not clipped:
            raise ValueError("Failed to triangulate seam loop")

    if len(indices) == 3:
        if ccw:
            faces.append(
                [loop[indices[0]], loop[indices[1]], loop[indices[2]]]
            )
        else:
            faces.append(
                [loop[indices[0]], loop[indices[2]], loop[indices[1]]]
            )
    return faces


def cap_planar_boundary_loops(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Cap small planar seam loops left after chunk assembly.

    Args:
        mesh: Assembled mesh that may contain tiny planar seam holes.

    Returns:
        Mesh with eligible seam loops capped.
    """
    # Return early in the common case where the mesh is already watertight.
    if mesh.is_watertight:
        return mesh

    try:
        loops = mesh_boundary_loops(mesh)
    except ValueError:
        return mesh
    if not loops:
        return mesh

    # Only cap small nearly planar seam loops. More complicated holes are left
    # untouched rather than guessed incorrectly.
    cap_faces: list[list[int]] = []
    for loop in loops:
        points = mesh.vertices[loop]
        spans = points.max(axis=0) - points.min(axis=0)
        if np.min(spans) > 1e-8 or len(loop) > 32:
            return mesh
        cap_faces.extend(triangulate_loop(loop, mesh.vertices))

    capped = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=np.vstack([mesh.faces, np.asarray(cap_faces, dtype=np.int64)]),
        process=False,
    )
    capped.merge_vertices()
    capped.update_faces(capped.unique_faces())
    capped.update_faces(capped.nondegenerate_faces())
    capped.remove_unreferenced_vertices()
    return capped
