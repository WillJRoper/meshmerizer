"""HDF5 serialization for the adaptive octree pipeline.

This module provides functions to save and reload the full adaptive
octree state (particles, cells, contributors, and optionally a mesh)
to an HDF5 file.  All I/O is done in Python with h5py; the C++
extension has no HDF5 dependency.

The on-disk layout is documented in ``adaptive-plan.md`` under
"HDF5 Schema".
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "h5py is required for octree serialization. "
        "Install it with: pip install h5py"
    ) from exc


# ---------------------------------------------------------------------------
# Version tag written into every file so readers can detect format changes.
# ---------------------------------------------------------------------------
_SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Public type aliases (kept deliberately simple for Python 3.8).
# ---------------------------------------------------------------------------
CellDict = Dict[str, object]
Vec3 = Tuple[float, float, float]
MeshVertex = Tuple[Vec3, Vec3]
Triangle = Tuple[int, int, int]


# ===================================================================
# Export
# ===================================================================


def export_octree(
    path: str,
    *,
    isovalue: float,
    base_resolution: int,
    max_depth: int,
    domain_minimum: Vec3,
    domain_maximum: Vec3,
    positions: Sequence[Vec3],
    smoothing_lengths: Sequence[float],
    cells: Sequence[CellDict],
    contributors: Sequence[int],
    vertices: Optional[Sequence[MeshVertex]] = None,
    triangles: Optional[Sequence[Triangle]] = None,
    version: str = "",
) -> None:
    """Write the full adaptive octree state to an HDF5 file.

    Args:
        path: Destination file path (will be overwritten if it exists).
        isovalue: The target isosurface level.
        base_resolution: Number of top-level cells per axis.
        max_depth: Maximum octree refinement depth.
        domain_minimum: Inclusive lower corner of the working domain.
        domain_maximum: Exclusive upper corner of the working domain.
        positions: Particle positions, shape ``(N, 3)``.
        smoothing_lengths: Per-particle support radii, length ``N``.
        cells: Sequence of octree cell dictionaries (from
            ``refine_octree``).
        contributors: Flat contributor index array (from
            ``refine_octree``).
        vertices: Optional mesh vertices as ``((px,py,pz),(nx,ny,nz))``
            pairs.
        triangles: Optional mesh triangles as ``(i0, i1, i2)`` triples.
        version: Optional version string written into metadata.  If
            empty the field is still written as an empty string.
    """
    # ------------------------------------------------------------------
    # Convert Python sequences to NumPy arrays for efficient HDF5 I/O.
    # ------------------------------------------------------------------

    # Particles ---
    pos_arr = np.asarray(positions, dtype=np.float64)
    if pos_arr.ndim == 1 and pos_arr.size == 0:
        pos_arr = pos_arr.reshape((0, 3))
    sml_arr = np.asarray(smoothing_lengths, dtype=np.float64)

    # Contributors ---
    contrib_arr = np.asarray(contributors, dtype=np.int64)

    # Cells: decompose the list of dicts into columnar arrays.
    n_cells = len(cells)
    morton_keys = np.empty(n_cells, dtype=np.uint64)
    depths = np.empty(n_cells, dtype=np.uint32)
    bounds_min = np.empty((n_cells, 3), dtype=np.float64)
    bounds_max = np.empty((n_cells, 3), dtype=np.float64)
    is_leaf = np.empty(n_cells, dtype=np.bool_)
    is_active = np.empty(n_cells, dtype=np.bool_)
    has_surface = np.empty(n_cells, dtype=np.bool_)
    child_begin = np.empty(n_cells, dtype=np.int64)
    corner_sign_mask = np.empty(n_cells, dtype=np.uint8)
    corner_values = np.empty((n_cells, 8), dtype=np.float64)
    contributor_begin = np.empty(n_cells, dtype=np.int64)
    contributor_end = np.empty(n_cells, dtype=np.int64)

    # We need to reconstruct contributor begin/end from the per-cell
    # contributors tuples.  The flat ``contributors`` array passed in
    # is the authoritative ordering.  We walk through cells and match
    # each cell's contributor tuple to a contiguous slice of the flat
    # array.
    #
    # Because ``refine_octree`` returns cells whose ``contributors``
    # field is already derived from contiguous slices of the flat
    # contributors array, we can rebuild begin/end by scanning.
    flat_offset = 0
    for i, cell in enumerate(cells):
        morton_keys[i] = cell["morton_key"]
        depths[i] = cell["depth"]
        bmin, bmax = cell["bounds"]
        bounds_min[i] = bmin
        bounds_max[i] = bmax
        is_leaf[i] = bool(cell["is_leaf"])
        is_active[i] = bool(cell["is_active"])
        has_surface[i] = bool(cell["has_surface"])
        child_begin[i] = cell["child_begin"]
        corner_sign_mask[i] = cell["corner_sign_mask"]
        corner_values[i] = cell["corner_values"]

        # Contributor range: each cell's contributors tuple contains
        # the particle indices for that cell.  These correspond to a
        # contiguous slice of the flat contributors array, but the
        # cell dict doesn't store the offsets directly.  We need to
        # find where this cell's contributors sit in the flat array.
        cell_contribs = cell["contributors"]
        n_contribs = len(cell_contribs)

        # For non-leaf cells (internal nodes), the contributors tuple
        # is empty and the range is (0, 0).  For leaf cells, find
        # the matching slice in the flat array by searching from the
        # current offset.
        if n_contribs == 0:
            contributor_begin[i] = 0
            contributor_end[i] = 0
        else:
            # The flat contributors array preserves the order cells
            # were created during refinement.  We scan forward to
            # find the matching contiguous run.
            found = False
            search_start = 0 if flat_offset == 0 else flat_offset
            for start in range(search_start, len(contributors)):
                if start + n_contribs > len(contributors):
                    break
                match = True
                for j in range(n_contribs):
                    if contributors[start + j] != cell_contribs[j]:
                        match = False
                        break
                if match:
                    contributor_begin[i] = start
                    contributor_end[i] = start + n_contribs
                    flat_offset = start + n_contribs
                    found = True
                    break
            if not found:
                # Fallback: store the contributors inline by
                # appending them.  This should not happen with
                # well-formed refine_octree output.
                raise ValueError(
                    f"Cell {i} contributors not found in flat "
                    f"contributor array at expected offset"
                )

    # ------------------------------------------------------------------
    # Write the HDF5 file.
    # ------------------------------------------------------------------
    with h5py.File(path, "w") as f:
        # -- metadata --
        meta = f.create_group("metadata")
        meta.attrs["version"] = version
        meta.attrs["schema_version"] = _SCHEMA_VERSION
        meta.attrs["isovalue"] = isovalue
        meta.attrs["base_resolution"] = np.int64(base_resolution)
        meta.attrs["max_depth"] = np.int64(max_depth)
        meta.attrs["kernel_type"] = "wendland_c2"
        meta.create_dataset(
            "domain_minimum",
            data=np.asarray(domain_minimum, dtype=np.float64),
        )
        meta.create_dataset(
            "domain_maximum",
            data=np.asarray(domain_maximum, dtype=np.float64),
        )

        # -- particles --
        ptcl = f.create_group("particles")
        ptcl.create_dataset("positions", data=pos_arr)
        ptcl.create_dataset("smoothing_lengths", data=sml_arr)

        # -- octree --
        tree = f.create_group("octree")
        tree.create_dataset("morton_keys", data=morton_keys)
        tree.create_dataset("depths", data=depths)
        tree.create_dataset("bounds_min", data=bounds_min)
        tree.create_dataset("bounds_max", data=bounds_max)
        tree.create_dataset("is_leaf", data=is_leaf)
        tree.create_dataset("is_active", data=is_active)
        tree.create_dataset("has_surface", data=has_surface)
        tree.create_dataset("child_begin", data=child_begin)
        tree.create_dataset("corner_sign_mask", data=corner_sign_mask)
        tree.create_dataset("corner_values", data=corner_values)
        tree.create_dataset("contributor_begin", data=contributor_begin)
        tree.create_dataset("contributor_end", data=contributor_end)

        # -- contributors --
        cgrp = f.create_group("contributors")
        cgrp.create_dataset("indices", data=contrib_arr)

        # -- mesh (optional) --
        if vertices is not None and triangles is not None:
            mgrp = f.create_group("mesh")
            vert_pos = np.array([v[0] for v in vertices], dtype=np.float64)
            vert_nrm = np.array([v[1] for v in vertices], dtype=np.float64)
            tri_arr = np.asarray(triangles, dtype=np.int64)
            if vert_pos.ndim == 1 and vert_pos.size == 0:
                vert_pos = vert_pos.reshape((0, 3))
                vert_nrm = vert_nrm.reshape((0, 3))
            if tri_arr.ndim == 1 and tri_arr.size == 0:
                tri_arr = tri_arr.reshape((0, 3))
            mgrp.create_dataset("vertices", data=vert_pos)
            mgrp.create_dataset("normals", data=vert_nrm)
            mgrp.create_dataset("triangles", data=tri_arr)


# ===================================================================
# Import
# ===================================================================


def import_octree(
    path: str,
) -> dict:
    """Reload the adaptive octree state from an HDF5 file.

    Args:
        path: Path to the HDF5 file written by ``export_octree``.

    Returns:
        A dictionary with the following keys:

        - ``isovalue`` (float)
        - ``base_resolution`` (int)
        - ``max_depth`` (int)
        - ``domain_minimum`` (tuple of 3 floats)
        - ``domain_maximum`` (tuple of 3 floats)
        - ``kernel_type`` (str)
        - ``version`` (str)
        - ``positions`` (list of 3-tuples)
        - ``smoothing_lengths`` (list of floats)
        - ``cells`` (list of cell dicts compatible with
          ``adaptive_core.generate_mesh``)
        - ``contributors`` (list of ints)
        - ``vertices`` (list of ``((px,py,pz),(nx,ny,nz))`` or None)
        - ``triangles`` (list of ``(i0,i1,i2)`` or None)
    """
    with h5py.File(path, "r") as f:
        # -- metadata --
        meta = f["metadata"]
        isovalue = float(meta.attrs["isovalue"])
        base_resolution = int(meta.attrs["base_resolution"])
        max_depth = int(meta.attrs["max_depth"])
        kernel_type = str(meta.attrs["kernel_type"])
        version = str(meta.attrs.get("version", ""))
        domain_minimum = tuple(meta["domain_minimum"][:].tolist())
        domain_maximum = tuple(meta["domain_maximum"][:].tolist())

        # -- particles --
        ptcl = f["particles"]
        pos_arr = ptcl["positions"][:]
        sml_arr = ptcl["smoothing_lengths"][:]
        positions = [tuple(row) for row in pos_arr.tolist()]
        smoothing_lengths = sml_arr.tolist()

        # -- contributors --
        contrib_arr = f["contributors"]["indices"][:]
        contributors_list = contrib_arr.tolist()

        # -- octree cells --
        tree = f["octree"]
        morton_keys = tree["morton_keys"][:]
        depths = tree["depths"][:]
        bounds_min_arr = tree["bounds_min"][:]
        bounds_max_arr = tree["bounds_max"][:]
        is_leaf_arr = tree["is_leaf"][:]
        is_active_arr = tree["is_active"][:]
        has_surface_arr = tree["has_surface"][:]
        child_begin_arr = tree["child_begin"][:]
        csm_arr = tree["corner_sign_mask"][:]
        cv_arr = tree["corner_values"][:]
        cb_arr = tree["contributor_begin"][:]
        ce_arr = tree["contributor_end"][:]

        n_cells = len(morton_keys)
        cells = []
        for i in range(n_cells):
            bmin = tuple(bounds_min_arr[i].tolist())
            bmax = tuple(bounds_max_arr[i].tolist())
            cb = int(cb_arr[i])
            ce = int(ce_arr[i])
            cell_contribs = tuple(int(x) for x in contrib_arr[cb:ce].tolist())
            cell = {
                "morton_key": int(morton_keys[i]),
                "depth": int(depths[i]),
                "bounds": (bmin, bmax),
                "is_leaf": int(bool(is_leaf_arr[i])),
                "is_active": int(bool(is_active_arr[i])),
                "has_surface": int(bool(has_surface_arr[i])),
                "child_begin": int(child_begin_arr[i]),
                "corner_sign_mask": int(csm_arr[i]),
                "corner_values": tuple(cv_arr[i].tolist()),
                "contributors": cell_contribs,
            }
            cells.append(cell)

        # -- mesh (optional) --
        vertices = None
        triangles_out = None
        if "mesh" in f:
            mgrp = f["mesh"]
            vert_pos = mgrp["vertices"][:]
            vert_nrm = mgrp["normals"][:]
            tri_arr = mgrp["triangles"][:]
            vertices = [
                (
                    tuple(vert_pos[i].tolist()),
                    tuple(vert_nrm[i].tolist()),
                )
                for i in range(len(vert_pos))
            ]
            triangles_out = [
                tuple(int(x) for x in row) for row in tri_arr.tolist()
            ]

    return {
        "isovalue": isovalue,
        "base_resolution": base_resolution,
        "max_depth": max_depth,
        "domain_minimum": domain_minimum,
        "domain_maximum": domain_maximum,
        "kernel_type": kernel_type,
        "version": version,
        "positions": positions,
        "smoothing_lengths": smoothing_lengths,
        "cells": cells,
        "contributors": contributors_list,
        "vertices": vertices,
        "triangles": triangles_out,
    }
