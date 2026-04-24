"""HDF5 serialization for adaptive octree state.

This module is the single source of truth for the on-disk octree schema.

Schema overview
---------------

Each file contains the following top-level groups:

- ``metadata``: scalar configuration and domain information
- ``particles``: particle positions and smoothing lengths
- ``octree``: columnar octree cell properties
- ``contributors``: flat contributor index array referenced by cells
- ``qef_vertices``: optional QEF vertex diagnostics

Dataset layout
--------------

``metadata``
    Attributes
        ``version``: producer version string
        ``schema_version``: octree schema version string
        ``isovalue``: extraction threshold
        ``base_resolution``: number of top-level cells per axis
        ``max_depth``: maximum refinement depth
        ``kernel_type``: kernel name used by the scalar field
    Datasets
        ``domain_minimum``: float64[3]
        ``domain_maximum``: float64[3]

``particles``
    ``positions``: float64[N, 3]
    ``smoothing_lengths``: float64[N]

``octree``
    ``morton_keys``: uint64[C]
    ``depths``: uint32[C]
    ``bounds_min``: float64[C, 3]
    ``bounds_max``: float64[C, 3]
    ``is_leaf``: bool[C]
    ``is_active``: bool[C]
    ``has_surface``: bool[C]
    ``child_begin``: int64[C]
    ``corner_sign_mask``: uint8[C]
    ``corner_values``: float64[C, 8]
    ``contributor_begin``: int64[C]
    ``contributor_end``: int64[C]

``contributors``
    ``indices``: int64[K]

``qef_vertices`` (optional)
    ``positions``: float64[V, 3]
    ``normals``: float64[V, 3]
    ``group_labels``: int64[V]

The imported Python representation mirrors the historical dictionary-based API
so existing reconstruction and diagnostic code can continue to consume it.
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


SCHEMA_VERSION = "1.0"

CellDict = Dict[str, object]
Vec3 = Tuple[float, float, float]
MeshVertex = Tuple[Vec3, Vec3]


def _as_position_array(positions: Sequence[Vec3]) -> np.ndarray:
    """Normalize particle positions for HDF5 storage."""
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim == 1 and array.size == 0:
        return array.reshape((0, 3))
    return array


def _as_optional_vertex_array(
    values: Optional[Sequence],
) -> Optional[np.ndarray]:
    """Normalize optional ``(N, 3)`` vertex-like arrays for storage."""
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1 and array.size == 0:
        return array.reshape((0, 3))
    return array


def _encode_cells(
    cells: Sequence[CellDict],
    contributors: Sequence[int],
) -> dict[str, np.ndarray]:
    """Convert octree cell dictionaries into columnar arrays.

    The exported file stores cell properties in a columnar layout for compact
    HDF5 writes and predictable schema evolution. Contributor slices are
    reconstructed by matching each cell's contributor tuple against the flat
    contributor array returned by the native octree builder.
    """
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

    flat_offset = 0
    for index, cell in enumerate(cells):
        morton_keys[index] = cell["morton_key"]
        depths[index] = cell["depth"]
        bounds_min[index], bounds_max[index] = cell["bounds"]
        is_leaf[index] = bool(cell["is_leaf"])
        is_active[index] = bool(cell["is_active"])
        has_surface[index] = bool(cell["has_surface"])
        child_begin[index] = cell["child_begin"]
        corner_sign_mask[index] = cell["corner_sign_mask"]
        corner_values[index] = cell["corner_values"]

        cell_contributors = cell["contributors"]
        n_contributors = len(cell_contributors)
        if n_contributors == 0:
            contributor_begin[index] = 0
            contributor_end[index] = 0
            continue

        search_start = flat_offset
        found = False
        for start in range(search_start, len(contributors)):
            stop = start + n_contributors
            if stop > len(contributors):
                break
            if all(
                contributors[start + offset] == cell_contributors[offset]
                for offset in range(n_contributors)
            ):
                contributor_begin[index] = start
                contributor_end[index] = stop
                flat_offset = stop
                found = True
                break

        if not found:
            raise ValueError(
                f"Cell {index} contributors not found in flat contributor "
                "array at expected offset"
            )

    return {
        "morton_keys": morton_keys,
        "depths": depths,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "is_leaf": is_leaf,
        "is_active": is_active,
        "has_surface": has_surface,
        "child_begin": child_begin,
        "corner_sign_mask": corner_sign_mask,
        "corner_values": corner_values,
        "contributor_begin": contributor_begin,
        "contributor_end": contributor_end,
    }


def _decode_cells(tree, contributor_indices: np.ndarray) -> list[CellDict]:
    """Reconstruct historical cell dictionaries from the columnar layout."""
    morton_keys = tree["morton_keys"][:]
    depths = tree["depths"][:]
    bounds_min = tree["bounds_min"][:]
    bounds_max = tree["bounds_max"][:]
    is_leaf = tree["is_leaf"][:]
    is_active = tree["is_active"][:]
    has_surface = tree["has_surface"][:]
    child_begin = tree["child_begin"][:]
    sign_masks = tree["corner_sign_mask"][:]
    corner_values = tree["corner_values"][:]
    contributor_begin = tree["contributor_begin"][:]
    contributor_end = tree["contributor_end"][:]

    cells: list[CellDict] = []
    for index in range(len(morton_keys)):
        begin = int(contributor_begin[index])
        end = int(contributor_end[index])
        cells.append(
            {
                "morton_key": int(morton_keys[index]),
                "depth": int(depths[index]),
                "bounds": (
                    tuple(bounds_min[index].tolist()),
                    tuple(bounds_max[index].tolist()),
                ),
                "is_leaf": int(bool(is_leaf[index])),
                "is_active": int(bool(is_active[index])),
                "has_surface": int(bool(has_surface[index])),
                "child_begin": int(child_begin[index]),
                "corner_sign_mask": int(sign_masks[index]),
                "corner_values": tuple(corner_values[index].tolist()),
                "contributors": tuple(
                    int(value)
                    for value in contributor_indices[begin:end].tolist()
                ),
            }
        )
    return cells


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
    vertices: Optional[np.ndarray] = None,
    normals: Optional[Sequence] = None,
    group_labels: Optional[Sequence[int]] = None,
    version: str = "",
) -> None:
    """Write the full adaptive octree state to an HDF5 file."""
    position_array = _as_position_array(positions)
    smoothing_array = np.asarray(smoothing_lengths, dtype=np.float64)
    contributor_array = np.asarray(contributors, dtype=np.int64)
    encoded_cells = _encode_cells(cells, contributors)

    with h5py.File(path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.attrs["version"] = version
        metadata.attrs["schema_version"] = SCHEMA_VERSION
        metadata.attrs["isovalue"] = isovalue
        metadata.attrs["base_resolution"] = np.int64(base_resolution)
        metadata.attrs["max_depth"] = np.int64(max_depth)
        metadata.attrs["kernel_type"] = "wendland_c2"
        metadata.create_dataset(
            "domain_minimum",
            data=np.asarray(domain_minimum, dtype=np.float64),
        )
        metadata.create_dataset(
            "domain_maximum",
            data=np.asarray(domain_maximum, dtype=np.float64),
        )

        particles = handle.create_group("particles")
        particles.create_dataset("positions", data=position_array)
        particles.create_dataset("smoothing_lengths", data=smoothing_array)

        tree = handle.create_group("octree")
        for name, values in encoded_cells.items():
            tree.create_dataset(name, data=values)

        contributor_group = handle.create_group("contributors")
        contributor_group.create_dataset("indices", data=contributor_array)

        vertex_positions = _as_optional_vertex_array(vertices)
        if vertex_positions is not None:
            qef_vertices = handle.create_group("qef_vertices")
            qef_vertices.create_dataset("positions", data=vertex_positions)

            vertex_normals = _as_optional_vertex_array(normals)
            if vertex_normals is None:
                vertex_normals = np.zeros_like(vertex_positions)
            qef_vertices.create_dataset("normals", data=vertex_normals)

            if group_labels is not None:
                qef_vertices.create_dataset(
                    "group_labels",
                    data=np.asarray(group_labels, dtype=np.int64),
                )


def import_octree(path: str) -> dict:
    """Reload the adaptive octree state from an HDF5 file."""
    with h5py.File(path, "r") as handle:
        metadata = handle["metadata"]
        isovalue = float(metadata.attrs["isovalue"])
        base_resolution = int(metadata.attrs["base_resolution"])
        max_depth = int(metadata.attrs["max_depth"])
        kernel_type = str(metadata.attrs["kernel_type"])
        version = str(metadata.attrs.get("version", ""))
        domain_minimum = tuple(metadata["domain_minimum"][:].tolist())
        domain_maximum = tuple(metadata["domain_maximum"][:].tolist())

        particles = handle["particles"]
        position_array = particles["positions"][:]
        smoothing_array = particles["smoothing_lengths"][:]

        contributor_indices = handle["contributors"]["indices"][:]
        cells = _decode_cells(handle["octree"], contributor_indices)

        vertices = None
        normals = None
        group_labels = None
        if "qef_vertices" in handle:
            qef_vertices = handle["qef_vertices"]
            vertices = qef_vertices["positions"][:]
            normals = qef_vertices["normals"][:]
            if "group_labels" in qef_vertices:
                group_labels = qef_vertices["group_labels"][:]

    return {
        "isovalue": isovalue,
        "base_resolution": base_resolution,
        "max_depth": max_depth,
        "domain_minimum": domain_minimum,
        "domain_maximum": domain_maximum,
        "kernel_type": kernel_type,
        "version": version,
        "positions": [tuple(row) for row in position_array.tolist()],
        "smoothing_lengths": smoothing_array.tolist(),
        "cells": cells,
        "contributors": contributor_indices.tolist(),
        "vertices": vertices,
        "normals": normals,
        "group_labels": group_labels,
    }


__all__ = [
    "CellDict",
    "MeshVertex",
    "SCHEMA_VERSION",
    "Vec3",
    "export_octree",
    "import_octree",
]
