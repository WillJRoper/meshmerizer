"""Chunked meshing helpers grouped by concern.

This package contains the hard-chunk pipeline used for low-memory watertight
chunked STL generation, along with the supporting geometry and assembly
helpers needed to union chunk-local meshes.
"""

from .assembly import keep_largest_mesh_component, union_hard_chunk_meshes
from .geometry import (
    AxisChunk,
    Chunk,
    ChunkSamples,
    HardChunkBounds,
    VirtualGrid,
    chunk_world_bounds,
    crop_grid_to_chunk_bounds,
    expand_hard_chunk_bounds,
    iter_hard_chunk_bounds,
)
from .hard import (
    clip_mesh_to_hard_chunk,
    generate_hard_chunk_meshes,
    mesh_hard_chunk_sdf,
    select_particles_in_hard_chunk,
    voxelize_hard_chunk,
)
from .processing import chunk_halo_voxels, preprocess_chunk_grid

__all__ = [
    "AxisChunk",
    "Chunk",
    "ChunkSamples",
    "HardChunkBounds",
    "VirtualGrid",
    "chunk_halo_voxels",
    "chunk_world_bounds",
    "clip_mesh_to_hard_chunk",
    "crop_grid_to_chunk_bounds",
    "expand_hard_chunk_bounds",
    "generate_hard_chunk_meshes",
    "iter_hard_chunk_bounds",
    "keep_largest_mesh_component",
    "mesh_hard_chunk_sdf",
    "preprocess_chunk_grid",
    "select_particles_in_hard_chunk",
    "union_hard_chunk_meshes",
    "voxelize_hard_chunk",
]
