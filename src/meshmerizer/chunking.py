"""Chunk geometry, voxelization, and assembly helpers.

This module contains the low-memory chunked meshing machinery used by the STL
CLI. It describes chunk geometry without allocating a full global grid,
voxelizes particles into chunk-local fields, extracts per-chunk meshes, and
assembles watertight unioned results from overlapping chunk meshes.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure
from trimesh import repair

try:
    from . import _voxelize
except ImportError:
    # Leave the module importable so non-smoothed code paths still work, but
    # smoothed chunk deposition will raise when invoked.
    _voxelize = None

from .mesh import Mesh, voxels_to_stl_via_sdf
from .voxels import (
    process_filament_filter,
    process_gaussian_smoothing,
    process_log_scale,
    process_remove_halos,
)


@dataclass(frozen=True)
class AxisChunk:
    """Owned marching-cubes cell interval for one axis.

    Attributes:
        cell_start: Inclusive first owned marching-cubes cell.
        cell_stop: Exclusive upper marching-cubes cell bound.
    """

    cell_start: int
    cell_stop: int

    @property
    def sample_start(self) -> int:
        """Return the first owned sample index.

        Returns:
            Inclusive lower sample index for this axis chunk.
        """
        return self.cell_start

    @property
    def sample_stop(self) -> int:
        """Return the exclusive upper sample index.

        Returns:
            Exclusive upper sample index for this axis chunk.
        """
        return self.cell_stop + 1


@dataclass(frozen=True)
class Chunk:
    """A single chunk in the virtual global grid.

    Attributes:
        index: Chunk index in ``(x, y, z)`` order.
        x: Owned interval on the x axis.
        y: Owned interval on the y axis.
        z: Owned interval on the z axis.
    """

    index: tuple[int, int, int]
    x: AxisChunk
    y: AxisChunk
    z: AxisChunk


@dataclass(frozen=True)
class ChunkSamples:
    """Owned and halo-expanded sample ranges for a chunk.

    Attributes:
        index: Chunk index in ``(x, y, z)`` order.
        owned_start: Inclusive owned sample start.
        owned_stop: Exclusive owned sample stop.
        start: Inclusive start after halo expansion.
        stop: Exclusive stop after halo expansion.
    """

    index: tuple[int, int, int]
    owned_start: np.ndarray
    owned_stop: np.ndarray
    start: np.ndarray
    stop: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the local sample-grid shape including halo.

        Returns:
            Sample-grid shape for the expanded chunk block.
        """
        return tuple((self.stop - self.start).astype(int))

    @property
    def owned_local_start(self) -> np.ndarray:
        """Return the owned region start in local coordinates.

        Returns:
            Inclusive local start index of the owned chunk region.
        """
        return self.owned_start - self.start

    @property
    def owned_local_stop(self) -> np.ndarray:
        """Return the owned region stop in local coordinates.

        Returns:
            Exclusive local stop index of the owned chunk region.
        """
        return self.owned_local_start + (self.owned_stop - self.owned_start)


@dataclass(frozen=True)
class HardChunkBounds:
    """Hard chunk bounds in sample, local, and world coordinates.

    Attributes:
        index: Chunk index in ``(x, y, z)`` order.
        nchunks: Number of chunks per axis in the parent virtual grid.
        sample_start: Inclusive sample-grid start.
        sample_stop: Exclusive sample-grid stop.
        local_start: Inclusive chunk start in local box coordinates.
        local_stop: Exclusive chunk stop in local box coordinates.
        world_start: Inclusive chunk start in world coordinates.
        world_stop: Exclusive chunk stop in world coordinates.
    """

    index: tuple[int, int, int]
    nchunks: int
    sample_start: np.ndarray
    sample_stop: np.ndarray
    local_start: np.ndarray
    local_stop: np.ndarray
    world_start: np.ndarray
    world_stop: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the chunk sample-grid shape.

        Returns:
            Sample-grid shape implied by the hard chunk bounds.
        """
        return tuple((self.sample_stop - self.sample_start).astype(int))

    @property
    def extent(self) -> np.ndarray:
        """Return the chunk side lengths in world units.

        Returns:
            Per-axis side lengths of the chunk box.
        """
        return self.world_stop - self.world_start

    @property
    def sample_voxel_size(self) -> np.ndarray:
        """Return the per-axis voxel size implied by the bounds.

        Returns:
            Per-axis voxel size inferred from the chunk extent and sample span.
        """
        return self.extent / np.asarray(self.shape, dtype=np.float64)

    @property
    def hard_local_start(self) -> np.ndarray:
        """Return the owned hard-boundary start in local coordinates.

        Returns:
            Inclusive local-space start of the owned hard chunk region.
        """
        return self.local_start

    @property
    def hard_local_stop(self) -> np.ndarray:
        """Return the owned hard-boundary stop in local coordinates.

        Returns:
            Exclusive local-space stop of the owned hard chunk region.
        """
        return self.local_stop - self.sample_voxel_size

    @property
    def hard_world_start(self) -> np.ndarray:
        """Return the owned hard-boundary start in world coordinates.

        Returns:
            Inclusive world-space start of the owned hard chunk region.
        """
        return self.world_start

    @property
    def hard_world_stop(self) -> np.ndarray:
        """Return the owned hard-boundary stop in world coordinates.

        Returns:
            Exclusive world-space stop of the owned hard chunk region.
        """
        return self.world_stop - self.sample_voxel_size


@dataclass(frozen=True)
class VirtualGrid:
    """Global voxel geometry without allocating the full grid.

    Attributes:
        origin: World-space origin of the voxel cube.
        box_size: Physical side length of the cube.
        resolution: Number of voxel samples per axis.
        nchunks: Number of chunks per axis.
    """

    origin: np.ndarray
    box_size: float
    resolution: int
    nchunks: int

    def __post_init__(self) -> None:
        """Validate the virtual grid definition."""
        # Normalize the origin once so every downstream helper can assume a
        # float64 ``(3,)`` vector.
        origin_arr = np.asarray(self.origin, dtype=np.float64)
        if origin_arr.shape != (3,):
            raise ValueError("origin must have shape (3,)")
        object.__setattr__(self, "origin", origin_arr)

        if self.box_size <= 0:
            raise ValueError("box_size must be > 0")
        if self.resolution < 2:
            raise ValueError("resolution must be >= 2")
        if self.nchunks < 1:
            raise ValueError("nchunks must be >= 1")
        # Require at least one marching-cubes cell per chunk so each chunk owns
        # a non-empty meshing region.
        if self.nchunks > self.cell_resolution:
            raise ValueError(
                "nchunks must be <= resolution - 1 so every chunk owns at "
                "least one marching-cubes cell"
            )

    @property
    def voxel_size(self) -> float:
        """Return the physical width of one voxel.

        Returns:
            Voxel width in world units.
        """
        return self.box_size / self.resolution

    @property
    def cell_resolution(self) -> int:
        """Return the number of marching-cubes cells per axis.

        Returns:
            Number of cell intervals, equal to ``resolution - 1``.
        """
        return self.resolution - 1

    def iter_chunks(self) -> Iterator[Chunk]:
        """Yield all chunks with owned marching-cubes cell ranges.

        Yields:
            Chunk definitions covering the whole virtual grid.
        """
        axis_chunks = _partition_axis(self.cell_resolution, self.nchunks)
        for ix, x in enumerate(axis_chunks):
            for iy, y in enumerate(axis_chunks):
                for iz, z in enumerate(axis_chunks):
                    yield Chunk(index=(ix, iy, iz), x=x, y=y, z=z)


def chunk_samples(
    grid: VirtualGrid,
    chunk: Chunk,
    halo: int = 0,
) -> ChunkSamples:
    """Return owned and halo-expanded sample ranges for a chunk.

    Args:
        grid: Parent virtual grid.
        chunk: Chunk whose sample ranges should be computed.
        halo: Number of extra samples to include on each side.

    Returns:
        Sample ranges for the requested chunk.

    Raises:
        ValueError: If ``halo`` is negative.
    """
    if halo < 0:
        raise ValueError("halo must be >= 0")

    # Compute the exact owned sample interval first, then enlarge it by the
    # halo while clamping to the global grid bounds.
    owned_start = np.array(
        [chunk.x.sample_start, chunk.y.sample_start, chunk.z.sample_start],
        dtype=np.int64,
    )
    owned_stop = np.array(
        [chunk.x.sample_stop, chunk.y.sample_stop, chunk.z.sample_stop],
        dtype=np.int64,
    )
    start = np.maximum(0, owned_start - halo)
    stop = np.minimum(grid.resolution, owned_stop + halo)

    return ChunkSamples(
        index=chunk.index,
        owned_start=owned_start,
        owned_stop=owned_stop,
        start=start,
        stop=stop,
    )


def chunk_origin(grid: VirtualGrid, samples: ChunkSamples) -> np.ndarray:
    """Return the world-space origin of a chunk sample block.

    Args:
        grid: Parent virtual grid.
        samples: Sample ranges describing the chunk block.

    Returns:
        World-space origin of the expanded chunk block.
    """
    return grid.origin + samples.start.astype(np.float64) * grid.voxel_size


def chunk_world_bounds(grid: VirtualGrid, chunk: Chunk) -> HardChunkBounds:
    """Return the hard bounds of a chunk with no halo expansion.

    Args:
        grid: Parent virtual grid.
        chunk: Chunk whose hard bounds should be computed.

    Returns:
        Hard chunk bounds in sample, local, and world coordinates.
    """
    # Convert the chunk's sample ownership into local and world-space bounds so
    # later code can work in whichever coordinate frame is most convenient.
    sample_start = np.array(
        [chunk.x.sample_start, chunk.y.sample_start, chunk.z.sample_start],
        dtype=np.int64,
    )
    sample_stop = np.array(
        [chunk.x.sample_stop, chunk.y.sample_stop, chunk.z.sample_stop],
        dtype=np.int64,
    )
    local_start = sample_start.astype(np.float64) * grid.voxel_size
    local_stop = sample_stop.astype(np.float64) * grid.voxel_size
    world_start = grid.origin + local_start
    world_stop = grid.origin + sample_stop.astype(np.float64) * grid.voxel_size
    return HardChunkBounds(
        index=chunk.index,
        nchunks=grid.nchunks,
        sample_start=sample_start,
        sample_stop=sample_stop,
        local_start=local_start,
        local_stop=local_stop,
        world_start=world_start,
        world_stop=world_stop,
    )


def iter_hard_chunk_bounds(grid: VirtualGrid) -> Iterator[HardChunkBounds]:
    """Yield all hard chunk bounds for the virtual grid.

    Args:
        grid: Parent virtual grid.

    Yields:
        Hard bounds for each chunk in ``x, y, z`` nested-loop order.
    """
    for chunk in grid.iter_chunks():
        yield chunk_world_bounds(grid, chunk)


def expand_hard_chunk_bounds(
    grid: VirtualGrid,
    chunk_bounds: HardChunkBounds,
    *,
    overlap_voxels: int,
) -> HardChunkBounds:
    """Expand a hard chunk by a small overlap on interior faces only.

    Args:
        grid: Parent virtual grid.
        chunk_bounds: Hard bounds to expand.
        overlap_voxels: Number of voxels to add on interior faces.

    Returns:
        Expanded hard chunk bounds.

    Raises:
        ValueError: If ``overlap_voxels`` is negative.
    """
    if overlap_voxels < 0:
        raise ValueError("overlap_voxels must be >= 0")
    if overlap_voxels == 0:
        return chunk_bounds

    sample_start = chunk_bounds.sample_start.copy()
    sample_stop = chunk_bounds.sample_stop.copy()

    # Expand only interior faces so the global domain boundary remains fixed.
    for axis in range(3):
        if chunk_bounds.index[axis] > 0:
            sample_start[axis] = max(0, sample_start[axis] - overlap_voxels)
        if chunk_bounds.index[axis] < grid.nchunks - 1:
            sample_stop[axis] = min(
                grid.resolution,
                sample_stop[axis] + overlap_voxels,
            )

    local_start = sample_start.astype(np.float64) * grid.voxel_size
    local_stop = sample_stop.astype(np.float64) * grid.voxel_size
    world_start = grid.origin + local_start
    world_stop = grid.origin + local_stop
    return HardChunkBounds(
        index=chunk_bounds.index,
        nchunks=grid.nchunks,
        sample_start=sample_start,
        sample_stop=sample_stop,
        local_start=local_start,
        local_stop=local_stop,
        world_start=world_start,
        world_stop=world_stop,
    )


def crop_grid_to_chunk_bounds(
    expanded_grid: np.ndarray,
    expanded_bounds: HardChunkBounds,
    target_bounds: HardChunkBounds,
) -> np.ndarray:
    """Crop an expanded hard-chunk grid back to target bounds.

    Args:
        expanded_grid: Grid sampled on the expanded chunk bounds.
        expanded_bounds: Bounds used to create ``expanded_grid``.
        target_bounds: Bounds to crop back to.

    Returns:
        Cropped grid covering only ``target_bounds``.
    """
    # Convert the target sample interval into indices relative to the expanded
    # chunk grid, then slice out just the owned block.
    start = target_bounds.sample_start - expanded_bounds.sample_start
    stop = start + (target_bounds.sample_stop - target_bounds.sample_start)
    return expanded_grid[
        start[0] : stop[0],
        start[1] : stop[1],
        start[2] : stop[2],
    ]


def select_particles_in_hard_chunk(
    coordinates: np.ndarray,
    chunk_bounds: HardChunkBounds,
    *,
    smoothing_lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Select particles that intersect a hard chunk boundary box.

    When smoothing lengths are supplied, the selection includes any particle
    whose support overlaps the hard chunk box.

    Args:
        coordinates: Particle coordinates with shape ``(N, 3)``.
        chunk_bounds: Hard bounds used for the selection.
        smoothing_lengths: Optional per-particle support radii.

    Returns:
        Integer particle indices selected for the chunk.

    Raises:
        ValueError: If array shapes are invalid.
    """
    # Work with NumPy arrays throughout so the overlap test is fully
    # vectorised.
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")

    lower = chunk_bounds.hard_local_start
    upper = chunk_bounds.hard_local_stop

    if smoothing_lengths is None:
        # In the non-smoothed path, inclusion is just a point-in-box test on
        # the hard owned chunk domain.
        mask = np.all(coords_arr >= lower, axis=1) & np.all(
            coords_arr < upper,
            axis=1,
        )
        return np.nonzero(mask)[0].astype(np.int64)

    h = np.asarray(smoothing_lengths, dtype=np.float64)
    if h.shape != (coords_arr.shape[0],):
        raise ValueError("smoothing_lengths must have shape (N,)")

    # In the smoothed path, include any particle whose support intersects the
    # chunk box rather than only those whose centre lies inside it.
    mins = coords_arr - h[:, None]
    maxs = coords_arr + h[:, None]
    mask = np.all(maxs >= lower, axis=1) & np.all(mins < upper, axis=1)
    return np.nonzero(mask)[0].astype(np.int64)


def voxelize_hard_chunk(
    data: np.ndarray,
    coordinates: np.ndarray,
    chunk_bounds: HardChunkBounds,
    *,
    smoothing_lengths: np.ndarray | None = None,
    nthreads: int = 1,
) -> tuple[np.ndarray, float]:
    """Voxelize particles into a hard chunk-local grid.

    The chunk's own world-space bounds define the voxelization cube.

    Args:
        data: Per-particle scalar values.
        coordinates: Particle coordinates with shape ``(N, 3)``.
        chunk_bounds: Bounds defining the local chunk voxel lattice.
        smoothing_lengths: Optional per-particle smoothing lengths.
        nthreads: Number of threads requested for the C deposition kernel.

    Returns:
        Tuple containing the chunk-local grid and voxel size.

    Raises:
        ValueError: If the inputs are malformed.
        RuntimeError: If smoothed deposition is requested without the C
            extension.
    """
    # Validate and normalize the particle arrays once before constructing the
    # chunk-local voxel lattice.
    data_arr = np.asarray(data)
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    if data_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError("data must have shape (N,)")

    resolution = int(
        chunk_bounds.sample_stop[0] - chunk_bounds.sample_start[0]
    )
    if resolution <= 0:
        raise ValueError("chunk resolution must be > 0")

    # Shift into chunk-local coordinates so deposition happens on a small local
    # cube rather than on a global lattice.
    local_coords = coords_arr - chunk_bounds.local_start
    local_box_size = float(chunk_bounds.extent[0])
    voxel_size = local_box_size / resolution
    local_grid = np.zeros(
        (resolution, resolution, resolution), dtype=np.float64
    )

    # Convert local coordinates into local voxel indices for either point or
    # smoothed deposition.
    scaled = local_coords / local_box_size * resolution
    vox_indices = np.floor(scaled).astype(np.int64)

    if smoothing_lengths is None:
        # The simple path deposits each particle into exactly one voxel.
        vox_indices = np.clip(vox_indices, 0, resolution - 1)
        np.add.at(
            local_grid,
            (vox_indices[:, 0], vox_indices[:, 1], vox_indices[:, 2]),
            data_arr,
        )
        return local_grid, voxel_size

    if _voxelize is None:
        raise RuntimeError(
            "Chunk-local smoothing requires the _voxelize C extension."
        )

    smoothing_arr = np.asarray(smoothing_lengths, dtype=np.float64)
    if smoothing_arr.shape != (coords_arr.shape[0],):
        raise ValueError("smoothing_lengths must have shape (N,)")

    # Convert smoothing lengths into voxel units before calling the C kernel.
    if voxel_size > 0:
        smoothing_lengths_vox = (smoothing_arr / voxel_size).astype(np.int64)
    else:
        smoothing_lengths_vox = np.zeros_like(smoothing_arr, dtype=np.int64)

    _voxelize.box_deposition_local(
        local_grid,
        data_arr.astype(np.float64),
        vox_indices.astype(np.int64),
        smoothing_lengths_vox,
        resolution,
        resolution,
        resolution,
        int(nthreads),
    )
    return local_grid, voxel_size


def mesh_hard_chunk_sdf(
    chunk_grid: np.ndarray,
    chunk_bounds: HardChunkBounds,
    *,
    threshold: float,
    closing_radius: int = 1,
) -> list[Mesh]:
    """Generate watertight mesh(es) for one hard chunk via local SDF.

    Returned meshes are placed in global coordinates using the chunk's world
    origin.

    Args:
        chunk_grid: Chunk-local scalar field.
        chunk_bounds: Hard bounds describing the chunk placement.
        threshold: Isovalue used for surface extraction.
        closing_radius: Binary closing radius passed through to SDF meshing.

    Returns:
        Extracted chunk meshes in world coordinates.
    """
    voxel_size = float(chunk_bounds.extent[0] / chunk_bounds.shape[0])
    # Reuse the dense SDF meshing helper so chunk and dense extraction stay as
    # consistent as possible.
    try:
        meshes = voxels_to_stl_via_sdf(
            chunk_grid,
            threshold=threshold,
            closing_radius=closing_radius,
            split_islands=True,
            voxel_size=voxel_size,
        )
    except ValueError as exc:
        if "No meshes created via SDF" in str(exc):
            return []
        raise

    # Move each local chunk mesh into the global world-space coordinate frame.
    world_origin = chunk_bounds.world_start
    for mesh in meshes:
        mesh.vertices[:] = mesh.vertices + world_origin
    return meshes


def clip_mesh_to_hard_chunk(mesh: Mesh, chunk_bounds: HardChunkBounds) -> Mesh:
    """Clip a mesh to the exact hard chunk box and cap the cut faces.

    Args:
        mesh: Mesh to clip.
        chunk_bounds: Owned hard chunk bounds.

    Returns:
        Clipped mesh.

    Raises:
        ValueError: If clipping removes all geometry.
    """
    extents = chunk_bounds.hard_world_stop - chunk_bounds.hard_world_start
    center = 0.5 * (
        chunk_bounds.hard_world_start + chunk_bounds.hard_world_stop
    )
    # Use a watertight box intersection so the clipped chunk remains closed.
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)

    clipped = trimesh.boolean.intersection(
        [mesh.to_trimesh(), box],
        engine="blender",
        check_volume=True,
    )
    if clipped is None or len(clipped.faces) == 0:
        raise ValueError("Chunk clipping removed all mesh geometry")

    clipped.process()
    clipped.fix_normals()
    return Mesh(mesh=clipped)


def generate_hard_chunk_meshes(
    data: np.ndarray,
    coordinates: np.ndarray,
    smoothing_lengths: np.ndarray | None,
    grid: VirtualGrid,
    *,
    threshold: float,
    preprocess: str,
    clip_halos: float | None,
    gaussian_sigma: float,
    nthreads: int = 1,
    overlap_voxels: int = 0,
    clip_to_bounds: bool = False,
) -> list[tuple[HardChunkBounds, list[Mesh]]]:
    """Generate per-chunk meshes for the hard-boundary chunk pipeline.

    Args:
        data: Per-particle scalar values.
        coordinates: Particle coordinates with shape ``(N, 3)``.
        smoothing_lengths: Optional per-particle smoothing lengths.
        grid: Parent virtual grid.
        threshold: Isovalue used for surface extraction.
        preprocess: Preprocessing mode applied to each chunk field.
        clip_halos: Optional clipping percentile.
        gaussian_sigma: Gaussian smoothing width in voxel units.
        nthreads: Number of worker threads to use across chunks.
        overlap_voxels: Additional overlap retained around chunk boundaries.
        clip_to_bounds: Whether to clip the extracted mesh back to the hard
            chunk box.

    Returns:
        List of ``(bounds, meshes)`` pairs for non-empty chunks.

    Raises:
        ValueError: If the input array shapes or thread count are invalid.
    """
    total_start = time.perf_counter()
    pass_label = "Chunk pass"
    # Normalize the particle arrays once so the per-chunk loop can stay focused
    # on chunk-specific work.
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    data_arr = np.asarray(data)
    if data_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError("data must have shape (N,)")

    if smoothing_lengths is None:
        smoothing_arr = None
    else:
        smoothing_arr = np.asarray(smoothing_lengths, dtype=np.float64)
        if smoothing_arr.shape != (coords_arr.shape[0],):
            raise ValueError("smoothing_lengths must have shape (N,)")
    if nthreads < 1:
        raise ValueError("nthreads must be >= 1")
    deposition_threads = 1

    halo_voxels = (
        chunk_halo_voxels(gaussian_sigma) if gaussian_sigma > 0 else 0
    )

    def _process_chunk(
        chunk_bounds: HardChunkBounds,
    ) -> dict[str, object] | None:
        """Process one hard chunk and return its timing/result summary.

        Args:
            chunk_bounds: Chunk bounds to process.

        Returns:
            Summary dictionary for the processed chunk, or ``None`` if the
            chunk has no contributing particles.
        """
        effective_bounds = expand_hard_chunk_bounds(
            grid,
            chunk_bounds,
            overlap_voxels=overlap_voxels + halo_voxels,
        )
        chunk_start = time.perf_counter()

        # Select only the particles that influence this chunk's effective
        # support region before any local voxelization work.
        particle_indices = select_particles_in_hard_chunk(
            coords_arr,
            effective_bounds,
            smoothing_lengths=smoothing_arr,
        )
        if particle_indices.size == 0:
            return {
                "bounds": chunk_bounds,
                "particles": 0,
                "meshes": [],
                "voxelize_time": 0.0,
                "preprocess_time": 0.0,
                "meshing_time": 0.0,
                "elapsed": time.perf_counter() - chunk_start,
                "status": "skipped",
            }

        voxelize_start = time.perf_counter()
        chunk_grid, _voxel_size = voxelize_hard_chunk(
            data_arr[particle_indices],
            coords_arr[particle_indices],
            effective_bounds,
            smoothing_lengths=(
                None
                if smoothing_arr is None
                else smoothing_arr[particle_indices]
            ),
            nthreads=deposition_threads,
        )
        voxelize_time = time.perf_counter() - voxelize_start

        preprocess_start = time.perf_counter()
        chunk_grid = preprocess_chunk_grid(
            chunk_grid,
            preprocess=preprocess,
            clip_halos=clip_halos,
            gaussian_sigma=gaussian_sigma,
        )
        if halo_voxels > 0:
            # Preprocess on the halo-expanded field, then crop back so the mesh
            # sees the same interior values as the dense path would.
            chunk_grid = crop_grid_to_chunk_bounds(
                chunk_grid,
                effective_bounds,
                expand_hard_chunk_bounds(
                    grid,
                    chunk_bounds,
                    overlap_voxels=overlap_voxels,
                ),
            )
            effective_bounds = expand_hard_chunk_bounds(
                grid,
                chunk_bounds,
                overlap_voxels=overlap_voxels,
            )
        preprocess_time = time.perf_counter() - preprocess_start

        meshing_start = time.perf_counter()
        # Extract meshes only after chunk-local preprocessing is complete.
        meshes = mesh_hard_chunk_sdf(
            chunk_grid,
            effective_bounds,
            threshold=threshold,
        )
        if clip_to_bounds:
            meshes = [
                clip_mesh_to_hard_chunk(mesh, chunk_bounds) for mesh in meshes
            ]
        meshing_time = time.perf_counter() - meshing_start

        status = "meshed" if meshes else "empty"
        return {
            "bounds": chunk_bounds,
            "particles": int(particle_indices.size),
            "meshes": meshes,
            "voxelize_time": voxelize_time,
            "preprocess_time": preprocess_time,
            "meshing_time": meshing_time,
            "elapsed": time.perf_counter() - chunk_start,
            "status": status,
        }

    chunk_meshes: list[tuple[HardChunkBounds, list[Mesh]]] = []
    voxelize_total = 0.0
    preprocess_total = 0.0
    meshing_total = 0.0
    processed_chunks = 0
    nonempty_chunks = 0
    emitted_meshes = 0
    all_bounds = list(iter_hard_chunk_bounds(grid))
    processed_chunks = len(all_bounds)

    # Preserve deterministic output ordering by collecting results in the same
    # chunk order regardless of whether execution is serial or parallel.
    if nthreads == 1:
        results = [_process_chunk(chunk_bounds) for chunk_bounds in all_bounds]
    else:
        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            results = list(executor.map(_process_chunk, all_bounds))

    for result in results:
        if result is None:
            continue
        chunk_bounds = result["bounds"]
        particle_count = int(result["particles"])
        meshes = result["meshes"]
        voxelize_total += float(result["voxelize_time"])
        preprocess_total += float(result["preprocess_time"])
        meshing_total += float(result["meshing_time"])
        elapsed = float(result["elapsed"])
        status = str(result["status"])

        if status == "skipped":
            print(
                f"{pass_label:>10} {chunk_bounds.index}: "
                "0 particles -> skipped"
            )
            continue

        if meshes:
            chunk_meshes.append((chunk_bounds, meshes))
            nonempty_chunks += 1
            emitted_meshes += len(meshes)
            print(
                f"{pass_label:>10} {chunk_bounds.index}: "
                f"{particle_count} particles -> "
                f"{len(meshes)} mesh(es) in {elapsed:.3f} s"
            )
        else:
            print(
                f"{pass_label:>10} {chunk_bounds.index}: "
                f"{particle_count} particles -> no mesh"
            )

    print("Chunk summary:")
    print(f"  processed chunks: {processed_chunks}")
    print(f"  non-empty chunks: {nonempty_chunks}")
    print(f"  emitted meshes:   {emitted_meshes}")
    print(f"  voxelization:     {voxelize_total:.3f} s total")
    print(f"  preprocessing:    {preprocess_total:.3f} s total")
    print(f"  sdf meshing:      {meshing_total:.3f} s total")
    print(f"  chunk pass total: {time.perf_counter() - total_start:.3f} s")
    return chunk_meshes


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
    if not chunk_meshes:
        raise ValueError("No chunk meshes to union")

    assembled_parts: list[trimesh.Trimesh] = []
    # Assign each triangle to exactly one owning chunk using centroid tests on
    # the hard chunk boxes, then weld the surviving seams.
    for bounds, meshes in chunk_meshes:
        for mesh in meshes:
            trimesh_mesh = mesh.to_trimesh().copy()
            centroids = trimesh_mesh.triangles_center
            keep = np.ones(len(trimesh_mesh.faces), dtype=bool)
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

    unioned = trimesh.util.concatenate(assembled_parts)
    unioned.merge_vertices()
    unioned.update_faces(unioned.unique_faces())
    unioned.update_faces(unioned.nondegenerate_faces())
    unioned.remove_unreferenced_vertices()
    # Let trimesh close simple boundary cycles first. This handles more general
    # seam graphs than the custom planar-loop fallback alone.
    repair.fill_holes(unioned)
    unioned = _cap_planar_boundary_loops(unioned)
    unioned.fix_normals()
    return Mesh(mesh=unioned)


def _mesh_boundary_loops(mesh: trimesh.Trimesh) -> list[list[int]]:
    """Return boundary vertex loops for a mesh with open boundaries.

    Args:
        mesh: Mesh whose open boundary loops should be extracted.

    Returns:
        Boundary loops as lists of vertex indices.

    Raises:
        ValueError: If the boundary graph is not a collection of simple loops.
    """
    # Count edge usages so open boundary edges can be isolated from manifold
    # interior edges.
    edge_counts: Counter[tuple[int, int]] = Counter()
    for tri in mesh.faces:
        a, b, c = tri
        edge_counts[tuple(sorted((a, b)))] += 1
        edge_counts[tuple(sorted((b, c)))] += 1
        edge_counts[tuple(sorted((a, c)))] += 1

    # Boundary edges appear exactly once across all triangles.
    boundary_edges = [
        edge for edge, count in edge_counts.items() if count == 1
    ]
    if not boundary_edges:
        return []

    # Convert the boundary edges into a simple graph so they can be walked as
    # vertex loops.
    adjacency: defaultdict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

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


def _polygon_area_2d(points: np.ndarray) -> float:
    """Return the signed area of a 2D polygon.

    Args:
        points: Polygon vertices in 2D.

    Returns:
        Signed polygon area.
    """
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def _point_in_triangle_2d(
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

    def _sign(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        # Compute the signed area of the triangle formed with ``point`` as one
        # corner. The sign pattern tells us which side of each edge the point
        # is on.
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (
            p1[1] - p3[1]
        )

    d1 = _sign(point, a, b)
    d2 = _sign(point, b, c)
    d3 = _sign(point, c, a)
    has_neg = (d1 < -1e-12) or (d2 < -1e-12) or (d3 < -1e-12)
    has_pos = (d1 > 1e-12) or (d2 > 1e-12) or (d3 > 1e-12)
    return not (has_neg and has_pos)


def _triangulate_loop(
    loop: list[int], vertices: np.ndarray
) -> list[list[int]]:
    """Triangulate one small planar boundary loop with ear clipping.

    Args:
        loop: Boundary loop as vertex indices into ``vertices``.
        vertices: Full vertex array.

    Returns:
        Triangles expressed as vertex-index triplets.

    Raises:
        ValueError: If ear clipping cannot triangulate the loop.
    """
    points = vertices[loop]
    spans = points.max(axis=0) - points.min(axis=0)
    flat_axis = int(np.argmin(spans))
    planar = points[:, [axis for axis in range(3) if axis != flat_axis]]

    # Project to the dominant plane and triangulate in 2D to keep the cap local
    # and cheap.
    indices = list(range(len(loop)))
    ccw = _polygon_area_2d(planar) > 0
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
                _point_in_triangle_2d(planar[j], a, b, c)
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


def _cap_planar_boundary_loops(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Cap small planar seam loops left after chunk assembly.

    Args:
        mesh: Assembled mesh that may contain tiny planar seam holes.

    Returns:
        Mesh with eligible seam loops capped.
    """
    # If trimesh already managed to close the seam holes, keep the mesh as-is.
    if mesh.is_watertight:
        return mesh

    # Only fall back to the custom planar capper for the remaining small seam
    # holes that trimesh repair did not handle.
    try:
        loops = _mesh_boundary_loops(mesh)
    except ValueError:
        return mesh
    if not loops:
        return mesh

    cap_faces: list[list[int]] = []
    # Only cap tiny planar seam holes introduced by chunk ownership. Larger or
    # non-planar openings are left untouched so the code does not guess.
    for loop in loops:
        points = mesh.vertices[loop]
        spans = points.max(axis=0) - points.min(axis=0)
        # Only patch tiny seam holes introduced by chunk assembly. Leave other
        # open boundaries untouched instead of guessing.
        if np.min(spans) > 1e-8 or len(loop) > 32:
            return mesh
        cap_faces.extend(_triangulate_loop(loop, mesh.vertices))

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


def generate_chunk_grid(
    data: np.ndarray,
    coordinates: np.ndarray,
    grid: VirtualGrid,
    chunk: Chunk,
    *,
    halo: int = 0,
    smoothing_lengths: np.ndarray | None = None,
) -> tuple[np.ndarray, ChunkSamples]:
    """Generate a chunk-local scalar field without allocating the full grid.

    Supports both point deposition and smoothed box deposition into a local
    chunk-sized grid.

    Args:
        data: Per-particle scalar values.
        coordinates: Particle coordinates with shape ``(N, 3)``.
        grid: Parent virtual grid.
        chunk: Chunk to voxelize.
        halo: Number of halo voxels to include.
        smoothing_lengths: Optional per-particle smoothing lengths.

    Returns:
        Tuple containing the chunk-local grid and its sample ranges.
    """
    # Normalize inputs before per-chunk processing so the inner loop only deals
    # with chunk logic, not repeated validation.
    data_arr = np.asarray(data)
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    if data_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError("data must have shape (N,)")

    samples = chunk_samples(grid, chunk, halo=halo)
    local_grid = np.zeros(samples.shape, dtype=np.float64)
    if coords_arr.shape[0] == 0:
        return local_grid, samples

    voxel_indices = particle_voxel_indices(coords_arr, grid)

    if smoothing_lengths is None:
        # The point-deposition path only keeps particles whose voxel indices
        # lie inside the expanded chunk block.
        mask = np.all(
            (voxel_indices >= samples.start) & (voxel_indices < samples.stop),
            axis=1,
        )
        if not np.any(mask):
            return local_grid, samples

        local_indices = voxel_indices[mask] - samples.start
        np.add.at(
            local_grid,
            (
                local_indices[:, 0],
                local_indices[:, 1],
                local_indices[:, 2],
            ),
            data_arr[mask],
        )
        return local_grid, samples

    if _voxelize is None:
        raise RuntimeError(
            "Chunk-local smoothing requires the _voxelize C extension."
        )

    smoothing_arr = np.asarray(smoothing_lengths, dtype=np.float64)
    if smoothing_arr.shape != (coords_arr.shape[0],):
        raise ValueError("smoothing_lengths must have shape (N,)")

    if grid.voxel_size > 0:
        smoothing_lengths_vox = (smoothing_arr / grid.voxel_size).astype(
            np.int64
        )
    else:
        smoothing_lengths_vox = np.zeros_like(smoothing_arr, dtype=np.int64)

    # For smoothed deposition, keep any particle whose support intersects the
    # chunk sample block.
    support_mins, support_maxs = particle_support_bounds(
        voxel_indices,
        smoothing_lengths_vox,
        grid.resolution,
    )
    mask = np.all(support_maxs >= samples.start, axis=1) & np.all(
        support_mins < samples.stop,
        axis=1,
    )
    if not np.any(mask):
        return local_grid, samples

    local_indices = voxel_indices[mask] - samples.start
    _voxelize.box_deposition_local(
        local_grid,
        data_arr[mask].astype(np.float64),
        local_indices.astype(np.int64),
        smoothing_lengths_vox[mask],
        local_grid.shape[0],
        local_grid.shape[1],
        local_grid.shape[2],
        1,
    )
    return local_grid, samples


def particle_voxel_indices(
    coordinates: np.ndarray,
    grid: VirtualGrid,
) -> np.ndarray:
    """Map particle coordinates onto the virtual global voxel lattice.

    Args:
        coordinates: Particle coordinates with shape ``(N, 3)``.
        grid: Parent virtual grid.

    Returns:
        Integer voxel indices clipped to the grid domain.
    """
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")

    voxel_indices = np.floor(
        (coords_arr - grid.origin) / grid.voxel_size
    ).astype(np.int64)
    return np.clip(voxel_indices, 0, grid.resolution - 1)


def particle_support_bounds(
    voxel_indices: np.ndarray,
    smoothing_lengths_vox: np.ndarray | None,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return inclusive voxel-index support bounds per particle.

    Args:
        voxel_indices: Central voxel index of each particle.
        smoothing_lengths_vox: Optional support radii in voxel units.
        resolution: Grid resolution per axis.

    Returns:
        Tuple of minimum and maximum voxel indices for each particle.
    """
    # Work with integer voxel coordinates because the support computation is
    # purely lattice-based.
    vox = np.asarray(voxel_indices, dtype=np.int64)
    if vox.ndim != 2 or vox.shape[1] != 3:
        raise ValueError("voxel_indices must have shape (N, 3)")
    if resolution < 1:
        raise ValueError("resolution must be >= 1")

    if smoothing_lengths_vox is None:
        mins = vox.copy()
        maxs = vox.copy()
    else:
        h = np.asarray(smoothing_lengths_vox, dtype=np.int64)
        if h.shape != (vox.shape[0],):
            raise ValueError("smoothing_lengths_vox must have shape (N,)")
        mins = vox - h[:, None]
        maxs = vox + h[:, None]

    mins = np.clip(mins, 0, resolution - 1)
    maxs = np.clip(maxs, 0, resolution - 1)
    return mins, maxs


def assign_particles_to_chunks(
    coordinates: np.ndarray,
    grid: VirtualGrid,
    *,
    halo: int = 0,
    smoothing_lengths_vox: np.ndarray | None = None,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Assign particles to every chunk whose sample block they influence.

    Args:
        coordinates: Particle coordinates with shape ``(N, 3)``.
        grid: Parent virtual grid.
        halo: Additional halo samples per chunk.
        smoothing_lengths_vox: Optional support radii in voxel units.

    Returns:
        Mapping from chunk index to particle indices.
    """
    if halo < 0:
        raise ValueError("halo must be >= 0")
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")

    n_particles = coords_arr.shape[0]
    # Precompute the chunk sample intervals along each axis so the batch loop
    # can use searchsorted instead of nested per-particle Python logic.
    axis_chunks = list(_partition_axis(grid.cell_resolution, grid.nchunks))
    axis_starts = np.maximum(
        0,
        np.array([chunk.sample_start for chunk in axis_chunks], dtype=np.int32)
        - halo,
    )
    axis_stops = np.minimum(
        grid.resolution,
        np.array([chunk.sample_stop for chunk in axis_chunks], dtype=np.int32)
        + halo,
    )

    if smoothing_lengths_vox is None:
        smoothing_arr = None
    else:
        smoothing_arr = np.asarray(smoothing_lengths_vox, dtype=np.int32)
        if smoothing_arr.shape != (n_particles,):
            raise ValueError("smoothing_lengths_vox must have shape (N,)")

    assignments: dict[tuple[int, int, int], list[np.ndarray]] = {
        chunk.index: [] for chunk in grid.iter_chunks()
    }
    batch_size = 1_000_000

    # Process particles in large batches to avoid giant temporary arrays while
    # keeping the assignment vectorised.
    for batch_start in range(0, n_particles, batch_size):
        batch_stop = min(batch_start + batch_size, n_particles)
        batch = coords_arr[batch_start:batch_stop]

        vox = np.floor((batch - grid.origin) / grid.voxel_size).astype(
            np.int32
        )
        vox = np.clip(vox, 0, grid.resolution - 1)

        if smoothing_arr is None:
            min_x = max_x = vox[:, 0]
            min_y = max_y = vox[:, 1]
            min_z = max_z = vox[:, 2]
        else:
            h = smoothing_arr[batch_start:batch_stop]
            min_x = np.clip(vox[:, 0] - h, 0, grid.resolution - 1)
            max_x = np.clip(vox[:, 0] + h, 0, grid.resolution - 1)
            min_y = np.clip(vox[:, 1] - h, 0, grid.resolution - 1)
            max_y = np.clip(vox[:, 1] + h, 0, grid.resolution - 1)
            min_z = np.clip(vox[:, 2] - h, 0, grid.resolution - 1)
            max_z = np.clip(vox[:, 2] + h, 0, grid.resolution - 1)

        x0 = np.searchsorted(axis_stops, min_x, side="right")
        x1 = np.searchsorted(axis_starts, max_x, side="left")
        y0 = np.searchsorted(axis_stops, min_y, side="right")
        y1 = np.searchsorted(axis_starts, max_y, side="left")
        z0 = np.searchsorted(axis_stops, min_z, side="right")
        z1 = np.searchsorted(axis_starts, max_z, side="left")

        x0 = np.clip(x0, 0, grid.nchunks - 1)
        x1 = np.clip(x1, 0, grid.nchunks - 1)
        y0 = np.clip(y0, 0, grid.nchunks - 1)
        y1 = np.clip(y1, 0, grid.nchunks - 1)
        z0 = np.clip(z0, 0, grid.nchunks - 1)
        z1 = np.clip(z1, 0, grid.nchunks - 1)

        particle_ids = np.arange(batch_start, batch_stop, dtype=np.int64)
        for ix in range(grid.nchunks):
            xmask = (x0 <= ix) & (x1 >= ix)
            if not np.any(xmask):
                continue
            for iy in range(grid.nchunks):
                xymask = xmask & (y0 <= iy) & (y1 >= iy)
                if not np.any(xymask):
                    continue
                for iz in range(grid.nchunks):
                    mask = xymask & (z0 <= iz) & (z1 >= iz)
                    if np.any(mask):
                        assignments[(ix, iy, iz)].append(particle_ids[mask])

    return {
        chunk_index: (
            np.concatenate(index_lists)
            if index_lists
            else np.empty(0, dtype=np.int64)
        )
        for chunk_index, index_lists in assignments.items()
    }


def preprocess_chunk_grid(
    grid: np.ndarray,
    *,
    preprocess: str,
    clip_halos: float | None,
    gaussian_sigma: float,
) -> np.ndarray:
    """Apply scalar-field preprocessing to a chunk-local grid.

    Args:
        grid: Chunk-local scalar field.
        preprocess: Preprocessing mode.
        clip_halos: Optional clipping percentile.
        gaussian_sigma: Gaussian smoothing width in voxel units.

    Returns:
        Preprocessed chunk-local field.

    Raises:
        ValueError: If the preprocess mode is unknown or sigma is negative.
    """
    # Always work on a float64 array so the downstream filters behave
    # consistently across input dtypes.
    out = np.asarray(grid, dtype=np.float64)

    if clip_halos is not None:
        out = process_remove_halos(out, threshold_percentile=clip_halos)

    if preprocess == "log":
        out = process_log_scale(out)
    elif preprocess == "filaments":
        out = process_filament_filter(out)
    elif preprocess != "none":
        raise ValueError(f"Unknown preprocess mode: {preprocess}")

    if gaussian_sigma < 0:
        raise ValueError("gaussian_sigma must be >= 0")
    out = process_gaussian_smoothing(out, sigma=gaussian_sigma)
    return out


def chunk_sdf_to_mesh(
    chunk_grid: np.ndarray,
    samples: ChunkSamples,
    grid: VirtualGrid,
    *,
    threshold: float,
) -> Mesh | None:
    """Convert a chunk-local scalar field to an SDF mesh.

    Args:
        chunk_grid: Chunk-local scalar field including halo.
        samples: Sample ranges describing the owned and halo regions.
        grid: Parent virtual grid.
        threshold: Isovalue used for extraction.

    Returns:
        Extracted mesh, or ``None`` if the owned region is empty.
    """
    # Skip meshing early if the chunk never crosses the requested isovalue.
    full_mask = chunk_grid > threshold
    if not np.any(full_mask):
        return None

    # Build the signed distance field on the full halo-expanded chunk, then
    # crop to the owned region for surface extraction.
    d_in = ndimage.distance_transform_edt(full_mask)
    d_out = ndimage.distance_transform_edt(~full_mask)
    sdf = d_in.astype(np.float64) - d_out.astype(np.float64)

    # Mesh only the chunk's owned region so adjacent chunks do not both emit
    # the same interior cells.
    owned = chunk_grid[
        samples.owned_local_start[0] : samples.owned_local_stop[0],
        samples.owned_local_start[1] : samples.owned_local_stop[1],
        samples.owned_local_start[2] : samples.owned_local_stop[2],
    ]
    if not np.any(owned > threshold):
        return None

    owned_sdf = sdf[
        samples.owned_local_start[0] : samples.owned_local_stop[0],
        samples.owned_local_start[1] : samples.owned_local_stop[1],
        samples.owned_local_start[2] : samples.owned_local_stop[2],
    ]

    verts, faces, normals, _ = measure.marching_cubes(
        owned_sdf,
        level=0.0,
        spacing=(grid.voxel_size, grid.voxel_size, grid.voxel_size),
        gradient_direction="ascent",
    )

    if faces.size == 0:
        return None

    verts += (
        chunk_origin(grid, samples)
        + samples.owned_local_start.astype(np.float64) * grid.voxel_size
    )
    return Mesh(vertices=verts, faces=faces, vertex_normals=normals)


def combine_chunk_meshes(meshes: list[Mesh]) -> Mesh:
    """Combine chunk mesh fragments into one assembled multi-body mesh.

    Args:
        meshes: Mesh fragments to concatenate.

    Returns:
        Combined multi-body mesh.

    Raises:
        ValueError: If no meshes are supplied.
    """
    if not meshes:
        raise ValueError("No chunk meshes to combine")
    if len(meshes) == 1:
        return meshes[0]

    combined = trimesh.util.concatenate([m.to_trimesh() for m in meshes])
    return Mesh(mesh=combined)


def keep_largest_mesh_component(mesh: Mesh) -> Mesh:
    """Keep only the largest connected component of a stitched mesh.

    Args:
        mesh: Mesh whose connected components should be filtered.

    Returns:
        Mesh containing only the largest component.

    Raises:
        ValueError: If no connected components exist.
    """
    # Work on connected components of the assembled trimesh rather than trying
    # to infer component size from disconnected faces manually.
    trimesh_mesh = mesh.to_trimesh()
    components = trimesh_mesh.split(only_watertight=False)
    if not components:
        raise ValueError("No connected mesh components found")
    if len(components) == 1:
        return mesh

    def component_size(component: trimesh.Trimesh) -> float:
        """Score a component by volume when possible, otherwise face count.

        Args:
            component: Candidate mesh component.

        Returns:
            Scalar size metric for ranking components.
        """
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


def chunk_halo_voxels(gaussian_sigma: float) -> int:
    """Return the halo size needed for chunk-local processing.

    Args:
        gaussian_sigma: Gaussian smoothing width in voxel units.

    Returns:
        Halo size in voxels.

    Raises:
        ValueError: If ``gaussian_sigma`` is negative.
    """
    if gaussian_sigma < 0:
        raise ValueError("gaussian_sigma must be >= 0")
    return max(1, int(math.ceil(4.0 * gaussian_sigma)))


def generate_chunked_mesh(
    data: np.ndarray,
    coordinates: np.ndarray,
    smoothing_lengths: np.ndarray | None,
    grid: VirtualGrid,
    *,
    threshold: float,
    preprocess: str,
    clip_halos: float | None,
    gaussian_sigma: float,
    remove_islands: bool = False,
) -> Mesh:
    """Generate a stitched mesh from chunk-local scalar fields.

    Args:
        data: Per-particle scalar values.
        coordinates: Particle coordinates with shape ``(N, 3)``.
        smoothing_lengths: Optional per-particle smoothing lengths.
        grid: Parent virtual grid.
        threshold: Isovalue used for extraction.
        preprocess: Preprocessing mode.
        clip_halos: Optional clipping percentile.
        gaussian_sigma: Gaussian smoothing width in voxel units.
        remove_islands: Whether to keep only the largest connected component.

    Returns:
        Stitched chunked mesh.

    Raises:
        ValueError: If no chunk meshes are created.
    """
    total_start = time.perf_counter()
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    data_arr = np.asarray(data)
    if data_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError("data must have shape (N,)")

    # Convert smoothing lengths to voxel units once so chunk assignment can use
    # lattice-space support bounds efficiently.
    smoothing_lengths_vox = None
    if smoothing_lengths is not None:
        smoothing_arr = np.asarray(smoothing_lengths, dtype=np.float64)
        if smoothing_arr.shape != (coords_arr.shape[0],):
            raise ValueError("smoothing_lengths must have shape (N,)")
        if grid.voxel_size > 0:
            smoothing_lengths_vox = (smoothing_arr / grid.voxel_size).astype(
                np.int64
            )
        else:
            smoothing_lengths_vox = np.zeros_like(
                smoothing_arr, dtype=np.int64
            )
    else:
        smoothing_arr = None

    # Compute chunk ownership once up front, then reuse it during per-chunk
    # deposition and meshing.
    halo = chunk_halo_voxels(gaussian_sigma)
    assign_start = time.perf_counter()
    assignments = assign_particles_to_chunks(
        coords_arr,
        grid,
        halo=halo,
        smoothing_lengths_vox=smoothing_lengths_vox,
    )
    print(
        "Chunk assignment took "
        f"{time.perf_counter() - assign_start:.3f} s "
        f"for {grid.nchunks**3} chunk(s)"
    )

    meshes: list[Mesh] = []
    deposition_total = 0.0
    preprocess_total = 0.0
    meshing_total = 0.0
    # Process each chunk independently and stitch their owned meshes at the
    # end.
    for chunk in grid.iter_chunks():
        particle_indices = assignments[chunk.index]
        if particle_indices.size == 0:
            continue

        deposition_start = time.perf_counter()
        chunk_grid, samples = generate_chunk_grid(
            data_arr[particle_indices],
            coords_arr[particle_indices],
            grid,
            chunk,
            halo=halo,
            smoothing_lengths=(
                None
                if smoothing_arr is None
                else smoothing_arr[particle_indices]
            ),
        )
        deposition_total += time.perf_counter() - deposition_start

        preprocess_start = time.perf_counter()
        chunk_grid = preprocess_chunk_grid(
            chunk_grid,
            preprocess=preprocess,
            clip_halos=clip_halos,
            gaussian_sigma=gaussian_sigma,
        )
        preprocess_total += time.perf_counter() - preprocess_start

        meshing_start = time.perf_counter()
        mesh = chunk_sdf_to_mesh(
            chunk_grid,
            samples,
            grid,
            threshold=threshold,
        )
        meshing_total += time.perf_counter() - meshing_start
        if mesh is not None:
            meshes.append(mesh)

    if not meshes:
        raise ValueError("No chunk meshes created. Check threshold and input.")

    print(f"Chunk deposition took {deposition_total:.3f} s total")
    print(f"Chunk preprocessing took {preprocess_total:.3f} s total")
    print(f"Chunk SDF meshing took {meshing_total:.3f} s total")

    stitch_start = time.perf_counter()
    combined = combine_chunk_meshes(meshes)
    print(f"Chunk stitch took {time.perf_counter() - stitch_start:.3f} s")
    if remove_islands:
        island_start = time.perf_counter()
        combined = keep_largest_mesh_component(combined)
        print(
            "Post-stitch island removal took "
            f"{time.perf_counter() - island_start:.3f} s"
        )
    print(
        f"Chunked mesh pipeline took {time.perf_counter() - total_start:.3f} s"
    )
    return combined


def _partition_axis(cell_resolution: int, nchunks: int) -> list[AxisChunk]:
    """Partition marching-cubes cells into contiguous owned chunks.

    Args:
        cell_resolution: Number of marching-cubes cells along one axis.
        nchunks: Number of chunks to create along that axis.

    Returns:
        Contiguous axis chunks that cover the full cell range.

    Raises:
        ValueError: If the partition request is invalid.
    """
    if cell_resolution < 1:
        raise ValueError("cell_resolution must be >= 1")
    if nchunks < 1:
        raise ValueError("nchunks must be >= 1")
    if nchunks > cell_resolution:
        raise ValueError("nchunks must be <= cell_resolution")

    # Distribute the remainder one cell at a time so chunk widths differ by at
    # most one cell.
    base = cell_resolution // nchunks
    remainder = cell_resolution % nchunks

    chunks: list[AxisChunk] = []
    start = 0
    for i in range(nchunks):
        width = base + (1 if i < remainder else 0)
        stop = start + width
        chunks.append(AxisChunk(cell_start=start, cell_stop=stop))
        start = stop

    return chunks
