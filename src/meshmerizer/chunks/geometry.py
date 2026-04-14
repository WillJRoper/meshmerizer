"""Chunk geometry and virtual-grid helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


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
        """Validate the virtual grid definition.

        Returns:
            ``None``. The dataclass is validated in place.

        Raises:
            ValueError: If the origin shape or grid parameters are invalid.
        """
        # Normalize the origin to one float64 vector so every later geometric
        # calculation can assume a stable representation.
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
        # Require at least one marching-cubes cell per chunk so chunk ownership
        # stays well defined.
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
        axis_chunks = partition_axis(self.cell_resolution, self.nchunks)
        for ix, x in enumerate(axis_chunks):
            for iy, y in enumerate(axis_chunks):
                for iz, z in enumerate(axis_chunks):
                    yield Chunk(index=(ix, iy, iz), x=x, y=y, z=z)


def chunk_world_bounds(grid: VirtualGrid, chunk: Chunk) -> HardChunkBounds:
    """Return the hard bounds of a chunk with no halo expansion.

    Args:
        grid: Parent virtual grid.
        chunk: Chunk whose hard bounds should be computed.

    Returns:
        Hard chunk bounds in sample, local, and world coordinates.
    """
    # Build the hard chunk bounds directly from the owned sample interval. No
    # overlap or halo is applied here.
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
    # Delegate the iteration order to ``VirtualGrid.iter_chunks`` so all chunk
    # traversals in the package stay consistent.
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
    # Expand only across interior chunk faces. Exterior faces stay fixed so the
    # chunked domain still matches the original global cube exactly.
    if overlap_voxels < 0:
        raise ValueError("overlap_voxels must be >= 0")
    if overlap_voxels == 0:
        return chunk_bounds

    sample_start = chunk_bounds.sample_start.copy()
    sample_stop = chunk_bounds.sample_stop.copy()

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
    # Convert the target chunk's sample coordinates into offsets inside the
    # expanded grid and slice the owned region back out.
    start = target_bounds.sample_start - expanded_bounds.sample_start
    stop = start + (target_bounds.sample_stop - target_bounds.sample_start)
    return expanded_grid[
        start[0] : stop[0],
        start[1] : stop[1],
        start[2] : stop[2],
    ]


def partition_axis(cell_resolution: int, nchunks: int) -> list[AxisChunk]:
    """Partition marching-cubes cells into contiguous owned chunks.

    Args:
        cell_resolution: Number of marching-cubes cells along one axis.
        nchunks: Number of chunks to create along that axis.

    Returns:
        Contiguous axis chunks that cover the full cell range.

    Raises:
        ValueError: If the partition request is invalid.
    """
    # Split the cells as evenly as possible and distribute the remainder one
    # cell at a time so neighbouring chunk widths differ by at most one cell.
    if cell_resolution < 1:
        raise ValueError("cell_resolution must be >= 1")
    if nchunks < 1:
        raise ValueError("nchunks must be >= 1")
    if nchunks > cell_resolution:
        raise ValueError("nchunks must be <= cell_resolution")

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
