"""Chunk geometry helpers for high-resolution virtual voxel grids."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

try:
    from . import _voxelize
except ImportError:
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
    """Owned marching-cubes cell interval for one axis."""

    cell_start: int
    cell_stop: int

    @property
    def sample_start(self) -> int:
        """First owned sample index for this chunk."""
        return self.cell_start

    @property
    def sample_stop(self) -> int:
        """Exclusive upper sample index for this chunk."""
        return self.cell_stop + 1


@dataclass(frozen=True)
class Chunk:
    """A single chunk in the virtual global grid."""

    index: tuple[int, int, int]
    x: AxisChunk
    y: AxisChunk
    z: AxisChunk


@dataclass(frozen=True)
class ChunkSamples:
    """Owned and halo-expanded sample ranges for a chunk."""

    index: tuple[int, int, int]
    owned_start: np.ndarray
    owned_stop: np.ndarray
    start: np.ndarray
    stop: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        """Local sample-grid shape including halo."""
        return tuple((self.stop - self.start).astype(int))

    @property
    def owned_local_start(self) -> np.ndarray:
        """Owned region start in local chunk-grid coordinates."""
        return self.owned_start - self.start

    @property
    def owned_local_stop(self) -> np.ndarray:
        """Owned region stop in local chunk-grid coordinates."""
        return self.owned_local_start + (self.owned_stop - self.owned_start)


@dataclass(frozen=True)
class HardChunkBounds:
    """Hard chunk bounds in voxel/sample and world coordinates."""

    index: tuple[int, int, int]
    sample_start: np.ndarray
    sample_stop: np.ndarray
    world_start: np.ndarray
    world_stop: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        """Chunk sample-grid shape."""
        return tuple((self.sample_stop - self.sample_start).astype(int))

    @property
    def extent(self) -> np.ndarray:
        """Chunk world-space side lengths."""
        return self.world_stop - self.world_start


@dataclass(frozen=True)
class VirtualGrid:
    """Global voxel geometry without allocating the full grid."""

    origin: np.ndarray
    box_size: float
    resolution: int
    nchunks: int

    def __post_init__(self) -> None:
        """Validate the virtual grid definition."""
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
        if self.nchunks > self.cell_resolution:
            raise ValueError(
                "nchunks must be <= resolution - 1 so every chunk owns at "
                "least one marching-cubes cell"
            )

    @property
    def voxel_size(self) -> float:
        """Physical width of one voxel."""
        return self.box_size / self.resolution

    @property
    def cell_resolution(self) -> int:
        """Number of marching-cubes cells per axis."""
        return self.resolution - 1

    def iter_chunks(self) -> Iterator[Chunk]:
        """Yield all chunks with owned marching-cubes cell ranges."""
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
    """Return owned and halo-expanded sample ranges for a chunk."""
    if halo < 0:
        raise ValueError("halo must be >= 0")

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
    """Return the world-space origin of a chunk sample block."""
    return grid.origin + samples.start.astype(np.float64) * grid.voxel_size


def chunk_world_bounds(grid: VirtualGrid, chunk: Chunk) -> HardChunkBounds:
    """Return the hard bounds of a chunk with no halo expansion."""
    sample_start = np.array(
        [chunk.x.sample_start, chunk.y.sample_start, chunk.z.sample_start],
        dtype=np.int64,
    )
    sample_stop = np.array(
        [chunk.x.sample_stop, chunk.y.sample_stop, chunk.z.sample_stop],
        dtype=np.int64,
    )
    world_start = (
        grid.origin + sample_start.astype(np.float64) * grid.voxel_size
    )
    world_stop = grid.origin + sample_stop.astype(np.float64) * grid.voxel_size
    return HardChunkBounds(
        index=chunk.index,
        sample_start=sample_start,
        sample_stop=sample_stop,
        world_start=world_start,
        world_stop=world_stop,
    )


def iter_hard_chunk_bounds(grid: VirtualGrid) -> Iterator[HardChunkBounds]:
    """Yield all hard chunk bounds for the virtual grid."""
    for chunk in grid.iter_chunks():
        yield chunk_world_bounds(grid, chunk)


def select_particles_in_hard_chunk(
    coordinates: np.ndarray,
    chunk_bounds: HardChunkBounds,
    *,
    smoothing_lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Select particles that intersect a hard chunk boundary box.

    When smoothing lengths are supplied, the chunk bounds are padded by the
    per-particle support radius so particles whose support overlaps the chunk
    are included.
    """
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")

    lower = chunk_bounds.world_start
    upper = chunk_bounds.world_stop

    if smoothing_lengths is None:
        mask = np.all(coords_arr >= lower, axis=1) & np.all(
            coords_arr < upper,
            axis=1,
        )
        return np.nonzero(mask)[0].astype(np.int64)

    h = np.asarray(smoothing_lengths, dtype=np.float64)
    if h.shape != (coords_arr.shape[0],):
        raise ValueError("smoothing_lengths must have shape (N,)")

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
    """
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

    local_coords = coords_arr - chunk_bounds.world_start
    local_box_size = float(chunk_bounds.extent[0])

    from .voxels import generate_voxel_grid

    return generate_voxel_grid(
        data=data_arr,
        coordinates=local_coords,
        resolution=resolution,
        smoothing_lengths=smoothing_lengths,
        box_size=local_box_size,
        nthreads=nthreads,
    )


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
    """
    voxel_size = float(chunk_bounds.extent[0] / chunk_bounds.shape[0])
    meshes = voxels_to_stl_via_sdf(
        chunk_grid,
        threshold=threshold,
        closing_radius=closing_radius,
        split_islands=True,
        voxel_size=voxel_size,
    )

    world_origin = chunk_bounds.world_start
    for mesh in meshes:
        mesh.vertices[:] = mesh.vertices + world_origin
    return meshes


def _drop_internal_plane_faces(
    mesh: Mesh,
    chunk_bounds: HardChunkBounds,
    interior_planes: np.ndarray,
    *,
    tol: float,
) -> Mesh:
    """Remove faces that lie exactly on interior chunk boundary planes."""
    trimesh_mesh = mesh.to_trimesh().copy()
    if len(trimesh_mesh.faces) == 0 or interior_planes.size == 0:
        return Mesh(mesh=trimesh_mesh)

    vertices = trimesh_mesh.vertices
    faces = trimesh_mesh.faces
    face_keep = np.ones(len(faces), dtype=bool)

    for axis in range(3):
        axis_planes = interior_planes[axis]
        if axis_planes.size == 0:
            continue

        face_axis = vertices[faces][:, :, axis]
        on_chunk_plane = np.isclose(
            face_axis,
            chunk_bounds.world_start[axis],
            atol=tol,
        ) | np.isclose(
            face_axis,
            chunk_bounds.world_stop[axis],
            atol=tol,
        )
        if not np.any(on_chunk_plane):
            continue

        for plane in axis_planes:
            on_plane = np.isclose(face_axis, plane, atol=tol)
            face_keep &= ~np.all(on_plane & on_chunk_plane, axis=1)

    filtered = trimesh.Trimesh(
        vertices=vertices.copy(),
        faces=faces[face_keep].copy(),
        process=False,
    )
    filtered.remove_unreferenced_vertices()
    filtered.process()
    filtered.fix_normals()
    return Mesh(mesh=filtered)


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
) -> list[tuple[HardChunkBounds, list[Mesh]]]:
    """Generate watertight meshes for each hard chunk independently."""
    total_start = time.perf_counter()
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

    chunk_meshes: list[tuple[HardChunkBounds, list[Mesh]]] = []
    voxelize_total = 0.0
    preprocess_total = 0.0
    meshing_total = 0.0
    for chunk_bounds in iter_hard_chunk_bounds(grid):
        chunk_start = time.perf_counter()
        particle_indices = select_particles_in_hard_chunk(
            coords_arr,
            chunk_bounds,
            smoothing_lengths=smoothing_arr,
        )
        if particle_indices.size == 0:
            continue

        voxelize_start = time.perf_counter()
        chunk_grid, _voxel_size = voxelize_hard_chunk(
            data_arr[particle_indices],
            coords_arr[particle_indices],
            chunk_bounds,
            smoothing_lengths=(
                None
                if smoothing_arr is None
                else smoothing_arr[particle_indices]
            ),
            nthreads=nthreads,
        )
        voxelize_total += time.perf_counter() - voxelize_start

        preprocess_start = time.perf_counter()
        chunk_grid = preprocess_chunk_grid(
            chunk_grid,
            preprocess=preprocess,
            clip_halos=clip_halos,
            gaussian_sigma=gaussian_sigma,
        )
        preprocess_total += time.perf_counter() - preprocess_start

        meshing_start = time.perf_counter()
        meshes = mesh_hard_chunk_sdf(
            chunk_grid,
            chunk_bounds,
            threshold=threshold,
        )
        meshing_total += time.perf_counter() - meshing_start
        if meshes:
            chunk_meshes.append((chunk_bounds, meshes))
            print(
                f"Chunk {chunk_bounds.index} kept {len(meshes)} mesh(es) "
                f"from {particle_indices.size} particle(s) in "
                f"{time.perf_counter() - chunk_start:.3f} s"
            )

    print(f"Hard chunk voxelization took {voxelize_total:.3f} s total")
    print(f"Hard chunk preprocessing took {preprocess_total:.3f} s total")
    print(f"Hard chunk SDF meshing took {meshing_total:.3f} s total")
    print(
        f"Hard chunk pipeline took {time.perf_counter() - total_start:.3f} s"
    )
    return chunk_meshes


def union_hard_chunk_meshes(
    chunk_meshes: list[tuple[HardChunkBounds, list[Mesh]]],
    grid: VirtualGrid,
) -> Mesh:
    """Combine hard chunk meshes after stripping internal interface faces."""
    if not chunk_meshes:
        raise ValueError("No chunk meshes to union")

    all_bounds = list(iter_hard_chunk_bounds(grid))
    interior_coords_x = np.unique([b.world_stop[0] for b in all_bounds])[:-1]
    interior_coords_y = np.unique([b.world_stop[1] for b in all_bounds])[:-1]
    interior_coords_z = np.unique([b.world_stop[2] for b in all_bounds])[:-1]
    interior_planes = np.array(
        [interior_coords_x, interior_coords_y, interior_coords_z],
        dtype=object,
    )
    tol = grid.voxel_size * 1e-3

    cleaned_meshes: list[Mesh] = []
    for bounds, meshes in chunk_meshes:
        for mesh in meshes:
            cleaned_meshes.append(
                _drop_internal_plane_faces(
                    mesh,
                    bounds,
                    interior_planes,
                    tol=tol,
                )
            )

    return combine_chunk_meshes(cleaned_meshes)


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
    """
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
    """Map particle coordinates onto the virtual global voxel lattice."""
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
    """Return inclusive voxel-index support bounds per particle."""
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
    """Assign particles to every chunk whose sample block they influence."""
    if halo < 0:
        raise ValueError("halo must be >= 0")
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")

    n_particles = coords_arr.shape[0]
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
    """Apply the existing scalar-field preprocessing to a chunk-local grid."""
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
    """Convert a chunk-local scalar field to an SDF mesh."""
    full_mask = chunk_grid > threshold
    if not np.any(full_mask):
        return None

    d_in = ndimage.distance_transform_edt(full_mask)
    d_out = ndimage.distance_transform_edt(~full_mask)
    sdf = d_in.astype(np.float64) - d_out.astype(np.float64)

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
    """Combine chunk mesh fragments into one mesh."""
    if not meshes:
        raise ValueError("No chunk meshes to combine")
    if len(meshes) == 1:
        return meshes[0]

    combined = trimesh.util.concatenate([m.to_trimesh() for m in meshes])
    merged = Mesh(mesh=combined)
    merged.to_trimesh().merge_vertices()
    merged.to_trimesh().process()
    merged.to_trimesh().fix_normals()
    return merged


def keep_largest_mesh_component(mesh: Mesh) -> Mesh:
    """Keep only the largest connected component of a stitched mesh."""
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


def chunk_halo_voxels(gaussian_sigma: float) -> int:
    """Return the halo size needed for chunk-local processing."""
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
    """Generate a stitched standard-MC mesh from chunk-local scalar fields."""
    total_start = time.perf_counter()
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    data_arr = np.asarray(data)
    if data_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError("data must have shape (N,)")

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
    """Partition marching-cubes cells into contiguous owned chunks."""
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
