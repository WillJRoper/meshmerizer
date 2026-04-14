"""Hard-boundary chunk meshing helpers.

This module contains the low-memory chunked meshing workflow. It handles
particle selection for individual hard chunks, chunk-local voxelization,
chunk-local SDF extraction, optional clipping back to exact hard bounds, and
parallel execution across chunks.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import trimesh

try:
    from meshmerizer import _voxelize
except ImportError:
    _voxelize = None

from meshmerizer.logging_utils import current_thread_number, log_status
from meshmerizer.mesh import Mesh, voxels_to_stl_via_sdf

from .geometry import (
    HardChunkBounds,
    VirtualGrid,
    crop_grid_to_chunk_bounds,
    expand_hard_chunk_bounds,
    iter_hard_chunk_bounds,
)
from .processing import chunk_halo_voxels, preprocess_chunk_grid


def select_particles_in_hard_chunk(
    coordinates: np.ndarray,
    chunk_bounds: HardChunkBounds,
    *,
    smoothing_lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Select particles that intersect a hard chunk boundary box.

    Args:
        coordinates: Particle coordinates in the chunk-local coordinate system.
        chunk_bounds: Hard chunk bounds that define the owned chunk box.
        smoothing_lengths: Optional per-particle smoothing lengths in world
            units.

    Returns:
        Integer indices of particles whose point position or smoothing support
        intersects the hard chunk box.

    Raises:
        ValueError: If the input arrays do not have the expected shapes.
    """
    # Work in plain NumPy arrays so both the point-deposition and smoothed
    # support-selection branches use the same validated coordinate data.
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")

    lower = chunk_bounds.hard_local_start
    upper = chunk_bounds.hard_local_stop

    # Treat the no-smoothing case as strict point ownership against the hard
    # chunk bounds.
    if smoothing_lengths is None:
        mask = np.all(coords_arr >= lower, axis=1) & np.all(
            coords_arr < upper,
            axis=1,
        )
        return np.nonzero(mask)[0].astype(np.int64)

    # When smoothing is present, select any particle whose support overlaps the
    # hard chunk rather than only particles whose centres lie inside it.
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

    Args:
        data: Per-particle scalar values.
        coordinates: Particle coordinates in the parent local cube.
        chunk_bounds: Bounds of the chunk grid to be voxelized.
        smoothing_lengths: Optional per-particle smoothing lengths.
        nthreads: Number of threads passed to the local deposition kernel.

    Returns:
        Tuple containing the local dense chunk grid and the isotropic voxel
        size implied by the chunk bounds.

    Raises:
        ValueError: If the inputs have invalid shapes or imply an empty chunk.
        RuntimeError: If smoothing is requested but the C extension is missing.
    """
    # Normalize array inputs before computing the local lattice mapping.
    data_arr = np.asarray(data)
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    if data_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError("data must have shape (N,)")

    # Convert the chunk bounds into per-axis grid dimensions. Hard chunks can
    # differ by one sample between axes when the global grid is not divisible
    # evenly by the requested chunk count, so the voxelizer must respect the
    # true ``(nx, ny, nz)`` shape rather than assuming a cube.
    grid_shape = np.asarray(chunk_bounds.shape, dtype=np.int64)
    if np.any(grid_shape <= 0):
        raise ValueError("chunk resolution must be > 0")
    if np.any(chunk_bounds.extent <= 0.0):
        raise ValueError("chunk extents must be > 0")

    # The package assumes one isotropic voxel size derived from the parent
    # virtual grid. Validate that the chunk bounds are consistent with that
    # assumption before using one scalar voxel size downstream.
    voxel_sizes = chunk_bounds.extent / grid_shape.astype(np.float64)
    if not np.allclose(voxel_sizes, voxel_sizes[0]):
        raise ValueError("chunk bounds imply anisotropic voxel sizes")

    # Map particle positions into chunk-local voxel coordinates.
    local_coords = coords_arr - chunk_bounds.local_start
    voxel_size = float(voxel_sizes[0])
    local_grid = np.zeros(tuple(grid_shape.tolist()), dtype=np.float64)

    # Scale each axis independently into the local chunk lattice.
    scaled = local_coords / chunk_bounds.extent * grid_shape.astype(np.float64)
    vox_indices = np.floor(scaled).astype(np.int64)

    # Use fast NumPy accumulation for the point-deposition case.
    if smoothing_lengths is None:
        vox_indices = np.clip(vox_indices, 0, grid_shape - 1)
        np.add.at(
            local_grid,
            (vox_indices[:, 0], vox_indices[:, 1], vox_indices[:, 2]),
            data_arr,
        )
        return local_grid, voxel_size

    # The smoothed deposition path uses the compiled local kernel because the
    # Python equivalent is too slow for production use.
    if _voxelize is None:
        raise RuntimeError(
            "Chunk-local smoothing requires the _voxelize C extension."
        )

    smoothing_arr = np.asarray(smoothing_lengths, dtype=np.float64)
    if smoothing_arr.shape != (coords_arr.shape[0],):
        raise ValueError("smoothing_lengths must have shape (N,)")

    if voxel_size > 0:
        smoothing_lengths_vox = (smoothing_arr / voxel_size).astype(np.int64)
    else:
        smoothing_lengths_vox = np.zeros_like(smoothing_arr, dtype=np.int64)

    _voxelize.box_deposition_local(
        local_grid,
        data_arr.astype(np.float64),
        vox_indices.astype(np.int64),
        smoothing_lengths_vox,
        int(grid_shape[0]),
        int(grid_shape[1]),
        int(grid_shape[2]),
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

    Args:
        chunk_grid: Dense scalar field for one chunk.
        chunk_bounds: World and local geometry for the chunk.
        threshold: Isovalue used for surface extraction.
        closing_radius: Binary closing radius passed to the SDF extractor.

    Returns:
        List of meshes emitted for the connected components found in the chunk.

    Raises:
        ValueError: If the chunk bounds imply anisotropic voxel sizes.
    """
    # Derive the physical voxel size from the chunk bounds so the extracted
    # mesh is scaled correctly in world space.
    voxel_sizes = chunk_bounds.extent / np.asarray(
        chunk_bounds.shape, dtype=np.float64
    )
    if not np.allclose(voxel_sizes, voxel_sizes[0]):
        raise ValueError("chunk bounds imply anisotropic voxel sizes")
    voxel_size = float(voxel_sizes[0])
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

    # The extractor returns chunk-local vertices, so translate them into the
    # global world-space coordinates described by the chunk bounds.
    world_origin = chunk_bounds.world_start
    for mesh in meshes:
        mesh.vertices[:] = mesh.vertices + world_origin
    return meshes


def clip_mesh_to_hard_chunk(mesh: Mesh, chunk_bounds: HardChunkBounds) -> Mesh:
    """Clip a mesh to the exact hard chunk box and cap the cut faces.

    Args:
        mesh: Chunk mesh to clip.
        chunk_bounds: Exact owned hard chunk box.

    Returns:
        Clipped mesh confined to the owned hard chunk bounds.

    Raises:
        ValueError: If clipping removes all geometry.
    """
    # Build the exact hard chunk box in world space and use a boolean
    # intersection to trim away any overlap region geometry.
    extents = chunk_bounds.hard_world_stop - chunk_bounds.hard_world_start
    center = 0.5 * (
        chunk_bounds.hard_world_start + chunk_bounds.hard_world_stop
    )
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)

    clipped = trimesh.boolean.intersection(
        [mesh.to_trimesh(), box],
        engine="blender",
        check_volume=True,
    )
    if clipped is None or len(clipped.faces) == 0:
        raise ValueError("Chunk clipping removed all mesh geometry")

    # Clean the boolean output before wrapping it so later repair and union
    # steps receive a consistent trimesh.
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
        coordinates: Particle coordinates in the voxelization cube.
        smoothing_lengths: Optional per-particle smoothing lengths.
        grid: Virtual grid describing the full chunked domain.
        threshold: Isovalue used for chunk-local surface extraction.
        preprocess: Chunk-local preprocessing mode.
        clip_halos: Optional percentile clip applied before preprocessing.
        gaussian_sigma: Gaussian smoothing width in voxel units.
        nthreads: Number of chunk workers to use.
        overlap_voxels: Number of interior overlap voxels between chunks.
        clip_to_bounds: Whether to boolean-clip each emitted mesh back to the
            exact hard chunk bounds.

    Returns:
        List of ``(chunk_bounds, meshes)`` pairs for chunks that emitted mesh
        geometry.

    Raises:
        ValueError: If the particle arrays or thread count are invalid.
    """
    # Normalize the particle arrays once so the per-chunk worker function only
    # handles chunk-specific selection and meshing logic.
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
    # Use chunk-level parallelism only.
    # The local deposition kernel is forced to one thread per chunk to avoid
    # nested oversubscription.
    if nthreads < 1:
        raise ValueError("nthreads must be >= 1")
    deposition_threads = 1

    # Grow the overlap by the Gaussian halo size so chunk-local smoothing has
    # enough context before the field is cropped back to the owned overlap box.
    halo_voxels = (
        chunk_halo_voxels(gaussian_sigma) if gaussian_sigma > 0 else 0
    )

    def process_chunk(
        chunk_bounds: HardChunkBounds,
    ) -> dict[str, object] | None:
        """Process one chunk through selection, voxelization, and meshing.

        Args:
            chunk_bounds: Hard chunk bounds for the chunk being processed.

        Returns:
            Dictionary containing timings, status information, and any emitted
            meshes for that chunk.
        """
        # Expand the chunk first so smoothing and overlap-sensitive extraction
        # have enough neighbouring support.
        effective_bounds = expand_hard_chunk_bounds(
            grid,
            chunk_bounds,
            overlap_voxels=overlap_voxels + halo_voxels,
        )
        chunk_start = time.perf_counter()

        # Skip chunks with no contributing particles before doing any voxel
        # work.
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
                "thread": current_thread_number() if nthreads > 1 else None,
            }

        # Deposit only the particles that influence this chunk's expanded box.
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

        # Preprocess on the expanded field, then crop away the extra Gaussian
        # halo while keeping the requested overlap region.
        preprocess_start = time.perf_counter()
        chunk_grid = preprocess_chunk_grid(
            chunk_grid,
            preprocess=preprocess,
            clip_halos=clip_halos,
            gaussian_sigma=gaussian_sigma,
        )
        if halo_voxels > 0:
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

        # Extract meshes from the processed chunk field.
        # Optionally clip the result back to the exact hard chunk bounds.
        meshing_start = time.perf_counter()
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
            "thread": current_thread_number() if nthreads > 1 else None,
        }

    # Compute the chunk list up front so serial and parallel execution see the
    # same processing order.
    chunk_meshes: list[tuple[HardChunkBounds, list[Mesh]]] = []
    voxelize_total = 0.0
    preprocess_total = 0.0
    meshing_total = 0.0
    nonempty_chunks = 0
    emitted_meshes = 0
    all_bounds = list(iter_hard_chunk_bounds(grid))
    processed_chunks = len(all_bounds)

    # Run either serially or with a thread pool depending on the requested
    # chunk worker count.
    if nthreads == 1:
        results = [process_chunk(chunk_bounds) for chunk_bounds in all_bounds]
    else:
        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            results = list(executor.map(process_chunk, all_bounds))

    # Accumulate timings and emit one structured status line per processed
    # chunk so long chunked runs remain understandable.
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
        thread = result["thread"]

        if status == "skipped":
            log_status(
                "Meshing",
                f"{chunk_bounds.index}: 0 particles -> skipped",
                thread=thread,
            )
            continue

        if meshes:
            chunk_meshes.append((chunk_bounds, meshes))
            nonempty_chunks += 1
            emitted_meshes += len(meshes)
            log_status(
                "Meshing",
                f"{chunk_bounds.index}: {particle_count} particles -> "
                f"{len(meshes)} mesh(es) in {elapsed:.3f} s",
                thread=thread,
            )
        else:
            log_status(
                "Meshing",
                f"{chunk_bounds.index}: {particle_count} particles -> no mesh",
                thread=thread,
            )

    log_status(
        "Meshing",
        "Chunk summary:\n"
        f"  processed chunks: {processed_chunks}\n"
        f"  non-empty chunks: {nonempty_chunks}\n"
        f"  emitted meshes:   {emitted_meshes}\n"
        f"  voxelization:     {voxelize_total:.3f} s total\n"
        f"  preprocessing:    {preprocess_total:.3f} s total\n"
        f"  sdf meshing:      {meshing_total:.3f} s total\n"
        f"  chunk pass total: {time.perf_counter() - total_start:.3f} s",
    )
    return chunk_meshes
