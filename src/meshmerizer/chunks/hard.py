"""Hard-boundary chunk meshing helpers.

This module contains the low-memory chunked meshing workflow. It handles
particle selection for individual hard chunks, chunk-local voxelization,
chunk-local SDF extraction, optional clipping back to exact hard bounds, and
parallel execution across chunks.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import trimesh

try:
    from meshmerizer import _voxelize
except ImportError:
    _voxelize = None

from meshmerizer.logging import (
    current_thread_number,
    log_debug_status,
    log_status,
    progress_bar,
    record_timing,
)
from meshmerizer.mesh import Mesh, voxels_to_stl_via_sdf

from .geometry import (
    HardChunkBounds,
    VirtualGrid,
    crop_grid_to_chunk_bounds,
    expand_hard_chunk_bounds,
    iter_hard_chunk_bounds,
    partition_axis,
)
from .processing import chunk_halo_voxels, preprocess_chunk_grid


def _axis_chunk_lookup(
    grid: VirtualGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-axis lookup tables from cell index to chunk index.

    Args:
        grid: Virtual grid whose chunk partition should be encoded.

    Returns:
        Tuple of three arrays mapping x, y, and z marching-cubes cell indices
        to the owning chunk index along that axis.
    """
    # Encode the chunk partition once so point particles can be mapped into
    # owned hard chunks without scanning every particle against every chunk.
    base_lookup = np.empty(grid.cell_resolution, dtype=np.int64)
    for chunk_index, axis_chunk in enumerate(
        partition_axis(grid.cell_resolution, grid.nchunks)
    ):
        base_lookup[axis_chunk.cell_start : axis_chunk.cell_stop] = chunk_index

    # The grid uses the same axis partition along x, y, and z, so one lookup
    # array can be copied for each axis.
    return base_lookup.copy(), base_lookup.copy(), base_lookup.copy()


def _bucket_point_particles_into_hard_chunks(
    coordinates: np.ndarray,
    grid: VirtualGrid,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Bucket point particles into owned hard chunks without per-chunk scans.

    Args:
        coordinates: Particle coordinates in the chunk-local coordinate system.
        grid: Virtual grid describing the global chunk partition.

    Returns:
        Mapping from chunk index to the particle indices owned by that chunk.

    Raises:
        ValueError: If ``coordinates`` does not have shape ``(N, 3)``.
    """
    # Convert particle positions to owning cell indices. Particles outside the
    # owned marching-cubes cell range are ignored because the current
    # hard-chunk ownership rule also excludes them.
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    if coords_arr.shape[0] == 0:
        return {}

    voxel_size = grid.voxel_size
    cell_indices = np.floor(coords_arr / voxel_size).astype(np.int64)
    valid_mask = np.all(cell_indices >= 0, axis=1) & np.all(
        cell_indices < grid.cell_resolution,
        axis=1,
    )
    if not np.any(valid_mask):
        return {}

    # Map each valid cell to its owning chunk on each axis, then group the
    # original particle indices by the resulting chunk triplet.
    x_lookup, y_lookup, z_lookup = _axis_chunk_lookup(grid)
    valid_particle_indices = np.nonzero(valid_mask)[0]
    valid_cells = cell_indices[valid_mask]
    chunk_ids = np.column_stack(
        (
            x_lookup[valid_cells[:, 0]],
            y_lookup[valid_cells[:, 1]],
            z_lookup[valid_cells[:, 2]],
        )
    )

    buckets: dict[tuple[int, int, int], list[int]] = {}
    for particle_index, chunk_id in zip(valid_particle_indices, chunk_ids):
        chunk_key = tuple(int(value) for value in chunk_id)
        buckets.setdefault(chunk_key, []).append(int(particle_index))

    return {
        chunk_key: np.asarray(particle_indices, dtype=np.int64)
        for chunk_key, particle_indices in buckets.items()
    }


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

    # Fast-path the common sparse point-particle case by assigning each
    # particle to its owned hard chunk once up front. Overlap or smoothing
    # disables this optimization because chunk influence then extends beyond
    # hard ownership.
    if coords_arr.shape[0] == 0:
        processed_chunks = grid.nchunks**3
        log_status(
            "Meshing",
            "Chunk summary:\n"
            f"  processed chunks: {processed_chunks}\n"
            "  non-empty chunks: 0\n"
            "  emitted meshes:   0\n"
            "  voxelization:     0.000 s total\n"
            "  preprocessing:    0.000 s total\n"
            "  sdf meshing:      0.000 s total\n"
            "  chunk pass total: 0.000 s",
        )
        record_timing("Chunk pass total", 0.0, operation="Meshing")
        return []
    if smoothing_arr is None and overlap_voxels == 0 and halo_voxels == 0:
        point_particle_buckets = _bucket_point_particles_into_hard_chunks(
            coords_arr,
            grid,
        )
    else:
        point_particle_buckets = None

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
        if point_particle_buckets is not None:
            particle_indices = point_particle_buckets.get(
                chunk_bounds.index,
                np.empty(0, dtype=np.int64),
            )
        else:
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
    chunk_mesh_map: dict[
        tuple[int, int, int], tuple[HardChunkBounds, list[Mesh]]
    ] = {}
    chunk_rows: list[dict[str, object]] = []
    voxelize_total = 0.0
    preprocess_total = 0.0
    meshing_total = 0.0
    nonempty_chunks = 0
    emitted_meshes = 0
    all_bounds = list(iter_hard_chunk_bounds(grid))
    processed_chunks = len(all_bounds)

    def handle_result(result: dict[str, object]) -> None:
        """Accumulate timings and emit chunk-level status output.

        Args:
            result: Result dictionary returned by ``process_chunk``.

        Returns:
            ``None``. Shared accumulators and status output are updated.
        """
        nonlocal emitted_meshes
        nonlocal meshing_total
        nonlocal nonempty_chunks
        nonlocal preprocess_total
        nonlocal voxelize_total

        chunk_bounds = result["bounds"]
        particle_count = int(result["particles"])
        meshes = result["meshes"]
        voxelize_total += float(result["voxelize_time"])
        preprocess_total += float(result["preprocess_time"])
        meshing_total += float(result["meshing_time"])
        elapsed = float(result["elapsed"])
        status = str(result["status"])
        thread = result["thread"]

        # Record a compact per-chunk row for the detailed log so long runs can
        # still be audited after the fact without printing each completion
        # live.
        chunk_rows.append(
            {
                "index": chunk_bounds.index,
                "particles": particle_count,
                "status": status,
                "meshes": len(meshes),
                "voxelize": float(result["voxelize_time"]),
                "preprocess": float(result["preprocess_time"]),
                "meshing": float(result["meshing_time"]),
                "elapsed": elapsed,
                "thread": thread,
            }
        )

        # Keep per-chunk completion output concise so the progress bar remains
        # the primary interactive signal during long runs.
        if status == "skipped":
            log_debug_status(
                "Meshing",
                f"{chunk_bounds.index}: skipped (0 particles)",
                thread=thread,
            )
            return

        if meshes:
            chunk_mesh_map[chunk_bounds.index] = (chunk_bounds, meshes)
            nonempty_chunks += 1
            emitted_meshes += len(meshes)
            log_debug_status(
                "Meshing",
                f"{chunk_bounds.index}: {particle_count} particles -> "
                f"{len(meshes)} mesh(es) in {elapsed:.3f} s",
                thread=thread,
            )
            return

        log_debug_status(
            "Meshing",
            f"{chunk_bounds.index}: {particle_count} particles -> no mesh",
            thread=thread,
        )

    # Run either serially or with a thread pool depending on the requested
    # chunk worker count.
    with progress_bar(
        processed_chunks,
        desc="Chunk meshing",
        unit="chunk",
        enabled=True,
    ) as bar:
        if nthreads == 1:
            for chunk_bounds in all_bounds:
                result = process_chunk(chunk_bounds)
                if result is None:
                    continue
                handle_result(result)
                bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=nthreads) as executor:
                future_map = {
                    executor.submit(process_chunk, chunk_bounds): chunk_bounds
                    for chunk_bounds in all_bounds
                }
                for future in as_completed(future_map):
                    result = future.result()
                    if result is None:
                        continue
                    handle_result(result)
                    bar.update(1)

    record_timing(
        "Chunk voxelization",
        voxelize_total,
        operation="Voxelising",
    )
    record_timing(
        "Chunk preprocessing",
        preprocess_total,
        operation="Cleaning",
    )
    record_timing("Chunk SDF meshing", meshing_total, operation="Meshing")
    record_timing(
        "Chunk pass total",
        time.perf_counter() - total_start,
        operation="Meshing",
    )
    if chunk_rows:
        table_lines = [
            "index      particles  status   meshes  voxelize  preprocess"
            "  meshing  elapsed  thread"
        ]
        for row in chunk_rows:
            thread_text = "-" if row["thread"] is None else str(row["thread"])
            table_lines.append(
                f"{str(row['index']):<10} "
                f"{int(row['particles']):>9} "
                f"{str(row['status']):<8} "
                f"{int(row['meshes']):>6} "
                f"{float(row['voxelize']):>9.3f} "
                f"{float(row['preprocess']):>11.3f} "
                f"{float(row['meshing']):>8.3f} "
                f"{float(row['elapsed']):>8.3f} "
                f"{thread_text:>6}"
            )
        log_status("Meshing", "Chunk details:\n" + "\n".join(table_lines))
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
    chunk_meshes = [
        chunk_mesh_map[chunk_bounds.index]
        for chunk_bounds in all_bounds
        if chunk_bounds.index in chunk_mesh_map
    ]
    return chunk_meshes
