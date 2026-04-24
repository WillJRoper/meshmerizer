"""Snapshot loading helpers for the CLI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from meshmerizer.logging import (
    log_debug_status,
    log_status,
    log_warning_status,
    record_elapsed,
)


def boxsize_to_float(boxsize: object) -> float:
    """Convert a SWIFT box-size object into a plain float.

    Args:
        boxsize: Snapshot metadata box-size value, which may be scalar, array,
            or quantity-like.

    Returns:
        Maximum box extent as a plain float.
    """
    # SWIFT metadata may expose the box size as a scalar, a vector, or a
    # quantity-like object. Normalize that variety to one float here so the
    # downstream geometry code stays simple.
    box = boxsize
    if hasattr(box, "value"):
        box = box.value
    arr = np.asarray(box)
    if arr.ndim == 0:
        return float(arr)
    return float(np.max(arr))


def apply_coordinate_shift(
    coords: np.ndarray,
    *,
    shift: np.ndarray,
    wrap_shift: bool,
    box_size: Optional[float],
) -> np.ndarray:
    """Apply a translation and optional periodic wrap to coordinates.

    Args:
        coords: Coordinate array with shape ``(N, 3)``.
        shift: Translation vector with shape ``(3,)``.
        wrap_shift: Whether to wrap translated coordinates back into the
            domain.
        box_size: Domain size used for wrapping.

    Returns:
        Shifted coordinate array.

    Raises:
        ValueError: If the coordinate shape or shift vector are invalid, or if
            wrapping is requested without a valid ``box_size``.
    """
    # Normalize inputs first so the shift and optional wrapping logic operates
    # on predictable numeric arrays.
    coords_arr = np.asarray(coords, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")

    shift_arr = np.asarray(shift, dtype=np.float64)
    if shift_arr.shape != (3,):
        raise ValueError("shift must have shape (3,)")

    # Apply the shift unconditionally. Wrapping is a separate step because some
    # workflows want a translation without periodic remapping.
    shifted = coords_arr + shift_arr
    if not wrap_shift:
        return shifted

    if box_size is None:
        raise ValueError(
            "--wrap-shift requested but box_size is not known. "
            "Pass --box-size (or ensure snapshot metadata includes boxsize), "
            "or use --no-wrap-shift."
        )
    box = float(box_size)
    if box <= 0.0:
        raise ValueError("box_size must be > 0 when using --wrap-shift")

    # When periodic wrapping is enabled, fold the translated coordinates back
    # into the simulation cube.
    return np.mod(shifted, box)


def crop_particles_to_region(
    coords: np.ndarray,
    smoothing_lengths: Optional[np.ndarray],
    *,
    center: np.ndarray,
    extent: float,
    box_size: float,
    periodic: bool,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Crop particle arrays to an axis-aligned cubic region.

    Args:
        coords: Particle coordinates with shape ``(N, 3)``.
        smoothing_lengths: Optional per-particle smoothing lengths.
        center: Region centre in world coordinates.
        extent: Cubic region side length.
        box_size: Full periodic box size.
        periodic: Whether to use periodic selection around the region centre.

    Returns:
        Tuple containing cropped local coordinates, cropped smoothing lengths,
        and the world-space origin of the selected region.

    Raises:
        ValueError: If the region specification or array shapes are invalid.
    """
    # Validate the region definition up front so the periodic and non-periodic
    # branches can assume a consistent cube specification.
    if extent <= 0:
        raise ValueError("extent must be > 0")
    if box_size <= 0:
        raise ValueError("box_size must be > 0")

    c = np.asarray(center, dtype=np.float64)
    if c.shape != (3,):
        raise ValueError("center must have shape (3,)")

    coords_arr = np.asarray(coords, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")

    half = 0.5 * float(extent)
    mins = c - half
    maxs = c + half

    # Build the cube either in periodic coordinates relative to the requested
    # centre or in straightforward box coordinates for the non-periodic path.
    if periodic:
        delta = coords_arr - c
        delta = (delta + 0.5 * box_size) % box_size - 0.5 * box_size
        mask = np.all(np.abs(delta) <= half, axis=1)

        # Re-express the periodic selection in a local cube ``[0, extent)`` so
        # the downstream adaptive code sees the same coordinate convention as a
        # non-periodic crop.
        local = delta + half
        origin = np.mod(mins, box_size)
    else:
        if np.any(mins < 0.0) or np.any(maxs > box_size):
            raise ValueError(
                "Non-periodic region must lie within [0, box_size]"
            )
        # In the non-periodic case the world-space minimum corner is
        # already the correct local origin.
        origin = mins
        mask = np.all((coords_arr >= mins) & (coords_arr < maxs), axis=1)
        local = coords_arr - origin

    # Apply the particle mask to every array together so the caller receives a
    # consistent cropped particle set.
    cropped_coords = local[mask]
    if smoothing_lengths is None:
        cropped_h = None
    else:
        # Preserve the original smoothing-length dtype here because this helper
        # only performs selection, not numeric kernel evaluation.
        h_arr = np.asarray(smoothing_lengths)
        cropped_h = h_arr[mask]

    return cropped_coords, cropped_h, origin


def tighten_working_bounds(
    coords: np.ndarray,
    smoothing_lengths: Optional[np.ndarray],
    *,
    box_size: float,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, float]:
    """Shrink the working cube to the occupied particle bounds.

    Args:
        coords: Particle coordinates with shape ``(N, 3)``.
        smoothing_lengths: Optional per-particle smoothing lengths.
        box_size: Current cubic box size.

    Returns:
        Tuple containing shifted coordinates, shifted smoothing lengths, the
        origin offset, and the tightened cubic working-domain size.

    Raises:
        ValueError: If the inputs are empty or malformed.
    """
    # Treat the optional smoothing lengths as part of the occupied support so
    # the tightened box still fully contains the deposited kernel support.
    if box_size <= 0:
        raise ValueError("box_size must be > 0")

    coords_arr = np.asarray(coords, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if coords_arr.shape[0] == 0:
        raise ValueError("coords must not be empty")

    mins = np.min(coords_arr, axis=0)
    maxs = np.max(coords_arr, axis=0)

    if smoothing_lengths is None:
        h_arr = None
    else:
        h_arr = np.asarray(smoothing_lengths, dtype=np.float64)
        if h_arr.shape != (coords_arr.shape[0],):
            raise ValueError("smoothing_lengths must have shape (N,)")
        mins = np.min(coords_arr - h_arr[:, None], axis=0)
        maxs = np.max(coords_arr + h_arr[:, None], axis=0)

    # Clamp to the original cube because the user may already have requested a
    # cropped subregion, and the tightened bounds must stay inside it.
    mins = np.clip(mins, 0.0, box_size)
    maxs = np.clip(maxs, 0.0, box_size)
    spans = maxs - mins
    tight_box_size = float(np.max(spans))

    if tight_box_size <= 0.0:
        return coords_arr, h_arr, np.zeros(3, dtype=np.float64), box_size

    return coords_arr - mins, h_arr, mins, tight_box_size


def raise_if_empty_subregion_selection(
    n_selected: int,
    *,
    particle_type: str,
    center: np.ndarray,
    extent: float,
    periodic: bool,
) -> None:
    """Raise a helpful error if a subregion selection is empty.

    Args:
        n_selected: Number of particles selected by the crop.
        particle_type: Selected particle family.
        center: Crop centre.
        extent: Crop extent.
        periodic: Whether the selection used periodic wrapping.

    Raises:
        RuntimeError: If no particles were selected.
    """
    # Return immediately in the common case so the expensive message formatting
    # below is only used for the actual error path.
    if n_selected != 0:
        return

    c = np.asarray(center, dtype=np.float64)
    msg = (
        "Subregion selection contains no particles: "
        f"particle_type={particle_type} "
        f"center=({c[0]:.6g},{c[1]:.6g},{c[2]:.6g}) "
        f"extent={float(extent):.6g} periodic={periodic} "
        f"selected_particles={n_selected}. "
        "Try increasing --extent or choosing a different --center "
        "(for example, the particle center-of-mass). "
        "If periodic wrapping is causing confusion, try --no-periodic."
    )
    raise RuntimeError(msg)


def load_swift_particles(
    filename: Path,
    particle_type: str,
    smoothing_factor: float,
    box_size: Optional[float],
    shift: list[float],
    wrap_shift: bool,
    center: Optional[list[float]],
    extent: Optional[float],
    periodic: bool,
    tight_bounds: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
]:
    """Load and preprocess particle arrays from a SWIFT snapshot.

    Args:
        filename: Snapshot filename.
        particle_type: Particle family to extract.
        smoothing_factor: Multiplier applied to smoothing lengths.
        box_size: Optional box size override.
        shift: Coordinate shift applied before cropping.
        wrap_shift: Whether to wrap shifted coordinates periodically.
        center: Optional crop centre.
        extent: Optional crop extent.
        periodic: Whether crop selection is periodic.
        tight_bounds: Whether to tighten the working cube to occupancy.

    Returns:
        Tuple containing prepared coordinates, smoothing lengths, effective box
        size, and the world-space origin.

    Raises:
        RuntimeError: If the snapshot cannot be loaded or has no usable box.
        ValueError: If argument combinations are inconsistent.
    """
    # Import SWIFT lazily so importing the package or using non-SWIFT codepaths
    # does not pay the dependency cost up front.
    import swiftsimio as sw
    from swiftsimio.visualisation import generate_smoothing_lengths

    total_start = time.perf_counter()

    # Load the snapshot first so every later validation error can report
    # against the actual metadata present in the file.
    log_status("Loading", f"Loading SWIFT data from {filename}...")
    load_start = time.perf_counter()
    try:
        data = sw.load(str(filename))
    except Exception as exc:
        raise RuntimeError(f"Error loading file: {exc}") from exc
    record_elapsed("Snapshot load", load_start, operation="Loading")

    if (center is None) != (extent is None):
        raise ValueError("--center and --extent must be provided together")

    # Resolve the effective box size once here so shifting, cropping, and tight
    # bounds all operate in one consistent coordinate system.
    full_box_size_source: str
    if box_size is not None:
        # Prefer an explicit user override when present so the CLI can repair
        # incomplete snapshot metadata.
        box_size_source = "--box-size"
    else:
        meta_box = None
        if hasattr(data, "metadata"):
            meta_box = getattr(data.metadata, "boxsize", None)

        if meta_box is None:
            raise RuntimeError(
                "Snapshot metadata does not include boxsize. "
                "Pass --box-size to define the physical volume."
            )

        # Normalize the metadata representation once here because SWIFT can
        # expose box size as either scalar or vector-like quantities.
        box_size = boxsize_to_float(meta_box)
        box_size_source = "snapshot metadata"

    full_box_size = float(box_size)
    full_box_size_source = box_size_source

    # Map the CLI particle-family name to the corresponding SWIFT dataset.
    if particle_type == "gas":
        # Map the CLI spelling directly to the loaded SWIFT particle container.
        part_data = data.gas
    elif particle_type == "dark_matter":
        part_data = data.dark_matter
    elif particle_type == "stars":
        part_data = data.stars
    elif particle_type == "black_holes":
        part_data = data.black_holes
    else:
        raise ValueError(f"Unknown particle type '{particle_type}'")

    # Extract coordinates for the selected particle family.
    log_status("Loading", f"Extracting {particle_type} particles...")
    extract_start = time.perf_counter()
    coords_cosmo = part_data.coordinates
    coords = coords_cosmo.value
    record_elapsed(
        "Particle extraction",
        extract_start,
        operation="Loading",
    )

    # Apply the requested coordinate shift before any subregion crop so the
    # crop is interpreted in the shifted coordinate system.
    shift_arr = np.asarray(shift, dtype=np.float64)
    if shift_arr.shape != (3,):
        raise ValueError("--shift must provide exactly 3 values: dx dy dz")
    log_debug_status(
        "Loading",
        "Coordinate shift: "
        f"({shift_arr[0]:.6g},{shift_arr[1]:.6g},{shift_arr[2]:.6g}) "
        f"wrap_shift={wrap_shift}",
    )
    shift_start = time.perf_counter()
    coords = apply_coordinate_shift(
        coords,
        shift=shift_arr,
        wrap_shift=wrap_shift,
        box_size=full_box_size,
    )
    record_elapsed("Coordinate shifting", shift_start, operation="Loading")

    # Prefer snapshot smoothing lengths when available. Only fall back to
    # regeneration when the particle family does not store them explicitly.
    if hasattr(part_data, "smoothing_lengths"):
        # If smoothing lengths are already present, keep the snapshot's own
        # values and only apply the CLI multiplier.
        smoothing_start = time.perf_counter()
        h = part_data.smoothing_lengths.value * smoothing_factor
        record_elapsed(
            "Smoothing-length extraction",
            smoothing_start,
            operation="Loading",
        )
    else:
        # Some particle families do not carry smoothing lengths. Fall back to a
        # generated estimate so the adaptive pipeline can still proceed.
        log_status("Loading", "Smoothing lengths not found. Generating...")
        boxsize = data.metadata.boxsize
        smoothing_start = time.perf_counter()
        try:
            h_cosmo = generate_smoothing_lengths(
                coords_cosmo,
                boxsize,
                kernel_gamma=1.8,
            )
            h = h_cosmo.value * smoothing_factor
            log_status("Loading", "Smoothing lengths generated.")
            record_elapsed(
                "Smoothing-length generation",
                smoothing_start,
                operation="Loading",
            )
        except Exception as exc:
            log_warning_status(
                "Loading",
                f"Error generating smoothing lengths: {exc}",
            )
            log_warning_status(
                "Loading",
                "Falling back to point deposition.",
            )
            # Preserve a ``None`` sentinel on failure so the caller can decide
            # whether this pipeline requires smoothing lengths or can degrade
            # gracefully.
            h = None
            record_elapsed(
                "Smoothing-length generation",
                smoothing_start,
                operation="Loading",
            )

    log_debug_status(
        "Loading",
        f"Snapshot box size: {full_box_size:.6g} "
        f"(sim units; from {full_box_size_source})",
    )

    # Track the current world-space origin and effective cube size explicitly.
    # Later crop and tighten operations update these values step by step.
    origin = np.zeros(3, dtype=np.float64)
    effective_box_size = full_box_size

    # When the user requests a subregion, remap the selected particles into the
    # local cube coordinates expected by the adaptive reconstruction code.
    if center is not None:
        crop_start = time.perf_counter()
        assert extent is not None
        center_arr = np.asarray(center, dtype=np.float64)
        extent_f = float(extent)
        # Convert the requested world-space crop into a local working cube and
        # track the world-space origin so final meshes can be shifted back.
        coords, h, origin = crop_particles_to_region(
            coords,
            h,
            center=center_arr,
            extent=extent_f,
            box_size=full_box_size,
            periodic=periodic,
        )
        effective_box_size = extent_f
        log_debug_status(
            "Cleaning",
            "Subregion: "
            f"center=({center_arr[0]:.6g},{center_arr[1]:.6g},"
            f"{center_arr[2]:.6g}) "
            f"extent={extent_f:.6g} periodic={periodic}",
        )
        log_debug_status(
            "Cleaning",
            "Subregion origin (world-space min corner): "
            f"({origin[0]:.6g},{origin[1]:.6g},{origin[2]:.6g})",
        )
        n_selected = int(np.asarray(coords).shape[0])
        raise_if_empty_subregion_selection(
            n_selected,
            particle_type=particle_type,
            center=center_arr,
            extent=extent_f,
            periodic=periodic,
        )
        log_debug_status(
            "Cleaning",
            f"Selected {n_selected} {particle_type} particles.",
        )
        record_elapsed("Subregion crop", crop_start, operation="Cleaning")

    # Tight bounds are applied after any explicit crop so the optimization acts
    # on the actual region that will be reconstructed.
    if tight_bounds:
        tighten_start = time.perf_counter()
        # Tight bounds are a second-stage optimization: first crop if
        # requested, then shrink the occupied cube.
        coords, h, origin_offset, effective_box_size = tighten_working_bounds(
            coords,
            h,
            box_size=effective_box_size,
        )
        origin = origin + origin_offset
        if center is not None and periodic:
            origin = np.mod(origin, full_box_size)
        log_debug_status(
            "Cleaning",
            "Tightened working bounds: "
            f"origin=({origin[0]:.6g},{origin[1]:.6g},{origin[2]:.6g}) "
            f"box_size={effective_box_size:.6g}",
        )
        record_elapsed("Tight bounds", tighten_start, operation="Cleaning")

    record_elapsed("Particle preparation", total_start, operation="Loading")
    return coords, h, effective_box_size, origin
