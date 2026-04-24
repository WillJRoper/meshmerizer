"""CLI helpers for loading and filtering particle inputs."""

from __future__ import annotations

import numpy as np

from meshmerizer.adaptive_core import fof_cluster
from meshmerizer.io.swift import load_swift_particles
from meshmerizer.logging import abort_with_error, log_status


def load_particles_for_adaptive(args):
    """Load and prepare particles for the adaptive pipeline.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Tuple ``(positions, smoothing_lengths, domain_min, domain_max,
        origin)`` ready for adaptive meshing.
    """
    # Delegate snapshot loading and preprocessing to the I/O layer so this CLI
    # helper only handles adaptive-pipeline-specific validation and shaping.
    coords, smoothing_lengths, effective_box_size, origin = (
        load_swift_particles(
            filename=args.filename,
            particle_type=args.particle_type,
            smoothing_factor=args.smoothing_factor,
            box_size=args.box_size,
            shift=list(args.shift),
            wrap_shift=args.wrap_shift,
            center=args.center,
            extent=args.extent,
            periodic=args.periodic,
            tight_bounds=args.tight_bounds,
        )
    )

    # The adaptive pipeline requires explicit smoothing lengths for SPH field
    # evaluation, so fail early if loading/generation could not produce them.
    if smoothing_lengths is None:
        abort_with_error(
            "Loading",
            "Smoothing lengths are required for the adaptive pipeline but "
            "could not be determined.",
        )

    # Refuse to proceed on an empty particle selection so later code can assume
    # a non-empty particle set.
    n_particles = coords.shape[0]
    if n_particles == 0:
        abort_with_error(
            "Loading",
            "No particles selected. Check domain selection flags.",
        )

    log_status(
        "Loading",
        f"Prepared {n_particles} particles for adaptive meshing.",
    )

    # Normalize arrays and domain bounds once here so all later CLI helpers see
    # the same contiguous float64 particle representation.
    positions = np.ascontiguousarray(coords, dtype=np.float64)
    smoothing_lengths = np.ascontiguousarray(
        smoothing_lengths,
        dtype=np.float64,
    )
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (
        float(effective_box_size),
        float(effective_box_size),
        float(effective_box_size),
    )

    return positions, smoothing_lengths, domain_min, domain_max, origin


def filter_small_particle_fof_clusters(
    positions: np.ndarray,
    smoothing_lengths: np.ndarray,
    domain_min: tuple[float, float, float],
    domain_max: tuple[float, float, float],
    linking_factor: float,
    min_cluster_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Discard particle FOF clusters smaller than ``min_cluster_size``.

    Args:
        positions: Particle positions with shape ``(N, 3)``.
        smoothing_lengths: Per-particle smoothing lengths with shape ``(N,)``.
        domain_min: Lower corner of the working domain.
        domain_max: Upper corner of the working domain.
        linking_factor: FOF linking-length multiplier.
        min_cluster_size: Minimum particle count required to keep a cluster.

    Returns:
        Tuple of filtered ``(positions, smoothing_lengths)`` arrays.

    Raises:
        ValueError: If ``min_cluster_size`` is not positive.
    """
    # Validate the threshold before clustering so later summary reporting can
    # assume it represents an actual minimum cluster size.
    if min_cluster_size <= 0:
        raise ValueError(
            f"min_cluster_size must be positive, got {min_cluster_size}"
        )

    # Short-circuit the empty case so we do not invoke the native clustering
    # path when there is nothing to cluster.
    if len(positions) == 0:
        return positions, smoothing_lengths

    # Cluster once, then compute both the keep mask and the human-readable
    # summary from the same label/count arrays.
    labels = fof_cluster(positions, domain_min, domain_max, linking_factor)
    unique_labels, counts = np.unique(labels, return_counts=True)
    kept_labels = unique_labels[counts >= min_cluster_size]

    n_groups = int(unique_labels.size)
    n_kept_groups = int(kept_labels.size)
    n_removed_groups = n_groups - n_kept_groups
    kept_mask = np.isin(labels, kept_labels)
    n_kept_particles = int(np.count_nonzero(kept_mask))
    n_removed_particles = int(len(positions) - n_kept_particles)

    log_status(
        "Clustering",
        (
            f"FOF particle filtering: kept {n_kept_groups}/{n_groups} groups "
            f"and {n_kept_particles}/{len(positions)} particles "
            f"(removed {n_removed_groups} groups, {n_removed_particles} "
            f"particles; min size = {min_cluster_size})."
        ),
    )

    # Preserve array rank on the all-removed path by slicing down to zero rows.
    if n_kept_particles == 0:
        return positions[:0], smoothing_lengths[:0]

    return positions[kept_mask], smoothing_lengths[kept_mask]


__all__ = [
    "filter_small_particle_fof_clusters",
    "load_particles_for_adaptive",
]
