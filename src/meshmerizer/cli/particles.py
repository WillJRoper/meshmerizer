"""CLI helpers for loading and filtering particle inputs."""

from __future__ import annotations

import numpy as np

from meshmerizer.adaptive_core import fof_cluster
from meshmerizer.io.swift import load_swift_particles
from meshmerizer.logging import abort_with_error, log_status


def load_particles_for_adaptive(args):
    """Load and prepare particles for the adaptive pipeline."""
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

    if smoothing_lengths is None:
        abort_with_error(
            "Loading",
            "Smoothing lengths are required for the adaptive pipeline but "
            "could not be determined.",
        )

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
    """Discard particle FOF clusters smaller than ``min_cluster_size``."""
    if min_cluster_size <= 0:
        raise ValueError(
            f"min_cluster_size must be positive, got {min_cluster_size}"
        )

    if len(positions) == 0:
        return positions, smoothing_lengths

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

    if n_kept_particles == 0:
        return positions[:0], smoothing_lengths[:0]

    return positions[kept_mask], smoothing_lengths[kept_mask]


__all__ = [
    "filter_small_particle_fof_clusters",
    "load_particles_for_adaptive",
]
