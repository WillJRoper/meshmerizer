"""Volume preprocessing shared by mesh extraction routines."""

import time
from typing import Iterable, Optional, Tuple

import numpy as np
from scipy import ndimage

from meshmerizer.logging_utils import log_status


def prepare_volume(
    volume: np.ndarray,
    threshold: float,
    closing_radius: int,
    split_islands: bool,
    remove_islands: Optional[int],
    mesh_index: Optional[int],
    padding: int = 1,
) -> Tuple[np.ndarray, Iterable[int]]:
    """Binarize, close, and label a voxel volume.

    Args:
        volume: Input scalar field.
        threshold: Threshold used to binarize the field.
        closing_radius: Radius for binary closing.
        split_islands: Whether to keep connected components separate.
        remove_islands: Island-removal mode. ``None`` keeps all components,
            ``0`` keeps only the largest one, and positive values remove
            components smaller than the given voxel count.
        mesh_index: Optional specific component label to extract.
        padding: Number of zero voxels padded around the volume.

    Returns:
        Tuple containing the labeled volume and the component labels selected
        for extraction.

    Raises:
        ValueError: If ``mesh_index`` does not exist in the selected labels.
    """
    # Pad the field before thresholding so surfaces touching the box boundary
    # can close cleanly during later extraction.
    if padding > 0:
        volume = np.pad(
            volume,
            pad_width=padding,
            mode="constant",
            constant_values=0,
        )

    # Convert the scalar field into a binary occupancy mask at the requested
    # isovalue.
    bin_start = time.perf_counter()
    bin_vol = volume > threshold
    bin_end = time.perf_counter()
    log_status(
        "Cleaning",
        f"Binarization took {bin_end - bin_start:.4f} seconds.",
    )

    # Close small holes and gaps before connected-component analysis.
    if closing_radius > 0:
        close_start = time.perf_counter()
        base_struct = ndimage.generate_binary_structure(3, 1)
        closing_struct = ndimage.iterate_structure(base_struct, closing_radius)
        bin_vol = ndimage.binary_closing(bin_vol, structure=closing_struct)
        close_end = time.perf_counter()
        log_status(
            "Cleaning",
            f"Binary closing took {close_end - close_start:.4f} seconds.",
        )

    # Label connected components only when the caller needs island splitting or
    # island filtering. Otherwise the whole mask is treated as one component.
    label_struct = ndimage.generate_binary_structure(3, 1)
    if split_islands or remove_islands is not None:
        split_start = time.perf_counter()
        labeled, num = ndimage.label(bin_vol, structure=label_struct)
        island_ids = range(1, num + 1)
        split_end = time.perf_counter()
        log_status(
            "Cleaning",
            f"Labeling took {split_end - split_start:.4f} seconds. "
            f"Found {num} islands.",
        )

        # Apply island filtering on the labeled voxel components before any
        # mesh extraction happens.
        if remove_islands is not None and num > 0:
            component_sizes = np.bincount(labeled.ravel())
            component_sizes[0] = 0
            if remove_islands == 0:
                largest_label = int(np.argmax(component_sizes))
                largest_size = int(component_sizes[largest_label])
                removed_count = num - 1
                labeled = np.where(labeled == largest_label, largest_label, 0)
                island_ids = [largest_label]
                log_status(
                    "Cleaning",
                    "Removing disconnected islands. "
                    f"Keeping largest component with {largest_size} voxels "
                    f"and discarding {removed_count} island(s).",
                )
            else:
                keep_labels = np.flatnonzero(component_sizes >= remove_islands)
                keep_labels = keep_labels[keep_labels != 0]
                keep_mask = np.isin(labeled, keep_labels)
                labeled = np.where(keep_mask, labeled, 0)
                island_ids = keep_labels.tolist()
                removed_count = int(num - len(island_ids))
                log_status(
                    "Cleaning",
                    "Removing disconnected islands smaller than "
                    f"{remove_islands} voxels. Keeping {len(island_ids)} "
                    f"component(s) and discarding {removed_count} island(s).",
                )
    else:
        labeled = bin_vol.astype(np.int32)
        if not np.any(labeled):
            island_ids = []
        else:
            island_ids = [1]

    # Optionally restrict extraction to one requested component label.
    if mesh_index is not None:
        all_ids = list(island_ids)
        if mesh_index not in all_ids:
            raise ValueError(
                f"Mesh index {mesh_index} not found in the volume."
            )
        island_ids = [mesh_index]

    return labeled, island_ids
