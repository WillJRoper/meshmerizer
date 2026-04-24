"""CLI-only diagnostics and visualization helpers."""

from __future__ import annotations

from meshmerizer.logging import (
    abort_with_error,
    log_status,
    log_summary_status,
)


def visualize_vertices(vert_positions, output_path: str) -> None:
    """Save a 6-panel figure showing QEF vertices from each face.

    Args:
        vert_positions: Vertex positions with shape ``(N, 3)``.
        output_path: Destination image path.

    Returns:
        ``None``. The figure is written to disk.
    """
    # Import matplotlib lazily because this is a diagnostic-only feature and
    # most CLI runs do not need the dependency.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        abort_with_error(
            "Meshing",
            "matplotlib is required for --visualise-verts. Install it with: "
            "pip install matplotlib",
        )

    # Define the six orthographic projections once so the plotting loop only
    # needs to map axes and labels into each panel.
    views = [
        ("+X face (Y-Z)", 1, 2, "Y", "Z"),
        ("-X face (Y-Z)", 1, 2, "Y", "Z"),
        ("+Y face (X-Z)", 0, 2, "X", "Z"),
        ("-Y face (X-Z)", 0, 2, "X", "Z"),
        ("+Z face (X-Y)", 0, 1, "X", "Y"),
        ("-Z face (X-Y)", 0, 1, "X", "Y"),
    ]

    # Build all subplots up front so the diagnostic output always has a stable
    # layout regardless of the number of vertices.
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"QEF Vertices ({len(vert_positions):,} points)",
        fontsize=14,
        fontweight="bold",
    )

    # Plot each face projection as a very small scatter so dense QEF
    # clouds stay readable without huge vector output files.
    for axis, (title, horizontal, vertical, xlabel, ylabel) in zip(
        axes.flat, views
    ):
        axis.scatter(
            vert_positions[:, horizontal],
            vert_positions[:, vertical],
            s=0.5,
            alpha=0.4,
            color="C0",
            edgecolors="none",
            rasterized=True,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_aspect("equal")

    # Tight layout keeps labels readable across the six-panel grid.
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log_summary_status(
        "Saving", f"Saved vertex visualisation to {output_path}"
    )


def emit_tree_structure_summary(cells) -> None:
    """Log a concise per-depth summary of the octree structure.

    Args:
        cells: Sequence of octree cell dictionaries.

    Returns:
        ``None``. A formatted summary is emitted through the CLI logger.
    """
    # Handle the empty-tree case explicitly so callers get a predictable
    # summary instead of an exception from ``max()`` below.
    if not cells:
        log_status(
            "Tree",
            "Summary:\n"
            "total=0 leaf=0 internal=0 active=0 inactive=0 surface=0",
        )
        return

    # Size the per-depth summary table from the deepest observed cell so the
    # later aggregation loop can index by depth directly.
    max_depth = max(int(cell.get("depth", 0)) for cell in cells)
    per_depth = [
        {"total": 0, "leaf": 0, "active": 0, "surface": 0}
        for _ in range(max_depth + 1)
    ]

    total_leaf = 0
    total_active = 0
    total_surface = 0

    # Aggregate the same counters used in the human-readable total summary and
    # the depth-by-depth breakdown.
    for cell in cells:
        depth = int(cell.get("depth", 0))
        summary = per_depth[depth]
        summary["total"] += 1
        is_leaf = bool(cell.get("is_leaf", False))
        is_active = bool(cell.get("is_active", False))
        has_surface = bool(cell.get("has_surface", False))
        if is_leaf:
            summary["leaf"] += 1
            total_leaf += 1
        if is_active:
            summary["active"] += 1
            total_active += 1
        if has_surface:
            summary["surface"] += 1
            total_surface += 1

    # Build the output as lines first so the final log call stays simple
    # and the summary formatting remains easy to tweak.
    lines = [
        "Summary:",
        (
            f"total={len(cells)} leaf={total_leaf} "
            f"internal={len(cells) - total_leaf} "
            f"active={total_active} inactive={len(cells) - total_active} "
            f"surface={total_surface}"
        ),
    ]
    for depth, summary in enumerate(per_depth):
        lines.append(
            (
                f"depth {depth}: total={summary['total']} "
                f"leaf={summary['leaf']} "
                f"internal={summary['total'] - summary['leaf']} "
                f"active={summary['active']} "
                f"inactive={summary['total'] - summary['active']} "
                f"surface={summary['surface']}"
            )
        )

    log_summary_status("Tree", "\n".join(lines))


__all__ = ["emit_tree_structure_summary", "visualize_vertices"]
