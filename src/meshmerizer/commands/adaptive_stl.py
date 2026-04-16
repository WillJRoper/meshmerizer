"""Adaptive meshing CLI command.

This module implements the ``adaptive`` subcommand which runs the new
adaptive octree + dual contouring pipeline from a SWIFT snapshot to an
STL file.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import numpy as np

from meshmerizer.adaptive_core import (
    create_top_level_cells,
    generate_mesh,
    query_cell_contributors,
    refine_octree,
)
from meshmerizer.logging import log_status, record_elapsed
from meshmerizer.mesh.core import Mesh
from meshmerizer.printing import scale_mesh_to_print
from meshmerizer.serialize import export_octree, import_octree


def _remove_islands(
    mesh: Mesh,
    remove_islands_fraction: Optional[float],
) -> Mesh:
    """Remove small disconnected components from a mesh.

    Args:
        mesh: Input mesh.
        remove_islands_fraction: Fraction of the total volume below
            which a connected component is discarded.  ``0.0`` keeps
            only the largest component.  ``None`` disables island
            removal entirely.

    Returns:
        A new ``Mesh`` with small islands removed, or the original
        mesh unchanged when removal is disabled.
    """
    if remove_islands_fraction is None:
        return mesh

    # Split into connected components via trimesh.
    tm = mesh.mesh
    components = tm.split(only_watertight=False)
    if len(components) <= 1:
        return mesh

    if remove_islands_fraction == 0.0:
        # Keep only the single largest component by volume.
        largest = max(
            components,
            key=lambda c: abs(c.volume) if c.is_watertight else 0.0,
        )
        log_status(
            "Cleaning",
            f"Kept largest of {len(components)} components.",
        )
        return Mesh(mesh=largest)

    # Keep components whose volume fraction exceeds the threshold.
    volumes = []
    for comp in components:
        vol = abs(comp.volume) if comp.is_watertight else 0.0
        volumes.append(vol)
    total_volume = sum(volumes)
    if total_volume == 0.0:
        return mesh

    kept = [
        c
        for c, v in zip(components, volumes)
        if v / total_volume >= remove_islands_fraction
    ]
    if not kept:
        # Keep at least the largest if nothing passes the threshold.
        kept = [
            max(
                components,
                key=lambda c: abs(c.volume) if c.is_watertight else 0.0,
            )
        ]
    log_status(
        "Cleaning",
        f"Kept {len(kept)} of {len(components)} components "
        f"(fraction >= {remove_islands_fraction}).",
    )
    import trimesh

    combined = trimesh.util.concatenate(kept)
    return Mesh(mesh=combined)


def _load_particles_for_adaptive(args):
    """Load and prepare particles for the adaptive pipeline.

    This reuses the existing ``load_swift_particles`` function from
    the loading module, then converts the outputs into the format
    expected by the adaptive C++ core.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Tuple of ``(positions, smoothing_lengths, domain_min,
        domain_max, origin)`` where positions is a list of 3-tuples,
        smoothing_lengths is a list of floats, domain_min and
        domain_max are 3-tuples, and origin is a numpy array.
    """
    from meshmerizer.commands.loading import load_swift_particles

    field_data, coords, h, effective_box_size, origin = load_swift_particles(
        filename=args.filename,
        particle_type=args.particle_type,
        field=args.field,
        smoothing_factor=args.smoothing_factor,
        box_size=args.box_size,
        shift=list(args.shift),
        wrap_shift=args.wrap_shift,
        center=args.center,
        extent=args.extent,
        periodic=args.periodic,
        tight_bounds=args.tight_bounds,
    )

    if h is None:
        print(
            "Error: Smoothing lengths are required for the "
            "adaptive pipeline but could not be determined.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_particles = coords.shape[0]
    if n_particles == 0:
        print(
            "Error: No particles selected. Check domain selection flags.",
            file=sys.stderr,
        )
        sys.exit(1)

    log_status(
        "Loading",
        f"Prepared {n_particles} particles for adaptive meshing.",
    )

    # Convert to the list-of-tuples format the C++ extension expects.
    positions = [tuple(row) for row in coords.tolist()]
    smoothing_lengths = h.tolist()

    # The adaptive pipeline uses an explicit domain bounding box.
    # After loading, coordinates live in [0, effective_box_size)^3.
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (
        float(effective_box_size),
        float(effective_box_size),
        float(effective_box_size),
    )

    return positions, smoothing_lengths, domain_min, domain_max, origin


def _visualize_vertices(vertices) -> None:
    """Open a 3D scatter plot of QEF vertex positions.

    Each vertex is a ``(position, normal)`` tuple as returned by
    ``generate_mesh``.  Only the position is plotted.

    Args:
        vertices: List of ``((x, y, z), (nx, ny, nz))`` tuples.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Error: matplotlib is required for --visualize. "
            "Install it with: pip install matplotlib",
            file=sys.stderr,
        )
        return

    positions = np.array([v[0] for v in vertices], dtype=np.float64)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=1,
        alpha=0.6,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"QEF Vertices ({len(vertices)} points)")
    plt.tight_layout()
    plt.show()


def run_adaptive(args) -> None:
    """Run the adaptive meshing pipeline from CLI arguments.

    Args:
        args: Parsed CLI namespace containing all adaptive flags.

    Returns:
        ``None``. The output STL is written to disk.
    """
    total_start = time.perf_counter()

    # Set OpenMP thread count if requested.
    if args.nthreads is not None:
        from meshmerizer._adaptive import set_num_threads

        set_num_threads(args.nthreads)
        log_status(
            "Config",
            f"OpenMP threads set to {args.nthreads}.",
        )

    # ------------------------------------------------------------------
    # Step 1: Load from HDF5 octree or from a SWIFT snapshot.
    # ------------------------------------------------------------------
    if args.load_octree is not None:
        log_status("Loading", f"Loading octree from {args.load_octree}")
        load_start = time.perf_counter()
        state = import_octree(str(args.load_octree))
        positions = state["positions"]
        smoothing_lengths = state["smoothing_lengths"]
        domain_min = state["domain_minimum"]
        domain_max = state["domain_maximum"]
        cells = state["cells"]
        contributors = state["contributors"]
        isovalue = state["isovalue"]
        max_depth = state["max_depth"]
        base_resolution = state["base_resolution"]
        origin = np.zeros(3, dtype=np.float64)
        record_elapsed("Octree load", load_start, operation="Loading")

        log_status(
            "Loading",
            f"Loaded {len(cells)} cells, "
            f"{len(positions)} particles from HDF5.",
        )
    else:
        # Load particles from the SWIFT snapshot.
        positions, smoothing_lengths, domain_min, domain_max, origin = (
            _load_particles_for_adaptive(args)
        )
        isovalue = args.isovalue
        max_depth = args.max_depth
        base_resolution = args.base_resolution

        # ----------------------------------------------------------
        # Step 2: Build the octree.
        # ----------------------------------------------------------
        log_status(
            "Building",
            f"Building octree: base_resolution={base_resolution}, "
            f"max_depth={max_depth}, isovalue={isovalue}",
        )
        tree_start = time.perf_counter()

        # Create top-level cells and attach contributors.
        top_cells = create_top_level_cells(
            domain_min, domain_max, base_resolution
        )
        initial_cells = []
        for cell in top_cells:
            cell_dict = dict(cell)
            # Attach all particles as contributors for each
            # top-level cell; refine_octree will filter during
            # refinement.  For large particle counts a spatial
            # query per cell would be faster, but the current
            # C++ contributor filtering handles this correctly.
            contribs = query_cell_contributors(
                positions,
                smoothing_lengths,
                domain_min,
                domain_max,
                base_resolution,
                cell["bounds"][0],
                cell["bounds"][1],
            )
            cell_dict["contributor_begin"] = 0
            cell_dict["contributor_end"] = len(contribs)
            # Store the actual contributor indices for
            # refine_octree to use.
            cell_dict["contributors"] = contribs
            initial_cells.append(cell_dict)

        # Run breadth-first refinement with balancing.
        cells, contributors = refine_octree(
            initial_cells,
            positions,
            smoothing_lengths,
            isovalue,
            max_depth,
        )
        record_elapsed("Octree construction", tree_start, operation="Building")

        log_status(
            "Building",
            f"Octree built: {len(cells)} cells, "
            f"{sum(1 for c in cells if c['is_leaf'])} leaves.",
        )

    # ------------------------------------------------------------------
    # Step 3: Optionally save the octree state.
    # ------------------------------------------------------------------
    if args.save_octree is not None:
        log_status("Saving", f"Saving octree to {args.save_octree}")
        save_start = time.perf_counter()
        export_octree(
            str(args.save_octree),
            isovalue=isovalue,
            base_resolution=base_resolution,
            max_depth=max_depth,
            domain_minimum=domain_min,
            domain_maximum=domain_max,
            positions=positions,
            smoothing_lengths=smoothing_lengths,
            cells=cells,
            contributors=contributors,
        )
        record_elapsed("Octree save", save_start, operation="Saving")

    # ------------------------------------------------------------------
    # Step 4: Generate the mesh via dual contouring.
    # ------------------------------------------------------------------
    log_status("Meshing", "Generating mesh via dual contouring...")
    mesh_start = time.perf_counter()
    vertices, triangles = generate_mesh(
        cells,
        contributors,
        positions,
        smoothing_lengths,
        isovalue,
        domain_min,
        domain_max,
        max_depth,
        base_resolution,
    )
    record_elapsed("Mesh generation", mesh_start, operation="Meshing")

    n_verts = len(vertices)
    n_tris = len(triangles)
    log_status(
        "Meshing",
        f"Generated mesh: {n_verts} vertices, {n_tris} triangles.",
    )

    # Optionally visualize QEF vertices as a 3D scatter plot.
    if getattr(args, "visualize", False):
        _visualize_vertices(vertices)

    if n_tris == 0:
        print(
            "Warning: Mesh generation produced no triangles. "
            "Check isovalue and domain selection.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Convert to a Mesh object and apply post-processing.
    # ------------------------------------------------------------------

    # Build numpy arrays from the C++ output.
    vert_positions = np.array([v[0] for v in vertices], dtype=np.float64)
    vert_normals = np.array([v[1] for v in vertices], dtype=np.float64)
    tri_indices = np.array(triangles, dtype=np.int64)

    # Translate vertex positions back to world-space coordinates.
    vert_positions += origin

    mesh = Mesh(
        vertices=vert_positions,
        faces=tri_indices,
        vertex_normals=vert_normals,
    )

    # Remove small disconnected islands if requested.
    mesh = _remove_islands(mesh, args.remove_islands_fraction)

    # Scale the mesh to a physical print size if requested.
    if args.target_size is not None:
        log_status(
            "Cleaning",
            f"Scaling mesh to {args.target_size} cm...",
        )
        mesh = scale_mesh_to_print(mesh, args.target_size)

    # ------------------------------------------------------------------
    # Step 6: Save the STL.
    # ------------------------------------------------------------------
    output_path = args.output
    if output_path is None:
        output_path = args.filename.with_suffix(".stl")

    log_status("Saving", f"Writing STL to {output_path}...")
    mesh.save(str(output_path))

    record_elapsed("Total pipeline", total_start, operation="Done")
    log_status("Done", f"Adaptive mesh saved to {output_path}")
