# Reconstruction Workflow

Meshmerizer does not build a dense uniform voxel grid for the whole domain
first. Instead, it adaptively refines only where the surface needs more
resolution.

That is the main reason the code can represent detailed surfaces without paying
the full memory cost of a uniformly fine grid. It is also the reason runtime can
change a lot when you adjust refinement settings.

## High-level stages

At a high level, Meshmerizer does this:

1. partitions the working domain into top-level cells,
2. assigns particles to those cells,
3. adaptively refines an octree where the field varies rapidly,
4. finds leaf cells that contain the target isosurface,
5. interpolates surface samples inside those leaves,
6. solves vertex positions and connects them into faces,
7. optionally regularizes thin features and post-processes the final mesh.

## Stage 1: Build the base grid

The reconstruction starts from a coarse Cartesian layout with
`base_resolution` cells per axis.

This base grid defines the starting spatial scale of the octree. A larger
`base_resolution` means the algorithm begins from a finer top-level partition.

## Stage 2: Sort particles into cells

Particles are associated with the cells they can influence through their
smoothing lengths.

This gives each part of the domain a local list of contributors, so later field
evaluation and refinement decisions do not have to treat the whole particle set
as equally relevant everywhere.

## Stage 3: Refine the octree adaptively

This is the central step of the algorithm.

Meshmerizer evaluates the field and its local surface information, then refines
cells where the surface is not yet represented well enough. In practice,
refinement is driven by the local field behavior and by surface-fit quality
criteria such as:

- whether the cell appears to contain the isosurface,
- whether enough usable Hermite samples exist,
- whether the local QEF fit is good enough,
- whether local surface normals are sufficiently consistent.

Refinement continues until either:

- the local surface estimate is considered good enough, or
- the cell reaches `max_depth`.

This is why Meshmerizer behaves a lot like an AMR code. In difficult regions,
the refined leaf size can become much smaller than the base grid size.

## Stage 4: Identify surface leaves

Once refinement stops, the code walks the leaf cells and finds those that hold a
crossing of the target `isovalue`.

These are the cells that contribute to the extracted surface. Leaves that are
completely inside or outside the solid do not directly produce surface
vertices.

## Stage 5: Interpolate surface information

For surface leaves, Meshmerizer interpolates positions and normals associated
with the isosurface crossing.

Those samples are the geometric constraints used to place vertices inside each
leaf.

## Stage 6: Solve vertices and build faces

Meshmerizer then solves for one representative surface vertex per active leaf
and connects neighboring leaves to build faces.

This extraction step is one of the most expensive parts of the pipeline,
especially once refinement has created many small leaf cells.

## Stage 7: Optional regularization and cleanup

After extraction, Meshmerizer can optionally:

- remove or suppress thin features,
- smooth the surface,
- remove small disconnected islands,
- simplify the final mesh,
- scale the output for printing.

These steps do not usually dominate total runtime in the way adaptive
refinement and face construction do, but they matter for final mesh quality.

## Where the time usually goes

For most expensive runs, the main cost is in:

- adaptive refinement,
- repeated local field evaluation during refinement,
- surface extraction and face construction.

That means the most important performance controls are usually:

- `base_resolution`,
- `max_depth`,
- the size of the selected working domain,
- whether OpenMP threading is enabled and used.

See [Tuning Performance](tuning-performance.md) for practical guidance.

## How this maps to the Python API

The staged Python API exposes the same broad phases:

- `build_tree(...)` handles the expensive adaptive tree construction,
- `regularize(...)` builds the opened-solid topology for thickness cleanup,
- `extract_mesh(...)` extracts the final surface mesh.

If tree construction is the expensive part of your workflow, keep the
`TreeState` around and reuse it rather than rebuilding from particles for every
experiment.
