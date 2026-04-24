# CLI Option Reference

This section documents Meshmerizer CLI options one page at a time.

The goal is to make complex controls easier to browse, cross-reference, and
understand in context. Each option page includes:

- a short usage example,
- the effect of the option,
- a **Requires** section when the option only makes sense with another option,
- a **Related** section for options that commonly interact.

## Input and output

- [`filename`](filename.md)
- [`--output`](output.md)

## Region selection and particle preparation

- [`--center`](center.md)
- [`--extent`](extent.md)
- [`--tight-bounds`](tight-bounds.md)
- [`--shift`](shift.md)
- [`--wrap-shift`](wrap-shift.md)
- [`--no-wrap-shift`](no-wrap-shift.md)
- [`--no-periodic`](no-periodic.md)
- [`--particle-type`](particle-type.md)
- [`--box-size`](box-size.md)
- [`--smoothing-factor`](smoothing-factor.md)

## Refinement and surface definition

- [`--base-resolution`](base-resolution.md)
- [`--max-depth`](max-depth.md)
- [`--isovalue`](isovalue.md)
- [`--surface-percentile`](surface-percentile.md)
- [`--min-usable-hermite-samples`](min-usable-hermite-samples.md)
- [`--max-qef-rms-residual-ratio`](max-qef-rms-residual-ratio.md)
- [`--min-normal-alignment-threshold`](min-normal-alignment-threshold.md)

## Topology and mesh cleanup

- [`--smoothing-iterations`](smoothing-iterations.md)
- [`--smoothing-strength`](smoothing-strength.md)
- [`--min-feature-thickness`](min-feature-thickness.md)
- [`--pre-thickening-radius`](pre-thickening-radius.md)
- [`--max-edge-ratio`](max-edge-ratio.md)
- [`--remove-islands-fraction`](remove-islands-fraction.md)
- [`--simplify-factor`](simplify-factor.md)
- [`--target-size`](target-size.md)

## Clustering

- [`--fof`](fof.md)
- [`--min-fof-cluster-size`](min-fof-cluster-size.md)
- [`--linking-factor`](linking-factor.md)

## Saved octrees and diagnostics

- [`--save-octree`](save-octree.md)
- [`--load-octree`](load-octree.md)
- [`--visualise-verts`](visualise-verts.md)
- [`--nthreads`](nthreads.md)
- [`--silent`](silent.md)
