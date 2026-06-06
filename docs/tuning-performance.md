# Tuning Performance

Meshmerizer runtime is dominated by adaptive refinement and mesh extraction, not
by a single dense-grid pass.

That means performance tuning is mostly about controlling how much of the domain
gets refined, how deep that refinement is allowed to go, and how much parallel
native work is available.

See [Reconstruction Workflow](reconstruction-workflow.md) if you want the full
conceptual picture first.

## The most important knobs

In practice, these are the controls that matter most:

- `max_depth`: strongest runtime and memory lever
- `base_resolution`: controls the starting grid scale
- working-domain size: smaller crops are often much faster
- `nthreads` or `--nthreads`: controls native parallelism

If you only change one thing first, change `max_depth`.

## Start with `max_depth`

Lowering `max_depth` is usually the fastest way to reduce runtime.

- Lower values stop refinement earlier.
- Fewer refined leaves means less extraction work.
- Memory use also drops.

The tradeoff is that the surface becomes less locally refined, so you may see
more cell-scale artifacts or loss of small features.

A good first experiment for a slow run is to try `max_depth=3` and compare the
result.

## Balance `base_resolution` against `max_depth`

`base_resolution` and `max_depth` work together.

- Higher `base_resolution` gives a finer starting grid.
- Higher `max_depth` allows much smaller local cells after refinement.

One useful tradeoff is:

- increase `base_resolution`,
- decrease `max_depth`.

That often preserves large-scale surface quality while avoiding the worst cost
of very deep local refinement.

In other words, if deep refinement is too expensive, try moving some of that
resolution budget into the base grid instead.

## Use smoothing to compensate for shallower refinement

If you reduce `max_depth`, the extracted surface may show more blocky or
cell-shaped artifacts.

That can often be improved with more aggressive mesh smoothing after extraction.

This is a common quality/performance tradeoff:

- shallower refinement for speed,
- stronger smoothing to clean up visible discretization artifacts.

The smoothing stage is usually a cheaper knob to turn than asking the adaptive
tree to refine much more deeply.

## Reduce the domain before increasing resolution

If you only care about one object or one region, crop first.

For the CLI, use options such as:

- `--center`
- `--extent`
- `--tight-bounds`

For the Python API, pass smaller `domain_min` and `domain_max` bounds.

Reducing empty space in the working domain often improves both runtime and
effective resolution, because the same refinement budget is spent on a smaller
region.

## Use native threading

Meshmerizer can use native OpenMP workers for the expensive reconstruction
steps, but only if it was built with OpenMP support.

Build with threading enabled:

```bash
WITH_OPENMP=1 pip install -e .
```

On macOS with Homebrew `libomp`:

```bash
WITH_OPENMP=/opt/homebrew/opt/libomp pip install -e .
```

Then use:

- `--nthreads 8` in the CLI, or
- `nthreads=8` in the Python API.

If Meshmerizer was not built with OpenMP support, increasing the thread count
will not help.

## Reuse expensive intermediate state

If you are experimenting with cleanup or export settings, avoid rebuilding the
tree every time.

For the Python API:

- use `build_tree(...)` once,
- keep the returned `TreeState`,
- reuse it with `extract_mesh(...)` or `regularize(...)`.

For the CLI:

- use `--save-octree` after an expensive run,
- reload with `--load-octree` for later experiments.

This helps most when the particle data, domain, isovalue, and refinement
controls stay fixed while you iterate on cleanup and export behavior.

## A practical tuning order

When a Python or CLI workflow is too slow, try this order:

1. reduce `max_depth`,
2. crop the domain more tightly,
3. increase `base_resolution` a bit if quality dropped too far,
4. add smoothing to recover surface appearance,
5. enable OpenMP and raise thread count,
6. save or reuse octree state for repeated experiments.

## Python API example

```python
from meshmerizer import generate_mesh

result = generate_mesh(
    positions,
    smoothing_lengths,
    domain_min=(0.0, 0.0, 0.0),
    domain_max=(4.0, 4.0, 4.0),
    base_resolution=96,
    max_depth=3,
    isovalue=0.01,
    nthreads=8,
    smoothing_iterations=10,
    smoothing_strength=0.6,
)
```

This is a typical "run faster, then smooth" configuration: use a moderate base
grid, cap refinement fairly early, and rely on post-processing to recover some
surface appearance.

## Recognizing when you are over-resolving

You may be asking for too much refinement if:

- runtime rises very sharply after a small parameter change,
- memory use grows faster than expected,
- visible mesh quality improves only slightly,
- the final use case is visualization or printing rather than precision
  measurement.

In those cases, the best result is often a cheaper tree plus modest smoothing,
not deeper refinement everywhere.
