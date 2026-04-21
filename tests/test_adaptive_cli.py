"""Tests for adaptive CLI helpers."""

import numpy as np
import trimesh

from meshmerizer.commands.adaptive_stl import _remove_islands
from meshmerizer.mesh.core import Mesh


def test_remove_islands_fraction_uses_largest_component_reference() -> None:
    """Small fluff should be compared to the largest component volume."""
    main = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    fluff = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    fluff.apply_translation((30.0, 0.0, 0.0))
    combined = trimesh.util.concatenate([main, fluff])

    cleaned = _remove_islands(
        Mesh(mesh=combined.copy()), remove_islands_fraction=0.1
    )

    components = cleaned.mesh.split(only_watertight=False)
    assert len(components) == 1
    assert np.allclose(components[0].extents, main.extents)


def test_remove_islands_keeps_large_nonwatertight_main_component() -> None:
    """A large imperfect main body should not lose out to tiny fluff."""
    main = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    main_faces = np.delete(main.faces.copy(), 0, axis=0)
    nonwatertight_main = trimesh.Trimesh(
        vertices=main.vertices.copy(), faces=main_faces, process=False
    )

    fluff = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    fluff.apply_translation((30.0, 0.0, 0.0))
    combined = trimesh.util.concatenate([nonwatertight_main, fluff])

    cleaned = _remove_islands(
        Mesh(mesh=combined.copy()), remove_islands_fraction=0.1
    )

    components = cleaned.mesh.split(only_watertight=False)
    assert len(components) == 1
    assert cleaned.mesh.bounds[1][0] < 20.0
