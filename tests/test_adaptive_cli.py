"""Tests for adaptive CLI helpers."""

import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import trimesh

from meshmerizer.cli import adaptive as adaptive_cli
from meshmerizer.cli.adaptive import run_adaptive
from meshmerizer.cli.args import build_parser
from meshmerizer.cli.main import main
from meshmerizer.cli.units import convert_print_length_to_native_units
from meshmerizer.logging import (
    _STATE,
    cli_logging_context,
    emit_warning_summary,
    format_status_prefix,
    log_warning_status,
)
from meshmerizer.mesh.core import Mesh
from meshmerizer.mesh.operations import remove_islands, simplify_mesh


def test_remove_islands_fraction_uses_largest_component_reference() -> None:
    """Small fluff should be compared to the largest component volume."""
    main = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    fluff = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    fluff.apply_translation((30.0, 0.0, 0.0))
    combined = trimesh.util.concatenate([main, fluff])

    cleaned = remove_islands(
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

    cleaned = remove_islands(
        Mesh(mesh=combined.copy()), remove_islands_fraction=0.1
    )

    components = cleaned.mesh.split(only_watertight=False)
    assert len(components) == 1
    assert cleaned.mesh.bounds[1][0] < 20.0


def test_remove_islands_ignores_flat_degenerate_component_without_warnings() -> (
    None
):
    """Flat zero-volume fluff should not trigger trimesh mass warnings."""
    main = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    flat = trimesh.Trimesh(
        vertices=np.array(
            [
                [20.0, 0.0, 0.0],
                [21.0, 0.0, 0.0],
                [21.0, 1.0, 0.0],
                [20.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    combined = trimesh.util.concatenate([main, flat])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cleaned = remove_islands(
            Mesh(mesh=combined.copy()), remove_islands_fraction=0.1
        )

    runtime_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
    ]
    assert runtime_warnings == []

    components = cleaned.mesh.split(only_watertight=False)
    assert len(components) == 1
    assert np.allclose(components[0].extents, main.extents)


def test_remove_islands_avoids_watertight_mass_property_warnings() -> None:
    """Watertight-marked zero-volume fluff should not hit trimesh volume."""
    main = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    flat = trimesh.Trimesh(
        vertices=np.array(
            [
                [20.0, 0.0, 0.0],
                [21.0, 0.0, 0.0],
                [21.0, 1.0, 0.0],
                [20.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array(
            [[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]], dtype=np.int64
        ),
        process=False,
    )
    combined = trimesh.util.concatenate([main, flat])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cleaned = remove_islands(
            Mesh(mesh=combined.copy()), remove_islands_fraction=0.1
        )

    runtime_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
    ]
    assert runtime_warnings == []

    components = cleaned.mesh.split(only_watertight=False)
    assert len(components) == 1
    assert np.allclose(components[0].extents, main.extents)


def test_postprocess_mesh_repairs_before_island_filtering(tmp_path) -> None:
    """CLI postprocessing should repair invalid topology before filtering."""
    from meshmerizer.cli.adaptive import _postprocess_mesh

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    bad = trimesh.Trimesh(
        vertices=box.vertices.copy(),
        faces=np.vstack([box.faces.copy(), box.faces[[0]][:, ::-1]]),
        process=False,
    )
    mesh = Mesh(mesh=bad)

    args = Namespace(
        remove_islands_fraction=None,
        simplify_factor=1.0,
        target_size=None,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        repaired = _postprocess_mesh(mesh, args)
        _ = repaired.mesh.is_watertight

    runtime_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
    ]
    assert runtime_warnings == []


def test_run_adaptive_passes_pre_thickening_radius(
    monkeypatch, tmp_path
) -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    smoothing_lengths = np.full(3, 0.1)
    captured = {}

    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.load_particles_for_adaptive",
        lambda args: (
            positions.copy(),
            smoothing_lengths.copy(),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            np.zeros(3),
        ),
    )
    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.compute_isovalue_from_percentile",
        lambda smoothing_lengths, percentile: 0.01,
    )

    def fake_reconstruct_mesh(*args, **kwargs):
        captured.update(kwargs)
        return (
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            np.array([[0, 1, 2]], dtype=np.uint32),
        )

    monkeypatch.setattr(
        "meshmerizer.cli.adaptive._reconstruct_mesh",
        fake_reconstruct_mesh,
    )
    monkeypatch.setattr(
        "meshmerizer.mesh.core.trimesh.Trimesh.export",
        lambda self, *args, **kwargs: None,
    )

    args = Namespace(
        nthreads=None,
        table_cadence=10.0,
        load_octree=None,
        save_octree=None,
        filename=Path("snapshot.hdf5"),
        output=tmp_path / "out.stl",
        min_feature_thickness=0.2,
        pre_thickening_radius=0.3,
        simplify_factor=1.0,
        target_size=None,
        max_depth=2,
        base_resolution=2,
        isovalue=None,
        surface_percentile=5.0,
        fof=False,
        linking_factor=0.2,
        smoothing_iterations=0,
        smoothing_strength=0.5,
        max_edge_ratio=1.5,
        min_usable_hermite_samples=3,
        max_qef_rms_residual_ratio=0.1,
        min_normal_alignment_threshold=0.97,
        remove_islands_fraction=None,
        visualise_verts=None,
        min_fof_cluster_size=None,
    )

    run_adaptive(args)

    assert captured["min_feature_thickness"] == 0.2
    assert captured["pre_thickening_radius"] == 0.3


def test_run_adaptive_converts_pre_thickening_in_print_units(
    monkeypatch, tmp_path
) -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    smoothing_lengths = np.full(3, 0.1)
    captured = {}

    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.load_particles_for_adaptive",
        lambda args: (
            positions.copy(),
            smoothing_lengths.copy(),
            (0.0, 0.0, 0.0),
            (2.0, 1.0, 1.0),
            np.zeros(3),
        ),
    )
    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.compute_isovalue_from_percentile",
        lambda smoothing_lengths, percentile: 0.01,
    )

    def fake_reconstruct_mesh(*args, **kwargs):
        captured.update(kwargs)
        return (
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            np.array([[0, 1, 2]], dtype=np.uint32),
        )

    monkeypatch.setattr(
        "meshmerizer.cli.adaptive._reconstruct_mesh",
        fake_reconstruct_mesh,
    )
    monkeypatch.setattr(
        "meshmerizer.mesh.core.trimesh.Trimesh.export",
        lambda self, *args, **kwargs: None,
    )

    args = Namespace(
        nthreads=None,
        table_cadence=10.0,
        load_octree=None,
        save_octree=None,
        filename=Path("snapshot.hdf5"),
        output=tmp_path / "out.stl",
        min_feature_thickness=0.2,
        pre_thickening_radius=0.4,
        simplify_factor=1.0,
        target_size=10.0,
        max_depth=2,
        base_resolution=2,
        isovalue=None,
        surface_percentile=5.0,
        fof=False,
        linking_factor=0.2,
        smoothing_iterations=0,
        smoothing_strength=0.5,
        max_edge_ratio=1.5,
        min_usable_hermite_samples=3,
        max_qef_rms_residual_ratio=0.1,
        min_normal_alignment_threshold=0.97,
        remove_islands_fraction=None,
        visualise_verts=None,
        min_fof_cluster_size=None,
    )

    run_adaptive(args)

    expected = convert_print_length_to_native_units(
        0.4,
        (0.0, 0.0, 0.0),
        (2.0, 1.0, 1.0),
        10.0,
    )
    assert np.isclose(captured["pre_thickening_radius"], expected)


def test_run_adaptive_converts_regularization_lengths_for_loaded_octree(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.import_octree",
        lambda path: {
            "positions": np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
                dtype=np.float64,
            ),
            "smoothing_lengths": np.full(3, 0.1, dtype=np.float64),
            "domain_minimum": (0.0, 0.0, 0.0),
            "domain_maximum": (2.0, 1.0, 1.0),
            "cells": [],
            "contributors": [],
            "isovalue": 0.01,
            "max_depth": 2,
            "base_resolution": 2,
        },
    )
    monkeypatch.setattr(
        "meshmerizer.cli.adaptive._reconstruct_mesh",
        lambda *args, **kwargs: (
            captured.update(kwargs),
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            np.array([[0, 1, 2]], dtype=np.uint32),
        )[1:],
    )
    monkeypatch.setattr(
        "meshmerizer.mesh.core.trimesh.Trimesh.export",
        lambda self, *args, **kwargs: None,
    )

    args = Namespace(
        nthreads=None,
        table_cadence=10.0,
        load_octree=tmp_path / "tree.hdf5",
        save_octree=None,
        filename=Path("snapshot.hdf5"),
        output=tmp_path / "out.stl",
        min_feature_thickness=0.2,
        pre_thickening_radius=0.4,
        simplify_factor=1.0,
        target_size=10.0,
        max_depth=2,
        base_resolution=2,
        isovalue=None,
        surface_percentile=5.0,
        fof=False,
        linking_factor=0.2,
        smoothing_iterations=0,
        smoothing_strength=0.5,
        max_edge_ratio=1.5,
        min_usable_hermite_samples=3,
        max_qef_rms_residual_ratio=0.1,
        min_normal_alignment_threshold=0.97,
        remove_islands_fraction=None,
        visualise_verts=None,
        min_fof_cluster_size=None,
        center=None,
        extent=None,
        silent=False,
    )

    run_adaptive(args)

    expected_thickening = convert_print_length_to_native_units(
        0.4,
        (0.0, 0.0, 0.0),
        (2.0, 1.0, 1.0),
        10.0,
    )
    expected_thickness = convert_print_length_to_native_units(
        0.2,
        (0.0, 0.0, 0.0),
        (2.0, 1.0, 1.0),
        10.0,
    )
    assert np.isclose(captured["pre_thickening_radius"], expected_thickening)
    assert np.isclose(captured["min_feature_thickness"], expected_thickness)


def test_build_parser_supports_top_level_help() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_build_parser_accepts_table_cadence() -> None:
    parser = build_parser()

    args = parser.parse_args(["snapshot.hdf5", "--table-cadence", "7.5"])

    assert args.table_cadence == pytest.approx(7.5)


def test_main_without_arguments_prints_usage(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Provide a snapshot filename or use --load-octree." in captured.out


def test_run_adaptive_allows_missing_filename_with_loaded_octree(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.import_octree",
        lambda path: {
            "positions": np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
                dtype=np.float64,
            ),
            "smoothing_lengths": np.full(3, 0.1, dtype=np.float64),
            "domain_minimum": (0.0, 0.0, 0.0),
            "domain_maximum": (2.0, 1.0, 1.0),
            "cells": [],
            "contributors": [],
            "isovalue": 0.01,
            "max_depth": 2,
            "base_resolution": 2,
        },
    )
    monkeypatch.setattr(
        "meshmerizer.cli.adaptive._reconstruct_mesh",
        lambda *args, **kwargs: (
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            np.array([[0, 1, 2]], dtype=np.uint32),
        ),
    )
    monkeypatch.setattr(
        "meshmerizer.mesh.core.trimesh.Trimesh.export",
        lambda self, *args, **kwargs: None,
    )

    args = Namespace(
        nthreads=None,
        table_cadence=10.0,
        load_octree=tmp_path / "tree.hdf5",
        save_octree=None,
        filename=None,
        output=None,
        min_feature_thickness=0.0,
        pre_thickening_radius=0.0,
        simplify_factor=1.0,
        target_size=None,
        max_depth=2,
        base_resolution=2,
        isovalue=None,
        surface_percentile=5.0,
        fof=False,
        linking_factor=0.2,
        smoothing_iterations=0,
        smoothing_strength=0.5,
        max_edge_ratio=1.5,
        min_usable_hermite_samples=3,
        max_qef_rms_residual_ratio=0.1,
        min_normal_alignment_threshold=0.97,
        remove_islands_fraction=None,
        visualise_verts=None,
        min_fof_cluster_size=None,
        center=None,
        extent=None,
        silent=False,
    )

    run_adaptive(args)


def test_run_adaptive_loaded_octree_reuses_tree_for_direct_meshing(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.import_octree",
        lambda path: {
            "positions": np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
                dtype=np.float64,
            ),
            "smoothing_lengths": np.full(3, 0.1, dtype=np.float64),
            "domain_minimum": (0.0, 0.0, 0.0),
            "domain_maximum": (2.0, 1.0, 1.0),
            "cells": [{"is_leaf": True}],
            "contributors": np.array([0, 1, 2], dtype=np.int64),
            "isovalue": 0.01,
            "max_depth": 2,
            "base_resolution": 2,
        },
    )
    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.generate_mesh",
        lambda *args, **kwargs: (
            captured.setdefault("generate_mesh_called", True),
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            np.zeros((3, 3), dtype=np.float64),
            np.array([[0, 1, 2]], dtype=np.uint32),
        )[1:],
    )
    monkeypatch.setattr(
        "meshmerizer.cli.adaptive._reconstruct_mesh",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("_reconstruct_mesh should not be called")
        ),
    )
    monkeypatch.setattr(
        "meshmerizer.mesh.core.trimesh.Trimesh.export",
        lambda self, *args, **kwargs: None,
    )

    args = Namespace(
        nthreads=None,
        table_cadence=10.0,
        load_octree=tmp_path / "tree.hdf5",
        save_octree=None,
        filename=None,
        output=tmp_path / "out.stl",
        min_feature_thickness=0.0,
        pre_thickening_radius=0.0,
        simplify_factor=1.0,
        target_size=None,
        max_depth=2,
        base_resolution=2,
        isovalue=None,
        surface_percentile=5.0,
        fof=False,
        linking_factor=0.2,
        smoothing_iterations=0,
        smoothing_strength=0.5,
        max_edge_ratio=0.0,
        min_usable_hermite_samples=3,
        max_qef_rms_residual_ratio=0.1,
        min_normal_alignment_threshold=0.97,
        remove_islands_fraction=None,
        visualise_verts=None,
        min_fof_cluster_size=None,
        center=None,
        extent=None,
        silent=False,
    )

    run_adaptive(args)

    assert captured["generate_mesh_called"] is True


def test_run_adaptive_errors_when_no_input_source_is_given() -> None:
    args = Namespace(
        nthreads=None,
        table_cadence=10.0,
        load_octree=None,
        save_octree=None,
        filename=None,
        output=None,
        min_feature_thickness=0.0,
        pre_thickening_radius=0.0,
        simplify_factor=1.0,
        target_size=None,
        max_depth=2,
        base_resolution=2,
        isovalue=None,
        surface_percentile=5.0,
        fof=False,
        linking_factor=0.2,
        smoothing_iterations=0,
        smoothing_strength=0.5,
        max_edge_ratio=1.5,
        min_usable_hermite_samples=3,
        max_qef_rms_residual_ratio=0.1,
        min_normal_alignment_threshold=0.97,
        remove_islands_fraction=None,
        visualise_verts=None,
        min_fof_cluster_size=None,
        center=None,
        extent=None,
        silent=False,
    )

    with pytest.raises(SystemExit):
        run_adaptive(args)


def test_main_accepts_snapshot_without_subcommand(monkeypatch) -> None:
    called = {}

    monkeypatch.setattr(
        "meshmerizer.cli.args.run_adaptive",
        lambda args: called.setdefault("filename", args.filename),
    )

    main(["snapshot.hdf5", "--silent"])

    assert str(called["filename"]) == "snapshot.hdf5"


def test_main_exits_cleanly_on_keyboard_interrupt(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "meshmerizer.cli.args.run_adaptive",
        lambda args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(SystemExit) as excinfo:
        main(["snapshot.hdf5"])

    assert excinfo.value.code == 130
    captured = capsys.readouterr()
    assert (
        "Interrupted by user; cancelled without writing output."
        in captured.out
    )


def test_run_adaptive_removes_temporary_output_on_interrupt(
    monkeypatch, tmp_path
) -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    smoothing_lengths = np.full(3, 0.1)
    output_path = tmp_path / "out.stl"
    temp_path = tmp_path / "out.tmp.stl"

    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.load_particles_for_adaptive",
        lambda args: (
            positions.copy(),
            smoothing_lengths.copy(),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            np.zeros(3),
        ),
    )
    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.compute_isovalue_from_percentile",
        lambda smoothing_lengths, percentile: 0.01,
    )
    monkeypatch.setattr(
        "meshmerizer.cli.adaptive._reconstruct_mesh",
        lambda *args, **kwargs: (
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            np.array([[0, 1, 2]], dtype=np.uint32),
        ),
    )

    def interrupting_export(self, filename, *args, **kwargs):
        Path(filename).write_text("partial")
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        "meshmerizer.mesh.core.trimesh.Trimesh.export",
        interrupting_export,
    )

    args = Namespace(
        nthreads=None,
        table_cadence=10.0,
        load_octree=None,
        save_octree=None,
        filename=Path("snapshot.hdf5"),
        output=output_path,
        min_feature_thickness=0.0,
        pre_thickening_radius=0.0,
        simplify_factor=1.0,
        target_size=None,
        max_depth=2,
        base_resolution=2,
        isovalue=None,
        surface_percentile=5.0,
        fof=False,
        linking_factor=0.2,
        smoothing_iterations=0,
        smoothing_strength=0.5,
        max_edge_ratio=1.5,
        min_usable_hermite_samples=3,
        max_qef_rms_residual_ratio=0.1,
        min_normal_alignment_threshold=0.97,
        remove_islands_fraction=None,
        visualise_verts=None,
        min_fof_cluster_size=None,
    )

    with pytest.raises(KeyboardInterrupt):
        run_adaptive(args)

    assert not output_path.exists()
    assert not temp_path.exists()


def test_format_status_prefix_includes_operation_function_and_thread() -> None:
    prefix = format_status_prefix("Loading", func="run_adaptive")

    assert prefix == "[Loading][run_adaptive][main]"


def test_simplify_mesh_noop_when_factor_is_one() -> None:
    mesh = Mesh(mesh=trimesh.creation.box())
    original_faces = len(mesh.faces)

    result = simplify_mesh(mesh, 1.0)

    assert result is mesh
    assert len(mesh.faces) == original_faces


def test_run_adaptive_applies_simplify_factor(monkeypatch, tmp_path) -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    smoothing_lengths = np.full(3, 0.1)
    simplified = {}

    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.load_particles_for_adaptive",
        lambda args: (
            positions.copy(),
            smoothing_lengths.copy(),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            np.zeros(3),
        ),
    )
    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.compute_isovalue_from_percentile",
        lambda smoothing_lengths, percentile: 0.01,
    )
    monkeypatch.setattr(
        "meshmerizer.cli.adaptive._reconstruct_mesh",
        lambda *args, **kwargs: (
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            np.array([[0, 1, 2]], dtype=np.uint32),
        ),
    )
    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.simplify_mesh",
        lambda mesh, factor: (simplified.setdefault("factor", factor), mesh)[
            1
        ],
    )
    monkeypatch.setattr(
        "meshmerizer.mesh.core.trimesh.Trimesh.export",
        lambda self, *args, **kwargs: None,
    )

    args = Namespace(
        nthreads=None,
        table_cadence=10.0,
        load_octree=None,
        save_octree=None,
        filename=Path("snapshot.hdf5"),
        output=tmp_path / "out.stl",
        min_feature_thickness=0.0,
        pre_thickening_radius=0.0,
        simplify_factor=0.4,
        target_size=None,
        max_depth=2,
        base_resolution=2,
        isovalue=None,
        surface_percentile=5.0,
        fof=False,
        linking_factor=0.2,
        smoothing_iterations=0,
        smoothing_strength=0.5,
        max_edge_ratio=1.5,
        min_usable_hermite_samples=3,
        max_qef_rms_residual_ratio=0.1,
        min_normal_alignment_threshold=0.97,
        remove_islands_fraction=None,
        visualise_verts=None,
        min_fof_cluster_size=None,
    )

    run_adaptive(args)

    assert simplified["factor"] == 0.4


def test_warning_summary_is_deferred_until_end(monkeypatch) -> None:
    records = []

    def capture_summary(operation, message, *, thread=None):
        records.append(message)

    with cli_logging_context():
        monkeypatch.setattr(
            "meshmerizer.logging.log_summary_status", capture_summary
        )
        log_warning_status("Cleaning", "deferred warning")
        assert not any("deferred warning" in message for message in records)
        emit_warning_summary()

    assert any("deferred warning" in message for message in records)


def test_progress_bar_respects_silent_mode() -> None:
    with cli_logging_context(silent=True):
        assert _STATE.silent is True


def test_reconstruct_mesh_runs_fof_groups_in_parallel_and_merges_faces(
    monkeypatch,
) -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [1.0, 1.0, 1.0],
            [1.1, 1.0, 1.0],
            [1.0, 1.1, 1.0],
        ],
        dtype=np.float64,
    )
    smoothing_lengths = np.full(6, 0.1, dtype=np.float64)
    labels = np.array([4, 4, 4, 9, 9, 9], dtype=np.int64)
    seen_groups = []

    def fake_run_full_pipeline(
        group_pos,
        group_sml,
        domain_min,
        domain_max,
        base_resolution,
        isovalue,
        max_depth,
        **kwargs,
    ):
        seen_groups.append(group_pos.copy())
        if np.allclose(group_pos[:, 0], [0.0, 0.1, 0.0]):
            return {
                "vertices": np.array(
                    [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
                    dtype=np.float64,
                ),
                "faces": np.array([[0, 1, 2]], dtype=np.uint32),
            }
        return {
            "vertices": np.array(
                [[1.0, 1.0, 1.0], [1.2, 1.0, 1.0], [1.0, 1.2, 1.0]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.uint32),
        }

    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.run_full_pipeline",
        fake_run_full_pipeline,
    )

    vertices, faces = adaptive_cli._reconstruct_mesh(
        positions,
        smoothing_lengths,
        (0.0, 0.0, 0.0),
        (2.0, 2.0, 2.0),
        4,
        0.01,
        2,
        worker_count=2,
        group_labels=labels,
    )

    assert len(seen_groups) == 2
    assert vertices.shape == (6, 3)
    np.testing.assert_array_equal(
        faces,
        np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
    )


def test_reconstruct_mesh_skips_undersized_fof_groups(monkeypatch) -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    smoothing_lengths = np.full(4, 0.1, dtype=np.float64)
    labels = np.array([1, 1, 1, 2], dtype=np.int64)
    call_sizes = []

    def fake_run_full_pipeline(
        group_pos,
        group_sml,
        domain_min,
        domain_max,
        base_resolution,
        isovalue,
        max_depth,
        **kwargs,
    ):
        call_sizes.append(group_pos.shape[0])
        return {
            "vertices": np.array(
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.uint32),
        }

    monkeypatch.setattr(
        "meshmerizer.cli.adaptive.run_full_pipeline",
        fake_run_full_pipeline,
    )

    vertices, faces = adaptive_cli._reconstruct_mesh(
        positions,
        smoothing_lengths,
        (0.0, 0.0, 0.0),
        (2.0, 2.0, 2.0),
        4,
        0.01,
        2,
        worker_count=2,
        group_labels=labels,
    )

    assert call_sizes == [3]
    assert vertices.shape == (3, 3)
    np.testing.assert_array_equal(faces, np.array([[0, 1, 2]], dtype=np.int64))
