"""Test convenience functions for creating and reading meshes."""

import shlex
import subprocess
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import pyvista as pv

import tommos_adapter as ta


@pytest.mark.parametrize(
    "geom, extent, h, out_name, additional_arguments",
    [("box", [1, 2, 3], 1, "box.vtu", "--backend grid"), ("ellipsoid", [4, 5, 6], 2, "ellip", "--n 3")],
)
def test_create_mesh_call_tommos(monkeypatch, geom, extent, h, out_name, additional_arguments):
    """Test call to tommos via `create_mesh` with possible different arguments."""
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr("tommos_adapter._mesh.subprocess.run", run)
    monkeypatch.setattr("tommos_adapter._mesh.pv.read", lambda x: None)
    read_mesh = Mock(return_value=pv.UnstructuredGrid())
    monkeypatch.setattr("tommos_adapter._mesh.read_mesh", read_mesh)
    extent_string = f"{extent[0]},{extent[1]},{extent[2]}"
    expected_cmd = shlex.split(
        f"tommos mesh --geom {geom} --extent {extent_string} --h {h} --out-name {out_name} {additional_arguments}"
    )

    ta.create_mesh(geom, extent, h=h, out_name=out_name, additional_arguments=additional_arguments)
    run.assert_called_once_with(
        expected_cmd,
        cwd=Path("."),
        capture_output=True,
    )
    read_mesh.assert_called_once_with(None, mesh_unit_length=1e-9, mat_id_name_mapping=None)


@pytest.mark.parametrize("geom", ["non_supported"])
def test_create_mesh_wrong_geom(geom):
    with pytest.raises(ValueError, match="Argument `geom` must be one of"):
        ta.create_mesh(geom=geom, extent=(1, 1, 1), h=1, out_name="test")


@pytest.mark.parametrize("extent", ["string", (1, 2), (5, 4, 3 + 2j), (1, 4, -1), (10, np.nan, 3)])
def test_create_mesh_wrong_extent(extent):
    with pytest.raises(ValueError, match="Argument `extent` must be an array-like of length 3 of finite positive"):
        ta.create_mesh(geom="box", extent=extent, h=1, out_name="test")


@pytest.mark.parametrize("h", ["string", (1, 2), 3 + 2j, np.nan, np.inf])
def test_create_mesh_wrong_h(h):
    with pytest.raises(ValueError, match="Argument `h` must be a positive finite real number"):
        ta.create_mesh(geom="box", extent=(20, 20, 20), h=h, out_name="test")


def test_create_mesh_process_fails():
    """Check that stderr of failing process is printed."""
    with pytest.raises(RuntimeError, match=r"tommos mesh: error: argument --geom: invalid choice"):
        ta.create_mesh(geom="box", extent=(1, 1, 1), h=1, out_name="test", additional_arguments="--geom error")


@pytest.mark.parametrize("mesh_unit_length, mat_id, mat_id_name_mapping", [(1e-6, 3, None), (1e-9, 1, {1: "region_1"})])
def test_read_mesh_possible_args(mesh_files, mesh_unit_length, mat_id, mat_id_name_mapping):
    """Test `read_mesh` with different possible arguments."""
    mesh = pv.read(mesh_files["vtu"])
    mesh.cell_data["mat_id"] = np.full(mesh.n_cells, mat_id, dtype=np.int32)
    mesh = ta.read_mesh(mesh, mesh_unit_length=mesh_unit_length, mat_id_name_mapping=mat_id_name_mapping)
    assert mesh.field_data["mesh_unit_length"] == mesh_unit_length
    np.testing.assert_equal(mesh.cell_data["mat_id"], mat_id)
    np.testing.assert_equal(mesh.field_data["mat_ids"], [mat_id])
    expected_mat_names = [f"grain_{mat_id}"] if mat_id_name_mapping is None else list(mat_id_name_mapping.values())
    assert all(mesh.field_data["mat_names"] == expected_mat_names)


def test_read_mesh_wrong_mat_id(mesh_files):
    """We modify the cell data `mat_id` in the fixture mesh to check that different dtypes are not accepted."""
    mesh = pv.read(mesh_files["vtu"])
    mesh.cell_data["mat_id"] = np.ones(mesh.n_cells, np.float64)
    with pytest.raises(ValueError, match="Cell data 'mat_id' should be an array of integers"):
        ta.read_mesh(mesh)


def test_no_mat_id_warning(mesh_files):
    mesh = pv.read(mesh_files["vtu"])
    del mesh.cell_data["mat_id"]
    with pytest.warns(UserWarning, match="Given mesh does not have a 'mat_id' field data"):
        ta.read_mesh(mesh)


def test_read_mesh_multigrain(mesh_files):
    """We simulate a multigrain material by modifying a cell to have a different `mat_id`."""
    mesh = pv.read(mesh_files["vtu"])
    mesh.cell_data["mat_id"][0] = mesh.cell_data["mat_id"][0] + 1
    with pytest.raises(NotImplementedError, match="does not support multigrain materials"):
        ta.read_mesh(mesh)


@pytest.mark.parametrize("mesh_unit_length", ["string", np.array(1e-9), -1, 0, np.nan])
def test_read_mesh_wrong_mesh_unit_length(mesh_files, mesh_unit_length):
    """Test failing values of `mesh_unit_length`."""
    mesh = pv.read(mesh_files["vtu"])
    with pytest.raises(ValueError, match="`mesh_unit_length` must be a positive finite real number"):
        ta.read_mesh(mesh, mesh_unit_length=mesh_unit_length)


@pytest.mark.parametrize("mat_id_name_mapping", [{2: "grain_2"}, {"1": "grain_1"}, {1: "grain_1", 2: "grain_2"}])
def test_read_mesh_wrong_mat_id_name_mapping(mesh_files, mat_id_name_mapping):
    """Test failing values of `mesh_unit_length`."""
    mesh = pv.read(mesh_files["vtu"])
    with pytest.raises(
        ValueError, match="Inconsistency between detected 'mat_ids' field data and given `mat_id_name_mapping`"
    ):
        ta.read_mesh(mesh, mat_id_name_mapping=mat_id_name_mapping)
