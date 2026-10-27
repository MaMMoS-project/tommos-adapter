"""Test the micromagneticdata plugins without running TOMMOS."""

import json

import micromagneticdata as mdata
import numpy as np
import pytest
import pyvista as pv
from micromagneticdata.testing.drive import *  # noqa: F403

from tommos_adapter import scripts
from tommos_adapter.plugins.tommos_drive import TommosDrive

_SYSTEM_NAME = "tommos_plugin"
_MAMMOS_CSV = """# mammos csv v3
Index,MagneticFluxDensity,MagneticPolarisation,EnergyDensity,EnergyDensity,
,,,,,
,,,,,
,T,T,J / m3,J / m3,
config,B_ext_T,J_par_T,E,gnorm,future_column
0,-1,-0.9,120000,400,7
1,1,0.95,90000,4,9
"""


def _initial_state() -> pv.UnstructuredGrid:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1e-9, 0.0, 0.0],
            [0.0, 1e-9, 0.0],
            [0.0, 0.0, 1e-9],
        ]
    )
    cells = np.array([4, 0, 1, 2, 3])
    grid = pv.UnstructuredGrid(cells, [pv.CellType.TETRA], points)
    grid.point_data["m"] = np.tile((0.0, 0.0, 1.0), (grid.n_points, 1))
    grid.point_data["custom_point_data"] = np.arange(grid.n_points)
    grid.cell_data["region_id"] = np.array([10])
    grid.cell_data["Ms"] = np.array([8e5])
    grid.cell_data["custom_cell_data"] = np.array([42])
    return grid


def _write_snapshot(reference: pv.UnstructuredGrid, filename, magnetisation) -> None:
    state = reference.copy(deep=True)
    state.points /= scripts.DEFAULT_MESH_UNIT
    state.clear_data()
    state.point_data["m"] = np.tile(magnetisation, (state.n_points, 1))
    state.point_data["U"] = np.zeros(state.n_points)
    state.cell_data["mat_id"] = reference.cell_data["region_id"]
    state.save(filename)


def _write_drive(dirname) -> None:
    drive_path = dirname / _SYSTEM_NAME / "drive-0"
    output_path = drive_path / f"hyst_{_SYSTEM_NAME}"
    output_path.mkdir(parents=True)

    (drive_path / "info.json").write_text(
        json.dumps(
            {
                "drive_number": 0,
                "adapter": "tommos_adapter",
                "driver": "HysteresisDriver",
            }
        ),
        encoding="utf-8",
    )
    (drive_path / f"{_SYSTEM_NAME}.p2").write_text("[field]\nhstart = -1\nhfinal = 1\n", encoding="utf-8")
    (output_path / "mammos_hysteresis.csv").write_text(_MAMMOS_CSV, encoding="utf-8")

    initial = _initial_state()
    initial.save(drive_path / "m0.vtu")
    _write_snapshot(initial, output_path / "state_cfg00000_B-1.0000e+00T.vtu", (1.0, 0.0, 0.0))
    _write_snapshot(initial, output_path / "state_cfg00001_B+1.0000e+00T.vtu", (0.0, 1.0, 0.0))


@pytest.fixture
def drive(tmp_path):
    """Return a synthetic TOMMOS drive discovered through its entry point."""
    _write_drive(tmp_path)
    return mdata.Data(name=_SYSTEM_NAME, dirname=tmp_path)[0]


@pytest.fixture
def drive_with_reference(drive):
    """Return the shared micromagneticdata test references."""
    return drive, ("B_hysteresis", "J_par_T", "hstart = -1")


def test_plugin_returns_complete_unstructured_grids(drive):
    """Restore SI coordinates and initial-grid metadata around TOMMOS output."""
    assert isinstance(drive, TommosDrive)
    assert isinstance(drive.m0, pv.UnstructuredGrid)
    np.testing.assert_allclose(drive.m0.points, _initial_state().points)

    first_state = drive[0]
    np.testing.assert_allclose(first_state.points, drive.m0.points)
    np.testing.assert_allclose(first_state.point_data["m"], np.tile((1.0, 0.0, 0.0), (4, 1)))
    np.testing.assert_array_equal(first_state.point_data["custom_point_data"], np.arange(4))
    np.testing.assert_array_equal(first_state.cell_data["region_id"], [10])
    np.testing.assert_array_equal(first_state.cell_data["Ms"], [8e5])
    np.testing.assert_array_equal(first_state.cell_data["custom_cell_data"], [42])


def test_table_plugin_preserves_metadata(drive):
    """Expose MaMMoS CSV data using Ubermag hysteresis conventions."""
    assert list(drive.table.data.columns) == ["config", "B_hysteresis", "J_par_T", "E", "gnorm", "future_column"]
    assert drive.table.units["B_hysteresis"] == "T"
    assert drive.table.units["E"] == "J / m3"
    assert drive.table.attributes["mammos"]["metadata"]["E"]["ontology_label"] == "EnergyDensity"
