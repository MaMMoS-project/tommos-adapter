"""Test the micromagneticdata plugins with synthetic and simulated results."""

import json

import micromagneticdata as mdata
import micromagneticmodel as mm
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
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = np.array([4, 0, 1, 2, 3])
    grid = pv.UnstructuredGrid(cells, [pv.CellType.TETRA], points)
    grid.field_data["mesh_unit_length"] = 1e-9
    grid.point_data["m"] = np.tile((0.0, 0.0, 1.0), (grid.n_points, 1))
    grid.point_data["custom_point_data"] = np.arange(grid.n_points)
    grid.cell_data["region_id"] = np.array([10])
    grid.cell_data["Ms"] = np.array([8e5])
    grid.cell_data["custom_cell_data"] = np.array([42])
    return grid


def _write_snapshot(reference: pv.UnstructuredGrid, filename, magnetisation, *, mesh_unit: float) -> None:
    reference_points = scripts.mesh_data(mm.System(name=_SYSTEM_NAME, m=reference), mesh_unit=mesh_unit)["knt"]
    airbox_points = np.array([[-50, -50, -50], [50, -50, -50], [-50, 50, -50], [-50, -50, 50]])
    points = np.vstack([reference_points, airbox_points]).astype(np.float32)
    output_m = np.vstack([np.tile(magnetisation, (reference.n_points, 1)), np.zeros((len(airbox_points), 3))]).astype(
        np.float32
    )
    connectivity = np.vstack([reference.cell_connectivity.reshape(-1, 4), np.arange(4) + reference.n_points])
    order = np.random.default_rng(0).permutation(len(points))
    new_index = np.argsort(order)
    state = pv.UnstructuredGrid({pv.CellType.TETRA: new_index[connectivity]}, points[order])
    state.point_data["m"] = output_m[order]
    state.save(filename)


def _write_drive(dirname, *, mesh_unit_length: float = 1e-9) -> None:
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
    (drive_path / f"{_SYSTEM_NAME}.p2").write_text(
        f"[mesh]\nsize = {mesh_unit_length}\n\n[field]\nhstart = -1\nhfinal = 1\n", encoding="utf-8"
    )
    (output_path / "mammos_hysteresis.csv").write_text(_MAMMOS_CSV, encoding="utf-8")

    initial = _initial_state()
    initial.field_data["mesh_unit_length"] = mesh_unit_length
    initial.save(drive_path / "m0.vtu")
    _write_snapshot(
        initial, output_path / "state_cfg00000_B-1.0000e+00T.vtu", (1.0, 0.0, 0.0), mesh_unit=mesh_unit_length
    )
    _write_snapshot(
        initial, output_path / "state_cfg00001_B+1.0000e+00T.vtu", (0.0, 1.0, 0.0), mesh_unit=mesh_unit_length
    )


@pytest.fixture
def drive(tommos_simulation):
    """Return a fresh drive reading the shared tommos simulation results."""
    system, output_directory = tommos_simulation
    return mdata.Data(name=system.name, dirname=output_directory)[0]


@pytest.fixture
def drive_with_reference(drive):
    """Return the shared micromagneticdata test references."""
    return drive, ("B_hysteresis", "J_par", "[initial state]")


@pytest.fixture
def expected_field_type():
    """Return the magnetisation type provided by the tommos plugin."""
    return pv.UnstructuredGrid


@pytest.fixture
def synthetic_drive(tmp_path):
    """Return synthetic results for targeted grid, snapshot, and file-error tests."""
    _write_drive(tmp_path)
    return mdata.Data(name=_SYSTEM_NAME, dirname=tmp_path)[0]


def test_plugin_reads_simulation_results(drive, tommos_simulation):
    """Read actual calculator output through the registered data plugin."""
    system, _ = tommos_simulation
    assert isinstance(drive, TommosDrive)

    initial_grid = drive.m0
    np.testing.assert_array_equal(initial_grid.point_data["m"], np.tile((1, 0, 0), (system.m.n_points, 1)))
    output_grids = list(drive)
    assert len(output_grids) == 2
    for grid in [initial_grid, *output_grids]:
        assert isinstance(grid, pv.UnstructuredGrid)
        assert grid.field_data["mesh_unit_length"][0] == 1e-9
        np.testing.assert_array_equal(grid.points, system.m.points)
        np.testing.assert_array_equal(grid.cells, system.m.cells)
        for name in ("region_id", "Ms"):
            np.testing.assert_array_equal(grid.cell_data[name], system.m.cell_data[name])
        assert grid.point_data["m"].shape == (system.m.n_points, 3)
        assert np.all(np.isfinite(grid.point_data["m"]))
    np.testing.assert_array_equal(output_grids[-1].point_data["m"], system.m.point_data["m"])
    reloaded_mesh = scripts.mesh_data(mm.System(name="reloaded", m=initial_grid))
    np.testing.assert_array_equal(reloaded_mesh["knt"], initial_grid.points)

    table = drive.table
    assert drive.n == len(table.data) == 2
    assert table.data.equals(system.table.data)
    assert table.units == system.table.units
    assert table.units["B_hysteresis"] == "T"
    assert table.attributes["mammos"]["metadata"]["E"]["ontology_label"] == "EnergyDensity"
    assert "[initial state]" in drive.calculator_script


def test_plugin_returns_complete_unstructured_grids(synthetic_drive):
    """Drop the airbox while retaining mesh coordinates, unit length, and metadata."""
    drive = synthetic_drive
    assert isinstance(drive, TommosDrive)
    assert isinstance(drive.m0, pv.UnstructuredGrid)
    np.testing.assert_allclose(drive.m0.points, _initial_state().points)

    first_state = drive[0]
    assert first_state.field_data["mesh_unit_length"][0] == 1e-9
    np.testing.assert_allclose(first_state.points, drive.m0.points)
    np.testing.assert_allclose(first_state.point_data["m"], np.tile((1.0, 0.0, 0.0), (4, 1)))
    np.testing.assert_array_equal(first_state.point_data["custom_point_data"], np.arange(4))
    np.testing.assert_array_equal(first_state.cell_data["region_id"], [10])
    np.testing.assert_array_equal(first_state.cell_data["Ms"], [8e5])
    np.testing.assert_array_equal(first_state.cell_data["custom_cell_data"], [42])


@pytest.mark.parametrize("initial_unit_length", [None, 1e-9, 1e-6])
def test_plugin_uses_configured_mesh_unit(tmp_path, initial_unit_length):
    """Read a non-default P2 mesh size, including initial grids saved in other units."""
    _write_drive(tmp_path, mesh_unit_length=1e-6)
    drive = mdata.Data(name=_SYSTEM_NAME, dirname=tmp_path)[0]
    initial = pv.read(drive._m0_path)
    if initial_unit_length is None:
        initial.points *= 1e-6
        del initial.field_data["mesh_unit_length"]
    else:
        initial.points *= 1e-6 / initial_unit_length
        initial.field_data["mesh_unit_length"] = initial_unit_length
    initial.save(drive._m0_path)

    for grid in [drive.m0, *drive]:
        np.testing.assert_allclose(grid.points, _initial_state().points, rtol=1e-14, atol=0)
        assert grid.field_data["mesh_unit_length"][0] == 1e-6
    np.testing.assert_array_equal(drive[1].point_data["m"], np.tile((0, 1, 0), (4, 1)))


@pytest.mark.parametrize("mesh_size", ["0", "nan", None])
def test_plugin_rejects_invalid_mesh_unit(synthetic_drive, mesh_size):
    """Reject missing or invalid P2 scales instead of guessing the physical size."""
    config = "[mesh]\n" if mesh_size is None else f"[mesh]\nsize = {mesh_size}\n"
    (synthetic_drive.drive_path / f"{synthetic_drive.name}.p2").write_text(config, encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid mesh unit length"):
        _ = synthetic_drive.m0


def test_missing_initial_grid_reports_filename(synthetic_drive):
    """Include the filename and original read error when the initial grid is missing."""
    drive = synthetic_drive
    drive._m0_path.unlink()
    with pytest.raises(RuntimeError, match="Cannot read tommos magnetisation grid") as error:
        _ = drive.m0
    assert str(drive._m0_path) in str(error.value)
    assert isinstance(error.value.__cause__, FileNotFoundError)


@pytest.fixture
def sparse_drive(synthetic_drive):
    """Include a field step without a new snapshot, as produced with ``mstep``."""
    drive = synthetic_drive
    csv = _MAMMOS_CSV.replace("1,1,0.95", "0,0,0.1,100000,40,8\n1,1,0.95")
    drive._table_path.write_text(csv, encoding="utf-8")
    return drive


@pytest.mark.parametrize("use_cache", [False, True])
def test_sparse_drive_preserves_complete_table(sparse_drive, use_cache):
    """Keep every field step even when fewer magnetisation snapshots are saved."""
    drive = sparse_drive
    drive.use_cache = use_cache
    assert drive.n == 3
    assert len(list(drive)) == 2
    np.testing.assert_array_equal(drive.table.data["config"], [0, 0, 1])
    np.testing.assert_array_equal(drive.table.data["B_hysteresis"], [-1, 0, 1])
    np.testing.assert_array_equal(drive.table.data["J_par"], [-0.9, 0.1, 0.95])
    np.testing.assert_array_equal(drive.table.data["future_column"], [7, 8, 9])
    assert drive.table.units["B_hysteresis"] == "T"
    assert drive.table.attributes["mammos"]["metadata"]["E"]["ontology_label"] == "EnergyDensity"
    np.testing.assert_array_equal(drive[1].point_data["m"], np.tile((0, 1, 0), (4, 1)))
    with pytest.raises(IndexError):
        drive[2]


@pytest.mark.parametrize("snapshots", ["missing", "none", "unmatched"])
def test_table_is_independent_of_snapshot_files(sparse_drive, snapshots):
    """Read the complete scalar data regardless of which state files exist."""
    drive = sparse_drive
    files = drive._step_files
    if snapshots == "missing":
        files[0].unlink()
    elif snapshots == "none":
        for filename in files:
            filename.unlink()
    else:
        files[1].rename(files[1].with_name("state_cfg00002_B+1.0000e+00T.vtu"))

    assert drive.n == 3
    np.testing.assert_array_equal(drive.table.data["config"], [0, 0, 1])
    np.testing.assert_array_equal(drive.table.data["B_hysteresis"], [-1, 0, 1])


@pytest.mark.parametrize("selection", ["cached", "slice", "callback"])
def test_changing_x_updates_cached_table(drive, selection):
    """Update the cached independent variable without changing the source drive."""
    if selection == "slice":
        selected = drive[1:]
    else:
        drive.use_cache = True
        selected = drive if selection == "cached" else drive.register_callback(lambda state: state)
    original_data = selected.table.data.copy()
    selected.x = "J_par"
    assert selected.x == selected.table.x == "J_par"
    assert selected.table.data.equals(original_data)
    if selected is not drive:
        assert drive.x == drive.table.x == "B_hysteresis"

    with pytest.raises(ValueError):
        selected.x = "missing_column"
    assert selected.x == selected.table.x == "J_par"

    selected.x = None
    assert selected.x == selected.table.x == "B_hysteresis"


def test_initial_x_with_cache(drive):
    """Validate a custom independent variable during initialisation."""
    selected = mdata.Drive(name=drive.name, number=drive.number, dirname=drive.dirname, x="J_par", use_cache=True)
    assert selected.x == selected.table.x == "J_par"
    with pytest.raises(ValueError):
        mdata.Drive(name=drive.name, number=drive.number, dirname=drive.dirname, x="missing_column", use_cache=True)
