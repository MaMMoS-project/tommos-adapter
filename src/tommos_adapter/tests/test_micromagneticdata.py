"""Test the micromagneticdata plugins without running tommos."""

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
    reference_points = scripts.mesh_data(mm.System(name=_SYSTEM_NAME, m=reference))["knt"]
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
    """Return a synthetic tommos drive discovered through its entry point."""
    _write_drive(tmp_path)
    return mdata.Data(name=_SYSTEM_NAME, dirname=tmp_path)[0]


@pytest.fixture
def drive_with_reference(drive):
    """Return the shared micromagneticdata test references."""
    return drive, ("B_hysteresis", "J_par", "hstart = -1")


@pytest.fixture
def expected_field_type():
    """Return the magnetisation type provided by the tommos plugin."""
    return pv.UnstructuredGrid


def test_plugin_returns_complete_unstructured_grids(drive):
    """Drop the airbox while restoring SI coordinates and initial-grid metadata."""
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
    assert list(drive.table.data.columns) == ["config", "B_hysteresis", "J_par", "E", "gnorm", "future_column"]
    assert drive.table.units["B_hysteresis"] == "T"
    assert drive.table.units["E"] == "J / m3"
    assert drive.table.attributes["mammos"]["metadata"]["E"]["ontology_label"] == "EnergyDensity"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_snapshots_use_mesh_input_scaling(drive, dtype):
    """Match coordinates written from the calculator input, including Float32 grids."""
    initial = _initial_state()
    initial.points = (initial.points * 1.19).astype(dtype)
    initial.save(drive._m0_path)
    _write_snapshot(initial, drive._step_files[0], (1.0, 0.0, 0.0))

    state = drive[0]
    np.testing.assert_array_equal(state.points, initial.points)
    np.testing.assert_array_equal(state.point_data["m"], np.tile((1.0, 0.0, 0.0), (4, 1)))


@pytest.fixture
def sparse_drive(drive):
    """Include a field step without a new snapshot, as produced with ``mstep``."""
    csv = _MAMMOS_CSV.replace("1,1,0.95", "0,0,0.1,100000,40,8\n1,1,0.95")
    drive._table_path.write_text(csv, encoding="utf-8")
    return drive


@pytest.mark.parametrize("use_cache", [False, True])
def test_sparse_snapshots_align_with_table(sparse_drive, use_cache):
    """Associate each saved state with the field at which it was written."""
    drive = sparse_drive
    drive.use_cache = use_cache
    assert drive.n == 2
    assert len(list(drive)) == 2
    np.testing.assert_array_equal(drive.table.data["config"], [0, 1])
    np.testing.assert_array_equal(drive.table.data["B_hysteresis"], [-1, 1])
    np.testing.assert_array_equal(drive[1].point_data["m"], np.tile((0, 1, 0), (4, 1)))
    with pytest.raises(IndexError):
        drive[2]

    np.testing.assert_array_equal(drive.full_table.data["config"], [0, 0, 1])
    np.testing.assert_array_equal(drive.full_table.data["B_hysteresis"], [-1, 0, 1])
    assert drive.full_table.units == drive.table.units
    assert drive.full_table.attributes == drive.table.attributes


def test_sparse_drive_slices_keep_matching_states(sparse_drive):
    """Slice and concatenate snapshots with the corresponding scalar rows."""
    last = sparse_drive[1:]
    assert last.n == 1
    np.testing.assert_array_equal(last.table.data["B_hysteresis"], [1])
    np.testing.assert_array_equal(last[0].point_data["m"], sparse_drive[1].point_data["m"])
    assert len(last.full_table.data) == 3

    reversed_drive = sparse_drive[::-1]
    np.testing.assert_array_equal(reversed_drive.table.data["B_hysteresis"], [1, -1])
    np.testing.assert_array_equal(reversed_drive[0].point_data["m"], sparse_drive[1].point_data["m"])
    assert sparse_drive[:0].n == 0
    assert list(sparse_drive[:0]) == []
    assert list(sparse_drive[:0].register_callback(lambda state: state)) == []

    combined = sparse_drive << sparse_drive
    assert combined.n == 4
    assert len(list(combined)) == 4
    np.testing.assert_array_equal(combined.table.data["config"], [0, 1, 0, 1])


def test_snapshot_configuration_controls_table_mapping(sparse_drive):
    """Use numeric configuration identifiers even if a state file is missing."""
    files = sparse_drive._step_files
    files[0].unlink()
    assert sparse_drive.n == 1
    np.testing.assert_array_equal(sparse_drive.table.data["B_hysteresis"], [1])
    assert len(sparse_drive.full_table.data) == 3


def test_snapshot_without_table_row_is_rejected(drive):
    """Do not silently pair a snapshot with an unrelated table row."""
    filename = drive._step_files[1]
    filename.rename(filename.with_name("state_cfg00002_B+1.0000e+00T.vtu"))
    with pytest.raises(RuntimeError, match="configuration.*2"):
        _ = drive.table


@pytest.mark.parametrize("selection", ["cached", "slice", "callback"])
def test_changing_x_updates_cached_tables(drive, selection):
    """Keep both table axes consistent without changing the source drive."""
    if selection == "slice":
        selected = drive[1:]
    else:
        drive.use_cache = True
        selected = drive if selection == "cached" else drive.register_callback(lambda state: state)
    _ = selected.table
    selected.x = "J_par"
    assert selected.x == selected.table.x == selected.full_table.x == "J_par"
    if selected is not drive:
        assert drive.x == drive.table.x == "B_hysteresis"

    with pytest.raises(ValueError):
        selected.x = "missing_column"
    assert selected.x == selected.table.x == selected.full_table.x == "J_par"

    selected.x = None
    assert selected.x == selected.table.x == selected.full_table.x == "B_hysteresis"


def test_initial_x_with_cache(drive):
    """Validate a custom axis before the default axis has been initialised."""
    selected = mdata.Drive(name=drive.name, number=drive.number, dirname=drive.dirname, x="J_par", use_cache=True)
    assert selected.x == selected.table.x == selected.full_table.x == "J_par"
    with pytest.raises(ValueError):
        mdata.Drive(name=drive.name, number=drive.number, dirname=drive.dirname, x="missing_column", use_cache=True)


def test_disabling_cache_refreshes_full_table(drive):
    """Refresh both scalar tables and snapshot discovery when caching is disabled."""
    drive.use_cache = True
    assert len(drive.table.data) == len(drive.full_table.data) == 2
    csv = _MAMMOS_CSV.replace("1,1,0.95", "0,0,0.1,100000,40,8\n1,1,0.95")
    drive._table_path.write_text(csv, encoding="utf-8")
    assert len(drive.full_table.data) == 2

    drive.use_cache = False
    assert len(drive.full_table.data) == 3
    assert len(drive.table.data) == 2
