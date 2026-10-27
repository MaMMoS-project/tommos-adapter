"""Test TOMMOS table output and reusable Ubermag table behaviour."""

import subprocess

import micromagneticmodel as mm
import numpy as np
import pytest
import pyvista as pv

# The remaining shared tests assume LLG-specific columns such as t, mx, and my,
# which are not part of TOMMOS hysteresis output.
from ubermagtable.testing.table import test_table_attributes as test_table_attributes
from ubermagtable.testing.table import test_table_columns as test_table_columns
from ubermagtable.testing.table import test_table_init as test_table_init

import tommos_adapter as ta
from tommos_adapter.plugins import table_from_file


@pytest.fixture(scope="session")
def hysteresis_output(tmp_path_factory):
    """Compute an inexpensive TOMMOS table instead of storing sample output."""
    dirname = tmp_path_factory.mktemp("table_hysteresis")

    subprocess.run(
        [
            "tommos",
            "mesh",
            "--geom",
            "box",
            "--extent",
            "2,2,2",
            "--h",
            "2",
            "--out-name",
            "table_mesh",
        ],
        cwd=dirname,
        check=True,
        capture_output=True,
        text=True,
    )
    state = pv.read(dirname / "table_mesh.vtu")
    state.points *= ta.scripts.DEFAULT_MESH_UNIT
    state.cell_data["region_id"] = state.cell_data["mat_id"]
    state.cell_data.remove("mat_id")
    state.point_data["m"] = np.tile((0.0, 0.0, 1.0), (state.n_points, 1))
    state.cell_data["Ms"] = np.full(state.n_cells, 1 / mm.consts.mu0)

    system = mm.System(name="table_hysteresis", m=state, energy=mm.Demag())
    driver = ta.HysteresisDriver(max_iter=100)
    driver.drive(
        system,
        dirname=dirname,
        Hmin=(0, 0, -1 / mm.consts.mu0),
        Hmax=(0, 0, 1 / mm.consts.mu0),
        n=2,
    )

    return dirname / system.name / "drive-0" / f"hyst_{system.name}" / "mammos_hysteresis.csv"


@pytest.fixture
def table_factory(hysteresis_output):
    """Return a factory backed by output from a real TOMMOS calculation."""

    def _factory(**kwargs):
        return table_from_file(hysteresis_output, **kwargs)

    return _factory


def test_table_from_file(table_factory):
    """Preserve TOMMOS columns, units, and MaMMoS metadata."""
    table = table_factory(x="B_hysteresis")

    assert table.x == "B_hysteresis"
    assert list(table.data.columns) == ["config", "B_hysteresis", "J_par_T", "E", "gnorm"]
    assert table.units == {
        "config": "",
        "B_hysteresis": "T",
        "J_par_T": "T",
        "E": "J / m3",
        "gnorm": "J / m3",
    }
    assert table.attributes["mammos"]["metadata"]["E"]["ontology_label"] == "EnergyDensity"


def test_table_from_file_without_renaming(table_factory):
    """Retain native TOMMOS column names when renaming is disabled."""
    table = table_factory(x="B_ext_T", rename=False)

    assert table.x == "B_ext_T"
    assert "B_ext_T" in table.data
    assert "B_hysteresis" not in table.data
