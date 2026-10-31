"""Test TOMMOS table output and reusable Ubermag table behaviour."""

import subprocess

import micromagneticmodel as mm
import numpy as np
import pytest
import pyvista as pv
from ubermagtable.testing import table as table_testing
from ubermagtable.testing.table import *  # noqa: F403

from tommos_adapter import HysteresisDriver
from tommos_adapter.plugins import table_from_file

test_table_units = pytest.mark.skip(
    reason=(
        "The shared test assumes an energy column in J, but TOMMOS "
        "hysteresis output contains energy-density columns in J/m3."
    )
)(table_testing.test_table_units)

test_table_mpl = pytest.mark.skip(
    reason=(
        "The shared plotting test assumes LLG-specific mx and my columns, "
        "which TOMMOS hysteresis output does not provide."
    )
)(table_testing.test_table_mpl)


@pytest.fixture(scope="session")
def tommos_table_file(tmp_path_factory):
    """Run a minimal TOMMOS simulation and return its table output."""
    output_directory = tmp_path_factory.mktemp("tommos_table")
    subprocess.run(
        ["tommos", "mesh", "--geom", "box", "--extent", "2,2,2", "--h", "2", "--out-name", "table"],
        cwd=output_directory,
        check=True,
    )
    state = pv.read(output_directory / "table.vtu")
    state.points *= 1e-9
    state.cell_data["region_id"] = state.cell_data.pop("mat_id")
    state.point_data["m"] = np.tile((1, 0, 0), (state.n_points, 1)).astype(np.float32)
    state.cell_data["Ms"] = np.full(state.n_cells, 1 / mm.consts.mu0)

    system = mm.System(name="table_output")
    system.m = state
    system.energy = mm.Exchange(A=1e-12) + mm.UniaxialAnisotropy(K=4e5, u=(1, 0, 0)) + mm.Demag()

    HysteresisDriver().drive(
        system,
        dirname=output_directory,
        append=False,
        verbose=0,
        Hmin=(0.5 / mm.consts.mu0, 0, 0),
        Hmax=(1 / mm.consts.mu0, 0, 0),
        n=2,
    )
    return output_directory / system.name / "drive-0" / f"hyst_{system.name}" / "mammos_hysteresis.csv"


@pytest.fixture
def table_factory(tommos_table_file):
    """Return a factory backed by actual TOMMOS output."""

    def _factory(**kwargs):
        return table_from_file(tommos_table_file, **kwargs)

    return _factory


@pytest.fixture
def synthetic_table_factory(tmp_path, write_mammos_table):
    """Return a factory backed by synthetic MaMMoS output."""
    table_file = write_mammos_table(tmp_path / "mammos_hysteresis.csv")

    def _factory(**kwargs):
        return table_from_file(table_file, **kwargs)

    return _factory


@pytest.fixture
def table_llg_factory():
    """Skip shared table tests for unsupported LLG output."""
    pytest.skip("TOMMOS does not provide LLG table output.")


@pytest.fixture
def table_llg_25ps():
    """Skip shared table tests for unsupported LLG output."""
    pytest.skip("TOMMOS does not provide LLG table output.")


@pytest.fixture
def table_minimisation_factory():
    """Skip shared table tests for unsupported minimisation output."""
    pytest.skip("TOMMOS does not provide minimisation table output.")


@pytest.fixture
def table_hysteresis_factory(table_factory):
    """Return the supported TOMMOS hysteresis table factory."""
    return table_factory


def test_table_from_file(synthetic_table_factory):
    """Preserve TOMMOS columns, units, and MaMMoS metadata."""
    table = synthetic_table_factory(x="B_hysteresis")

    assert table.x == "B_hysteresis"
    assert list(table.data.columns) == ["config", "B_hysteresis", "J_par", "E", "gnorm", "additional_column"]
    assert table.units == {
        "config": "",
        "B_hysteresis": "T",
        "J_par": "T",
        "E": "J / m3",
        "gnorm": "J / m3",
        "additional_column": "",
    }
    np.testing.assert_allclose(table.data["additional_column"], (7, 8, 9))
    assert table.attributes["mammos"]["metadata"]["config"]["ontology_label"] == "Index"
    assert table.attributes["mammos"]["metadata"]["E"]["ontology_label"] == "EnergyDensity"


def test_table_from_file_without_renaming(synthetic_table_factory):
    """Retain native TOMMOS column names when renaming is disabled."""
    table = synthetic_table_factory(x="B_ext_T", rename=False)

    assert table.x == "B_ext_T"
    assert "B_ext_T" in table.data
    assert "B_hysteresis" not in table.data
