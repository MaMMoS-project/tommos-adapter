"""Test TOMMOS table output and reusable Ubermag table behaviour."""

import numpy as np
import pytest
from ubermagtable.testing import table as table_testing
from ubermagtable.testing.table import *  # noqa: F403

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


@pytest.fixture
def table_factory(tmp_path, write_mammos_table):
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


def test_table_from_file(table_factory):
    """Preserve TOMMOS columns, units, and MaMMoS metadata."""
    table = table_factory(x="B_hysteresis")

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


def test_table_from_file_without_renaming(table_factory):
    """Retain native TOMMOS column names when renaming is disabled."""
    table = table_factory(x="B_ext_T", rename=False)

    assert table.x == "B_ext_T"
    assert "B_ext_T" in table.data
    assert "B_hysteresis" not in table.data
