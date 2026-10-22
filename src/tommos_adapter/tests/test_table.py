from pathlib import Path

import numpy as np

from tommos_adapter.plugins import table_from_file

SAMPLE_FILE = Path(__file__).parent / "test_sample" / "mammos_hysteresis.csv"


def test_table_from_file():
    table = table_from_file(SAMPLE_FILE, x="B_hysteresis")

    assert table.x == "B_hysteresis"
    assert list(table.data.columns) == ["config", "B_hysteresis", "J_par_T", "E", "gnorm", "future_column"]
    assert table.units == {
        "config": "",
        "B_hysteresis": "T",
        "J_par_T": "T",
        "E": "J / m3",
        "gnorm": "J / m3",
        "future_column": "",
    }
    np.testing.assert_allclose(table.data["B_hysteresis"], (-1, 0, 1))
    np.testing.assert_allclose(table.data["future_column"], (7, 8, 9))
    assert table.attributes["mammos"]["metadata"]["E"]["ontology_label"] == "EnergyDensity"


def test_table_from_file_without_renaming():
    table = table_from_file(SAMPLE_FILE, x="B_ext_T", rename=False)

    assert table.x == "B_ext_T"
    assert "B_ext_T" in table.data
    assert "B_hysteresis" not in table.data
