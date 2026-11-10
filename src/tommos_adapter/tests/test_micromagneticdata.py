"""Test the micromagneticdata plugin with simulated tommos results."""

import micromagneticdata as mdata
import numpy as np
import pytest
import pyvista as pv
from micromagneticdata.testing.drive import *  # noqa: F403

from tommos_adapter import scripts
from tommos_adapter.plugins.tommos_drive import TommosDrive


@pytest.fixture
def drive(tommos_simulation):
    """Return the shared tommos simulation through micromagneticdata."""
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


def test_plugin_reads_simulation_results(drive, tommos_simulation):
    """Restore the input grid and magnetisation from actual tommos output."""
    system, _ = tommos_simulation
    assert isinstance(drive, TommosDrive)

    initial_grid = drive.m0
    output_grids = list(drive)
    assert len(output_grids) == 2

    expected_points = system.m.points / scripts.DEFAULT_MESH_UNIT
    for grid in [initial_grid, *output_grids]:
        assert isinstance(grid, pv.UnstructuredGrid)
        assert grid.field_data["mesh_unit_length"][0] == scripts.DEFAULT_MESH_UNIT
        np.testing.assert_allclose(grid.points, expected_points, rtol=1e-14, atol=0)
        np.testing.assert_array_equal(grid.cells, system.m.cells)
        np.testing.assert_array_equal(grid.cell_data["region_id"], system.m.cell_data["region_id"])
        np.testing.assert_array_equal(grid.cell_data["Ms"], system.m.cell_data["Ms"])

    np.testing.assert_array_equal(initial_grid.point_data["m"], np.tile((1, 0, 0), (system.m.n_points, 1)))
    np.testing.assert_array_equal(output_grids[-1].point_data["m"], system.m.point_data["m"])
