"""Test configurations."""

import shlex
import subprocess
from pathlib import Path

import mammos_entity as me
import micromagneticmodel as mm
import numpy as np
import pytest
import pyvista as pv

import tommos_adapter as ta
from tommos_adapter import HysteresisDriver


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """Define a temporary data directory."""
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def cube_20_path(data_dir):
    """Define a cubic mesh of side length 20 and mesh size 2."""
    cmd = "tommos mesh --geom box --extent 20,20,20 --h 2 --out-name cube_20"
    subprocess.run(shlex.split(cmd), cwd=data_dir, check=True)
    return data_dir / "cube_20.vtu"


@pytest.fixture(scope="function")
def mesh_cube_20(cube_20_path):
    """Read mesh of cube of side length 20.

    This fixture has 'function' scope, while the fixture creating the cubic mesh has 'session' scope.
    Therefore, the mesh creation only happens once per `pytest` call, while this function gets called
    for each test, allowing mesh modifications in any test without modifying the mesh stored on disk.
    """
    return pv.read(cube_20_path)


@pytest.fixture(scope="function")
def m_cube_20_001(mesh_cube_20):
    """Initialize uniform (0, 0, 1) magnetization on cube of side length 20.

    The saturation magnetization corresponds to a spontaneous polarization of 1 T everywhere.
    """
    mesh_cube_20.point_data["m"] = np.tile((0.0, 0.0, 1.0), (mesh_cube_20.n_points, 1))
    mesh_cube_20.cell_data["Ms"] = np.full(mesh_cube_20.n_cells, 1 / mm.consts.mu0)
    mesh_cube_20.field_data["mesh_unit_length"] = 1e-9
    mesh_cube_20.field_data["mat_ids"] = [1]
    mesh_cube_20.field_data["mat_names"] = ["grain_1"]
    return mesh_cube_20


@pytest.fixture(scope="session")
def cube_20_multigrain_n3_gb_path(data_dir: Path) -> Path:
    """Define a multigrain cubic mesh.

    Side length 20, mesh size 2, 3 grains, and with grain boundary.
    """
    cmd = "tommos mesh --geom poly_gb --extent 20,20,20 --h 2 --n 3 --out-name poly_gb_20"
    subprocess.run(shlex.split(cmd), cwd=data_dir, check=True)
    return data_dir / "poly_gb_20.vtu"


@pytest.fixture(scope="function")
def mesh_cube_mg(cube_20_multigrain_n3_gb_path: Path) -> pv.UnstructuredGrid:
    """Read multigrain mesh and set `mat_names`."""
    mesh = ta.read_mesh(
        pv.read(cube_20_multigrain_n3_gb_path),
        mat_id_name_mapping={1: "grain_1", 2: "grain_2", 3: "grain_3", 4: "grain_boundary"},
    )
    return mesh


@pytest.fixture(scope="session")
def tommos_simulation(tmp_path_factory):
    """Run a small two-step tommos simulation once per test session."""
    output_directory = tmp_path_factory.mktemp("tommos_simulation")
    subprocess.run(
        ["tommos", "mesh", "--geom", "box", "--extent", "2,2,2", "--h", "2", "--out-name", "table"],
        cwd=output_directory,
        check=True,
    )
    grid = pv.read(output_directory / "table.vtu")
    grid.point_data["m"] = np.tile((1, 0, 0), (grid.n_points, 1))
    grid.cell_data["Ms"] = np.full(grid.n_cells, 1 / mm.consts.mu0)
    grid.field_data["mesh_unit_length"] = 1e-9
    grid.field_data["mat_ids"] = [1]
    grid.field_data["mat_names"] = ["grain_1"]

    system = mm.System(name="table_output", m=grid)
    system.energy = mm.Exchange(A=1e-12) + mm.UniaxialAnisotropy(K=4e5, u=(1, 0, 0)) + mm.Demag()
    HysteresisDriver().drive(
        system,
        dirname=output_directory,
        append=False,
        verbose=0,
        h_direction=(0, 0, 1),
        h_start=0.5,
        h_final=1,
        n=2,
    )
    return system, output_directory


@pytest.fixture(scope="session")
def cube_tiny_path(data_dir):
    """Create very small cubic mesh.

    The mesh is a cube of side length 2nm with 2nm mesh size.

    This mesh can be used in tests with calls to drivers where the physical result is not important.
    """
    cmd = "tommos mesh --geom box --extent 2,2,2 --h 2 --out-name tiny"
    subprocess.run(shlex.split(cmd), cwd=data_dir, check=True)
    return data_dir / "tiny.vtu"


@pytest.fixture(scope="function")
def m_tiny(cube_tiny_path):
    """Initialize uniform magnetization on a very small cube.

    The default magnetization is uniform and set to (0, 0, 1). The saturation magnetization
    corresponds to a spontaneous polarization of 1 T everywhere.
    """
    m = pv.read(cube_tiny_path)
    m.point_data["m"] = np.tile(np.array([0.0, 0.0, 1.0]), (m.n_points, 1))
    m.cell_data["Ms"] = np.full(m.n_cells, 1 / mm.consts.mu0)
    m.field_data["mesh_unit_length"] = 1e-9
    m.field_data["mat_ids"] = np.unique(m.cell_data["mat_id"])
    m.field_data["mat_names"] = ["grain_1"]
    return m


@pytest.fixture
def write_mammos_table():
    """Return a factory writing synthetic TOMMOS tabular output."""

    def _write(filename):
        collection = me.EntityCollection(
            description="Synthetic TOMMOS hysteresis output.",
            config=me.Entity("Index", [0, 1, 2]),
            B_ext_T=me.Entity("MagneticFluxDensity", [-1, 0, 1], "T"),
            J_par_T=me.Entity("MagneticPolarisation", [-0.9, 0.1, 0.95], "T"),
            E=me.Entity("EnergyDensity", [120_000, 100_000, 90_000], "J / m3"),
            gnorm=me.Entity("EnergyDensity", [400, 40, 4], "J / m3"),
            additional_column=[7, 8, 9],
        )
        collection.to_csv(filename)
        return filename

    return _write
