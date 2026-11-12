"""Regression tests for the demagnetisation energy."""

import subprocess

import micromagneticmodel as mm
import numpy as np
import pyvista as pv

import tommos_adapter as ta
from tommos_adapter.plugins import table_from_file


def test_uniformly_magnetised_cube_energy(tmp_path, monkeypatch):
    """A uniformly magnetised cube has magnetometric factor one third."""
    name = "uniform_cube"
    subprocess.run(
        [
            "tommos",
            "mesh",
            "--geom",
            "box",
            "--extent",
            "20,20,20",
            "--h",
            "1",
            "--out-name",
            name,
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    state = pv.read(tmp_path / f"{name}.vtu")
    state.points *= ta.scripts.DEFAULT_MESH_UNIT
    state.cell_data["region_id"] = state.cell_data.pop("mat_id")
    state.point_data["m"] = np.tile((0.0, 0.0, 1.0), (state.n_points, 1))

    spontaneous_polarisation = 1.0  # T
    state.cell_data["Ms"] = np.full(
        state.n_cells,
        spontaneous_polarisation / mm.consts.mu0,
    )

    system = mm.System(name=name, energy=mm.Demag(), m=state)
    driver = ta.HysteresisDriver()
    monkeypatch.chdir(tmp_path)
    driver._write_input_files(
        system,
        hx=0,
        hy=0,
        hz=1,
        hstart=0,
        hfinal=0,
        hstep=1,
    )

    argstr = driver.get_argstr(system, num_devices=1)
    result = ta.TommosRunner()._call(
        f"{argstr} --no-loop --B-start 0 --B-end 0 --dB 1 "
        "--max-iter 0 --poisson-solver jax --snapshot-every 0"
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()

    table = table_from_file(tmp_path / f"hyst_{name}" / "mammos_hysteresis.csv")
    energy_density = table.data["E"].iloc[0]
    expected_energy_density = spontaneous_polarisation**2 / (6 * mm.consts.mu0)

    np.testing.assert_allclose(energy_density, expected_energy_density, rtol=0.04)
