"""Regression tests for the demagnetisation energy."""

import micromagneticmodel as mm
import numpy as np

import tommos_adapter as ta
from tommos_adapter.plugins import table_from_file


def test_uniformly_magnetised_cube_energy(tmp_path, monkeypatch):
    """A uniformly magnetised cube has magnetometric factor one third."""
    name = "uniform_cube"
    state = ta.create_mesh(
        geom="box",
        extent=(20, 20, 20),
        h=1,
        out_name=tmp_path / name,
    )

    spontaneous_polarisation = 1.0  # T
    state = ta.set_magnetization(
        state,
        m=(0, 0, 1),
        Ms=spontaneous_polarisation / mm.consts.mu0,
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
    # The C++ minimizer does not evaluate the initial energy when max_iter is zero.
    result = ta.TommosRunner()._call(
        f"{argstr} --no-loop --B-start 0 --B-end 0 --dB 1 "
        "--max-iter 0 --no-cpp-mkl --cpu-spmv-backend scipy "
        "--poisson-solver jax --snapshot-every 0"
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()

    table = table_from_file(tmp_path / f"hyst_{name}" / "mammos_hysteresis.csv")
    energy_density = table.data["E"].iloc[0]
    expected_energy_density = spontaneous_polarisation**2 / (6 * mm.consts.mu0)

    np.testing.assert_allclose(energy_density, expected_energy_density, rtol=0.04)
