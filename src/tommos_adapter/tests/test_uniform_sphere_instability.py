"""Regression test for the instability of a uniformly magnetised sphere."""

import subprocess

import micromagneticmodel as mm
import numpy as np
import pytest
import pyvista as pv

import tommos_adapter as ta


@pytest.mark.regression
def test_uniform_sphere_instability_field(tmp_path, monkeypatch):
    """A uniformly magnetised soft sphere becomes unstable near Js/3."""
    name = "uniform_sphere"
    subprocess.run(
        [
            "tommos",
            "mesh",
            "--geom",
            "ellipsoid",
            "--extent",
            "20,20,20",
            "--h",
            "3",
            "--backend",
            "meshpy",
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

    spontaneous_polarisation = 1.0  # Js = mu0 Ms in T
    state.cell_data["Ms"] = np.full(
        state.n_cells,
        spontaneous_polarisation / mm.consts.mu0,
    )

    system = mm.System(
        name=name,
        energy=mm.Exchange(A=0) + mm.UniaxialAnisotropy(K=0, u=(0, 0, 1)) + mm.Demag(),
        m=state,
    )
    driver = ta.HysteresisDriver(max_iter=1000, tol_fun=1e-8)
    monkeypatch.chdir(tmp_path)
    driver._write_input_files(
        system,
        hx=0,
        hy=0,
        hz=1,
        hstart=0.34,
        hfinal=0.30,
        hstep=-0.01,
    )

    result = ta.TommosRunner()._call(
        # For this 20 nm sphere and h=3 nm, KL=10 gives a 227.8 nm air box.
        f"{name} --add-shell --KL 10 --K 1.5 --hmax 20 --no-loop "
        "--no-cpp-mkl --cpu-spmv-backend scipy --poisson-solver jax "
        "--snapshot-every 0 --num-devices 1"
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()

    output = np.atleast_2d(np.loadtxt(tmp_path / f"hyst_{name}" / f"{name}.mh"))
    field = output[:, 0]
    parallel_polarisation = output[:, 1]
    stable = parallel_polarisation >= 0.99 * spontaneous_polarisation

    assert stable[0], "The uniform state is already unstable at the upper sweep boundary."
    assert not stable[-1], "The uniform state remained stable throughout the field sweep."
    first_unstable = np.flatnonzero(~stable)[0]
    numerical_instability_field = np.mean(field[first_unstable - 1 : first_unstable + 1])
    expected_instability_field = spontaneous_polarisation / 3
    np.testing.assert_allclose(
        numerical_instability_field,
        expected_instability_field,
        rtol=0.10,
    )
