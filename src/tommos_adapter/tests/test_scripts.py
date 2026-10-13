"""Test scripts submodule."""

import os
from pathlib import Path
from textwrap import dedent

import micromagneticmodel as mm
import numpy as np
import pandas as pd
import pytest

import tommos_adapter as ta
from tommos_adapter.scripts import save_mesh_npz, write_input_files, write_krn, write_p2


def test_write_input_files(state_zero, tmp_path):
    """Test that the `write_input_files` script produces the three expected input files."""
    os.chdir(tmp_path)
    system = mm.System(name="test_write_input_files")
    system.m = state_zero.copy()
    driver = ta.HysteresisDriver()
    write_input_files(driver, system)
    assert Path(tmp_path / "test_write_input_files.npz").is_file()
    assert Path(tmp_path / "test_write_input_files.krn").is_file()
    assert Path(tmp_path / "test_write_input_files.p2").is_file()


def test_save_npz(state_zero, tmp_path):
    """Test the generate `npz` mesh.

    We only check that the saved mesh has the same number of points and cells of the system mesh.
    - `knt` collects the coordinates of all mesh points.
    - `ijk` collects collectivity + region ID for each tetrahedron.
    """
    os.chdir(tmp_path)
    system = mm.System(name="test_save_npz")
    system.m = state_zero.copy()
    save_mesh_npz(system)
    assert Path(tmp_path / "test_save_npz.npz")
    mesh = np.load(tmp_path / "test_save_npz.npz")
    assert "knt" in mesh and "ijk" in mesh
    assert mesh["knt"].shape == (system.m.n_points, 3)
    assert mesh["ijk"].shape == (system.m.n_cells, 5)


@pytest.mark.parametrize(
    "Js,A,K,theta,phi",
    [
        (1.0, 2.0, 3.0, np.pi / 4, np.pi / 3),
        (1.76, 1e-12, 4e5, 0.0, 0.0),
    ],
)
def test_write_krn(Js, A, K, theta, phi, state_zero, tmp_path):
    """Test script creating `krn` input file."""
    os.chdir(tmp_path)
    system = mm.System(name="test_write_krn")
    state = state_zero.copy()
    state.cell_data["Js"] *= Js
    system.m = state
    u = (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    system.energy = mm.Exchange(A=A) + mm.UniaxialAnisotropy(K=K, u=u)
    write_krn(system)
    read_array = (
        pd.read_csv(
            tmp_path / "test_write_krn.krn", comment="#", sep=" ", names=["theta", "phi", "K1", "unused", "Js", "A"]
        )
        .to_numpy()
        .flatten()
    )
    np.testing.assert_allclose(read_array, [theta, phi, K, 0, Js, A])


@pytest.mark.parametrize(
    "size,m,hx,hy,hz,hstart,hfinal,hstep,max_iter,cg_tol",
    [
        (
            1e-6,
            (0, 0, 1),
            np.float64(1),
            np.float64(0),
            np.float64(0),
            np.float64(-1),
            np.float64(2),
            np.float64(0.6),
            1000,
            1e-6,
        ),
        (
            1e-9,
            (0, 1, 1),
            np.float64(-1),
            np.float64(0),
            np.float64(0),
            np.float64(-1),
            np.float64(3),
            np.float64(1),
            40000,
            1e-9,
        ),
    ],
)
def test_write_p2(size, m, hx, hy, hz, hstart, hfinal, hstep, max_iter, cg_tol, state_zero, tmp_path):
    """Test script creating `p2` input file."""
    os.chdir(tmp_path)
    system = mm.System(name="test_write_p2")
    state = state_zero.copy()
    m_array = np.array(m, dtype=np.float32) / np.linalg.norm(m)
    state.point_data["m"][:, 0] = m_array[0]
    state.point_data["m"][:, 1] = m_array[1]
    state.point_data["m"][:, 2] = m_array[2]
    system.m = state
    average_magnetization = state.point_data["m"].sum(axis=0) / state.n_points
    kwargs = {
        "size": size,
        "hx": hx,
        "hy": hy,
        "hz": hz,
        "hstart": hstart,
        "hfinal": hfinal,
        "hstep": hstep,
        "max_iter": max_iter,
        "cg_tol": cg_tol,
    }

    write_p2(system, **kwargs)
    assert (tmp_path / "test_write_p2.p2").read_text() == dedent(
        f"""\
        [mesh]
        size = {size}

        [initial state]
        mx = {average_magnetization[0]!s}
        my = {average_magnetization[1]!s}
        mz = {average_magnetization[2]!s}

        [field]
        hx = {hx}
        hy = {hy}
        hz = {hz}
        hstart = {hstart}
        hfinal = {hfinal}
        hstep = {hstep}

        [minimizer]
        max_iter = {max_iter}

        [poisson]
        cg_tol = {cg_tol}

        """
    )
