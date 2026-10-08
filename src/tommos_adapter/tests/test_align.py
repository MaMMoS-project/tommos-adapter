"""Test hysteresis loop."""

import micromagneticmodel as mm
import numpy as np
import pytest
import pyvista as pv

import tommos_adapter as ta


@pytest.mark.parametrize("sign", [+1, -1])
def test_align_pos(mesh, tmp_path, sign):
    """Test alignment with external field."""
    # Intrinsic properties
    Js = 1.76
    A = 1e-12
    K = 4e5
    u_theta = 0
    u_phi = 0

    # Read mesh and define magnetization
    state = pv.read(mesh["vtu"])
    id_array = state.get_array("mat_id")
    state.cell_data["id"] = id_array  # Rename `mat_id` cell data to `id`
    state.cell_data.remove("mat_id")  # remove old data with name `mat_id`
    m = np.zeros((state.n_points, 3), dtype=np.float32)
    m[:, [1, 2]] = 1 / np.sqrt(2)
    state.point_data["m"] = m
    Js_array = Js * np.ones_like(state.n_cells, dtype=np.float32)
    state.cell_data["Js"] = Js_array

    # Define micromagnetic system
    system = mm.System(name="test_relaxation")
    system.m = state

    # Define energies
    u = (np.sin(u_theta) * np.cos(u_phi), np.sin(u_theta) * np.sin(u_phi), np.cos(u_theta))
    system.energy = mm.Exchange(A=A) + mm.UniaxialAnisotropy(K=K, u=u)

    # Drive system
    Hmin = (-sign * 2 / mm.consts.mu0, 0, 0)
    Hmax = (sign * 1 / mm.consts.mu0, 0, 0)
    n = 5
    hd = ta.HysteresisDriver()
    hd.drive(system, dirname=tmp_path, Hmin=Hmin, Hmax=Hmax, n=n)

    # Read final magnetization
    mag_array = system.m.get_array("m")
    avg_magnetization = sum(mag_array) / mag_array.shape[0]
    avg_magnetization /= np.linalg.norm(avg_magnetization)
    np.testing.assert_allclose(avg_magnetization[0], sign * 1)
