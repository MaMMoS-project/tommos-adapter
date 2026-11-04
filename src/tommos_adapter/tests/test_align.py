"""Test hysteresis loop."""

import micromagneticmodel as mm
import numpy as np
import pytest

import tommos_adapter as ta


@pytest.mark.parametrize("sign", [+1, -1])
def test_align_external_field(state_zero, tmp_path, sign):
    """Test alignment with external field."""
    # Intrinsic properties
    Js = 1.76
    A = 1e-12
    K = 4e5
    u_theta = 0
    u_phi = 0

    # Read mesh and define magnetization
    state = state_zero.copy()
    state.point_data["m"][:, [1, 2]] = 1 / np.sqrt(2)
    state.cell_data["Ms"] *= Js

    # Define micromagnetic system
    system = mm.System(name="test_relaxation")
    system.m = state

    # Define energies
    u = (np.sin(u_theta) * np.cos(u_phi), np.sin(u_theta) * np.sin(u_phi), np.cos(u_theta))
    system.energy = mm.Exchange(A=A) + mm.UniaxialAnisotropy(K=K, u=u) + mm.Demag()

    # Drive system
    hd = ta.HysteresisDriver()
    hd.drive(
        system,
        dirname=tmp_path,
        h_direction=(1, 0, 0),
        h_start=-sign / mm.consts.mu0,
        h_final=sign / mm.consts.mu0,
        n=5,
    )

    # Check orientation of normalized average magnetization
    avg_magnetization = np.mean(system.m.point_data["m"], axis=0)
    normalized_avg_magnetization = avg_magnetization / np.linalg.norm(avg_magnetization)
    np.testing.assert_allclose(normalized_avg_magnetization[0], sign, rtol=1e-3)
