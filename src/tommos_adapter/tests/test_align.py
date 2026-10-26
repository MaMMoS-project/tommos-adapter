"""Test hysteresis loop."""

import micromagneticmodel as mm
import numpy as np
import pytest

import tommos_adapter as ta


@pytest.mark.parametrize("theta_deg, phi_deg", [(0, 0), (45, 45), (135, 135)])
def test_align_external_field(monkeypatch, state_zero, tmp_path, theta_deg, phi_deg):
    """Test alignment with external field."""
    monkeypatch.chdir(tmp_path)
    # Intrinsic properties
    Js = 1.76
    A = 1e-12
    K = 0
    h_theta = np.deg2rad(theta_deg)
    h_phi = np.deg2rad(phi_deg)
    h_direction = (
        np.sin(h_theta) * np.cos(h_phi),
        np.sin(h_theta) * np.sin(h_phi),
        np.cos(h_theta),
    )
    h_field = np.array(h_direction) / mm.consts.mu0

    # Read mesh and define magnetization
    state = state_zero.copy()
    state.point_data["m"][:, [1, 2]] = 1 / np.sqrt(2)
    state.cell_data["Ms"] *= Js

    # Define micromagnetic system
    system = mm.System(name="test_align")
    system.m = state

    # Define energies
    system.energy = mm.Exchange(A=A) + mm.UniaxialAnisotropy(K=K, u=(0, 0, 1)) + mm.Zeeman(H=h_field) + mm.Demag()

    # Drive system
    md = ta.MinDriver()
    md.drive(system)

    # Check the averaged magnetization directly so that reduced alignment is visible.
    average_magnetization = np.mean(system.m.get_array("m"), axis=0)
    assert np.vdot(average_magnetization, h_direction) > 0.9


@pytest.mark.parametrize("theta_deg, phi_deg", [(0, 0), (45, 45), (135, 135)])
def test_align_anisotropy_axis(monkeypatch, state_zero, tmp_path, theta_deg, phi_deg):
    """Test alignment with anisotropy axis."""
    monkeypatch.chdir(tmp_path)
    # Intrinsic properties
    Js = 1.76
    A = 1e-12
    K = 4e6
    u_theta = np.deg2rad(theta_deg)
    u_phi = np.deg2rad(phi_deg)
    u_direction = (
        np.sin(u_theta) * np.cos(u_phi),
        np.sin(u_theta) * np.sin(u_phi),
        np.cos(u_theta),
    )

    # Read mesh and define magnetization
    state = state_zero.copy()
    state.point_data["m"][:, [1, 2]] = 1 / np.sqrt(2)
    state.cell_data["Ms"] *= Js

    # Define micromagnetic system
    system = mm.System(name="test_align")
    system.m = state

    # Define energies
    system.energy = mm.Exchange(A=A) + mm.UniaxialAnisotropy(K=K, u=u_direction) + mm.Demag()

    # Drive system
    md = ta.MinDriver()
    md.drive(system)

    # Check the averaged magnetization directly so that reduced alignment is visible.
    average_magnetization = np.mean(system.m.get_array("m"), axis=0)
    assert abs(np.vdot(average_magnetization, u_direction)) > 0.9
