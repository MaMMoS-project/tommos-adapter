"""Test hysteresis loop."""

import micromagneticmodel as mm
import numpy as np
import pytest

import tommos_adapter as ta


@pytest.mark.parametrize("theta_deg, phi_deg", [(0, 0), (45, 45), (135, 135)])
def test_align_external_field(m_cube_20_001, tmp_path, theta_deg, phi_deg):
    """Test alignment with external field.

    We start from uniform magnetization (0, 1/√2, 1/√2) and test that in absence of anisotropy the magnetization
    points to the external field, whose direction is given by the angles `theta` and `phi`.
    """
    # Intrinsic properties
    Js = 1.76
    A = 1e-12
    h_theta = np.deg2rad(theta_deg)
    h_phi = np.deg2rad(phi_deg)
    h_direction = (
        np.sin(h_theta) * np.cos(h_phi),
        np.sin(h_theta) * np.sin(h_phi),
        np.cos(h_theta),
    )
    h_field = np.array(h_direction) / mm.consts.mu0

    # Define micromagnetic system
    system = mm.System(name="test_align")
    system.m = m_cube_20_001
    system.m.point_data["m"] = np.tile(np.array([0, 1, 1], dtype=np.float32) / np.sqrt(2), (system.m.n_points, 1))
    system.m.cell_data["Ms"] *= Js

    # Define energies
    system.energy = mm.Exchange(A=A) + mm.Zeeman(H=h_field) + mm.Demag()

    # Drive system
    md = ta.MinDriver()
    md.drive(system, dirname=tmp_path)

    # Check the averaged magnetization directly so that reduced alignment is visible.
    average_magnetization = np.mean(system.m.get_array("m"), axis=0)
    assert np.vdot(average_magnetization, h_direction) > 0.9


@pytest.mark.parametrize("theta_deg, phi_deg", [(0, 0), (45, 45), (135, 135)])
def test_align_anisotropy_axis(m_cube_20_001, tmp_path, theta_deg, phi_deg):
    """Test alignment with anisotropy axis.

    We start from uniform magnetization (0, 1/√2, 1/√2) and test that in absence of external field the magnetization
    points in the direction of the anisotropy axis, whose direction is given by the angles `theta` and `phi`.
    """
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

    # Define micromagnetic system
    system = mm.System(name="test_align")
    system.m = m_cube_20_001
    system.m.point_data["m"] = np.tile(np.array([0, 1, 1], dtype=np.float32) / np.sqrt(2), (system.m.n_points, 1))
    system.m.cell_data["Ms"] *= Js

    # Define energies
    system.energy = mm.Exchange(A=A) + mm.UniaxialAnisotropy(K=K, u=u_direction) + mm.Demag()

    # Drive system
    md = ta.MinDriver()
    md.drive(system, dirname=tmp_path)

    # Check the averaged magnetization directly so that reduced alignment is visible.
    average_magnetization = np.mean(system.m.get_array("m"), axis=0)
    assert abs(np.vdot(average_magnetization, u_direction)) > 0.9
