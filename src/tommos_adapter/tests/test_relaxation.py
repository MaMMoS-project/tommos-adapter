"""Test hysteresis loop."""

import micromagneticmodel as mm
import numpy as np
import pyvista as pv


def test_relaxation(mesh, tmp_path):
    """Test relaxation.

    In absence of an external magnetic field, we expect the magnetization to relax to the anisotropy direction.
    In this case, the anisotropy unit vector is `(1, 0, 0)`, so we test that the first component of the magnetization
    is close to `1`.
    """
    # Intrinsic properties
    Js = 0.51
    A = 1.4e-12
    K = 930000.0
    u_theta = 0
    u_phi = 0

    # Read mesh and define magnetization
    state = pv.read(mesh["vtu"])
    m = np.zeros((state.n_points, 3), dtype=np.float32)
    m[:, [0, 2]] = 1 / np.sqrt(2)
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
    # TODO: Define MinDriver
