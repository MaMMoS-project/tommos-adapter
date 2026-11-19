"""Integration tests for externally generated meshes."""

import micromagneticmodel as mm
import numpy as np
import pyvista as pv

import tommos_adapter as ta


def test_hysteresis_with_pyvista_mesh(tmp_path):
    """Run a hysteresis calculation with a mesh created directly by PyVista."""
    points = np.array(
        [
            [0, 0, 0],
            [20, 0, 0],
            [0, 20, 0],
            [0, 0, 20],
        ],
        dtype=float,
    )
    # tommos requires the mesh to be centered around the origin.
    points -= points.mean(axis=0)

    cells = np.array([4, 0, 1, 2, 3])
    celltypes = np.array([pv.CellType.TETRA])
    mesh = pv.UnstructuredGrid(cells, celltypes, points)

    mesh.point_data["m"] = np.tile((0.0, 0.0, 1.0), (mesh.n_points, 1))
    mesh.cell_data["mat_id"] = np.array([1])
    mesh.cell_data["Ms"] = np.array([8e5])
    mesh.field_data["mesh_unit_length"] = 1e-9

    system = mm.System(name="pyvista_mesh", m=mesh)
    system.energy = mm.Exchange(A=1e-11) + mm.UniaxialAnisotropy(K=5e4, u=(0, 0, 1)) + mm.Demag()
    system.T = 0

    ta.HysteresisDriver().drive(
        system,
        dirname=tmp_path,
        append=False,
        verbose=0,
        h_direction=(0, 0, 1),
        h_start=1.0 / mm.consts.mu0,
        h_final=-1.0 / mm.consts.mu0,
        n=2,
    )

    np.testing.assert_allclose(system.table.data.B_hysteresis, (1.0, -1.0))
    np.testing.assert_allclose(system.m.points, points)
    np.testing.assert_allclose(np.linalg.norm(system.m.point_data["m"], axis=1), 1.0)
