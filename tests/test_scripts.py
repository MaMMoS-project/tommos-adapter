import configparser
import io

import micromagneticmodel as mm
import numpy as np
import pytest
import pyvista as pv

from tommos_adapter import scripts


@pytest.fixture
def system():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1e-9, 0.0, 0.0],
            [0.0, 1e-9, 0.0],
            [0.0, 0.0, 1e-9],
            [1e-9, 1e-9, 1e-9],
        ]
    )
    cells = np.array([4, 0, 1, 2, 3, 4, 1, 2, 3, 4])
    cell_types = np.array([pv.CellType.TETRA, pv.CellType.TETRA], dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.point_data["m"] = np.tile((0.0, 0.0, 1.0), (grid.n_points, 1))
    grid.cell_data["region_id"] = np.array([10, 20])
    grid.cell_data["Ms"] = np.array([1e6, 0.5e6])
    grid.field_data["region_ids"] = np.array([10, 20])
    grid.field_data["region_names"] = np.array(["hard", "soft"])

    energy = (
        mm.Exchange(A={"hard": 1e-11, "soft": 5e-12})
        + mm.UniaxialAnisotropy(
            K={"hard": 1e6, "soft": 2e5},
            u={"hard": (0, 0, 1), "soft": (1, 0, 0)},
        )
        + mm.Demag()
        + mm.Zeeman(H=(0, 0, 1e6))
    )
    return mm.System(name="two_regions", energy=energy, m=grid)


def test_mesh_data_scales_points_and_remaps_regions(system):
    data = scripts.mesh_data(system, mesh_unit=1e-9)

    assert data.keys() == {"knt", "ijk"}
    assert np.allclose(data["knt"], system.m.points / 1e-9)
    assert np.array_equal(data["ijk"][:, :4], np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))
    assert np.array_equal(data["ijk"][:, 4], np.array([1, 2]))


def test_krn_script_translates_regional_materials(system):
    table = np.loadtxt(io.StringIO(scripts.krn_script(system)))

    assert table.shape == (2, 6)
    assert table[0, 0] == pytest.approx(0.0)
    assert table[1, 0] == pytest.approx(np.pi / 2)
    assert table[:, 1] == pytest.approx((0.0, 0.0))
    assert table[:, 2] == pytest.approx((1e6, 2e5))
    assert table[:, 3] == pytest.approx((0.0, 0.0))
    assert table[:, 4] == pytest.approx(scripts.MU0 * np.array((1e6, 0.5e6)))
    assert table[:, 5] == pytest.approx((1e-11, 5e-12))


def test_p2_script_creates_single_field_step(system):
    config = configparser.ConfigParser()
    config.read_string(scripts.p2_script(system, max_iter=1000, tol_fun=1e-8, eps_a="auto"))

    assert config.getfloat("mesh", "size") == pytest.approx(1e-9)
    assert config.getfloat("initial state", "mx") == pytest.approx(0.0)
    assert config.getfloat("initial state", "my") == pytest.approx(0.0)
    assert config.getfloat("initial state", "mz") == pytest.approx(1.0)
    assert config.getfloat("field", "hstart") == pytest.approx(scripts.MU0 * 1e6)
    assert config.getfloat("field", "hfinal") == pytest.approx(scripts.MU0 * 1e6)
    assert config.getfloat("field", "hz") == pytest.approx(1.0)
    assert not config.getboolean("field", "loop")
    assert config.getint("minimizer", "max_iter") == 1000
    assert config.get("minimizer", "eps_a") == "auto"


def test_p2_script_uses_restart_for_nonuniform_m(system):
    system.m.point_data["m"][-1] = (1.0, 0.0, 0.0)

    config = configparser.ConfigParser()
    config.read_string(scripts.p2_script(system))

    assert scripts.needs_initial_state_file(system)
    assert config.getint("initial state", "ini") == 0
    assert not config.has_option("initial state", "mx")


def test_krn_script_rejects_missing_demag(system):
    system.energy = mm.Exchange(A=1e-11)

    with pytest.raises(ValueError, match="always computes demagnetisation"):
        scripts.krn_script(system)
