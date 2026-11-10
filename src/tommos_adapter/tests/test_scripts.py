"""Tests for the TOMMOS input translation functions."""

import configparser
import io

import micromagneticmodel as mm
import numpy as np
import pytest
import pyvista as pv

from tommos_adapter import scripts


@pytest.fixture
def system():
    """Return a tetrahedral test system with two materials."""
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
    grid.cell_data["mat_id"] = np.array([10, 20])
    grid.cell_data["Ms"] = np.array([1e6, 0.5e6])
    grid.field_data["mat_ids"] = np.array([10, 20])
    grid.field_data["mat_names"] = np.array(["hard", "soft"])
    grid.field_data["mesh_unit_length"] = 1e-9

    energy = (
        mm.Exchange(A={"hard": 1e-11, "soft": 5e-12})
        + mm.UniaxialAnisotropy(
            K={"hard": 1e6, "soft": 2e5},
            u={"hard": (0, 0, 1), "soft": (1, 0, 0)},
        )
        + mm.Demag()
    )
    return mm.System(name="two_materials", energy=energy, m=grid)


@pytest.mark.parametrize(
    "mesh_unit_length, error_string",
    [
        ([1, 2], "Mesh unit length should be a single-valued field data"),
        (-1e-9, "must be a positive finite number"),
        (np.nan, "must be a positive finite number"),
    ],
)
def test_wrong_mesh_unit_length(system, mesh_unit_length, error_string):
    system.m.field_data["mesh_unit_length"] = mesh_unit_length
    with pytest.raises(ValueError, match=error_string):
        scripts.p2_script(
            system,
            hx=0,
            hy=0,
            hz=1,
            hstart=-1,
            hfinal=1,
            hstep=0.5,
        )


def test_mesh_data_remaps_materials(system):
    """Test remapping of material identifiers."""
    data = scripts.mesh_data(system)

    assert data.keys() == {"knt", "ijk"}
    assert np.allclose(data["knt"], system.m.points)
    assert np.array_equal(data["ijk"][:, :4], np.array([[0, 1, 2, 3], [1, 2, 3, 4]]))
    assert np.array_equal(data["ijk"][:, 4], np.array([1, 2]))


def test_mesh_data_requires_material_ids(system):
    """The public translator enforces the documented material contract."""
    system.m.cell_data.remove("mat_id")

    with pytest.raises(ValueError, match="cell_data does not contain 'mat_id'"):
        scripts.mesh_data(system)


def test_mesh_data_does_not_write_files(system, tmp_path, monkeypatch):
    """Pure translation functions do not write into the working directory."""
    monkeypatch.chdir(tmp_path)

    scripts.mesh_data(system)
    scripts.krn_script(system)
    scripts.p2_script(
        system,
        hx=0,
        hy=0,
        hz=1,
        hstart=-1,
        hfinal=1,
        hstep=0.5,
    )

    assert not list(tmp_path.iterdir())


def test_krn_script_translates_multiple_materials(system):
    """Material translation preserves region-dependent parameters."""
    table = np.loadtxt(io.StringIO(scripts.krn_script(system)))

    assert table.shape == (2, 6)
    assert table[0, 0] == pytest.approx(0.0)
    assert table[1, 0] == pytest.approx(np.pi / 2)
    assert table[:, 1] == pytest.approx((0.0, 0.0))
    assert table[:, 2] == pytest.approx((1e6, 2e5))
    assert table[:, 3] == pytest.approx((0.0, 0.0))
    assert table[:, 4] == pytest.approx(mm.consts.mu0 * np.array((1e6, 0.5e6)))
    assert table[:, 5] == pytest.approx((1e-11, 5e-12))


def test_krn_script_translates_default_materials(system):
    """Explicit regions override Ubermag-style default material values."""
    system.energy = (
        mm.Exchange(A={"default": 1e-11, "soft": 5e-12})
        + mm.UniaxialAnisotropy(
            K={"default": 1e6, "soft": 2e5},
            u={"default": (0, 0, 1), "soft": (1, 0, 0)},
        )
        + mm.Demag()
    )

    table = np.loadtxt(io.StringIO(scripts.krn_script(system)))

    # KRN columns are theta, phi, K1, K2, Js, and A.
    assert table[:, 0] == pytest.approx((0.0, np.pi / 2))
    assert table[:, 2] == pytest.approx((1e6, 2e5))
    assert table[:, 5] == pytest.approx((1e-11, 5e-12))


def test_krn_script_matches_known_example(system):
    """KRN translation matches a known TOMMOS input file."""
    expected = (
        "0 0 1000000 0 1.2566370614359172 9.9999999999999994e-12\n"
        "1.5707963267948966 0 200000 0 0.62831853071795862 4.9999999999999997e-12\n"
    )

    assert scripts.krn_script(system) == expected


def test_p2_script_translates_hysteresis_parameters(system):
    """P2 translation represents the field sweep supplied by the driver."""
    config = configparser.ConfigParser()
    config.read_string(
        scripts.p2_script(
            system,
            hx=-1,
            hy=0,
            hz=0,
            hstart=-1,
            hfinal=2,
            hstep=0.3,
            mstep=1,
            method="tr",
            max_iter=1000,
            tol_fun=1e-8,
            eps_a="auto",
        )
    )

    assert config.getfloat("mesh", "size") == pytest.approx(1e-9)
    assert config.getfloat("initial state", "mx") == pytest.approx(0.0)
    assert config.getfloat("initial state", "my") == pytest.approx(0.0)
    assert config.getfloat("initial state", "mz") == pytest.approx(1.0)
    assert config.getfloat("field", "hstart") == pytest.approx(-1.0)
    assert config.getfloat("field", "hfinal") == pytest.approx(2.0)
    assert config.getfloat("field", "hstep") == pytest.approx(0.3)
    assert config.getfloat("field", "hx") == pytest.approx(-1.0)
    assert config.getfloat("field", "mstep") == pytest.approx(1)
    assert config.get("minimizer", "method") == "tr"
    assert config.getint("minimizer", "max_iter") == 1000
    assert config.getfloat("minimizer", "tol_fun") == pytest.approx(1e-8)
    assert config.get("minimizer", "eps_a") == "auto"


def test_p2_script_rejects_nonuniform_m(system):
    """Reject a nonuniform state until restart-VTU writing is implemented."""
    system.m.point_data["m"][-1] = (1.0, 0.0, 0.0)

    assert scripts.needs_initial_state_file(system)
    with pytest.raises(ValueError, match="only a uniform initial magnetisation"):
        scripts.p2_script(
            system,
            hx=0,
            hy=0,
            hz=1,
            hstart=-1,
            hfinal=1,
            hstep=0.5,
        )


@pytest.mark.parametrize(
    "field_kwargs,error_message",
    [
        ({"hx": 0, "hy": 0, "hz": 0}, "non-zero direction"),
        ({"hx": 1, "hy": 1, "hz": 0}, "unit vector"),
    ],
)
def test_p2_script_validates_field_direction(system, field_kwargs, error_message):
    """P2 translation accepts only a finite unit field direction."""
    with pytest.raises(ValueError, match=error_message):
        scripts.p2_script(
            system,
            **field_kwargs,
            hstart=-1,
            hfinal=1,
            hstep=0.5,
        )


def test_translation_requires_unstructured_grid():
    """Translation functions reject discretisedfield magnetisation objects."""
    system = mm.examples.macrospin()

    with pytest.raises(TypeError, match=r"pyvista\.UnstructuredGrid"):
        scripts.mesh_data(system)
