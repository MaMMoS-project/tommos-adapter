"""Test min driver."""

import sys

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest

import tommos_adapter as ta


def test_allowed_attributes():
    assert ta.MinDriver()._allowed_attributes == ["mstep", "method", "max_iter", "tol_fun", "eps_a"]
    driver = ta.MinDriver(max_iter=1000, tol_fun=1e-8, eps_a="auto")
    assert dict(driver) == {"max_iter": 1000, "tol_fun": 1e-8, "eps_a": "auto"}
    with pytest.raises(AttributeError, match="Invalid attribute key"):
        ta.MinDriver(unexpected_argument=1)


@pytest.mark.parametrize(
    "H",
    [
        (1e6, 0, 0),
        np.array([0, -2e6, 0]),
        [1e6, 1e6, 1e6],
        [0, 0, 0],
    ],
)
def test_correct_zeeman(H):
    """Test `MinDriver` argument processing with correct Zeeman energy term."""
    system = mm.System(name="test_correct_zeeman")
    system.energy = mm.Demag() + mm.Zeeman(H=H)
    hd = ta.MinDriver()
    args = hd._get_required_p2_arguments(system, {})
    H_norm = np.linalg.norm(H)
    h = np.array(H) / H_norm if H_norm else np.array(H)
    assert args == {
        "hx": h[0],
        "hy": h[1],
        "hz": h[2],
        "hstart": H_norm * mm.consts.mu0,
        "hfinal": H_norm * mm.consts.mu0,
        "hstep": 0,
    }


def test_without_zeeman():
    """Test `MinDriver` argument processing without Zeeman energy term."""
    system = mm.System(name="test_no_zeeman")
    system.energy = mm.Demag()
    hd = ta.MinDriver()
    assert hd._get_required_p2_arguments(system, {}) == {}


def test_wrong_zeeman_dict():
    """Test `MinDriver` argument processing with wrong Zeeman energy term (defined by dictionary)."""
    system = mm.System(name="test_dict_zeeman")
    system.energy = mm.Demag() + mm.Zeeman(H={"region1": (0, 0, 1e6), "region2": (0, 0, -1e6)})
    hd = ta.MinDriver()
    with pytest.raises(ValueError, match="Multi-domain external field is not supported."):
        hd._get_required_p2_arguments(system, {})


def test_wrong_zeeman_changing():
    """Test `MinDriver` argument processing with wrong Zeeman energy term (changing via `func`)."""
    system = mm.System(name="test_changing_zeeman")

    system.energy = mm.Demag() + mm.Zeeman(H=(0, 0, 1e6), func="sin", f=1e9, t0=0)
    hd = ta.MinDriver()
    with pytest.raises(ValueError, match="Changing external field is not supported."):
        hd._get_required_p2_arguments(system, {})

    system.energy = mm.Demag() + mm.Zeeman(H=(0, 0, 1e6), func=lambda t: np.exp(-t / 1e-10), dt=1e-13)
    hd = ta.MinDriver()
    with pytest.raises(ValueError, match="Changing external field is not supported."):
        hd._get_required_p2_arguments(system, {})


def test_wrong_zeeman_discretisedfield():
    """Test `MinDriver` argument processing with wrong Zeeman energy term (defined via discretisedfield)."""
    system = mm.System(name="test_discretisedfield_zeeman")

    region = df.Region(p1=(0, 0, 0), p2=(5e-9, 5e-9, 10e-9))
    mesh = df.Mesh(region=region, n=(5, 5, 10))
    H = df.Field(mesh, nvdim=3, value=(1e6, -1e6, 0))

    system.energy = mm.Demag() + mm.Zeeman(H=H)
    hd = ta.MinDriver()
    with pytest.raises(ValueError, match="External field given as a `discretisedfield.Field` is not supported."):
        hd._get_required_p2_arguments(system, {})


def test_check_system_unsupported_energies(state_zero):
    """Test that `MinDriver` does not support RKKY energy terms."""
    system = mm.System(name="test_unsupported_energies")
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    system.m = state
    system.energy = mm.Demag() + mm.RKKY()
    hd = ta.HysteresisDriver()
    with pytest.raises(ValueError, match=r"Energy term RKKY\(\) is not supported by this driver"):
        hd._check_system(system)


def test_check_system_requires_unstructured_grid():
    """`MinDriver` does not accept a discretisedfield magnetization."""
    system = mm.System(name="test_discretisedfield")
    system.energy = mm.Demag()
    region = df.Region(p1=(0, 0, 0), p2=(1e-9, 1e-9, 1e-9))
    mesh = df.Mesh(region=region, n=(1, 1, 1))
    system.m = df.Field(mesh, nvdim=3, value=(0, 1, 1), norm=1e6)

    with pytest.raises(TypeError, match="requires system.m to be a pyvista.UnstructuredGrid"):
        ta.MinDriver()._check_system(system)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"Hmin": (-1, 0, 0)},
        {"Hmax": (1, 0, 0)},
        {"n": 3},
    ],
)
def test_wrong_args(kwargs):
    """Test `MinDriver` with not allowed arguments."""
    with pytest.raises(RuntimeError, match="Arguments `Hmin`, `Hmax`, `n` are not supported"):
        ta.MinDriver()._checkargs(kwargs)


def test_write_input_files_after_checking(state_zero, tmp_path, monkeypatch):
    """Check inputs after processing the field arguments."""
    system = mm.System(name="test_write_input_files_after_checking")
    state = state_zero.copy()
    state.point_data["m"][:, 2] = 1
    system.m = state
    system.energy = mm.Demag() + mm.Zeeman(H=(0, 0, 2e6))
    hd = ta.MinDriver(max_iter=1000, tol_fun=1e-8, eps_a="auto")
    kwargs = {}
    monkeypatch.chdir(tmp_path)
    hd._checkargs(kwargs)
    hd._write_input_files(system, **kwargs)
    assert (tmp_path / "test_write_input_files_after_checking.npz").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.krn").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.p2").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.krn").read_text() == "0 0 0 0 1 0\n"
    p2_text = (tmp_path / "test_write_input_files_after_checking.p2").read_text()
    assert "size = 1.0000000000000001e-09" in p2_text
    assert "mx = 0" in p2_text
    assert "my = 0" in p2_text
    assert "mz = 1" in p2_text
    assert "hz = 1" in p2_text
    assert "hstart = 2" in p2_text
    assert "hfinal = 2" in p2_text
    assert "max_iter = 1000" in p2_text
    assert "tol_fun = 1e-08" in p2_text
    assert "eps_a = auto" in p2_text


def test_schedule_command():
    """Test driver schedule command."""
    system = mm.System(name="test_schedule_command")
    hd = ta.MinDriver()
    tr = ta.TommosRunner()
    schedule_command = "tommos loop test_schedule_command"
    if sys.platform == "darwin":
        schedule_command += " --cpu-spmv-backend scipy --no-cpp-mkl"
    assert hd._schedule_commands(system, tr) == [schedule_command]
    assert hd._schedule_commands(system, None) == [schedule_command]


def test_x():
    """Test `x` label."""
    assert ta.MinDriver()._x == "B_hysteresis"
