"""Test min driver."""

import sys

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest

import tommos_adapter as ta


def test_allowed_attributes():
    assert ta.MinDriver()._allowed_attributes == ["mstep", "method", "max_iter", "tol_fun", "eps_a", "num_devices"]
    driver = ta.MinDriver(max_iter=1000, tol_fun=1e-8, eps_a="auto")
    assert dict(driver) == {"max_iter": 1000, "tol_fun": 1e-8, "eps_a": "auto"}
    with pytest.raises(AttributeError, match="Invalid attribute key"):
        ta.MinDriver(unexpected_argument=1)


@pytest.mark.parametrize(
    "H, hx, hy, hz, h_val",
    [
        ((1e6, 0, 0), 1, 0, 0, 1e6),
        (np.array([0, -2e6, 0]), 0, -1, 0, 2e6),
        ([1e6, 1e6, 1e6], 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), 1e6 * np.sqrt(3)),
    ],
)
def test_correct_zeeman(H, hx, hy, hz, h_val):
    """Test `MinDriver` argument processing with correct Zeeman energy term."""
    system = mm.System(name="test_correct_zeeman")
    system.energy = mm.Demag() + mm.Zeeman(H=H)
    md = ta.MinDriver()
    args = md._get_required_p2_arguments(system, {})
    np.testing.assert_allclose(args["hx"], hx)
    np.testing.assert_allclose(args["hy"], hy)
    np.testing.assert_allclose(args["hz"], hz)
    np.testing.assert_allclose(args["hstart"], h_val * mm.consts.mu0)
    np.testing.assert_allclose(args["hfinal"], h_val * mm.consts.mu0)


def test_without_zeeman():
    """Test `MinDriver` argument processing without Zeeman energy term."""
    system = mm.System(name="test_no_zeeman")
    system.energy = mm.Demag()
    md = ta.MinDriver()
    assert md._get_required_p2_arguments(system, {}) == {"hstart": 0.0, "hfinal": 0.0}


def test_zeeman_zero_H():
    """Test `MinDriver` argument processing without Zeeman energy term."""
    system = mm.System(name="test_zeeman_zero")
    system.energy = mm.Demag() + mm.Zeeman(H=(0, 0, 0))
    md = ta.MinDriver()
    args = md._get_required_p2_arguments(system, {})
    assert args == {"hx": 0, "hy": 0, "hz": 1, "hstart": 0, "hfinal": 0}


def test_wrong_zeeman_dict():
    """Test `MinDriver` argument processing with wrong Zeeman energy term (defined by dictionary)."""
    system = mm.System(name="test_dict_zeeman")
    system.energy = mm.Demag() + mm.Zeeman(H={"region1": (0, 0, 1e6), "region2": (0, 0, -1e6)})
    md = ta.MinDriver()
    with pytest.raises(ValueError, match="Multi-domain external field is not supported."):
        md._get_required_p2_arguments(system, {})


def test_wrong_zeeman_changing():
    """Test `MinDriver` argument processing with wrong Zeeman energy term (changing via `func`)."""
    system = mm.System(name="test_changing_zeeman")

    system.energy = mm.Demag() + mm.Zeeman(H=(0, 0, 1e6), func="sin", f=1e9, t0=0)
    md = ta.MinDriver()
    with pytest.raises(ValueError, match="Changing external field is not supported."):
        md._get_required_p2_arguments(system, {})

    system.energy = mm.Demag() + mm.Zeeman(H=(0, 0, 1e6), func=lambda t: np.exp(-t / 1e-10), dt=1e-13)
    md = ta.MinDriver()
    with pytest.raises(ValueError, match="Changing external field is not supported."):
        md._get_required_p2_arguments(system, {})


def test_wrong_zeeman_discretisedfield():
    """Test `MinDriver` argument processing with wrong Zeeman energy term (defined via discretisedfield)."""
    system = mm.System(name="test_discretisedfield_zeeman")

    region = df.Region(p1=(0, 0, 0), p2=(5e-9, 5e-9, 10e-9))
    mesh = df.Mesh(region=region, n=(5, 5, 10))
    H = df.Field(mesh, nvdim=3, value=(1e6, -1e6, 0))

    system.energy = mm.Demag() + mm.Zeeman(H=H)
    md = ta.MinDriver()
    with pytest.raises(ValueError, match="External field given as a `discretisedfield.Field` is not supported."):
        md._get_required_p2_arguments(system, {})


def test_check_system_unsupported_energies(state_zero):
    """Test that `MinDriver` does not support RKKY energy terms."""
    system = mm.System(name="test_unsupported_energies")
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    system.m = state
    system.energy = mm.Demag() + mm.RKKY()
    md = ta.MinDriver()
    with pytest.raises(ValueError, match=r"Energy term RKKY\(\) is not supported by this driver"):
        md._check_system(system)


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
        {"h_direction": (1, 0, 0)},
        {"h_start": -1.0},
        {"h_final": 1.0},
        {"n": 3},
    ],
)
def test_wrong_args(kwargs):
    """Test `MinDriver` with not allowed arguments."""
    with pytest.raises(TypeError, match="Arguments `h_direction`, `h_start`, `h_final`, `n` are not supported"):
        ta.MinDriver()._checkargs(kwargs)


def test_write_input_files_after_checking(state_zero, tmp_path, monkeypatch):
    """Check inputs after processing the field arguments."""
    system = mm.System(name="test_write_input_files_after_checking")
    state = state_zero.copy()
    state.point_data["m"][:, 2] = 1
    system.m = state
    system.energy = mm.Demag() + mm.Zeeman(H=(0, 0, 2e6))
    md = ta.MinDriver(max_iter=1000, tol_fun=1e-8, eps_a="auto")
    kwargs = {}
    monkeypatch.chdir(tmp_path)
    md._checkargs(kwargs)
    md._write_input_files(system, **kwargs)
    assert (tmp_path / "test_write_input_files_after_checking.npz").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.krn").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.p2").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.krn").read_text() == "0 0 0 0 1 0\n"
    p2_text = (tmp_path / "test_write_input_files_after_checking.p2").read_text()
    assert "size = 1e-09" in p2_text
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
    md = ta.MinDriver()
    tr = ta.TommosRunner()
    schedule_command = "tommos loop test_schedule_command --add-shell"
    if sys.platform == "darwin":
        schedule_command += " --cpu-spmv-backend scipy --no-cpp-mkl"
    assert md._schedule_commands(system, tr) == [schedule_command]
    assert md._schedule_commands(system, None) == [schedule_command]


def test_x():
    """Test `x` label."""
    assert ta.MinDriver()._x == "B_hysteresis"
