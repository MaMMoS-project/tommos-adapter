"""Test hysteresis driver."""

import sys
from unittest.mock import Mock

import micromagneticmodel as mm
import numpy as np
import pytest

import tommos_adapter as ta


def test_allowed_attributes():
    assert ta.HysteresisDriver()._allowed_attributes == ["mstep", "method", "max_iter", "tol_fun", "eps_a"]
    driver = ta.HysteresisDriver(mstep=1, method="tr", max_iter=1000, tol_fun=1e-8, eps_a="auto")
    assert dict(driver) == {"mstep": 1, "method": "tr", "max_iter": 1000, "tol_fun": 1e-8, "eps_a": "auto"}
    with pytest.raises(AttributeError, match="Invalid attribute key"):
        ta.HysteresisDriver(unexpected_argument=1)


@pytest.mark.parametrize(
    "h_direction,h_start,h_final,n,hx,hy,hz,hstart,hfinal,hstep",
    [
        ((1, 0, 0), -1, 2, 6, 1, 0, 0, -1, 2, 0.6),
        ((-1, 1, 0), 1, -3, 5, -1 / np.sqrt(2), 1 / np.sqrt(2), 0, -1, 3, 1),
        ((0, 0, 3), -8, 2, 21, 0, 0, 1, -8, 2, 0.5),
        ((-1, 0, 0), 2, 0, 3, -1, 0, 0, -2, 0, 1),
    ],
)
@pytest.mark.parametrize("processor", ["_checkargs", "schedule_kwargs_setup", "drive_kwargs_setup"])
def test_process_hysteresis_arguments_single_sweep(
    processor, h_direction, h_start, h_final, n, hx, hy, hz, hstart, hfinal, hstep
):
    """Test that field arguments are processed correctly."""
    kwargs = {"h_direction": h_direction, "h_start": h_start, "h_final": h_final, "n": n}
    hd = ta.HysteresisDriver()
    getattr(hd, processor)(kwargs)
    assert kwargs == {
        "hx": hx,
        "hy": hy,
        "hz": hz,
        "hstart": hstart,
        "hfinal": hfinal,
        "hstep": hstep,
        "n": n,
    }


@pytest.mark.parametrize("arg_to_remove", ["h_direction", "h_start", "h_final", "n"])
def test_missing_required_parameters(arg_to_remove):
    """Test error when missing required arguments."""
    kwargs = {"h_direction": (0, 0, 1), "h_start": 1e6, "h_final": -2e6, "n": 13}
    del kwargs[arg_to_remove]
    with pytest.raises(ValueError, match="`h_direction`, `h_start`, `h_final`, and `n` must all be provided"):
        ta.HysteresisDriver()._checkargs(kwargs)


def test_check_direction():
    pass  # TODO: test


def test_check_direction_raises(h_direction):
    """Test that wrong `n` argument raise errors."""
    # TODO: parametrize
    with pytest.raises(ValueError, match=r"`n` \(number of field step\) must be an integer bigger than 2."):
        ta.tommos_driver._check_direction(h_direction)


def test_check_field_magnitude():
    pass  # TODO: test


def test_check_field_magnitude_raises(h_val):
    """Test that wrong `n` argument raise errors."""
    # TODO: parametrize
    with pytest.raises(ValueError, match=r"`n` \(number of field step\) must be an integer bigger than 2."):
        ta.tommos_driver._check_field_magnitude(h_val)


@pytest.mark.parametrize("n", [np.int64(3), 104])
def test_number_field_steps(n):
    ta.tommos_driver._check_number_field_steps(n)
    pass


@pytest.mark.parametrize("n", [3.0, -1, np.inf, np.nan, [3]])
def test_number_field_steps_raises(n):
    """Test that wrong `n` argument raise errors."""
    with pytest.raises(ValueError, match=r"`n` \(number of field step\) must be an integer bigger than 2."):
        ta.tommos_driver._check_number_field_steps(n)


# TODO: check all tests below


@pytest.mark.parametrize(
    "H, expected",
    [
        ([1, 0, 0], np.array([1.0, 0.0, 0.0])),
        ((0, 0, 8), np.array([0.0, 0.0, 8.0])),
        (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])),
    ],
)
def test_check_field_value(H, expected):
    """Test that correct field values are correctly converted to NumPy arrays."""
    out = ta.tommos_driver._check_field_value(H, "")
    assert isinstance(out, np.ndarray)
    assert out.dtype == expected.dtype
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    "H, expected_error, error_message",
    [
        ("string", RuntimeError, r"Conversion of field .* to NumPy array failed"),
        (
            (-1, 0),
            ValueError,
            "must be convertible to a one-dimensional array containing three numerical values",
        ),
        ((np.nan, 0, 0), ValueError, "must contain only finite values."),
        ((0, 0, -np.nan), ValueError, "must contain only finite values."),
    ],
)
def test_check_field_value_raises(H, expected_error, error_message):
    """Test that incorrect field values raise errors."""
    with pytest.raises(expected_error, match=error_message):
        ta.tommos_driver._check_field_value(H, "")


@pytest.mark.parametrize(
    "Hstart, Hend, h",
    [
        (np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([-1, 0, 0])),
        (np.array([0, 0, -1]), np.array([0, 0, 1]), np.array([0, 0, 1])),
        (np.array([1, 1, 0]), np.array([-1, -1, 0]), np.array([-1 / np.sqrt(2), -1 / np.sqrt(2), 0])),
    ],
)
def test_get_field_direction(Hstart, Hend, h):
    """Test calculation of field direction from `Hstart` to `Hend`."""
    np.testing.assert_allclose(ta.tommos_driver._get_field_direction(Hstart, Hend), h)


@pytest.mark.parametrize(
    "Hstart, Hend, expected_error, error_message",
    [
        (np.array([1, 0, 0]), np.array([1, 0, 0]), ValueError, "Hstart and Hend must differ."),
        (np.array([1, 0, 0]), np.array([0, 1, 0]), ValueError, "Hstart and Hend must be collinear with the origin"),
    ],
)
def test_get_field_direction_raises(Hstart, Hend, expected_error, error_message):
    """Test that errors are raised whne the field direction is unacceptable."""
    with pytest.raises(expected_error, match=error_message):
        ta.tommos_driver._get_field_direction(Hstart, Hend)


@pytest.mark.parametrize(
    "Hstart, Hend, Hn, expected",
    [
        (
            np.array([1, 0, 0]),
            np.array([-1, 0, 0]),
            3,
            {"hx": -1.0, "hy": 0.0, "hz": 0.0, "hstart": -1.0 * mm.consts.mu0, "hfinal": 1.0 * mm.consts.mu0},
        ),
        (
            np.array([0, 0, -3]),
            np.array([0, 0, 4]),
            11,
            {"hx": 0.0, "hy": 0.0, "hz": 1.0, "hstart": -3.0 * mm.consts.mu0, "hfinal": 4.0 * mm.consts.mu0},
        ),
    ],
)
def test_get_field_parameters(Hstart, Hend, Hn, expected):
    """Test parsed field parameters in the right format for the `p2` input file."""
    expected["n"] = Hn
    expected["hstep"] = (expected["hfinal"] - expected["hstart"]) / (Hn - 1)
    assert ta.tommos_driver._get_field_parameters(Hstart, Hend, Hn) == expected


def test_invalid_temperature(state_zero):
    """Raise error on temperature different than zero."""
    system = mm.System(name="wrong_temperature")
    state = state_zero.copy()
    state.point_data["m"][:, 2] = 1
    system.m = state
    system.energy = mm.Demag()
    system.T = 300.0
    hd = ta.HysteresisDriver()
    with pytest.raises(RuntimeError, match="`tommos` supports only T=0 K. Given temperature: 300.0 K"):
        hd._check_system(system)


def test_write_input_files_after_checking(state_zero, tmp_path, monkeypatch):
    """Check inputs after processing the field arguments."""
    system = mm.System(name="test_write_input_files_after_checking")
    state = state_zero.copy()
    state.point_data["m"][:, 2] = 1
    system.m = state
    system.energy = mm.Demag()
    hd = ta.HysteresisDriver(mstep=1, method="tr", max_iter=1000, tol_fun=1e-8, eps_a="auto")
    kwargs = {"Hsteps": [[(1 / mm.consts.mu0, 0, 0), (-2 / mm.consts.mu0, 0, 0), 11]]}
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
    assert "hx = -1" in p2_text
    assert "hstart = -1" in p2_text
    assert "hfinal = 2" in p2_text
    assert "hstep = 0.29999999999999999" in p2_text
    assert "mstep = 1" in p2_text
    assert "method = tr" in p2_text
    assert "max_iter = 1000" in p2_text
    assert "tol_fun = 1e-08" in p2_text
    assert "eps_a = auto" in p2_text


def test_schedule_command():
    """Test driver schedule command."""
    system = mm.System(name="test_schedule_command")
    hd = ta.HysteresisDriver()
    tr = ta.TommosRunner()
    schedule_command = "tommos loop test_schedule_command"
    if sys.platform == "darwin":
        schedule_command += " --cpu-spmv-backend scipy --no-cpp-mkl"
    assert hd._schedule_commands(system, tr) == [schedule_command]
    assert hd._schedule_commands(system, None) == [schedule_command]


def test_call_passes_progress_information():
    """Pass the number and location of TOMMOS snapshots to the runner."""
    system = mm.System(name="test_progress")
    runner = Mock()

    ta.HysteresisDriver()._call(system, runner, verbose=2, n=7)

    argstr = "test_progress"
    if sys.platform == "darwin":
        argstr += " --cpu-spmv-backend scipy --no-cpp-mkl"

    runner.call.assert_called_once_with(
        argstr=argstr,
        verbose=2,
        total=7,
        glob_name="hyst_test_progress/*.vtu",
    )


def _write_tommos_output(state, output_file, magnetization):
    """Write a minimal VTU file matching TOMMOS output."""
    output_state = state.copy()
    output_state.points /= ta.scripts.DEFAULT_MESH_UNIT
    output_state.clear_data()
    output_state.point_data["m"] = np.broadcast_to(magnetization, (state.n_points, 3))
    output_state.point_data["U"] = np.zeros(state.n_points)
    output_state.cell_data["mat_id"] = state.cell_data["region_id"]
    output_state.save(output_file)


def test_read_data_updates_only_magnetization(state_zero, tmp_path, monkeypatch):
    """Read the last TOMMOS state while retaining user grid metadata."""
    state = state_zero.copy()
    state.point_data["m"][:, 2] = 1
    state.point_data["custom_point_data"] = np.arange(state.n_points)
    state.cell_data["region"] = np.arange(state.n_cells)
    system = mm.System(name="read_data", m=state)
    original_state = system.m
    original_region_id = state.cell_data["region_id"].copy()
    original_Ms = state.cell_data["Ms"].copy()
    original_region = state.cell_data["region"].copy()
    original_point_data = state.point_data["custom_point_data"].copy()

    output_directory = tmp_path / "hyst_read_data"
    output_directory.mkdir()
    _write_tommos_output(state, output_directory / "state_cfg00010_B+1.0000e+00T.vtu", (0, 1, 0))
    _write_tommos_output(state, output_directory / "state_cfg00002_B-1.0000e+00T.vtu", (1, 0, 0))
    monkeypatch.chdir(tmp_path)

    ta.HysteresisDriver()._read_data(system)

    assert system.m is original_state
    np.testing.assert_array_equal(system.m.point_data["m"], np.tile((0, 1, 0), (state.n_points, 1)))
    np.testing.assert_array_equal(system.m.point_data["custom_point_data"], original_point_data)
    np.testing.assert_array_equal(system.m.cell_data["region_id"], original_region_id)
    np.testing.assert_array_equal(system.m.cell_data["Ms"], original_Ms)
    np.testing.assert_array_equal(system.m.cell_data["region"], original_region)


def test_read_data_requires_output(state_zero, tmp_path, monkeypatch):
    """Missing TOMMOS snapshots produce a helpful error without changing the system."""
    state = state_zero.copy()
    state.point_data["m"][:, 2] = 1
    system = mm.System(name="missing_output", m=state)
    magnetization_before_read = state.point_data["m"].copy()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="No TOMMOS VTU output found"):
        ta.HysteresisDriver()._read_data(system)

    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_read)


@pytest.mark.parametrize("invalid_output", ["missing_m", "wrong_shape", "different_mesh"])
def test_read_data_rejects_invalid_output(state_zero, tmp_path, monkeypatch, invalid_output):
    """Invalid TOMMOS snapshots do not partially update the system."""
    state = state_zero.copy()
    state.point_data["m"][:, 2] = 1
    system = mm.System(name="invalid_output", m=state)
    magnetization_before_read = state.point_data["m"].copy()

    output_state = state.copy()
    output_state.points /= ta.scripts.DEFAULT_MESH_UNIT
    if invalid_output == "missing_m":
        output_state.point_data.remove("m")
        error_message = "does not contain point data 'm'"
    elif invalid_output == "wrong_shape":
        output_state.point_data["m"] = np.ones((state.n_points, 2))
        error_message = "output magnetization has shape"
    else:
        output_state.points[0, 0] += 1
        error_message = "does not match system.m"

    output_directory = tmp_path / "hyst_invalid_output"
    output_directory.mkdir()
    output_state.save(output_directory / "state_cfg00000_B+0.0000e+00T.vtu")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match=error_message):
        ta.HysteresisDriver()._read_data(system)

    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_read)


def test_check_system_missing_energies(state_zero):
    """Test that the system is not well defined if energies are not defined."""
    system = mm.System(name="test_invalid_magnetization")
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    system.m = state
    hd = ta.HysteresisDriver()
    with pytest.raises(RuntimeError, match="System's energy is not defined"):
        hd._check_system(system)


def test_check_system_energies_unsupported_parameters(state_zero):
    """Test that the system energies raise errors on unsupported parameters."""
    system = mm.System(name="test_unsupported_parameters")
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    system.m = state
    system.energy = mm.Demag(asymptotic_radius=3)
    hd = ta.HysteresisDriver()
    with pytest.raises(RuntimeError, match="Attribute `asymptotic_radius` is not supported."):
        hd._check_system(system)


def test_check_system_missing_magnetization():
    """A system without magnetization is invalid."""
    system = mm.System(name="test_invalid_magnetization")
    system.energy = mm.Exchange(A=1e-12)

    with pytest.raises(RuntimeError, match="Undefined magnetization state"):
        ta.HysteresisDriver()._check_system(system)


def test_check_system_requires_unstructured_grid():
    """TOMMOS does not accept a discretisedfield magnetization."""
    system = mm.examples.macrospin()

    with pytest.raises(TypeError, match=r"pyvista\.UnstructuredGrid"):
        ta.HysteresisDriver()._check_system(system)


@pytest.mark.parametrize(
    "association, name, error_message",
    [
        ("point_data", "m", "magnetization array 'm' is missing"),
        ("cell_data", "region_id", "material array 'region_id' is missing"),
        ("cell_data", "Ms", "material array 'Ms' is missing"),
    ],
)
def test_check_system_requires_grid_data(state_zero, association, name, error_message):
    """All arrays needed to create TOMMOS inputs must be present."""
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    getattr(state, association).remove(name)
    system = mm.System(name="test_missing_grid_data", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=state)

    with pytest.raises(RuntimeError, match=error_message):
        ta.HysteresisDriver()._check_system(system)


def test_check_system_magnetization_shape(state_zero):
    """Magnetization must contain one three-dimensional vector per point."""
    state = state_zero.copy()
    state.point_data["m"] = np.ones((state.n_points, 2))
    system = mm.System(name="test_invalid_shape", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=state)

    with pytest.raises(RuntimeError, match="must have shape"):
        ta.HysteresisDriver()._check_system(system)


def test_check_system_finite_magnetization(state_zero):
    """Magnetization values must be finite."""
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    state.point_data["m"][0, 0] = np.nan
    system = mm.System(name="test_non_finite_m", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=state)

    with pytest.raises(RuntimeError, match="must contain only finite values"):
        ta.HysteresisDriver()._check_system(system)


@pytest.mark.parametrize("value", [0, 2])
def test_check_system_normalized_magnetization(state_zero, value):
    """Magnetization must be normalized and is not modified during validation."""
    state = state_zero.copy()
    state.point_data["m"][:, 0] = value
    magnetization_before_check = state.point_data["m"].copy()
    system = mm.System(name="test_non_normalized_m", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=state)

    with pytest.raises(RuntimeError, match="must contain unit vectors"):
        ta.HysteresisDriver()._check_system(system)
    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_check)


def test_check_system_does_not_modify_valid_system(state_zero):
    """Validation accepts pointwise-normalized magnetization without changing it."""
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    magnetization_before_check = state.point_data["m"].copy()
    system = mm.System(name="test_valid_system", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=state)

    ta.HysteresisDriver()._check_system(system)

    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_check)


def test_x():
    hd = ta.HysteresisDriver()
    assert hd._x == "B_hysteresis"
