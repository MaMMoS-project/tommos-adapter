"""Test hysteresis driver."""

import sys
from unittest.mock import Mock

import discretisedfield as df
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
        ((1, 0, 0), -1 / mm.consts.mu0, 2 / mm.consts.mu0, 6, 1.0, 0, 0, -1.0, 2.0, 0.6),
        ((-1, 1, 0), 1 / mm.consts.mu0, -3 / mm.consts.mu0, 5, -1 / np.sqrt(2), 1 / np.sqrt(2), 0, 1.0, -3.0, -1.0),
        ((0, 0, 3), -8 / mm.consts.mu0, 2 / mm.consts.mu0, 21, 0, 0, 1.0, -8.0, 2.0, 0.5),
        ((-1, 0, 0), 2 / mm.consts.mu0, 0, 3, -1.0, 0, 0, 2.0, 0, -1.0),
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
    system = mm.System(name="empty_system")
    assert hd._get_required_p2_arguments(system, kwargs) == {
        "hx": hx,
        "hy": hy,
        "hz": hz,
        "hstart": hstart,
        "hfinal": hfinal,
        "hstep": hstep,
    }


@pytest.mark.parametrize("arg_to_remove", ["h_direction", "h_start", "h_final", "n"])
def test_missing_required_parameters(arg_to_remove):
    """Test error when missing required arguments."""
    kwargs = {"h_direction": (0, 0, 1), "h_start": 1e6, "h_final": -2e6, "n": 13}
    del kwargs[arg_to_remove]
    with pytest.raises(ValueError, match="`h_direction`, `h_start`, `h_final`, and `n` must all be provided"):
        ta.HysteresisDriver()._checkargs(kwargs)


@pytest.mark.parametrize(
    "h_direction, expected",
    [
        ((0, 0, 1), np.array([0.0, 0.0, 1.0])),
        ([0, 0, 0.5], np.array([0.0, 0.0, 1.0])),
        (np.array([0, 3, 3]), np.array([0.0, 1 / np.sqrt(2), 1 / np.sqrt(2)])),
    ],
)
def test_check_direction(h_direction, expected):
    """Test correct conversion of `_check_direction` function."""
    np.testing.assert_allclose(ta.tommos_driver._check_direction(h_direction), expected)


@pytest.mark.parametrize(
    "h_direction, expected_error, error_message",
    [
        ("string", RuntimeError, "Conversion of direction `h_direction` to NumPy array failed"),
        ((-1, 0), RuntimeError, "must be a one-dimensional array containing three numerical values"),
        ((np.nan, 0, 0), ValueError, "must contain only finite values"),
        ((0, 0, -np.inf), ValueError, "must contain only finite values"),
        (
            (
                0,
                0,
                0,
            ),
            ValueError,
            "cannot have zero norm",
        ),
    ],
)
def test_check_direction_raises(h_direction, expected_error, error_message):
    """Test that wrong `h-direction` argument raise errors."""
    with pytest.raises(expected_error, match=error_message):
        ta.tommos_driver._check_direction(h_direction)


@pytest.mark.parametrize(
    "h, expected",
    [
        (1, 1.0),
        (np.int64(1), 1.0),
        (8.0, 8.0),
        (np.float64(8.0), 8.0),
    ],
)
def test_check_field_magnitude(h, expected):
    """Test correct conversion of `_check_field_magnitude` function."""
    np.testing.assert_equal(ta.tommos_driver._check_field_magnitude(h, "h_start"), expected)


@pytest.mark.parametrize(
    "h, name",
    [("1", "h_start"), (np.nan, "h_final"), (np.inf, "some_other_name"), ((8.0,), "tuple"), (1 + 2j, "complex")],
)
def test_check_field_magnitude_raises(h, name):
    """Test that wrong value in `_check_field_magnitude` raise errors."""
    with pytest.raises(ValueError, match=f"Field magnitude `{name}` must be a single finite real number"):
        ta.tommos_driver._check_field_magnitude(h, name)


@pytest.mark.parametrize("n", [np.int64(3), 104])
def test_number_field_steps(n):
    """Test correct conversion of `_check_field_magnitude` function."""
    assert ta.tommos_driver._check_number_field_steps(n) == int(n)


@pytest.mark.parametrize("n", [3.0, -1, np.inf, np.nan, [3]])
def test_number_field_steps_raises(n):
    """Test that wrong `n` argument raise errors."""
    with pytest.raises(ValueError, match=r"`n` .* must be an integer bigger than 2"):
        ta.tommos_driver._check_number_field_steps(n)


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
    kwargs = {"h_direction": (1, 0, 0), "h_start": 1 / mm.consts.mu0, "h_final": -2 / mm.consts.mu0, "n": 11}
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
    assert "hx = 1" in p2_text
    assert "hstart = 1" in p2_text
    assert "hfinal = -2" in p2_text
    assert "hstep = -0.29999999999999999" in p2_text
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

    tr = ta.TommosRunner(command=["pixi", "run", "-e", "cuda", "tommos"])
    assert hd._schedule_commands(system, tr) == [f"pixi run -e cuda {schedule_command}"]


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


def test_read_data_updates_system(state_zero, tmp_path, monkeypatch, write_mammos_table):
    """Read the last TOMMOS state and table while retaining grid metadata."""
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
    write_mammos_table(output_directory / "mammos_hysteresis.csv")
    monkeypatch.chdir(tmp_path)

    ta.HysteresisDriver()._read_data(system)

    assert system.m is original_state
    np.testing.assert_array_equal(system.m.point_data["m"], np.tile((0, 1, 0), (state.n_points, 1)))
    np.testing.assert_array_equal(system.m.point_data["custom_point_data"], original_point_data)
    np.testing.assert_array_equal(system.m.cell_data["region_id"], original_region_id)
    np.testing.assert_array_equal(system.m.cell_data["Ms"], original_Ms)
    np.testing.assert_array_equal(system.m.cell_data["region"], original_region)
    assert system.table.x == "B_hysteresis"
    np.testing.assert_allclose(system.table.data["B_hysteresis"], (-1, 0, 1))


def test_read_data_requires_table(state_zero, tmp_path, monkeypatch):
    """Missing tabular output does not partially update the system."""
    state = state_zero.copy()
    state.point_data["m"][:, 2] = 1
    system = mm.System(name="missing_table", m=state)
    magnetization_before_read = state.point_data["m"].copy()

    output_directory = tmp_path / "hyst_missing_table"
    output_directory.mkdir()
    _write_tommos_output(state, output_directory / "state_cfg00000_B+0.0000e+00T.vtu", (0, 1, 0))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="Cannot read TOMMOS table"):
        ta.HysteresisDriver()._read_data(system)

    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_read)


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
    system = mm.System(name="test_missing_energies")
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    system.m = state
    hd = ta.HysteresisDriver()
    with pytest.raises(RuntimeError, match="System's energy is not defined"):
        hd._check_system(system)


def test_check_system_unsupported_energies(state_zero):
    """Test that `HysteresisDriver` does not support Zeeman energy terms."""
    system = mm.System(name="test_unsupported_energies")
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    system.m = state
    system.energy = mm.Demag() + mm.Zeeman(H=(0, 0, 1 / mm.consts.mu0))
    hd = ta.HysteresisDriver()
    with pytest.raises(ValueError, match="Energy term Zeeman.* is not supported by this driver"):
        hd._check_system(system)


def test_check_system_missing_magnetization():
    """A system without magnetization is invalid."""
    system = mm.System(name="test_missing_magnetization")
    system.energy = mm.Exchange(A=1e-12) + mm.Demag()

    with pytest.raises(RuntimeError, match="Undefined magnetization state"):
        ta.HysteresisDriver()._check_system(system)


def test_check_system_requires_unstructured_grid():
    """`HysteresisDriver` does not accept a discretisedfield magnetization."""
    system = mm.System(name="test_discretisedfield")
    system.energy = mm.Demag()
    region = df.Region(p1=(0, 0, 0), p2=(1e-9, 1e-9, 1e-9))
    mesh = df.Mesh(region=region, n=(1, 1, 1))
    system.m = df.Field(mesh, nvdim=3, value=(0, 1, 1), norm=1e6)

    with pytest.raises(TypeError, match="requires system.m to be a pyvista.UnstructuredGrid"):
        ta.HysteresisDriver()._check_system(system)


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
    """Test `x` label."""
    assert ta.HysteresisDriver()._x == "B_hysteresis"
