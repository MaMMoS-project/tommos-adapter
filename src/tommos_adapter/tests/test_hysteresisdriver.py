"""Test hysteresis driver."""

import sys
import textwrap
from unittest.mock import Mock

import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import pytest
import pyvista as pv

import tommos_adapter as ta


def test_allowed_attributes():
    assert ta.HysteresisDriver()._allowed_attributes == [
        "mstep",
        "method",
        "max_iter",
        "tol_fun",
        "eps_a",
        "num_devices",
    ]
    driver = ta.HysteresisDriver(mstep=1, method="tr", max_iter=1000, tol_fun=1e-8, eps_a="auto", num_devices=2)
    assert dict(driver) == {
        "mstep": 1,
        "method": "tr",
        "max_iter": 1000,
        "tol_fun": 1e-8,
        "eps_a": "auto",
        "num_devices": 2,
    }
    with pytest.raises(AttributeError, match="Invalid attribute key"):
        ta.HysteresisDriver(unexpected_argument=1)


def test_num_devices():
    system = mm.System(name="num_devices")
    with pytest.raises(ValueError, match=r"`num\_devices` should be a non-negative integer"):
        ta.HysteresisDriver().get_argstr(system, num_devices="mistake")
    assert ta.HysteresisDriver().get_argstr(system, num_devices=11) == "num_devices --add-shell --num-devices 11"


@pytest.mark.parametrize("name, full_loop", [["name_1", None], ["name_2", True], ["name_3", False]])
def test_get_argstr(name, full_loop):
    """Test string of arguments passed to the runner or to the scheduler in `HysteresisDriver`."""
    system = mm.System(name=name)
    hd = ta.HysteresisDriver()
    expected_argstr = f"{name} --add-shell"
    if full_loop:
        expected_argstr += " --loop"
    assert hd.get_argstr(system, full_loop=full_loop) == expected_argstr


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


def test_invalid_temperature(m_cube_20_zero):
    """Raise error on temperature different than zero."""
    system = mm.System(name="wrong_temperature")
    system.m = m_cube_20_zero
    system.m.point_data["m"][:, 2] = 1
    system.energy = mm.Demag()
    system.T = 300.0
    hd = ta.HysteresisDriver()
    with pytest.raises(RuntimeError, match="`tommos` supports only T=0 K. Given temperature: 300.0 K"):
        hd._check_system(system)


def test_write_input_files_after_checking(m_cube_20_zero, tmp_path, monkeypatch):
    """Check inputs after processing the field arguments."""
    system = mm.System(name="test_write_input_files_after_checking", m=m_cube_20_zero)
    input_points = system.m.points.copy()
    system.m.point_data["m"][:, 2] = 1
    system.energy = mm.Demag()
    hd = ta.HysteresisDriver(mstep=1, method="tr", max_iter=1000, tol_fun=1e-8, eps_a="auto")
    kwargs = {"h_direction": (1, 0, 0), "h_start": 3 / mm.consts.mu0, "h_final": -2 / mm.consts.mu0, "n": 6}
    monkeypatch.chdir(tmp_path)
    hd._checkargs(kwargs)
    hd._write_input_files(system, **kwargs)
    assert (tmp_path / "test_write_input_files_after_checking.npz").is_file()
    assert (tmp_path / "m0.vtu").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.krn").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.p2").is_file()
    saved_initial = pv.read(tmp_path / "m0.vtu")
    np.testing.assert_array_equal(saved_initial.points, input_points)
    np.testing.assert_array_equal(system.m.points, input_points)
    krn_text = (tmp_path / "test_write_input_files_after_checking.krn").read_text()
    assert krn_text == "# theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)\n0 0 0 0 1 0\n"
    p2_text = (tmp_path / "test_write_input_files_after_checking.p2").read_text()
    assert p2_text == textwrap.dedent(
        """\
        [mesh]
        size = 1e-09

        [initial state]
        mx = 0
        my = 0
        mz = 1

        [field]
        hstart = 3
        hfinal = -2
        hstep = -1
        hx = 1
        hy = 0
        hz = 0
        mstep = 1

        [minimizer]
        method = tr
        max_iter = 1000
        tol_fun = 1e-08
        eps_a = auto
        """
    )


def test_schedule_command():
    """Test driver schedule command."""
    system = mm.System(name="test_schedule_command")
    hd = ta.HysteresisDriver()
    tr = ta.TommosRunner()
    schedule_command = "tommos loop test_schedule_command --add-shell"
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

    argstr = "test_progress --add-shell"

    runner.call.assert_called_once_with(
        argstr=argstr,
        verbose=2,
        total=7,
        glob_name="hyst_test_progress/*.vtu",
    )


def _write_tommos_output(state, output_file, magnetization):
    """Write a minimal VTU file matching TOMMOS output."""
    output_state = state.copy()
    output_state.clear_data()
    output_state.point_data["m"] = np.broadcast_to(magnetization, (state.n_points, 3))
    output_state.point_data["U"] = np.zeros(state.n_points)
    output_state.save(output_file)


def test_read_data_updates_system(m_cube_20_zero, tmp_path, monkeypatch, write_mammos_table):
    """Read the last TOMMOS state and table while retaining grid metadata."""
    system = mm.System(name="read_data", m=m_cube_20_zero)
    system.m.point_data["m"][:, 2] = 1
    system.m.point_data["custom_point_data"] = np.arange(system.m.n_points)
    system.m.cell_data["region"] = np.arange(system.m.n_cells)
    original_state = system.m
    original_mat_id = system.m.cell_data["mat_id"].copy()
    original_Ms = system.m.cell_data["Ms"].copy()
    original_region = system.m.cell_data["region"].copy()
    original_point_data = system.m.point_data["custom_point_data"].copy()

    output_directory = tmp_path / "hyst_read_data"
    output_directory.mkdir()
    _write_tommos_output(system.m, output_directory / "state_cfg00010_B+1.0000e+00T.vtu", (0, 1, 0))
    _write_tommos_output(system.m, output_directory / "state_cfg00002_B-1.0000e+00T.vtu", (1, 0, 0))
    write_mammos_table(output_directory / "mammos_hysteresis.csv")
    monkeypatch.chdir(tmp_path)

    ta.HysteresisDriver()._read_data(system)

    assert system.m is original_state
    np.testing.assert_array_equal(system.m.point_data["m"], np.tile((0, 1, 0), (system.m.n_points, 1)))
    np.testing.assert_array_equal(system.m.point_data["custom_point_data"], original_point_data)
    np.testing.assert_array_equal(system.m.cell_data["mat_id"], original_mat_id)
    np.testing.assert_array_equal(system.m.cell_data["Ms"], original_Ms)
    np.testing.assert_array_equal(system.m.cell_data["region"], original_region)
    assert system.table.x == "B_hysteresis"
    np.testing.assert_allclose(system.table.data["B_hysteresis"], (-1, 0, 1))


def test_read_data_requires_table(m_cube_20_zero, tmp_path, monkeypatch):
    """Missing tabular output does not partially update the system."""
    system = mm.System(name="missing_table", m=m_cube_20_zero)
    system.m.point_data["m"][:, 2] = 1
    magnetization_before_read = system.m.point_data["m"].copy()

    output_directory = tmp_path / "hyst_missing_table"
    output_directory.mkdir()
    _write_tommos_output(system.m, output_directory / "state_cfg00000_B+0.0000e+00T.vtu", (0, 1, 0))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="Cannot read tommos table"):
        ta.HysteresisDriver()._read_data(system)

    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_read)


def test_read_data_drops_airbox(m_cube_20_zero, tmp_path, monkeypatch, write_mammos_table):
    """Map magnetization back from output with airbox nodes and reordered points."""
    system = mm.System(name="airbox", m=m_cube_20_zero)
    system.m.point_data["m"][:, 2] = 1
    expected = np.random.default_rng(0).normal(size=(system.m.n_points, 3))
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)

    airbox_points = np.array([[-50, -50, -50], [50, -50, -50], [-50, 50, -50], [-50, -50, 50]])
    points = np.vstack([system.m.points, airbox_points]).astype(np.float32)
    magnetization = np.vstack([expected, np.zeros((len(airbox_points), 3))]).astype(np.float32)
    connectivity = np.vstack([system.m.cell_connectivity.reshape(-1, 4), np.arange(4) + system.m.n_points])
    order = np.random.default_rng(1).permutation(len(points))
    new_index = np.argsort(order)
    output_state = pv.UnstructuredGrid({pv.CellType.TETRA: new_index[connectivity]}, points[order])
    output_state.point_data["m"] = magnetization[order]

    output_directory = tmp_path / "hyst_airbox"
    output_directory.mkdir()
    output_state.save(output_directory / "state_cfg00000_B+0.0000e+00T.vtu")
    write_mammos_table(output_directory / "mammos_hysteresis.csv")
    monkeypatch.chdir(tmp_path)

    ta.HysteresisDriver()._read_data(system)

    assert system.m.n_points == system.m.n_points
    np.testing.assert_allclose(system.m.point_data["m"], expected, atol=1e-7)


def test_read_data_requires_output(m_cube_20_zero, tmp_path, monkeypatch):
    """Missing TOMMOS snapshots produce a helpful error without changing the system."""
    system = mm.System(name="missing_output", m=m_cube_20_zero)
    system.m.point_data["m"][:, 2] = 1
    magnetization_before_read = system.m.point_data["m"].copy()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="No tommos VTU output found"):
        ta.HysteresisDriver()._read_data(system)

    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_read)


@pytest.mark.parametrize("invalid_output", ["missing_m", "wrong_shape", "different_mesh"])
def test_read_data_rejects_invalid_output(m_cube_20_zero, tmp_path, monkeypatch, invalid_output):
    """Invalid TOMMOS snapshots do not partially update the system."""
    system = mm.System(name="invalid_output", m=m_cube_20_zero)
    system.m.point_data["m"][:, 2] = 1
    magnetization_before_read = system.m.point_data["m"].copy()

    output_state = system.m.copy()
    if invalid_output == "missing_m":
        output_state.point_data.remove("m")
        error_message = "does not contain point data 'm'"
    elif invalid_output == "wrong_shape":
        output_state.point_data["m"] = np.ones((system.m.n_points, 2))
        error_message = "has magnetization shape"
    else:
        output_state.points[0, 0] += 1
        error_message = "does not match the reference mesh"

    output_directory = tmp_path / "hyst_invalid_output"
    output_directory.mkdir()
    output_state.save(output_directory / "state_cfg00000_B+0.0000e+00T.vtu")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match=error_message):
        ta.HysteresisDriver()._read_data(system)

    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_read)


def test_check_system_missing_energies(m_cube_20_zero):
    """Test that the system is not well defined if energies are not defined."""
    system = mm.System(name="test_missing_energies", m=m_cube_20_zero)
    system.m.point_data["m"][:, 0] = 1
    hd = ta.HysteresisDriver()
    with pytest.raises(RuntimeError, match="System's energy is not defined"):
        hd._check_system(system)


def test_check_system_unsupported_energies(m_cube_20_zero):
    """Test that `HysteresisDriver` does not support Zeeman energy terms."""
    system = mm.System(name="test_unsupported_energies", m=m_cube_20_zero)
    system.m.point_data["m"][:, 0] = 1
    system.energy = mm.Demag() + mm.Zeeman(H=(0, 0, 1 / mm.consts.mu0))
    hd = ta.HysteresisDriver()
    with pytest.raises(ValueError, match="Energy term Zeeman.* is not supported by this driver"):
        hd._check_system(system)


def test_check_system_multiple_energies_of_same_type(m_cube_20_zero):
    """Test that `HysteresisDriver` does not support multiple energy terms of the same type."""
    system = mm.System(name="test_unsupported_energies", m=m_cube_20_zero)
    system.m.point_data["m"][:, 0] = 1
    system.energy = mm.Demag() + mm.Exchange(A=1e-12) + mm.Exchange(name="different_exchange", A=2e-12)
    hd = ta.HysteresisDriver()
    with pytest.raises(ValueError, match=f"Multiple energy terms of type {type(mm.Exchange())} are not supported"):
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


def test_check_system_energies_unsupported_parameters(m_cube_20_zero):
    """Test that the system energies raise errors on unsupported parameters."""
    system = mm.System(name="test_unsupported_parameters", m=m_cube_20_zero)
    system.m.point_data["m"][:, 0] = 1
    system.energy = mm.Demag(asymptotic_radius=3)
    hd = ta.HysteresisDriver()
    with pytest.raises(RuntimeError, match="Attribute `asymptotic_radius` is not supported."):
        hd._check_system(system)


@pytest.mark.parametrize(
    "association, name, error_message",
    [
        ("point_data", "m", "magnetization array 'm' is missing"),
        ("cell_data", "mat_id", "material array 'mat_id' is missing"),
        ("cell_data", "Ms", "material array 'Ms' is missing"),
        ("field_data", "mesh_unit_length", "'mesh_unit_length' is missing"),
    ],
)
def test_check_system_requires_grid_data(m_cube_20_zero, association, name, error_message):
    """All arrays needed to create TOMMOS inputs must be present."""
    system = mm.System(name="test_missing_grid_data", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=m_cube_20_zero)
    system.m.point_data["m"][:, 0] = 1
    getattr(system.m, association).remove(name)

    with pytest.raises(RuntimeError, match=error_message):
        ta.HysteresisDriver()._check_system(system)


@pytest.mark.parametrize(
    "field_data, error, error_message",
    [
        (None, RuntimeError, "Field data 'mat_ids' and 'mat_names' are necessary"),
        ({"mat_ids": [1, 2]}, RuntimeError, "Field data 'mat_ids' and 'mat_names' are necessary"),
        ({"mat_names": ["grain_1", "grain_2"]}, RuntimeError, "Field data 'mat_ids' and 'mat_names' are necessary"),
    ],
)
def test_check_multigrain_field_data(m_cube_20_zero, field_data, error, error_message):
    """Check field data consistency for multirgain materials."""
    system = mm.System(name="test_check_multigrain_field_data", energy=mm.Demag(), m=m_cube_20_zero)
    system.m.cell_data["mat_id"][-1] = 2
    del system.m.field_data["mat_ids"]
    del system.m.field_data["mat_names"]
    if field_data is not None:
        for key, value in field_data.items():
            system.m.field_data[key] = value
    with pytest.raises(error, match=error_message):
        ta.HysteresisDriver()._check_system(system)


def test_check_system_magnetization_shape(m_cube_20_zero):
    """Magnetization must contain one three-dimensional vector per point."""
    system = mm.System(name="test_invalid_shape", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=m_cube_20_zero)
    system.m.point_data["m"] = np.ones((system.m.n_points, 2))

    with pytest.raises(RuntimeError, match="must have shape"):
        ta.HysteresisDriver()._check_system(system)


def test_check_system_finite_magnetization(m_cube_20_zero):
    """Magnetization values must be finite."""
    system = mm.System(name="test_non_finite_m", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=m_cube_20_zero)
    system.m.point_data["m"][:, 0] = 1
    system.m.point_data["m"][0, 0] = np.nan

    with pytest.raises(RuntimeError, match="must contain only finite values"):
        ta.HysteresisDriver()._check_system(system)


@pytest.mark.parametrize("value", [0, 2])
def test_check_system_normalized_magnetization(m_cube_20_zero, value):
    """Magnetization must be normalized and is not modified during validation."""
    system = mm.System(name="test_non_normalized_m", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=m_cube_20_zero)
    system.m.point_data["m"][:, 0] = value
    magnetization_before_check = system.m.point_data["m"].copy()

    with pytest.raises(RuntimeError, match="must contain unit vectors"):
        ta.HysteresisDriver()._check_system(system)
    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_check)


def test_check_system_does_not_modify_valid_system(m_cube_20_zero):
    """Validation accepts pointwise-normalized magnetization without changing it."""
    system = mm.System(name="test_valid_system", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=m_cube_20_zero)
    system.m.point_data["m"][:, 0] = 1
    magnetization_before_check = system.m.point_data["m"].copy()

    ta.HysteresisDriver()._check_system(system)

    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_check)


def test_full_loop(m_tiny, tmp_path):
    system = mm.System("test_full_loop", m=m_tiny)
    system.energy = mm.Exchange(A=1e-12) + mm.UniaxialAnisotropy(K=4e5, u=(0, 0, 1)) + mm.Demag()
    h_start_T = 3
    h_final_T = -2
    n = 7
    ta.HysteresisDriver().drive(
        system,
        dirname=tmp_path,
        h_direction=(1, 0, 0),
        h_start=h_start_T / mm.consts.mu0,
        h_final=h_final_T / mm.consts.mu0,
        n=n,
        full_loop=True,
    )
    expected_B_hysteresis = np.concatenate(
        (
            np.linspace(h_start_T, h_final_T, num=n - 1, endpoint=False),
            np.linspace(h_final_T, h_start_T, num=n, endpoint=True),
        )
    )
    np.testing.assert_allclose(system.table.data.B_hysteresis, expected_B_hysteresis)


def test_x():
    """Test `x` label."""
    assert ta.HysteresisDriver()._x == "B_hysteresis"
