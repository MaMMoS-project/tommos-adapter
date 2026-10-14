"""Test hysteresis driver."""

from textwrap import dedent
from unittest.mock import Mock

import micromagneticmodel as mm
import numpy as np
import pytest

import tommos_adapter as ta


def test_allowed_attributes():
    assert ta.HysteresisDriver()._allowed_attributes == []
    with pytest.raises(AttributeError, match="Invalid attribute key"):
        ta.HysteresisDriver(unexpected_argument=1)


@pytest.mark.parametrize(
    "Bmin,Bmax,n,hx,hy,hz,hstart,hfinal,hstep",
    [
        (
            (-1, 0, 0),
            (2, 0, 0),
            6,
            np.float64(1),
            np.float64(0),
            np.float64(0),
            np.float64(-1),
            np.float64(2),
            np.float64(0.6),
        ),
        (
            (1, 0, 0),
            (-3, 0, 0),
            5,
            np.float64(-1),
            np.float64(0),
            np.float64(0),
            np.float64(-1),
            np.float64(3),
            np.float64(1),
        ),
        (
            (0, 0, -8),
            (0, 0, 2),
            21,
            np.float64(0),
            np.float64(0),
            np.float64(1),
            np.float64(-8),
            np.float64(2),
            np.float64(0.5),
        ),
        (
            (2, 0, 0),
            (0, 0, 0),
            3,
            np.float64(-1),
            np.float64(0),
            np.float64(0),
            np.float64(-2),
            np.float64(0),
            np.float64(1),
        ),
    ],
)
@pytest.mark.parametrize("processor", ["_checkargs", "schedule_kwargs_setup", "drive_kwargs_setup"])
def test_process_hysteresis_arguments(processor, Bmin, Bmax, n, hx, hy, hz, hstart, hfinal, hstep):
    """Test that field arguments are processed correctly."""
    kwargs = {
        "Hmin": (Bmin[0] / mm.consts.mu0, Bmin[1] / mm.consts.mu0, Bmin[2] / mm.consts.mu0),
        "Hmax": (Bmax[0] / mm.consts.mu0, Bmax[1] / mm.consts.mu0, Bmax[2] / mm.consts.mu0),
        "n": n,
    }
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


@pytest.mark.parametrize(
    "kwargs, error_message",
    [
        ({}, "Hmin, Hmax, and n must all be provided"),
        ({"Hmin": (-1, 0, 0), "Hmax": (1, 0, 0)}, "Hmin, Hmax, and n must all be provided"),
        ({"Hmin": (-1, 0), "Hmax": (1, 0, 0), "n": 3}, "Hmin must contain three numeric values"),
        ({"Hmin": (-1, 0, 0), "Hmax": (1, np.nan, 0), "n": 3}, "Hmax must contain only finite values"),
        ({"Hmin": (-1, 0, 0), "Hmax": (1, 0, 0), "n": 3.0}, "n must be an integer"),
        ({"Hmin": (-1, 0, 0), "Hmax": (1, 0, 0), "n": 1}, "n must be at least 2"),
        ({"Hmin": (1, 0, 0), "Hmax": (1, 0, 0), "n": 3}, "Hmin and Hmax must differ"),
        (
            {"Hmin": (1, 0, 0), "Hmax": (0, 1, 0), "n": 3},
            "Hmin and Hmax must be collinear with the origin",
        ),
    ],
)
def test_invalid_hysteresis_arguments(kwargs, error_message):
    """Invalid hysteresis arguments must produce helpful errors."""
    with pytest.raises(ValueError, match=error_message):
        ta.HysteresisDriver()._checkargs(kwargs)


def test_write_input_files_before_checking(state_zero, tmp_path, monkeypatch):
    """Check inputs before processing the field arguments."""
    system = mm.System(name="test_write_input_files_before_checking")
    state = state_zero.copy()
    system.m = state
    hd = ta.HysteresisDriver()
    monkeypatch.chdir(tmp_path)
    hd._write_input_files(system)
    assert (tmp_path / "test_write_input_files_before_checking.npz").is_file()
    assert (tmp_path / "test_write_input_files_before_checking.krn").is_file()
    assert (tmp_path / "test_write_input_files_before_checking.p2").is_file()
    assert (tmp_path / "test_write_input_files_before_checking.krn").read_text() == dedent(
        """\
        # theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
        0.0 0.0 0.0 0.0 1.0 0.0"""
    )
    assert (tmp_path / "test_write_input_files_before_checking.p2").read_text() == dedent(
        """\
        [mesh]

        [initial state]
        mx = 0.0
        my = 0.0
        mz = 0.0

        [field]

        [minimizer]

        [poisson]

        """
    )
    # TODO: Consider rejecting input generation before hysteresis arguments are processed.


def test_write_input_files_after_checking(state_zero, tmp_path, monkeypatch):
    """Check inputs after processing the field arguments."""
    system = mm.System(name="test_write_input_files_after_checking")
    state = state_zero.copy()
    state.point_data["m"][:, 2] = 1
    system.m = state
    hd = ta.HysteresisDriver()
    kwargs = {
        "Hmin": (1 / mm.consts.mu0, 0, 0),
        "Hmax": (-2 / mm.consts.mu0, 0, 0),
        "n": 11,
    }
    monkeypatch.chdir(tmp_path)
    hd._checkargs(kwargs)
    hd._write_input_files(system, **kwargs)
    assert (tmp_path / "test_write_input_files_after_checking.npz").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.krn").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.p2").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.krn").read_text() == dedent(
        """\
        # theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
        0.0 0.0 0.0 0.0 1.0 0.0"""
    )
    assert (tmp_path / "test_write_input_files_after_checking.p2").read_text() == dedent(
        """\
        [mesh]

        [initial state]
        mx = 0.0
        my = 0.0
        mz = 1.0

        [field]
        hx = -1.0
        hy = 0.0
        hz = 0.0
        hstart = -1.0
        hfinal = 2.0
        hstep = 0.3

        [minimizer]

        [poisson]

        """
    )


def test_schedule_command():
    """Test driver schedule command."""
    system = mm.System(name="test_schedule_command")
    hd = ta.HysteresisDriver()
    tr = ta.TommosRunner()
    assert hd._schedule_commands(system, tr) == ["tommos loop test_schedule_command"]


def test_call_passes_progress_information():
    """Pass the number and location of TOMMOS snapshots to the runner."""
    system = mm.System(name="test_progress")
    runner = Mock()

    ta.HysteresisDriver()._call(system, runner, verbose=2, n=7)

    runner.call.assert_called_once_with(
        argstr="test_progress",
        verbose=2,
        total=7,
        glob_name="hyst_test_progress/*.vtu",
    )


def test_check_system_missing_energies(state_zero):
    """Test that the system is not well defined if energies are not defined."""
    system = mm.System(name="test_invalid_magnetization")
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    system.m = state
    hd = ta.HysteresisDriver()
    with pytest.raises(RuntimeError, match="System's energy is not defined"):
        hd._check_system(system)


def test_check_system_invalid_magnetization(state_zero):
    """Test that the system is not well defined if the magnetization is missing or with zero norm."""
    system = mm.System(name="test_invalid_magnetization")
    system.energy = mm.Exchange(A=1e-12)
    hd = ta.HysteresisDriver()

    # magnetization state not given
    with pytest.raises(RuntimeError, match="Undefined magnetization state"):
        hd._check_system(system)

    # given magnetization state has zero norm
    state = state_zero.copy()
    system.m = state
    with pytest.raises(RuntimeError, match="Invalid initial magnetization"):
        hd._check_system(system)


def test_x():
    hd = ta.HysteresisDriver()
    assert hd._x == "B_hysteresis"
