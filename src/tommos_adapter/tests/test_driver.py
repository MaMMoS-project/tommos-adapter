"""Test hysteresis driver."""

import os
from textwrap import dedent

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
    ],
)
def test_checkargs(Bmin, Bmax, n, hx, hy, hz, hstart, hfinal, hstep):
    """Test that field arguments are processed correctly."""
    kwargs = {
        "Hmin": (Bmin[0] / mm.consts.mu0, Bmin[1] / mm.consts.mu0, Bmin[2] / mm.consts.mu0),
        "Hmax": (Bmax[0] / mm.consts.mu0, Bmax[1] / mm.consts.mu0, Bmax[2] / mm.consts.mu0),
        "n": n,
    }
    hd = ta.HysteresisDriver()
    hd._checkargs(kwargs)
    assert kwargs == {
        "hx": hx,
        "hy": hy,
        "hz": hz,
        "hstart": hstart,
        "hfinal": hfinal,
        "hstep": hstep,
    }


def test_write_input_files_before_state():
    """Check that input writing fails if we do not define a field first."""
    system = mm.System(name="test_write_input_files_before_state")
    hd = ta.HysteresisDriver()
    with pytest.raises(AttributeError, match="no attribute 'points'"):
        hd._write_input_files(system)


def test_write_input_files_before_checking(state_zero, tmp_path):
    """Check that input writing fails if we do not process the field arguments first."""
    system = mm.System(name="test_write_input_files_before_checking")
    state = state_zero.copy()
    system.m = state
    hd = ta.HysteresisDriver()
    os.chdir(tmp_path)
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
    # TODO: Should we instead raise an error here?


def test_call():
    pass  # TODO: write


def test_schedule_command():
    pass  # TODO: write


def test_schedule_kwargs_setup():
    pass  # TODO: write


def test_drive_kwargs_setup():
    pass  # TODO: write


def test_check_system():
    pass  # TODO: write


def test_x():
    pass  # TODO: write
