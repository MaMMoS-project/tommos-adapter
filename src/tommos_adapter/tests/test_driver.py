"""Test hysteresis driver."""

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


def test_write_input_files_before_checking(state_zero):
    """Check that input writing fails if we do not process the field arguments first."""
    system = mm.System(name="test_write_input_files_before_checking")
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    system.m = state
    hd = ta.HysteresisDriver()
    # TODO: set temporary working directory
    hd._write_input_files(system)
    # TODO: check created npz file exists
    # TODO: check created krn
    # TODO: check create p2
    # TODO: Should we raise an error here?


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
