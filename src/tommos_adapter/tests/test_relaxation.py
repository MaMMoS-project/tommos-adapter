from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

import tommos_adapter as ta


@pytest.fixture(scope="module")
def relaxation_krn_path(data_dir):
    krn_path = data_dir / "test.krn"
    Path(krn_path).write_text(
        dedent(
            f"""\
            # theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
            {np.pi / 2} 0.0 930000.0 0.0 0.51019464687562 1.4e-12
            """
        )
    )
    return krn_path


@pytest.fixture(scope="module")
def relaxation_p2_path(data_dir):
    p2_path = data_dir / "test.p2"
    Path(p2_path).write_text(
        dedent(
            """\
            [mesh]
            size = 1e-9

            [initial state]
            mx = 0.707
            my = 0.0
            mz = 0.707

            [field]
            hstart = 0
            hfinal = 0
            """
        )
    )
    return p2_path


def test_relaxation(system, mesh_path, relaxation_krn_path, relaxation_p2_path, tmp_path):
    """Test relaxation.

    In absence of an external magnetic field, we expect the magnetization to relax to the anisotropy direction.
    In this case, the anisotropy unit vector is `(1, 0, 0)`, so we test that the first component of the magnetization
    is close to `1`.
    """
    hd = ta.HysteresisDriver(mesh_path=mesh_path, krn_path=relaxation_krn_path, p2_path=relaxation_p2_path)
    hd.drive(system, dirname=tmp_path)
    mag_array = system.m.get_array("m")
    avg_magnetization = sum(mag_array) / mag_array.shape[0]
    avg_magnetization /= avg_magnetization.dot(avg_magnetization)
    np.testing.assert_allclose(avg_magnetization[0], 1)
