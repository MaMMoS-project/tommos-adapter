"""Test configurations."""

import shlex
import subprocess
from pathlib import Path
from textwrap import dedent

import discretisedfield as df
import micromagneticmodel as mm
import pytest


@pytest.fixture
def system():
    system = mm.System(name="test")
    system.energy = mm.Zeeman(H=[0, 0, 1])
    system.m = df.Field(
        mesh=df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), n=(2, 2, 2)),
        nvdim=3,
        value=[1, 0, 0],
        norm=1e5,
    )
    return system


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def mesh_path(data_dir):
    cmd = "tommos mesh --geom box --extent 20,20,20 --h 2 --out-name test"
    subprocess.run(shlex.split(cmd), cwd=data_dir)
    return data_dir / "test.npz"


@pytest.fixture(scope="session")
def krn_path(data_dir):
    krn_path = data_dir / "test.krn"
    Path(krn_path).write_text(
        dedent(
            """\
            # theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
            0.0 0.0 930000.0 0.0 0.51019464687562 1.4e-12
            """
        )
    )
    return krn_path


@pytest.fixture(scope="session")
def p2_path(data_dir):
    p2_path = data_dir / "test.p2"
    Path(p2_path).write_text(
        dedent(
            """\
            [mesh]
            size = 1e-9

            [initial state]
            mx = 0.0
            my = 0.0
            mz = 1.0

            [field]
            hstart = 4.58128078817734
            hfinal = -4.58128078817734
            hstep = -0.229064039408867
            hx = 0.0017453283658983088
            hy = 0.0
            hz = 0.9999984769132877
            """
        )
    )
    return p2_path
