"""Collection of scripts to translate a system into `tommos` input files."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import micromagneticmodel as mm
import numpy as np

if TYPE_CHECKING:
    import micromagneticmodel


def write_input_files(driver, system, **kwargs):
    """Write input files.

    - Mesh in `npz` format.
    - Intrinsic properties in `krn` format.
    - Simulation and hysteresis parameters in `p2` format.
    """
    save_mesh_npz(system)
    write_krn(system)
    write_p2(system)


def save_mesh_npz(system: micromagneticmodel.System):
    """Save mesh in `npz` format.

    TODO: explain conversion??

    Args:
        system: TODO: write docstrings
    """
    knt = system.m.points
    ijk = np.c_[
        system.m.cell_connectivity.reshape(-1, 4),
        system.m.get_array("id"),
    ]
    np.savez(f"{system.name}.npz", knt=knt, ijk=ijk)


def write_krn(system):
    """Save intrinsic properties in `krn` format."""
    properties = {"Js": 0, "A": 0, "K1": 0, "theta": 0, "phi": 0}
    if "Js" in system.m.cell_data:
        properties["Js"] = np.unique(system.m["Js"]).item()  # only works for singlegrain
        # TODO: generalize for multigrain
    for energy in system.energy:
        if isinstance(energy, mm.Exchange):
            properties["A"] = energy.A
        if isinstance(energy, mm.UniaxialAnisotropy):
            properties["K"] = energy.K
            u_x, u_y, u_z = energy.u  # TODO: assumes normal. Is this always true?
            properties["theta"] = np.arccos(u_z)
            properties["phi"] = np.sign(u_y) * np.arccos(u_x) / np.sqrt(u_x * u_x + u_y * u_y)
    with open(f"{system.name}.krn", "w") as f:
        f.write(
            dedent(
                f"""\
                # theta (rad) phi (rad) K1 (J/m3) not used Js (Tesla) A (J/m)
                {properties["theta"]} {properties["phi"]} {properties["K"]} 0.0 {properties["Js"]} {properties["A"]}
                """
            )
        )
        # TODO: use jinja


def write_p2(system):
    """Save simulation parameters in `p2` format."""
    pass
    # shutil.copy(driver.p2_path, f"{system.name}.p2")
    # TODO: todo
