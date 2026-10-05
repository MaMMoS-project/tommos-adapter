"""Collection of scripts.

- Write `npz` mesh.
- Write `krn` input file.
- Write `p2` input file.
"""

import shutil


def write_mesh(driver, system, **kwargs):
    """Write `npz` mesh."""
    shutil.copy(driver.mesh_path, f"{system.name}.npz")


def write_krn(driver, system, **kwargs):
    """Write `krn` input file."""
    shutil.copy(driver.krn_path, f"{system.name}.krn")


def write_p2(driver, system, **kwargs):
    """Write `p2` input file."""
    shutil.copy(driver.p2_path, f"{system.name}.p2")
