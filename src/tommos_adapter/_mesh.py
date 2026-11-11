"""Internal helpers for reading and mapping tommos magnetization grids."""

from __future__ import annotations

import numbers
import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    import collections.abc

    import pyvista


def create_mesh(
    geom: str, extent: collections.abc.Sequence[numbers.Real], h: numbers.Real, out_name: str
) -> pyvista.UnstructuredGrid:
    """Create mesh using `tommos mesh`.

    Args:
        geom: String specifying the geometry. Possible choices are 'box', 'ellipsoid', 'eye', 'elliptic_cylinder',
            or 'poly_gb'.
        extent: Sequence of three real numbers specifying the three parameters defining the geometry.
        h: Mesh size.
        out_name: Name of the mesh file. Both a `.vtu` and a `.npz` file will be created, but only the former
            will be read using Pyvista.
    """
    if geom not in ["box", "ellipsoid", "eye", "elliptic_cylinder", "poly_gb"]:
        raise ValueError(
            "Argument `geom` must be one between 'box', 'ellipsoid', 'eye', 'elliptic_cylinder', or "
            f"'poly_gb'. Got {geom}."
        )

    if not (
        isinstance(extent, Sequence | np.ndarray)
        and len(extent) == 3
        and all(isinstance(L, numbers.Real) and L > 0 and np.isfinite(L) for L in extent)
    ):
        raise ValueError(
            f"Argument `extent` must be an array-like of length 3 of finite positive numbers. Got {extent}"
        )

    if not (isinstance(h, numbers.Real) and h > 0 and np.isfinite(h)):
        raise ValueError(f"Argument `h` must be a positive finite real number. Got {h}.")

    if geom == "poly_gb":
        if shutil.which("neper") is None:
            raise RuntimeError("Neper not found. Please install it.")
        extra_args = "--n 2"  # TODO: implement other multigrain arguments
        # we choose n=2 (number of grains) so tests run faster
    else:
        extra_args = ""

    extent_string = ",".join(str(L) for L in extent)
    res = subprocess.run(
        shlex.split(f"tommos mesh --geom {geom} --extent {extent_string} --h {h} --out-name {out_name} {extra_args}"),
        capture_output=True,
    )
    res.check_returncode()
    return pv.read(f"{out_name}.vtu")


def _read_magnetization_grid(filename: str | os.PathLike) -> pv.UnstructuredGrid:
    """Read a tommos VTU grid and validate its point magnetization.

    Args:
        filename: VTU file containing the magnetization grid.

    Returns:
        Unstructured grid with finite, three-component point data ``m``.

    Raises:
        RuntimeError: If the file cannot be read, is not an unstructured grid,
            or contains missing, incorrectly shaped, or non-finite magnetization.
    """
    try:
        grid = pv.read(filename)
    except Exception as error:
        raise RuntimeError(f"Cannot read tommos magnetization grid '{filename}'.") from error

    if not isinstance(grid, pv.UnstructuredGrid):
        raise RuntimeError(f"tommos magnetization grid '{filename}' is not an UnstructuredGrid.")
    if "m" not in grid.point_data:
        raise RuntimeError(f"tommos magnetization grid '{filename}' does not contain point data 'm'.")

    magnetization = np.asarray(grid.point_data["m"])
    expected_shape = (grid.n_points, 3)
    if magnetization.shape != expected_shape:
        raise RuntimeError(
            f"tommos magnetization grid '{filename}' has magnetization shape {magnetization.shape}; "
            f"expected {expected_shape}."
        )
    if not np.all(np.isfinite(magnetization)):
        raise RuntimeError(f"tommos magnetization grid '{filename}' contains non-finite magnetization values.")
    return grid


def _magnetization_at_reference_points(
    reference_points: np.ndarray,
    output_state: pv.UnstructuredGrid,
) -> np.ndarray:
    """Return output magnetization at the requested reference points."""
    output_points = np.asarray(output_state.points)
    # tommos writes float32 coordinates, so the reference points are rounded the same way.
    _, key = np.unique(
        np.vstack([reference_points.astype(output_points.dtype), output_points]),
        axis=0,
        return_inverse=True,
    )
    key = key.ravel()
    output_index = np.full(key.max() + 1, -1)
    output_index[key[len(reference_points) :]] = np.arange(len(output_points))
    point_map = output_index[key[: len(reference_points)]]
    if np.any(point_map < 0) or len(np.unique(point_map)) != len(reference_points):
        raise RuntimeError("Mesh in tommos output does not match the reference mesh.")
    return np.asarray(output_state.point_data["m"][point_map])
