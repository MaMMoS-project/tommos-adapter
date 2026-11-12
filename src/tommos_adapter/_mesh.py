"""Internal helpers for reading and mapping tommos magnetization grids."""

from __future__ import annotations

import numbers
import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    import collections.abc
    import os

    import pyvista


def create_mesh(
    geom: str,
    extent: collections.abc.Sequence[numbers.Real],
    h: numbers.Real,
    out_name: os.PathLike,
    *,
    additional_arguments: str = "",
) -> pyvista.UnstructuredGrid:
    """Create mesh using `tommos mesh`.

    Args:
        geom: String specifying the geometry. Possible choices are 'box', 'ellipsoid', 'eye', or 'elliptic_cylinder'.
        extent: Sequence of three real numbers specifying the three parameters defining the geometry.
        h: Mesh size.
        out_name: Name of the mesh file. Both a `.vtu` and a `.npz` file will be created, but only the former
            will be read using Pyvista.
        additional_arguments: string containing optional arguments to pass at the end of the `tommos mesh ...` command.

    Returns:
        Mesh read from the created `.vtu` file.

    Raises:
        ValueError: argument `geom` was not recognized.
        ValueError: argument `extent` has the wrong type, length, or content.
        ValueError: argument `h` is not a positive finite real number.

    Examples:
        1. Create a 20x20x20 cube of mesh size 2 whose file is called `cube.vtu`.
        >>> import tommos_adapter as ta
        ...
        >>> ta.create_mesh("box", extent=[20, 20, 20], h=2, out_name="cube")
        UnstructuredGrid...

        2. Create a sphere of radius 8 using a different `tommos mesh` backend.
        >>> import tommos_adapter as ta
        ...
        >>> ta.create_mesh("ellipsoid", extent=[8, 8, 8], h=2, out_name="sphere", additional_arguments="--backend grid")
        UnstructuredGrid...

    """
    if geom not in ["box", "ellipsoid", "eye", "elliptic_cylinder"]:
        raise ValueError(
            f"Argument `geom` must be one of 'box', 'ellipsoid', 'eye', or 'elliptic_cylinder'. Got {geom}."
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

    out_name = Path(out_name)
    out_name.parent.mkdir(parents=True, exist_ok=True)
    extent_string = ",".join(str(L) for L in extent)
    res = subprocess.run(
        shlex.split(
            f"tommos mesh --geom {geom} --extent {extent_string} --h {h}"
            f" --out-name {out_name.name} {additional_arguments}"
        ),
        cwd=out_name.parent,
        capture_output=True,
    )
    res.check_returncode()
    return pv.read(out_name.with_suffix(".vtu"))


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
