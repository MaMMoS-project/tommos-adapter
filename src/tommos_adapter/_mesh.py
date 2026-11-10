"""Internal helpers for reading and mapping tommos magnetisation grids."""

import os
from numbers import Real

import numpy as np
import pyvista as pv

DEFAULT_MESH_UNIT = 1e-9


def _mesh_unit_length(grid: pv.UnstructuredGrid, *, default: float = 1.0) -> float:
    """Return the length in metres of one grid coordinate unit.

    Grids without ``field_data["mesh_unit_length"]`` use ``default``.
    The default value of one preserves support for existing SI grids.
    """
    values = np.asarray(grid.field_data.get("mesh_unit_length", [default])).ravel()
    if (
        values.size != 1
        or not isinstance(values[0], Real)
        or isinstance(values[0], (bool, np.bool_))
        or not np.isfinite(values[0])
        or values[0] <= 0
    ):
        raise ValueError("field_data['mesh_unit_length'] must contain one positive finite length in metres.")
    return float(values[0])


def _mesh_points(grid: pv.UnstructuredGrid, *, mesh_unit: float) -> np.ndarray:
    """Return grid coordinates expressed in the requested mesh unit."""
    points = np.asarray(grid.points, dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise ValueError("Grid points must contain only finite values.")
    unit_length = _mesh_unit_length(grid)
    if unit_length == mesh_unit:
        return points
    return points * unit_length / float(mesh_unit)


def _read_magnetisation_grid(filename: str | os.PathLike) -> pv.UnstructuredGrid:
    """Read a tommos VTU grid and validate its point magnetisation.

    Args:
        filename: VTU file containing the magnetisation grid.

    Returns:
        Unstructured grid with finite, three-component point data ``m``.

    Raises:
        RuntimeError: If the file cannot be read, is not an unstructured grid,
            or contains missing, incorrectly shaped, or non-finite magnetisation.
    """
    try:
        grid = pv.read(filename)
    except Exception as error:
        raise RuntimeError(f"Cannot read tommos magnetisation grid '{filename}'.") from error

    if not isinstance(grid, pv.UnstructuredGrid):
        raise RuntimeError(f"tommos magnetisation grid '{filename}' is not an UnstructuredGrid.")
    if "m" not in grid.point_data:
        raise RuntimeError(f"tommos magnetisation grid '{filename}' does not contain point data 'm'.")

    magnetisation = np.asarray(grid.point_data["m"])
    expected_shape = (grid.n_points, 3)
    if magnetisation.shape != expected_shape:
        raise RuntimeError(
            f"tommos magnetisation grid '{filename}' has magnetisation shape {magnetisation.shape}; "
            f"expected {expected_shape}."
        )
    if not np.all(np.isfinite(magnetisation)):
        raise RuntimeError(f"tommos magnetisation grid '{filename}' contains non-finite magnetisation values.")
    return grid


def _magnetization_without_airbox(
    reference: pv.UnstructuredGrid,
    output_state: pv.UnstructuredGrid,
    *,
    mesh_unit: float = DEFAULT_MESH_UNIT,
) -> np.ndarray:
    """Return output magnetization at the points of a reference mesh."""
    # Match mesh_data(): scale in float64 before tommos rounds its output to float32.
    reference_points = _mesh_points(reference, mesh_unit=mesh_unit)
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
    return np.asarray(output_state.point_data["m"])[point_map]
