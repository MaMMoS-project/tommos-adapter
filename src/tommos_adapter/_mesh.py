"""Internal helpers for reading and mapping tommos magnetization grids."""

import os

import numpy as np
import pyvista as pv


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
