"""Internal helpers for mapping tommos output meshes."""

import numpy as np
import pyvista as pv

from tommos_adapter.scripts import DEFAULT_MESH_UNIT


def _magnetization_without_airbox(reference: pv.UnstructuredGrid, output_state: pv.UnstructuredGrid) -> np.ndarray:
    """Return output magnetization at the points of a reference mesh."""
    reference_points = np.asarray(reference.points) / DEFAULT_MESH_UNIT
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
