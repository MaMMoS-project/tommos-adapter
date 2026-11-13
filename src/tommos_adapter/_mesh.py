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
    additional_arguments: str = "",
    mesh_unit_length: numbers.Real = 1e-9,
    mat_id_name_mapping: collections.abc.Mapping[int, str] | None = None,
) -> pyvista.UnstructuredGrid:
    r"""Create and read mesh using `tommos mesh`.

    This function creates a mesh with the following attributes:

    - `m` (point-wise): defines the initial magnetization on each mesh node.
    - `Ms` (cell-wise): defines the spontaneous magnetization intrinsic magnetic property on each cell in `A/m`.
    - `mat_id` (cell-wise): defines the material or region integer ID. Different IDs define separated regions rather
      than different materials, i.e. they can have the same magnetic properties.
    - `mesh_unit_length` (field-wise): mesh unit length or scaling factor in terms of meters.
    - `mat_ids` (field-wise): list of all possible material IDs.
    - `mat_names` (field-wise): list of names of all material region.

    The mesh is created via the `tommos mesh` CLI tool. Argument `geom` is the type of geometry, `h` is the mesh size,
    `out_name` is the name or path where the mesh file will be saved. The argument `extent` means different thing for
    each geometry shape:

    - if `geom="box"`, the geometry is a parallelepiped where the three real numbers `extent=(Lx, Ly, Lz)` define the
      three side lengths. The geometry is then centred in the origin, so that the all geometry points
      :math:`(x, y, z)` satisfy:

      .. math::
        |x| \le L_\mathsf{x}/2, \\
        |y| \le L_\mathsf{y}/2, \\
        |z| \le L_\mathsf{z}/2.

    - if `geom="ellipsoid"`, the geometry is an ellipsoid with rotational symmetry in the xy-plane, centred in the
      origin. The three real numbers `extent=(Lx, Ly, Lz)` are a linear transformation of the semi-axes, and the
      geometry points :math:`(x, y, z)` satisfy:

      .. math::
        \left(\frac{x}{a}\right)^2 + \left(\frac{y}{b}\right)^2 + \left(\frac{z}{c}\right^2 \le 1,

      where:

      .. math::
        a = \frac{L_\mathsf{x} + L_\mathsf{y}}/4, \\
        b = \frac{L_\mathsf{x} + L_\mathsf{y}}/4, \\
        c = \frac{L_\mathsf{z}}/4.

    - if `geom="eye"`, the projection in the xy-plane is an "eye" shape formed by two symmetric Bézier curves, extruded
      in the z dimension. The three real numbers `extent=(l, h, Lz)` are the full eye length, the eye half-height, and
      the width in the z dimension, respectively, and the geometry points :math:`(x, y, z)` satisfy:

      .. math:
        y \le -2 \frac{h}{l^2} x + \frac{h}{2}, \\
        |z| \le L_\mathsf{z}.

    - of `geom="elliptic_cylinder"`, the geometry is a cylinder whose faces in the xy-planes are ellipses, centred in
      the origin. The three real numbers `extent=(A, B, Lz)` are the major axes of the ellipse in the xy-plane and the
      width in the z dimension, and the geometry points :math:`(x, y, z)` satisfy:

      .. math:
        \left(\frac{x}{A/2}\right)^2 + \right(\frac{y}{B/2}\right)^2 \le 1, \\
        |z| \le L_\mathsf{z}.

    For more information regarding `tommos mesh`, see https://github.com/MaMMoS-project/tommos#tommos-mesh-meshing-tool.

    Args:
        geom: String specifying the geometry. Possible choices are 'box', 'ellipsoid', 'eye', or 'elliptic_cylinder'.
        extent: Sequence of three real numbers specifying the three parameters defining the geometry.
        h: Mesh size.
        out_name: Name of the mesh file. Both a `.vtu` and a `.npz` file will be created, but only the former
            will be read using Pyvista.
        additional_arguments: string containing optional arguments to pass at the end of the `tommos mesh ...` command.
        mesh_unit_length: Mesh unit length indicating a scaling factor in meter. The default value 1e-9 means we
            assume nanometer scale unless specified otherwise.
        mat_id_name_mapping: Mapping linking each `mat_id` to its grain name. If not specified, the `mat_names` are
            set as `grain_1`, `grain_2`, etc.

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
    mesh = pv.read(out_name.with_suffix(".vtu"))
    return read_mesh(mesh, mesh_unit_length=mesh_unit_length, mat_id_name_mapping=mat_id_name_mapping)


def read_mesh(
    mesh: pyvista.UnstructuredGrid,
    mesh_unit_length: numbers.Real = 1e-9,
    mat_id_name_mapping: collections.abc.Mapping[int, str] | None = None,
) -> pyvista.UnstructuredGrid:
    """Read mesh using `tommos mesh`.

    TODO: args, returns, raises
    """
    pass  # TODO: body


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
