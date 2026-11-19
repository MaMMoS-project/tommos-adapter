"""Internal helpers for reading and mapping tommos magnetization grids."""

from __future__ import annotations

import numbers
import shlex
import shutil
import subprocess
import warnings
from collections.abc import Mapping, Sequence
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
    additional_mesher_arguments: str = "",
    mesh_unit_length: numbers.Real = 1e-9,
    mat_id_name_mapping: collections.abc.Mapping[int, str] | None = None,
) -> pyvista.UnstructuredGrid:
    r"""Create and read mesh using `tommos mesh`.

    This function creates a mesh with the following attributes:

    - `mat_id` (cell-wise): defines the material or region integer ID. Different IDs define separated regions rather
      than different materials, i.e. they can have the same magnetic properties.
    - `mesh_unit_length` (field-wise): mesh unit length.
    - `mat_ids` (field-wise): list of all unique material IDs.
    - `mat_names` (field-wise): list of names of all material region.

    The mesh is created via the `tommos mesh` CLI tool. Argument `geom` is the type of geometry, `h` is the mesh size,
    `out_name` is the name or path where the mesh file will be saved. The argument `extent` means different things for
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
      the thickness in the z dimension, respectively, and the geometry points :math:`(x, y, z)` satisfy:

      .. math:
        y \le -2 \frac{h}{l^2} x + \frac{h}{2}, \\
        |z| \le L_\mathsf{z}.

    - if `geom="elliptic_cylinder"`, the geometry is a cylinder whose faces in the xy-planes are ellipses, centred in
      the origin. The three real numbers `extent=(A, B, Lz)` are the major and minor axes of the ellipse in the
      xy-plane and the thickness in the z dimension, and the geometry points :math:`(x, y, z)` satisfy:

      .. math:
        \left(\frac{x}{A/2}\right)^2 + \right(\frac{y}{B/2}\right)^2 \le 1, \\
        |z| \le L_\mathsf{z}.

    - if `geom="poly"`, the geometry is a polycrystalline parallelepiped without grain boundary phase. The parameters
      are the same of geometry `"box"`. By default, this creates a material with 10 different grains generated randomly
      by Voronoi tessellation using the tessellation software Neper. To modify the number of grains, specify additional
      argument `--n`. For this geometry, Neper has to be installed on the system and its location must be in PATH.

    - if `geom="poly_gb"`, the geometry is a polycrystalline parallelepiped with grain boundary phase. The parameters
      are the same of geometry `"box"` and `"poly"`. By default, the number of grains are 10 and the grain boundary
      phase has thickness of 1 and element size of 1. To modify these variables, specify additional arguments
      `--gb-thickness` and `--gb-h`. For this geometry, Neper has to be installed on the system and its location must
      be in PATH. The total number of different IDs in the material will be the number of grains plus one.

    For more information regarding `tommos mesh`, see https://github.com/MaMMoS-project/tommos#tommos-mesh-meshing-tool.
    These equations specify geometries in `tommos` 0.1.0.

    Args:
        geom: String specifying the geometry. Possible choices are 'box', 'ellipsoid', 'eye', or 'elliptic_cylinder'.
        extent: Sequence of three real numbers specifying the three parameters defining the geometry.
        h: Mesh size.
        out_name: Name of the mesh file. Both a `.vtu` and a `.npz` file will be created, but only the former
            will be read using Pyvista.
        additional_mesher_arguments: string containing optional arguments to pass at the end of the `tommos mesh ...`
            command.
        mesh_unit_length: Mesh unit length indicating a scaling factor in unit of meter. The default value 1e-9 means
            nanometer scale.
        mat_id_name_mapping: Mapping linking each `mat_id` to its grain name. If not specified, the `mat_names` are
            set as `grain_1`, `grain_2`, etc.

    Returns:
        Mesh read from the created `.vtu` file.

    Raises:
        ValueError: Argument `geom` was not recognized.
        ValueError: Argument `extent` has the wrong type, length, or content.
        ValueError: Argument `h` is not a positive finite real number.
        RuntimeError: `neper` was not found.
        RuntimeError: Meshing with `tommos mesh` failed.

    Examples:
        1. Create a 20x20x20 cube of mesh size 2 whose file is called `cube.vtu`.
        >>> import tommos_adapter as ta
        ...
        >>> ta.create_mesh("box", extent=[20, 20, 20], h=2, out_name="cube")
        UnstructuredGrid...

        2. Create a mesh in a specific directory.
        >>> import tommos_adapter as ta
        ...
        >>> ta.create_mesh("elliptic_cylinder", extent=[8, 8, 8], h=2, out_name="meshes/cylinder")
        UnstructuredGrid...

        3. Create a sphere of radius 8 using a different `tommos mesh` backend.
        >>> import tommos_adapter as ta
        ...
        >>> ta.create_mesh("ellipsoid", extent=[8, 8, 8], h=2, out_name="sphere", additional_mesher_arguments="--backend grid")
        UnstructuredGrid...

        4. Create a polycrystalline cube of side length 20 with 6 grains and with grain boundary phase.
        >>> import tommos_adapter as ta
        ...
        >>> ta.create_mesh("poly_gb", extent=[20, 20, 20], h=2, out_name="poly", additional_mesher_arguments="--n 6")
        UnstructuredGrid...

    """  # noqa: E501
    if geom not in ["box", "ellipsoid", "eye", "elliptic_cylinder", "poly", "poly_gb"]:
        raise ValueError(
            "Argument `geom` must be one of 'box', 'ellipsoid', 'eye', 'elliptic_cylinder', 'poly', or 'poly_gb'. "
            f"Got {geom}."
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

    if "geom" in ["poly", "poly_gb"] and shutil.which("neper") is None:
        raise RuntimeError("Neper not found. For geometries 'poly' and 'poly_gb' the software Neper is necessary.")

    out_name = Path(out_name)
    out_name.parent.mkdir(parents=True, exist_ok=True)
    extent_string = ",".join(str(L) for L in extent)
    res = subprocess.run(
        shlex.split(
            f"tommos mesh --geom {geom} --extent {extent_string} --h {h}"
            f" --out-name {out_name.name} {additional_mesher_arguments}"
        ),
        cwd=out_name.parent,
        capture_output=True,
        text=True,
    )
    if res.returncode:
        raise RuntimeError(f"`tommos mesh` failed with the following stderr:\n{res.stderr}")
    res.check_returncode()
    mesh = pv.read(out_name.with_suffix(".vtu"))
    if mat_id_name_mapping is None and geom == "poly_gb":
        mat_ids = np.unique(mesh.cell_data["mat_id"])
        mat_id_name_mapping = {i: f"grain_{i}" for i in mat_ids}
        mat_id_name_mapping[mat_ids[-1]] = "grain_boundary"
    return read_mesh(mesh, mesh_unit_length=mesh_unit_length, mat_id_name_mapping=mat_id_name_mapping)


def read_mesh(
    mesh: pyvista.UnstructuredGrid,
    mesh_unit_length: numbers.Real = 1e-9,
    mat_id_name_mapping: collections.abc.Mapping[int, str] | None = None,
) -> pyvista.UnstructuredGrid:
    """Read mesh using `tommos mesh`.

    This function modifies the given mesh to set the following `field_data` attributes:

    - `mesh_unit_length`: mesh unit length.
    - `mat_ids`: list of all unique material IDs.
    - `mat_names`: list of names of all material region.

    The name `default` as a material name (`mat_name`) is not allowed and it will raise an error.

    Args:
        mesh: User-defined mesh or generated from `tommos mesh` as a :py:class:`pyvista.UnstructuredGrid`.
            The mesh should contain a `cell_data` `'mat_id'` containing an integer id for each cell, intended as
            a grain index. If not given, the material is assumed to be uniform single grain and the value is set to 1.
        mesh_unit_length: Mesh unit length indicating a scaling factor in unit of meter. The default value 1e-9 means
            nanometer scale.
        mat_id_name_mapping: Mapping linking each `mat_id` to its grain name. If not specified, the `mat_names` are
            set as `grain_1`, `grain_2`, etc.

    Returns:
        Modified mesh with set `field_data` attributes.

    Raises:
        ValueError: Cell data attribute `mat_id` contains values different than :py:class:`numpy.int32`.
        ValueError: Given argument `mesh_unit_length` is of the wrong type or value.
        ValueError: Keys of `mat_id_name_mapping` and detected mesh `mat_ids` do not match.
        ValueError: The material name 'default' in `mat_id_name_mapping` is not allowed.

    Warns:
        UserWarning: Given mesh does not a 'mat_id' `cell_data` attribute.
    """
    if "mat_id" in mesh.cell_data:
        if (mat_id := mesh.cell_data.get_array("mat_id")).dtype != np.int32:
            raise ValueError(f"Cell data 'mat_id' should be an array of integers. Got {mat_id}.")
    else:
        mesh.cell_data["mat_id"] = np.ones(mesh.n_cells, dtype=np.int32)
        warnings.warn(
            "Given mesh does not have a 'mat_id' field data. Defaulting to a singlegrain with 'mat_id'=1.",
            stacklevel=2,
        )

    mesh.field_data["mat_ids"] = np.unique(mesh.cell_data["mat_id"])

    if not (isinstance(mesh_unit_length, numbers.Real) and mesh_unit_length > 0 and np.isfinite(mesh_unit_length)):
        raise ValueError(f"Argument `mesh_unit_length` must be a positive finite real number. Got {mesh_unit_length}.")
    mesh.field_data["mesh_unit_length"] = mesh_unit_length

    if mat_id_name_mapping is None:
        mesh.field_data["mat_names"] = [f"grain_{i}" for i in mesh.field_data["mat_ids"]]
    else:
        if set(mat_id_name_mapping.keys()) != set(mesh.field_data["mat_ids"]):
            raise ValueError(
                "Inconsistency between detected 'mat_ids' field data and given `mat_id_name_mapping` argument. "
                f"Found 'mat_ids': {mesh.field_data['mat_ids']}. Given `mat_id_name_mapping`: {mat_id_name_mapping}."
            )
        if "default" in mat_id_name_mapping.values():
            raise ValueError(f"The material name 'default' is not allowed. Got {mat_id_name_mapping=}.")
        else:
            mesh.field_data["mat_names"] = [mat_id_name_mapping[_id] for _id in mesh.field_data["mat_ids"]]

    return mesh


def set_magnetization(
    mesh: pyvista.UnstructuredGrid,
    m: tuple,
    Ms: numbers.Real | collections.abc.Mapping[str, numbers.Real],
) -> pyvista.UnstructuredGrid:
    """Set values to initial magnetization and spontaneous magnetization.

    The mesh is modified in-place.

    Args:
        mesh: Mesh to fill with information.
        m: single magnetization array to spread to all mesh points. It gets normalized internally.
        Ms: single spontaneous magnetization value to spread to all mesh cells.

    Returns:
        Mesh with set values.

    Raises:
        ValueError: `m` is of the wrong type, wrong length, or contains non finite values.
        ValueError: `Ms` is not a finite real number.
    """
    if not (
        isinstance(m, Sequence | np.ndarray)
        and len(m) == 3
        and all(isinstance(m_i, numbers.Real) and np.isfinite(m_i) for m_i in m)
    ):
        raise ValueError(f"Argument `m` must be an array-like of length 3 of finite numbers. Got {m}")
    m_array = np.asarray(m, dtype=np.float64)
    if (m_norm := np.linalg.norm(m_array)) > 0:
        m_array /= m_norm
    else:
        raise ValueError("Argument `m` cannot be the zero vector.")
    mesh.point_data["m"] = np.tile(m_array, (mesh.n_points, 1))

    if isinstance(Ms, numbers.Real) and Ms > 0 and np.isfinite(Ms):
        mesh.cell_data["Ms"] = np.full(mesh.n_cells, Ms)
    elif isinstance(Ms, Mapping):
        Ms_keys = set(Ms.keys())
        mesh_names = set(mesh.field_data["mat_names"])
        assigned_Ms_keys = Ms_keys - {"default"}
        mesh.cell_data["Ms"] = np.zeros(mesh.n_cells, dtype=np.float64)

        # case 1: unrecognized key in `Ms`
        if not assigned_Ms_keys.issubset(mesh_names):
            raise ValueError(
                f"Invalid material name in `Ms`: {assigned_Ms_keys - mesh_names}. Valid material names: {mesh_names}"
            )
        # case 2: Ms_keys does not have all necessary keys nor 'default'
        if "default" not in Ms_keys and Ms_keys != mesh_names:
            raise ValueError(
                f"Argument `Ms` does not contain information for all materials. Given {Ms=}. "
                f"Missing materials: {[str(k) for k in mesh_names - Ms_keys]}."
            )
        # case 3: Ms_keys contains all keys or some keys + 'default'
        for i, mat_id in enumerate(mesh.field_data["mat_ids"]):
            mat_name = mesh.field_data["mat_names"][i]
            mesh.cell_data["Ms"][mesh.cell_data["mat_id"] == mat_id] = Ms.get(mat_name, Ms.get("default"))

    else:
        raise ValueError(
            "Invalid argument `Ms`. It must be either a positive finite real number or a dictionary assigning a value "
            f"to each material name. Got {Ms}."
        )

    return mesh


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
