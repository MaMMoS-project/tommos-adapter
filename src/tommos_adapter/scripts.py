"""Translate micromagnetic models into TOMMOS input data.

The functions in this module deliberately do not write files.  They produce
the text and arrays needed by TOMMOS so that the translation can be tested
independently from a driver and from the TOMMOS executable.

The provisional unstructured-grid contract is:

* ``point_data["m"]`` contains the nodal magnetisation directions.
* ``cell_data["mat_id"]`` contains one material identifier per tetrahedron.
* ``cell_data["Ms"]`` contains the saturation magnetisation in A/m.
* ``field_data["mesh_unit_length"]`` contains the mesh unit length.
* ``field_data["mat_ids"]`` and ``field_data["mat_names"]`` map numeric material
  identifiers to names used in parameter dictionaries. They are optional for singlegrain
  materials and mandatory for multigrain.

TOMMOS material identifiers are contiguous and one-based. Source material
identifiers are therefore remapped when mesh data are generated.

Terms `material` and `mat` do not indicate necessarily different materials,
but rather different regions with potentially different magnetic parameters.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import micromagneticmodel as mm
import numpy as np
import pyvista as pv


@dataclass(frozen=True)
class _Material:
    """Material information required by the TOMMOS input formats."""

    name: str
    source_id: int
    mat_id: int


def mesh_data(
    system: mm.System,
    *,
    mat_key: str = "mat_id",
) -> dict[str, np.ndarray]:
    """Return TOMMOS ``knt`` and ``ijk`` arrays for a system.

    Args:
        system: Micromagnetic system with a PyVista unstructured grid as
            magnetisation object.
        mat_key: Name of the cell-data array containing material identifiers.

    Returns:
        A dictionary with ``knt`` and ``ijk`` suitable for ``numpy.savez``.

    Raises:
        TypeError: If the magnetisation is not a PyVista unstructured grid.
        ValueError: If the grid is empty, contains non-tetrahedral cells, or
            contains invalid material information.
    """
    grid = _grid(system)
    connectivity = _tetra_connectivity(grid)
    source_ids, materials = _materials(grid, mat_key=mat_key)

    mat_ids = np.empty(grid.n_cells, dtype=np.int32)
    for material in materials:
        mat_ids[source_ids == material.source_id] = material.mat_id

    points = np.asarray(grid.points, dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise ValueError("Grid points must contain only finite values.")

    ijk = np.column_stack((connectivity, mat_ids)).astype(np.int64, copy=False)
    return {"knt": points, "ijk": ijk}


def initial_magnetisation(system: mm.System, *, key: str = "m") -> np.ndarray:
    """Return normalised nodal magnetisation vectors.

    Examples:
        >>> points = np.array(
        ...     [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ... )
        >>> cells = np.array([4, 0, 1, 2, 3])
        >>> grid = pv.UnstructuredGrid(cells, [pv.CellType.TETRA], points)
        >>> grid.point_data["m"] = np.tile((0.0, 0.0, 2.0), (4, 1))
        >>> system = mm.System(name="example", m=grid)
        >>> initial_magnetisation(system)
        array([[0., 0., 1.],
               [0., 0., 1.],
               [0., 0., 1.],
               [0., 0., 1.]])

    Args:
        system: Micromagnetic system with a PyVista unstructured grid.
        key: Name of the point-data array containing the magnetisation.

    Returns:
        An ``(n_points, 3)`` array of unit vectors.

    Raises:
        ValueError: If the array is missing, has the wrong shape, or contains
            zero-length or non-finite vectors.
    """
    grid = _grid(system)
    if key not in grid.point_data:
        raise ValueError(f"Grid point_data does not contain {key!r}.")

    magnetisation = np.asarray(grid.point_data[key], dtype=np.float64)
    expected_shape = (grid.n_points, 3)
    if magnetisation.shape != expected_shape:
        raise ValueError(f"point_data[{key!r}] must have shape {expected_shape}, got {magnetisation.shape}.")
    if not np.all(np.isfinite(magnetisation)):
        raise ValueError(f"point_data[{key!r}] must contain only finite values.")

    norms = np.linalg.norm(magnetisation, axis=1)
    if np.any(norms == 0):
        raise ValueError(f"point_data[{key!r}] contains a zero-length vector.")
    return magnetisation / norms[:, np.newaxis]


def needs_initial_state_file(system: mm.System, *, key: str = "m") -> bool:
    """Return whether TOMMOS needs a VTU file for the initial state.

    Uniform magnetisation can be represented directly in a ``.p2`` file.
    Non-uniform magnetisation currently has to use TOMMOS' restart-VTU path.

    Args:
        system: Micromagnetic system to inspect.
        key: Name of the point-data magnetisation array.

    Returns:
        ``True`` for a non-uniform initial state, otherwise ``False``.
    """
    magnetisation = initial_magnetisation(system, key=key)
    return not np.allclose(magnetisation, magnetisation[0], rtol=1e-12, atol=1e-14)


def krn_data(
    system: mm.System,
    *,
    mat_key: str = "mat_id",
    saturation_key: str = "Ms",
) -> np.ndarray:
    """Return the TOMMOS material table represented by a ``.krn`` file.

    Each row contains ``theta, phi, K1, K2, Js, A``.  TOMMOS expects ``Js``
    in tesla, whereas the provisional grid contract stores ``Ms`` in A/m.

    Args:
        system: Micromagnetic system to translate.
        mat_key: Name of the cell-data material array.
        saturation_key: Name of the cell-data saturation-magnetisation array.

    Returns:
        A floating-point array with one row per material.

    Raises:
        ValueError: If material parameters cannot be represented by TOMMOS.
    """
    grid = _grid(system)
    source_ids, materials = _materials(grid, mat_key=mat_key)
    mat_names = [material.name for material in materials]
    terms = _energy_terms(system)

    exchange = terms.get(mm.Exchange)
    if exchange is None:
        exchange_values = np.zeros(len(materials))
    else:
        exchange_values = _scalar_values(exchange.A, mat_names, parameter="A")

    anisotropy = terms.get(mm.UniaxialAnisotropy)
    if anisotropy is None:
        anisotropy_values = np.zeros(len(materials))
        easy_axes = np.tile((0.0, 0.0, 1.0), (len(materials), 1))
    else:
        anisotropy_values, easy_axes = _anisotropy_values(anisotropy, mat_names)

    theta = np.arccos(np.clip(easy_axes[:, 2], -1.0, 1.0))
    phi = np.arctan2(easy_axes[:, 1], easy_axes[:, 0])
    saturation = _saturation_values(
        grid,
        source_ids,
        materials,
        saturation_key=saturation_key,
    )

    return np.column_stack(
        (
            theta,
            phi,
            anisotropy_values,
            np.zeros(len(materials)),
            mm.consts.mu0 * saturation,
            exchange_values,
        )
    )


def krn_script(
    system: mm.System,
    *,
    mat_key: str = "mat_id",
    saturation_key: str = "Ms",
) -> str:
    """Return the contents of a TOMMOS ``.krn`` material file.

    Args:
        system: Micromagnetic system to translate.
        mat_key: Name of the cell-data material array.
        saturation_key: Name of the cell-data saturation-magnetisation array.

    Returns:
        Space-separated TOMMOS material rows ending with a newline.
    """
    output = io.StringIO()
    np.savetxt(
        output,
        krn_data(system, mat_key=mat_key, saturation_key=saturation_key),
        fmt="%.17g",
    )
    return output.getvalue()


def p2_script(
    system: mm.System,
    *,
    hx: Real,
    hy: Real,
    hz: Real,
    hstart: Real,
    hfinal: Real,
    hstep: Real,
    max_iter: int | None = None,
    mstep: float | None = None,
    method: str = "",
    tol_fun: float | None = None,
    eps_a: float | str | None = None,
) -> str:
    """Return a TOMMOS ``.p2`` file for a hysteresis calculation.

    The field arguments are TOMMOS parameters produced by
    :class:`tommos_adapter.HysteresisDriver`. Field magnitudes are magnetic
    flux densities in tesla and ``(hx, hy, hz)`` is a unit vector.

    Args:
        system: Micromagnetic system to translate.
        hx: Applied-field direction x component.
        hy: Applied-field direction y component.
        hz: Applied-field direction z component.
        hstart: Initial applied field in tesla.
        hfinal: Final applied field in tesla.
        hstep: Applied-field step in tesla.
        max_iter: Optional maximum number of minimiser iterations.
        mstep: Magnetization change threshold for saving ``.vtu`` snapshots. Setting this to a very large value
            (e.g. 10000) prevents VTU outputs to save disk space.
        method: Energy minimization algorithm. Possible choices: ``pcohen_hs``, ``tr``.
        tol_fun: Optional relative energy tolerance.
        eps_a: Optional absolute tangent-gradient tolerance or ``"auto"``.

    Returns:
        INI-formatted TOMMOS parameter text.
    """
    if max_iter is not None and (not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0):
        raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}.")
    if tol_fun is not None and (not isinstance(tol_fun, Real) or not np.isfinite(tol_fun) or tol_fun <= 0):
        raise ValueError(f"tol_fun must be a positive finite number, got {tol_fun!r}.")
    if mstep is not None and (not isinstance(mstep, Real) or not np.isfinite(mstep) or mstep <= 0):
        raise ValueError(f"mstep must be a positive finite number, got {mstep!r}.")
    if method != "" and (not isinstance(method, str) or method not in ["pcohen_hs", "tr"]):
        raise ValueError(f"method must be a string between 'pcohen_hs' and 'tr', got {method!r}.")

    direction = _field_direction(hx, hy, hz)
    field_values = {"hstart": hstart, "hfinal": hfinal, "hstep": hstep}
    for name, value in field_values.items():
        if not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}.")
    if hstep == 0:
        raise ValueError("hstep must be non-zero.")

    magnetisation = initial_magnetisation(system)
    if needs_initial_state_file(system):
        raise ValueError(
            "TOMMOS adapter currently supports only a uniform initial magnetisation; "
            "writing a restart VTU file is not implemented yet."
        )

    lines = ["[mesh]"]

    mesh_unit_length = system.m.field_data["mesh_unit_length"]
    if len(mesh_unit_length) != 1:
        raise ValueError(f"Mesh unit length should be a single-valued field data. Got {mesh_unit_length}")
    mul = mesh_unit_length[0]
    if not isinstance(mul, Real) or not np.isfinite(mul) or mul <= 0:
        raise ValueError(f"Mesh field data `mesh_unit_length` must be a positive finite number, got {mul!r}.")
    lines.append(f"size = {mul}")

    lines.extend(("", "[initial state]"))
    mx, my, mz = magnetisation[0]
    lines.extend(
        (
            f"mx = {_format_number(mx)}",
            f"my = {_format_number(my)}",
            f"mz = {_format_number(mz)}",
        )
    )

    hx, hy, hz = direction
    lines.extend(
        (
            "",
            "[field]",
            f"hstart = {_format_number(hstart)}",
            f"hfinal = {_format_number(hfinal)}",
            f"hstep = {_format_number(hstep)}",
            f"hx = {_format_number(hx)}",
            f"hy = {_format_number(hy)}",
            f"hz = {_format_number(hz)}",
        )
    )
    if mstep is not None:
        lines.append(f"mstep = {mstep}")

    minimizer_lines = []
    if method != "":
        minimizer_lines.append(f"method = {method}")
    if max_iter is not None:
        minimizer_lines.append(f"max_iter = {max_iter}")
    if tol_fun is not None:
        minimizer_lines.append(f"tol_fun = {_format_number(tol_fun)}")
    if eps_a is not None:
        if isinstance(eps_a, str):
            if eps_a != "auto":
                raise ValueError(f"eps_a must be a positive number or 'auto', got {eps_a!r}.")
            eps_a_value = eps_a
        elif isinstance(eps_a, Real) and np.isfinite(eps_a) and eps_a > 0:
            eps_a_value = _format_number(eps_a)
        else:
            raise ValueError(f"eps_a must be a positive number or 'auto', got {eps_a!r}.")
        minimizer_lines.append(f"eps_a = {eps_a_value}")
    if minimizer_lines:
        lines.extend(("", "[minimizer]", *minimizer_lines))

    return "\n".join(lines) + "\n"


def _grid(system: mm.System) -> pv.UnstructuredGrid:
    grid = system.m
    if not isinstance(grid, pv.UnstructuredGrid):
        raise TypeError(f"TOMMOS requires system.m to be a pyvista.UnstructuredGrid, got {type(grid).__name__}.")
    if grid.n_points == 0 or grid.n_cells == 0:
        raise ValueError("The unstructured grid must contain points and cells.")
    return grid


def _tetra_connectivity(grid: pv.UnstructuredGrid) -> np.ndarray:
    cell_types = np.asarray(grid.celltypes)
    tetra_type = int(pv.CellType.TETRA)
    if cell_types.shape != (grid.n_cells,) or np.any(cell_types != tetra_type):
        raise ValueError("TOMMOS currently supports tetrahedral cells only.")
    connectivity = np.asarray(grid.cells_dict[tetra_type], dtype=np.int64)
    if connectivity.shape != (grid.n_cells, 4):
        raise ValueError(f"Expected tetrahedral connectivity with shape {(grid.n_cells, 4)}, got {connectivity.shape}.")
    return connectivity


def _materials(grid: pv.UnstructuredGrid, *, mat_key: str) -> tuple[np.ndarray, list[_Material]]:
    if mat_key not in grid.cell_data:
        raise ValueError(f"Grid cell_data does not contain {mat_key!r}.")

    raw_ids = np.asarray(grid.cell_data[mat_key])
    if raw_ids.ndim == 2 and raw_ids.shape[1] == 1:
        raw_ids = raw_ids[:, 0]
    if raw_ids.shape != (grid.n_cells,):
        raise ValueError(f"cell_data[{mat_key!r}] must have shape {(grid.n_cells,)}, got {raw_ids.shape}.")
    if not np.issubdtype(raw_ids.dtype, np.integer):
        raise ValueError(f"cell_data[{mat_key!r}] must contain integer identifiers.")
    source_ids = raw_ids.astype(np.int64, copy=False)

    unique_ids = np.unique(source_ids)

    has_mat_ids = "mat_ids" in grid.field_data
    has_mat_names = "mat_names" in grid.field_data
    if has_mat_ids != has_mat_names:
        raise ValueError("field_data must contain both 'mat_ids' and 'mat_names'.")
    if has_mat_ids:
        metadata_ids = np.asarray(grid.field_data["mat_ids"]).reshape(-1)
        metadata_names = np.asarray(grid.field_data["mat_names"]).reshape(-1)
        if metadata_ids.size != metadata_names.size:
            raise ValueError("field_data['mat_ids'] and field_data['mat_names'] must have equal lengths.")
        if not np.issubdtype(metadata_ids.dtype, np.integer):
            raise ValueError("field_data['mat_ids'] must contain integers.")

        metadata_ids = metadata_ids.astype(np.int64, copy=False)
        if np.unique(metadata_ids).size != metadata_ids.size:
            raise ValueError("field_data['mat_ids'] contains duplicate identifiers.")
        names = [_as_string(name) for name in metadata_names]
        if len(set(names)) != len(names):
            raise ValueError("field_data['mat_names'] contains duplicate names.")
        names_by_id = dict(zip(map(int, metadata_ids), names, strict=True))
        missing_ids = sorted(set(map(int, unique_ids)) - set(names_by_id))
        extra_ids = sorted(set(names_by_id) - set(map(int, unique_ids)))
        if missing_ids or extra_ids:
            raise ValueError(
                f"Material metadata does not match the cell-data identifiers; missing={missing_ids}, extra={extra_ids}."
            )
    else:
        names_by_id = {int(mat_id): f"grain_{int(mat_id)}" for mat_id in unique_ids}

    materials = [
        _Material(name=names_by_id[int(source_id)], source_id=int(source_id), mat_id=index)
        for index, source_id in enumerate(unique_ids, start=1)
    ]
    return source_ids, materials


def _as_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _energy_terms(system: mm.System) -> dict[type, object]:
    supported = (mm.Exchange, mm.UniaxialAnisotropy, mm.Demag)
    result = {}
    for term in system.energy:
        if not isinstance(term, supported):
            raise ValueError(f"Energy term {term!r} is not supported by TOMMOS.")
        term_type = type(term)
        if term_type in result:
            raise ValueError(f"Only one {term_type.__name__} term is supported by TOMMOS.")
        result[term_type] = term

    if mm.Demag not in result:
        raise ValueError("TOMMOS currently always computes demagnetisation; system.energy must contain Demag().")
    return result


def _scalar_values(value: object, mat_names: list[str], *, parameter: str) -> np.ndarray:
    if isinstance(value, Mapping):
        keys = set(value)
        missing = [] if "default" in value else sorted(set(mat_names) - keys)
        extra = sorted(keys - set(mat_names) - {"default"})
        if missing or extra:
            raise ValueError(f"Parameter {parameter} has mismatching materials; missing={missing}, extra={extra}.")
        values = [value[name] if name in value else value["default"] for name in mat_names]
    else:
        values = [value] * len(mat_names)

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Parameter {parameter} must contain numeric scalar values.") from error
    if array.shape != (len(mat_names),) or not np.all(np.isfinite(array)):
        raise ValueError(f"Parameter {parameter} must contain one finite scalar per material.")
    return array


def _vector_values(value: object, mat_names: list[str], *, parameter: str) -> np.ndarray:
    if isinstance(value, Mapping):
        keys = set(value)
        missing = [] if "default" in value else sorted(set(mat_names) - keys)
        extra = sorted(keys - set(mat_names) - {"default"})
        if missing or extra:
            raise ValueError(f"Parameter {parameter} has mismatching materials; missing={missing}, extra={extra}.")
        values = [value[name] if name in value else value["default"] for name in mat_names]
    else:
        values = [value] * len(mat_names)

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Parameter {parameter} must contain numeric vectors.") from error
    expected_shape = (len(mat_names), 3)
    if array.shape != expected_shape or not np.all(np.isfinite(array)):
        raise ValueError(f"Parameter {parameter} must have shape {expected_shape} and contain finite values.")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0):
        raise ValueError(f"Parameter {parameter} contains a zero-length vector.")
    return array / norms[:, np.newaxis]


def _anisotropy_values(term: mm.UniaxialAnisotropy, mat_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    has_k = "K" in term.__dict__
    has_k1 = "K1" in term.__dict__
    if has_k and has_k1:
        raise ValueError("UniaxialAnisotropy must define either K or K1, not both.")
    anisotropy = _scalar_values(term.__dict__.get("K", term.__dict__.get("K1", 0.0)), mat_names, parameter="K1")

    if "K2" in term.__dict__:
        k2 = _scalar_values(term.__dict__["K2"], mat_names, parameter="K2")
        if np.any(k2 != 0):
            raise ValueError("Non-zero K2 is not supported because TOMMOS currently ignores the K2 material column.")

    axes = _vector_values(term.__dict__.get("u", (0.0, 0.0, 1.0)), mat_names, parameter="u")
    return anisotropy, axes


def _saturation_values(
    grid: pv.UnstructuredGrid,
    source_ids: np.ndarray,
    materials: list[_Material],
    *,
    saturation_key: str,
) -> np.ndarray:
    if saturation_key not in grid.cell_data:
        raise ValueError(f"Grid cell_data does not contain {saturation_key!r}.")
    saturation = np.asarray(grid.cell_data[saturation_key], dtype=np.float64)
    if saturation.ndim == 2 and saturation.shape[1] == 1:
        saturation = saturation[:, 0]
    if saturation.shape != (grid.n_cells,):
        raise ValueError(f"cell_data[{saturation_key!r}] must have shape {(grid.n_cells,)}, got {saturation.shape}.")
    if not np.all(np.isfinite(saturation)) or np.any(saturation < 0):
        raise ValueError(f"cell_data[{saturation_key!r}] must contain non-negative finite values.")

    result = []
    for material in materials:
        material_values = saturation[source_ids == material.source_id]
        if not np.allclose(material_values, material_values[0], rtol=1e-12, atol=0.0):
            raise ValueError(
                f"TOMMOS requires a constant {saturation_key} per material; "
                f"material {material.name!r} contains multiple values."
            )
        result.append(material_values[0])
    return np.asarray(result)


def _field_direction(hx: Real, hy: Real, hz: Real) -> np.ndarray:
    try:
        direction = np.asarray((hx, hy, hz), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("hx, hy, and hz must be numeric values.") from error
    if not np.all(np.isfinite(direction)):
        raise ValueError("hx, hy, and hz must be finite values.")
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("hx, hy, and hz must define a non-zero direction.")
    if not np.isclose(norm, 1.0, rtol=1e-12, atol=1e-14):
        raise ValueError("hx, hy, and hz must define a unit vector.")
    return direction


def _format_number(value: Real) -> str:
    return format(float(value), ".17g")
