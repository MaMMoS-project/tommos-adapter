"""Define drivers for tommos.

The only available driver is a `HysteresisDriver` as the only capability of `tommos` is to run hysteresis loops.
"""

from __future__ import annotations

import numbers
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import micromagneticmodel as mm
import numpy as np
import pyvista as pv

from tommos_adapter import scripts
from tommos_adapter.tommos_runner import TommosRunner

if TYPE_CHECKING:
    import micromagneticmodel
    import numpy.typing

    import tommos_adapter


class HysteresisDriver(mm.adapter_base.ExternalDriver):
    """Driver to run a hysteresis loop.

    The field sweep is defined when calling :meth:`drive` or :meth:`schedule` using the argument ``Hsteps``,
    specifying a curve in the form ``(Hstart, Hend, n)``. Objects ``Hstart`` and ``Hend`` are one-dimensional
    arrays of length 3, specified in A/m and must lie on a common axis through the origin.

    Examples:
        1. Defining a driver.

        >>> import tommos_adapter as ta
        ...
        >>> hd = ta.HysteresisDriver()

        2. Passing an argument which is not allowed.

        >>> import tommos_adapter as ta
        ...
        >>> md = ta.HysteresisDriver(myarg=1)
        Traceback (most recent call last):
           ...
        AttributeError: ...

        3. Getting the list of allowed attributes.

        >>> import tommos_adapter as ta
        ...
        >>> hd = ta.HysteresisDriver()
        >>> hd._allowed_attributes
        [...]

    """

    _allowed_attributes = ["mstep", "method", "max_iter", "tol_fun", "eps_a"]

    def get_argstr(self, system: micromagneticmodel.System) -> str:
        """Return string passed to the `tommos` call.

        Args:
            system: Micromagnetic system.

        Returns:
            Argument string passed to the `tommos loop` runner. It differs depending on the running platform.
        """
        argstr = f"{system.name}"
        if sys.platform == "darwin":
            argstr += " --cpu-spmv-backend scipy --no-cpp-mkl"
        return argstr

    def _checkargs(self, kwargs: dict[str, Any]) -> None:
        """Validate and convert hysteresis arguments to TOMMOS parameters."""
        if "Hsteps" not in kwargs:
            raise ValueError("`Hsteps` must be provided.")
        Hsteps = kwargs.pop("Hsteps")

        for field_argument in ("Hmin", "Hmax", "n"):
            if field_argument in kwargs:
                raise RuntimeError(f"Argument `{field_argument}` should not be provided.")

        if not isinstance(Hsteps, Sequence):
            raise ValueError("Argument `Hsteps` must be a sequence of `(Hstart, Hend, n)`.")
        match len(Hsteps):
            case 0:
                raise ValueError(f"List of field sweeps is empty: {Hsteps=}.")
            case 1:
                kwargs.update(_parse_single_field_sweep(Hsteps[0]))
            case _:
                raise NotImplementedError("Multi sweep is currently not implemented.")

    def _write_input_files(self, system: micromagneticmodel.System, **kwargs: Any) -> None:
        """Write input files."""
        mesh = scripts.mesh_data(system, mesh_unit=scripts.DEFAULT_MESH_UNIT)
        np.savez(f"{system.name}.npz", **mesh)
        Path(f"{system.name}.krn").write_text(scripts.krn_script(system), encoding="utf-8")

        p2_arguments = {name: kwargs[name] for name in ("hx", "hy", "hz", "hstart", "hfinal", "hstep")}
        p2_arguments.update(dict(self))
        Path(f"{system.name}.p2").write_text(
            scripts.p2_script(
                system,
                mesh_unit=scripts.DEFAULT_MESH_UNIT,
                **p2_arguments,
            ),
            encoding="utf-8",
        )

    def _call(
        self,
        system: micromagneticmodel.System,
        runner: tommos_adapter.tommos_runner.TommosRunner,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Call runner to launch the simulation."""
        if runner is None:
            runner = TommosRunner()
        runner.call(
            argstr=self.get_argstr(system),
            verbose=verbose,
            total=kwargs.get("n"),
            glob_name=f"hyst_{system.name}/*.vtu",
        )

    def _schedule_commands(
        self, system: micromagneticmodel.System, runner: tommos_adapter.tommos_runner.TommosRunner
    ) -> list[str]:
        """Get the TOMMOS command to add to a schedule script.

        Args:
            system: Micromagnetic system.
            runner: Runner object defining the calculator.

        Returns:
            Command that runs the TOMMOS hysteresis simulation.
        """
        if runner is None:
            runner = TommosRunner()
        return [runner._call(argstr=self.get_argstr(system), dry_run=True)]

    def _read_data(self, system: micromagneticmodel.System) -> None:
        """Read generated data.

        After this function is called, the state `system.m` is updated.

        Args:
            system: Micromagnetic system.
        """
        output_directory = Path(f"hyst_{system.name}")
        indexed_output_files = []
        for output_file in output_directory.glob("state_cfg*.vtu"):
            match = re.fullmatch(r"state_cfg(\d+)_.*\.vtu", output_file.name)
            if match is not None:
                indexed_output_files.append((int(match.group(1)), output_file))

        if not indexed_output_files:
            raise RuntimeError(f"No TOMMOS VTU output found in '{output_directory}'.")

        last_output_file = max(indexed_output_files, key=lambda item: item[0])[1]
        try:
            output_state = pv.read(last_output_file)
        except Exception as error:
            raise RuntimeError(f"Cannot read TOMMOS output '{last_output_file}'.") from error

        if not isinstance(output_state, pv.UnstructuredGrid):
            raise RuntimeError(f"TOMMOS output '{last_output_file}' is not an UnstructuredGrid.")
        if "m" not in output_state.point_data:
            raise RuntimeError(f"TOMMOS output '{last_output_file}' does not contain point data 'm'.")

        magnetization = np.asarray(output_state.point_data["m"])
        expected_shape = (system.m.n_points, 3)
        if magnetization.shape != expected_shape:
            raise RuntimeError(
                f"TOMMOS output magnetization has shape {magnetization.shape}; expected {expected_shape}."
            )
        if not np.all(np.isfinite(magnetization)):
            raise RuntimeError("TOMMOS output magnetization contains non-finite values.")

        matching_mesh = (
            output_state.n_points == system.m.n_points
            and output_state.n_cells == system.m.n_cells
            and np.allclose(
                output_state.points * scripts.DEFAULT_MESH_UNIT,
                system.m.points,
                rtol=1e-7,
                atol=scripts.DEFAULT_MESH_UNIT * 1e-5,
            )
            and np.array_equal(output_state.celltypes, system.m.celltypes)
            and np.array_equal(output_state.cell_connectivity, system.m.cell_connectivity)
        )
        if not matching_mesh:
            raise RuntimeError(f"Mesh in TOMMOS output '{last_output_file}' does not match system.m.")

        system.m.point_data["m"] = magnetization.copy()

        # TODO: update table information
        # system.table = table_from_file("output.csv", x=self._x)

    def schedule_kwargs_setup(self, schedule_kwargs: dict[str, Any]) -> None:
        """Validate and convert hysteresis arguments for a scheduled run.

        Args:
            schedule_kwargs: Arguments containing the starting field protocols ``Hsteps`` as a list of sweeps,
                each in the form of ``(Hstart, Hend, n)``, where the initial and final fields are one-dimensional
                arrays of 3 numbers (expressed in A/m) and the number of field steps ``n`` is an integer.

        Raises:
            ValueError: If arguments are missing or do not define a valid
                collinear field sweep.
        """
        self._checkargs(schedule_kwargs)

    def drive_kwargs_setup(self, drive_kwargs: dict[str, Any]) -> None:
        """Validate and convert hysteresis arguments for a direct run.

        Args:
            drive_kwargs: Arguments containing the starting field protocols ``Hsteps`` as a list of sweeps,
                each in the form of ``(Hstart, Hend, n)``, where the initial and final fields are one-dimensional
                arrays of 3 numbers (expressed in A/m) and the number of field steps ``n`` is an integer.

        Raises:
            ValueError: If arguments are missing or do not define a valid
                collinear field sweep.
        """
        self._checkargs(drive_kwargs)

    def _check_system(self, system: micromagneticmodel.System) -> None:
        """Check that the system is well defined.

        Args:
            system: Micromagnetic system.

        Raises:
            TypeError: If ``system.m`` is not a PyVista unstructured grid.
            RuntimeError: If energy, magnetization, or required grid data are
                missing or invalid.
        """
        if len(system.energy) == 0:
            raise RuntimeError("System's energy is not defined")
        for energy in system.energy:
            if isinstance(energy, mm.Demag) and hasattr(energy, "asymptotic_radius"):
                raise RuntimeError(
                    f"Attribute `asymptotic_radius` is not supported. Given value: {energy.asymptotic_radius}."
                )

        if system.m is None:
            raise RuntimeError(f"Undefined magnetization state in given system {system}.")
        if not isinstance(system.m, pv.UnstructuredGrid):
            raise TypeError("TOMMOS requires system.m to be a pyvista.UnstructuredGrid.")
        if "m" not in system.m.point_data:
            raise RuntimeError("The magnetization array 'm' is missing from system.m.point_data.")
        for name in ("region_id", "Ms"):
            if name not in system.m.cell_data:
                raise RuntimeError(f"The material array '{name}' is missing from system.m.cell_data.")

        magnetization = np.asarray(system.m.point_data["m"])
        expected_shape = (system.m.n_points, 3)
        if magnetization.shape != expected_shape:
            raise RuntimeError(f"The magnetization array 'm' must have shape {expected_shape}.")
        if not np.all(np.isfinite(magnetization)):
            raise RuntimeError("The magnetization array 'm' must contain only finite values.")

        magnetization_norm = np.linalg.norm(magnetization, axis=1)
        if not np.allclose(magnetization_norm, 1, rtol=1e-6, atol=1e-6):
            raise RuntimeError("The magnetization array 'm' must contain unit vectors at every mesh point.")
        if not np.allclose(magnetization, magnetization[0], rtol=1e-12, atol=1e-14):
            raise RuntimeError(
                "TOMMOS adapter currently supports only a uniform initial magnetization; "
                "writing a restart VTU file is not implemented yet."
            )

        if system.T != 0:
            raise RuntimeError(f"`tommos` supports only T=0 K. Given temperature: {system.T} K.")

        try:
            scripts.mesh_data(system, mesh_unit=scripts.DEFAULT_MESH_UNIT)
            scripts.krn_data(system)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"System cannot be translated to TOMMOS input: {error}") from error

    @property
    def _x(self) -> str:
        return "B_hysteresis"


def _parse_single_field_sweep(Hstep: Sequence[tuple, tuple, int]) -> dict[str, Any]:
    """Parse single field sweep.

    Args:
        Hstep: Field sweep parameters as a `(Hstart, Hend, n)`. Fields `Hstart` and `Hend` are sequences
            of three numbers representing the field vectors and integer `n` is the number of field steps.

    Returns:
        Parsed field parameters according to `p2` input file syntax.
    """
    Hstart, Hend, Hn = _check_sweep(Hstep)
    return _get_field_parameters(Hstart, Hend, Hn)


def _check_sweep(Hstep: Sequence[Any]) -> tuple(tuple, tuple, int):
    """Check field sweep structure and return its arguments.

    Args:
        Hstep: Field sweep parameters as a `(Hstart, Hend, n)`. Fields `Hstart` and `Hend` are sequences of three
            numbers representing the start and end field vectors, and integer `n` is the number of field steps.

    Returns:
        Tuple `(Hstart, Hend, n)` where the first two are one-dimensional NumPy arrays of length 3 and `n` is an
        integer.

    Raises:
        ValueError: Given argument `Hstep` does not contain 3 objects.
        ValueError: Number of field steps is not an integer bigger than 2.
    """
    if len(Hstep) != 3:
        raise ValueError(f"Field sweep must have the form `(Hstart, Hend, n)`. Given {Hstep=}")

    Hstart = _check_field_value(Hstep[0], "Hstart")
    Hend = _check_field_value(Hstep[1], "Hend")

    Hn = Hstep[2]
    if not isinstance(Hn, numbers.Integral) or Hn < 2:
        raise ValueError(f"Parameter `n` (number of field step) must be an integer bigger than 2. Got: {Hn}")
    return Hstart, Hend, Hn


def _check_field_value(H: Sequence[numbers.Real], name: str) -> numpy.typing.ArrayLike:
    """Check field value and convert it to NumPy array.

    Args:
        H: Field value to check and convert to NumPy array.
        name: Name of the field (useful for meaningful error messages).

    Returns:
        Checked NumPy array.

    Raises:
        RuntimeError: Conversion of one of the fields to NumPy array failed.
        ValueError: Given field is not convertible to a one-dimensional array of three numerical values.
        ValueError: Given field contain infinite values.
    """
    try:
        H_arr = np.asarray(H, dtype=float)
    except Exception as exc:
        raise RuntimeError(f"Conversion of field `{name}`={H} to NumPy array failed.") from exc
    if H_arr.shape != (3,):
        raise ValueError(
            f"Field `{name}`={H} must be convertible to a one-dimensional array containing three numerical values."
        )
    if not np.all(np.isfinite(H_arr)):
        raise ValueError(f"Field `{name}`={H} must contain only finite values.")
    return H_arr


def _get_field_direction(Hstart: numpy.typing.ArrayLike, Hend: numpy.typing.ArrayLike) -> numpy.typing.ArrayLike:
    """Get unit field direction given starting and final field.

    Args:
        Hstart: Starting external field array as a one-dimensional NumPy array of length 3.
        Hend: Ending external field array as a one-dimensional NumPy array of length 3.

    Returns:
        Unit array pointing from `Hstart` to `Hend`.

    Raises:
        ValueError: Fields `Hstart` and `Hend` are equal.
        ValueError: Fields `Hstart` and `Hend` are not collinear with the origin.

    """
    field_difference = Hend - Hstart
    difference_norm = np.linalg.norm(field_difference)
    if difference_norm == 0:
        raise ValueError("Hstart and Hend must differ.")

    Hstart_norm = np.linalg.norm(Hstart)
    Hend_norm = np.linalg.norm(Hend)
    if Hstart_norm > 0 and Hend_norm > 0 and np.linalg.norm(np.cross(Hstart / Hstart_norm, Hend / Hend_norm)) > 1e-12:
        raise ValueError("Hstart and Hend must be collinear with the origin.")
    return field_difference / difference_norm


def _get_field_parameters(Hstart: numpy.typing.ArrayLike, Hend: numpy.typing.ArrayLike, Hn: int) -> dict[str, Any]:
    """Get field parameters in the right format for the `p2` input file.

    Args:
        Hstart: Starting external field array as a one-dimensional NumPy array of length 3 in A/m.
        Hend: Ending external field array as a one-dimensional NumPy array of length 3 in A/m.
        Hn: Number of field steps.

    Returns:
        Dictionary of values to write to `p2` input files. Field values are understood as polarization in Tesla.
    """
    h = _get_field_direction(Hstart, Hend)
    hstart = np.vdot(h, Hstart) * mm.consts.mu0
    hfinal = np.vdot(h, Hend) * mm.consts.mu0
    return {
        "hx": float(h[0]),
        "hy": float(h[1]),
        "hz": float(h[2]),
        "hstart": float(hstart),
        "hfinal": float(hfinal),
        "hstep": float((hfinal - hstart) / (Hn - 1)),
        "n": int(Hn),
    }
