"""Define drivers for tommos.

The only available driver is a `HysteresisDriver` as the only capability of `tommos` is to run hysteresis loops.
"""

from __future__ import annotations

import numbers
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import micromagneticmodel as mm
import numpy as np
import pyvista as pv

from tommos_adapter import scripts
from tommos_adapter.plugins import table_from_file
from tommos_adapter.tommos_runner import TommosRunner

if TYPE_CHECKING:
    import micromagneticmodel
    import numpy.typing

    import tommos_adapter


def _magnetization_without_airbox(system: micromagneticmodel.System, output_state: pv.UnstructuredGrid) -> np.ndarray:
    """Return the output magnetization at the points of ``system.m``.

    With ``--add-shell`` tommos adds an airbox and reorders the points, but keeps the
    coordinates of the original ones. Delete this function once tommos writes its output
    without the airbox.
    """
    knt = scripts.mesh_data(system, mesh_unit=scripts.DEFAULT_MESH_UNIT)["knt"]
    output_points = np.asarray(output_state.points)
    # tommos writes float32 coordinates, so the input points are rounded the same way.
    _, key = np.unique(np.vstack([knt.astype(output_points.dtype), output_points]), axis=0, return_inverse=True)
    key = key.ravel()
    output_index = np.full(key.max() + 1, -1)
    output_index[key[len(knt) :]] = np.arange(len(output_points))
    point_map = output_index[key[: len(knt)]]
    if np.any(point_map < 0) or len(np.unique(point_map)) != len(knt):
        raise RuntimeError("Mesh in tommos output does not match system.m.")
    return np.asarray(output_state.point_data["m"])[point_map]


class HysteresisDriver(mm.adapter_base.ExternalDriver):
    """Driver to run a hysteresis loop.

    The field sweep is defined when calling :meth:`drive` or :meth:`schedule` using the arguments ``h_direction``,
    an iterable representing the unit vector of the direction of the external field, ``h_start``, the initial magnitude,
    ``h_final``, the final magnitude``, and ``n``, the number of field steps.

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

    _allowed_attributes = ["mstep", "method", "max_iter", "tol_fun", "eps_a", "num_devices"]

    def get_argstr(self, system: micromagneticmodel.System, **kwargs: Any) -> str:
        """Return string passed to the `tommos` call.

        Args:
            system: Micromagnetic system.
            **kwargs: Optional keyword arguments passed to the runner call.

        Returns:
            Argument string passed to the `tommos loop` runner. It differs depending on the running platform.
        """
        argstr = f"{system.name} --add-shell"
        if nd := kwargs.get("num_devices"):
            if not isinstance(nd, int) or nd < 0:
                raise ValueError(f"Parameter `num_devices` should be a non-negative integer. Got: {nd}")
            argstr += f" --num-devices {nd}"
        return argstr

    def _checkargs(self, kwargs: dict[str, Any]) -> None:
        """Validate and convert hysteresis arguments to TOMMOS parameters."""
        required_arguments = {"h_direction", "h_start", "h_final", "n"}
        if not required_arguments.issubset(kwargs):
            raise ValueError("`h_direction`, `h_start`, `h_final`, and `n` must all be provided.")

        h_direction = _check_direction(kwargs.pop("h_direction"))
        h_start = _check_field_magnitude(kwargs.pop("h_start"), "h_start")
        h_final = _check_field_magnitude(kwargs.pop("h_final"), "h_final")
        n = _check_number_field_steps(kwargs.get("n"))
        h_start_Tesla = h_start * mm.consts.mu0
        h_final_Tesla = h_final * mm.consts.mu0
        h_step_Tesla = (h_final_Tesla - h_start_Tesla) / (n - 1)

        kwargs.update(
            hx=h_direction[0],
            hy=h_direction[1],
            hz=h_direction[2],
            hstart=h_start_Tesla,
            hfinal=h_final_Tesla,
            hstep=h_step_Tesla,
        )

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
            raise RuntimeError(f"No tommos VTU output found in '{output_directory}'.")

        last_output_file = max(indexed_output_files, key=lambda item: item[0])[1]
        try:
            output_state = pv.read(last_output_file)
        except Exception as error:
            raise RuntimeError(f"Cannot read tommos output '{last_output_file}'.") from error

        if not isinstance(output_state, pv.UnstructuredGrid):
            raise RuntimeError(f"tommos output '{last_output_file}' is not an UnstructuredGrid.")
        if "m" not in output_state.point_data:
            raise RuntimeError(f"tommos output '{last_output_file}' does not contain point data 'm'.")

        output_magnetization = np.asarray(output_state.point_data["m"])
        expected_shape = (output_state.n_points, 3)
        if output_magnetization.shape != expected_shape:
            raise RuntimeError(
                f"tommos output magnetization has shape {output_magnetization.shape}; expected {expected_shape}."
            )
        if not np.all(np.isfinite(output_magnetization)):
            raise RuntimeError("tommos output magnetization contains non-finite values.")

        magnetization = _magnetization_without_airbox(system, output_state)

        table_file = output_directory / "mammos_hysteresis.csv"
        try:
            table = table_from_file(table_file, x=self._x)
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"Cannot read tommos table '{table_file}'.") from error

        system.m.point_data["m"] = magnetization.copy()
        system.table = table

    def schedule_kwargs_setup(self, schedule_kwargs: dict[str, Any]) -> None:
        """Validate and convert hysteresis arguments for a scheduled run.

        Args:
            schedule_kwargs: Arguments containing the starting field direction ``h_direction``, initial magnitude
                ``h_start`` in A/m, final magnitude ``h_final`` in A/m, and number of field steps ``n``.

        Raises:
            ValueError: If arguments are missing or do not define a valid
                collinear field sweep.
        """
        self._checkargs(schedule_kwargs)

    def drive_kwargs_setup(self, drive_kwargs: dict[str, Any]) -> None:
        """Validate and convert hysteresis arguments for a direct run.

        Args:
            drive_kwargs: Arguments containing the starting field direction ``h_direction``, initial magnitude
                ``h_start`` in A/m, final magnitude ``h_final`` in A/m, and number of field steps ``n``.

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


def _check_direction(h_direction: Sequence[numbers.Real]) -> numpy.ndarray:
    """Check field value and convert it to normalized NumPy array.

    Args:
        h_direction: Direction array to check and convert to NumPy array.

    Returns:
        Normalized NumPy array.

    Raises:
        RuntimeError: Given direction is not convertible to a numeric NumPy array.
        RuntimeError: Given direction is not a one-dimensional array of three numerical values.
        ValueError: Given direction contains infinite values.
        ValueError: Given direction has zero magnitude.
    """
    try:
        h = np.asarray(h_direction, dtype=float)
    except Exception as exc:
        raise RuntimeError(f"Conversion of direction `h_direction` to NumPy array failed. {h_direction=}") from exc
    if h.shape != (3,):
        raise RuntimeError(
            f"Direction `h_direction` must be a one-dimensional array containing three "
            f"numerical values. {h_direction=}."
        )
    if not np.all(np.isfinite(h)):
        raise ValueError(f"Direction `h_direction` must contain only finite values. Got {h_direction=}")
    if (h_norm := np.linalg.norm(h)) > 0:
        return h / h_norm
    else:
        raise ValueError(f"Direction `h_direction` cannot have zero norm. Got {h_direction=}")


def _check_field_magnitude(h: numbers.Real, name: str) -> float:
    """Check field magnitude and convert it to a float.

    Args:
        h: Field magnitude.
        name: String for error messages.

    Returns:
        Checked field magnitude as a Python float.

    Raises:
        ValueError: Given direction contains infinite values or NaNs.
    """
    if not (isinstance(h, numbers.Real) and np.isfinite(h)):
        raise ValueError(f"Field magnitude `{name}` must be a single finite real number. Got {h}")
    return float(h)


def _check_number_field_steps(n: numbers.Integral) -> int:
    """Check number of field steps and convert it to an int.

    Args:
        n: Number of field steps.

    Returns:
        Checked number of steps as a Python integer.

    Raises:
        ValueError: Given value is not an integer bigger than 2.
    """
    if not isinstance(n, numbers.Integral) or n < 2 or not np.isfinite(n):
        raise ValueError(f"Parameter `n` (number of field steps) must be an integer bigger than 2. Got: {n}")
    return int(n)
