"""Define drivers for tommos.

The only available driver is a `HysteresisDriver` as the only capability of `tommos` is to run hysteresis loops.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import micromagneticmodel as mm
import numpy as np
import pyvista as pv

from tommos_adapter.scripts import write_input_files
from tommos_adapter.tommos_runner import TommosRunner

if TYPE_CHECKING:
    import micromagneticmodel

    import tommos_adapter


class HysteresisDriver(mm.adapter_base.ExternalDriver):
    """Driver to run a hysteresis loop.

    Examples:
        1. Defining driver with a keyword argument.

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

    _allowed_attributes = []

    def _checkargs(self, kwargs: dict[str, Any]) -> None:
        """Validate and convert hysteresis arguments to TOMMOS parameters."""
        required_arguments = {"Hmin", "Hmax", "n"}
        if not required_arguments.issubset(kwargs):
            raise ValueError("Hmin, Hmax, and n must all be provided.")

        fields = {}
        for name in ("Hmin", "Hmax"):
            try:
                field = np.asarray(kwargs[name], dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must contain three numeric values.") from error
            if field.shape != (3,):
                raise ValueError(f"{name} must contain three numeric values.")
            if not np.all(np.isfinite(field)):
                raise ValueError(f"{name} must contain only finite values.")
            fields[name] = field

        n = kwargs["n"]
        if isinstance(n, bool) or not isinstance(n, int):
            raise ValueError(f"n must be an integer, not {type(n).__name__}.")
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n}.")

        Hmin = fields["Hmin"]
        Hmax = fields["Hmax"]
        field_difference = Hmax - Hmin
        difference_norm = np.linalg.norm(field_difference)
        if difference_norm == 0:
            raise ValueError("Hmin and Hmax must differ.")

        Hmin_norm = np.linalg.norm(Hmin)
        Hmax_norm = np.linalg.norm(Hmax)
        if Hmin_norm > 0 and Hmax_norm > 0:
            relative_cross_product = np.linalg.norm(np.cross(Hmin, Hmax)) / (Hmin_norm * Hmax_norm)
            if not np.isclose(relative_cross_product, 0, atol=1e-12):
                raise ValueError("Hmin and Hmax must be collinear with the origin.")

        h = field_difference / difference_norm
        hstart = float(np.vdot(h, Hmin) * mm.consts.mu0)
        hfinal = float(np.vdot(h, Hmax) * mm.consts.mu0)
        kwargs.update(
            hx=float(h[0]),
            hy=float(h[1]),
            hz=float(h[2]),
            hstart=hstart,
            hfinal=hfinal,
            hstep=(hfinal - hstart) / (n - 1),
        )

        # TOMMOS uses the converted field parameters, but n is retained for progress reporting.
        for key in ["Hmin", "Hmax"]:
            kwargs.pop(key)

        # For the moment symmetric hysteresis simulations or multi-step drivers are not allowed.
        # TODO: Allow symmetric hysteresis simulations?
        # TODO: Allow multi-step hysteresis drivers?

    def _write_input_files(self, system: micromagneticmodel.System, **kwargs: Any) -> None:
        """Write input files."""
        write_input_files(self, system, **kwargs)

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
            argstr=system.name,
            verbose=verbose,
            total=kwargs.get("n"),
            glob_name=f"hyst_{system.name}/*.vtu",
        )

    def _schedule_commands(
        self, system: micromagneticmodel.System, runner: tommos_adapter.tommos_runner.TommosRunner
    ) -> list[str]:
        """Get command to add to the schedule script.

        Python is used to test/simulate schedule during tests because there typically is no scheduling system
        and Python is always available. Therefore, we return a Python comment that can be added to the schedule
        script without breaking the execution.

        Args:
            system: Micromagnetic system.
            runner: Runner object defining the calculator.

        Returns:
            Python comment to add to the schedule script.
        """
        if runner is None:
            runner = tommos_adapter.TommosRunner()
        return [runner._call(argstr=system.name, dry_run=True)]

    def _read_data(self, system: micromagneticmodel.System) -> None:
        """Read generated data.

        After this function is called, the state `system.m` is updated.

        Args:
            system: Micromagnetic system.
        """
        output_files = Path(f"hyst_{system.name}").glob("*.vtu")
        last_output_file = sorted(output_files)[-1]
        old_state = system.m.copy()
        system.m = pv.read(last_output_file)
        system.m.cell_data["Js"] = old_state.cell_data["Js"]

        # TODO: update table information
        # system.table = table_from_file("output.csv", x=self._x)

    def schedule_kwargs_setup(self, schedule_kwargs: dict[str, Any]) -> None:
        """Check argument for schedule.

        Args:
            schedule_kwargs: Passed keyword arguments.
        """
        self._checkargs(schedule_kwargs)

    def drive_kwargs_setup(self, drive_kwargs: dict[str, Any]) -> None:
        """Check argument for drive.

        Args:
            drive_kwargs: Passed keyword arguments.
        """
        self._checkargs(drive_kwargs)

    def _check_system(self, system: micromagneticmodel.System) -> None:
        """Check that the system is well defined.

        Args:
            system: Micromagnetic system.

        Raises:
            RuntimeError: System's energy is not defined.
        """
        if len(system.energy) == 0:
            raise RuntimeError("System's energy is not defined")
        if system.m is None:
            raise RuntimeError(f"Undefined magnetization state in given system {system}.")
        if not isinstance(system.m, pv.UnstructuredGrid):
            raise TypeError("TOMMOS requires system.m to be a pyvista.UnstructuredGrid.")
        if "m" not in system.m.point_data:
            raise RuntimeError("The magnetization array 'm' is missing from system.m.point_data.")
        for name in ("id", "Js"):
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

    @property
    def _x(self) -> str:
        return "B_hysteresis"
