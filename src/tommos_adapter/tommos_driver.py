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
        """Check all given argument."""
        # The next steps assume that the axis between Hmax and Hmin always passes via the origin.
        H_array = np.array(kwargs["Hmax"])
        h = H_array / np.linalg.norm(H_array)  # unit vector in the direction of Hmax
        kwargs["hx"] = h[0]
        kwargs["hy"] = h[1]
        kwargs["hz"] = h[2]
        kwargs["hstart"] = np.vdot(h, np.array(kwargs["Hmin"])) * mm.consts.mu0
        kwargs["hfinal"] = np.vdot(h, np.array(kwargs["Hmax"])) * mm.consts.mu0
        kwargs["hstep"] = (kwargs["hfinal"] - kwargs["hstart"]) / (kwargs["n"] - 1)

        # Remove unwanted parameters
        for key in ["Hmin", "Hmax", "n"]:
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
            glob_name=f"{system.name}*.omf",
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
        return [
            "# calculator-specific setup, e.g. setting environment variables",
            "# " + runner._call(argstr=self._inputfilename(system), dry_run=True),
        ]

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
        if (norm := np.linalg.norm(system.m.point_data["m"])) > 0:
            system.m.point_data["m"] /= norm
        else:
            raise RuntimeError(f"Invalid initial magnetization: {system.m.point_data['m']}.")

    @property
    def _x(self) -> str:
        return "B_hysteresis"
