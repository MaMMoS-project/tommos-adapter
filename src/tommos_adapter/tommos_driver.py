"""Define drivers for tommos.

The only available driver is a `HysteresisDriver` as the only capability of `tommos` is to run hysteresis loops.
"""

from pathlib import Path

import pyvista as pv
from micromagneticmodel import adapter_base

import tommos_adapter


class HysteresisDriver(adapter_base.ExternalDriver):
    """Driver to run a hysteresis loop.

    Attributes:
        input_krn: Input `krn` file.
        input_p2: Input `p2` file.
        mesh: Input mesh in `npz` format.

    Examples:
        1. Defining driver with a keyword argument.

        >>> from pathlib import Path
        >>> import tommos_adapter as ta
        ...
        >>> hd = ta.HysteresisDriver(mesh_path=Path("cube.npz").resolve())

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

    _allowed_attributes = [
        "mesh_path",
        "krn_path",
        "p2_path",
    ]

    def _checkargs(self, kwargs):
        pass  # TODO: checkargs

    def _write_input_files(self, system, **kwargs):
        """Write input files."""
        tommos_adapter.scripts.write_mesh(self, system, **kwargs)
        tommos_adapter.scripts.write_krn(self, system, **kwargs)
        tommos_adapter.scripts.write_p2(self, system, **kwargs)

    def _call(self, system, runner, verbose=1, **kwargs):
        if runner is None:
            runner = tommos_adapter.tommos_runner.TommosRunner()
        runner.call(
            argstr=system.name,
            verbose=verbose,
            total=kwargs.get("n"),
            glob_name=f"{system.name}*.omf",
        )

    def _schedule_commands(self, system, runner):
        """Get command to add to the schedule script.

        Python is used to test/simulate schedule during tests because there typically is no scheduling system
        and Python is always available. Therefore, we return a Python comment that can be added to the schedule
        script without breaking the execution.
        """
        if runner is None:
            runner = tommos_adapter.TommosRunner()
        return [
            "# calculator-specific setup, e.g. setting environment variables",
            "# " + runner._call(argstr=self._inputfilename(system), dry_run=True),
        ]

    def _read_data(self, system):
        """Read generated data.

        After this function is called, the following attributes are updated:
        - system.m
        """
        output_files = Path(f"hyst_{system.name}").glob("*.vtu")
        last_output_file = sorted(output_files)[-1]
        system.m = pv.read(last_output_file)

        # update table information
        # system.table = table_from_file("output.csv", x=self._x)  # TODO: update table

    def schedule_kwargs_setup(self, schedule_kwargs):
        """HysteresisDriver takes no special keyword arguments."""
        pass

    def drive_kwargs_setup(self, drive_kwargs):
        """MinDriver takes no special keyword arguments."""
        pass

    def _check_system(self, system):
        """Check that system.energy is defined."""
        pass  # TODO: reintroduce system checks

    @property
    def _x(self):
        return "B_hysteresis"
