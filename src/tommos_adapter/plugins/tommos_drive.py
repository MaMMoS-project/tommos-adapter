"""Read tommos drives with micromagneticdata."""

from __future__ import annotations

import os
import pathlib

import micromagneticdata as mdata
import pyvista as pv

from tommos_adapter._mesh import _magnetization_at_reference_points, _read_magnetization_grid


class TommosDrive(mdata.Drive):
    """Provide access to tommos simulation results."""

    @property
    def x(self) -> str:
        """Independent table variable name."""
        return self._x

    @x.setter
    def x(self, value: str | None) -> None:
        """Set the independent table variable.

        Args:
            value: Table column name. Defaults to ``B_hysteresis``.

        Raises:
            ValueError: If the requested column does not exist.
        """
        if value is None:
            self._x = "B_hysteresis"
        else:
            old_value = getattr(self, "_x", None)
            self._x = value
            if value not in self.table.data.columns:
                if old_value is None:
                    del self._x
                else:
                    self._x = old_value
                raise ValueError(f"Column value={value!r} does not exist in data.")

    @property
    def _table_path(self):
        return self.drive_path / f"hyst_{self.name}" / "mammos_hysteresis.csv"

    @property
    def _step_file_glob(self):
        return (self.drive_path / f"hyst_{self.name}").glob("state_cfg*.vtu")

    @property
    def _m0_path(self):
        return self.drive_path / "m0.vtu"

    @property
    def calculator_script(self) -> str:
        """Return the tommos P2 configuration."""
        return (self.drive_path / f"{self.name}.p2").read_text(encoding="utf-8")

    def _read_m(self, filename: str | os.PathLike) -> pv.UnstructuredGrid:
        """Read a complete magnetization grid without the tommos airbox."""
        output_grid = _read_magnetization_grid(filename)
        if pathlib.Path(filename) == self._m0_path:
            return output_grid

        reference_grid = _read_magnetization_grid(self._m0_path)
        reference_points = reference_grid.points
        reference_grid.point_data["m"] = _magnetization_at_reference_points(reference_points, output_grid)
        return reference_grid
