"""Read tommos drives with micromagneticdata."""

from __future__ import annotations

import configparser
import copy
import math
import os
import pathlib

import micromagneticdata as mdata
import pyvista as pv

from tommos_adapter._mesh import _magnetization_without_airbox, _mesh_points, _read_magnetisation_grid
from tommos_adapter.plugins.read_table import table_from_file


class TommosDrive(mdata.Drive):
    """Read tommos snapshots as PyVista unstructured grids."""

    @property
    def x(self) -> str:
        """Independent variable name."""
        return self._x

    @x.setter
    def x(self, value: str | None) -> None:
        """Set the independent table variable.

        Args:
            value: Table column name. Defaults to ``B_hysteresis``.

        Raises:
            ValueError: If the requested column does not exist.
        """
        if value is not None:
            table = self._table if self._table is not None else table_from_file(self._table_path)
            if value not in table.data.columns:
                raise ValueError(f"Column value={value!r} does not exist in data.")
        self._x = "B_hysteresis" if value is None else value
        if self._table is not None:
            # Callbacks may share a Table object with their source drive.
            self._table = copy.copy(self._table)
            self._table.x = self._x

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

    @property
    def _mesh_unit_length(self) -> float:
        """Read the length of one simulation coordinate unit from the P2 file."""
        config = configparser.ConfigParser()
        try:
            config.read_string(self.calculator_script)
            mesh_unit = config.getfloat("mesh", "size")
            if not math.isfinite(mesh_unit) or mesh_unit <= 0:
                raise ValueError("Mesh size must be positive and finite.")
        except (configparser.Error, ValueError) as error:
            raise RuntimeError(f"Invalid mesh unit length in '{self.drive_path / f'{self.name}.p2'}'.") from error
        return mesh_unit

    def _read_m(self, filename: str | os.PathLike) -> pv.UnstructuredGrid:
        """Read a complete magnetisation grid in simulation coordinates.

        The initial grid supplies the mesh and its metadata. The P2 mesh size
        is stored in ``field_data["mesh_unit_length"]`` as metres per coordinate
        unit. Snapshots supply the computed magnetisation.

        Args:
            filename: VTU state file.

        Returns:
            Complete unstructured grid with an explicit mesh unit length.

        Raises:
            RuntimeError: If the VTU data is missing, invalid, or incompatible
                with the initial grid, or the P2 mesh size is invalid.
            ValueError: If the initial grid has an invalid mesh unit length.
        """
        output_grid = _read_magnetisation_grid(filename)
        mesh_unit = self._mesh_unit_length
        if pathlib.Path(filename) == self._m0_path:
            reference_grid = output_grid
        else:
            reference_grid = _read_magnetisation_grid(self._m0_path)
            reference_grid.point_data["m"] = _magnetization_without_airbox(
                reference_grid, output_grid, mesh_unit=mesh_unit
            )
        reference_grid.points = _mesh_points(reference_grid, mesh_unit=mesh_unit)
        reference_grid.field_data["mesh_unit_length"] = mesh_unit
        return reference_grid
