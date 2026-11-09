"""Read tommos drives with micromagneticdata."""

from __future__ import annotations

import copy
import os
import pathlib

import micromagneticdata as mdata
import numpy as np
import pyvista as pv

from tommos_adapter._mesh import _magnetization_without_airbox
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

    def _read_m(self, filename: str | os.PathLike) -> pv.UnstructuredGrid:
        """Read a complete state in SI coordinates.

        tommos snapshots store calculator-scaled coordinates and do not retain
        all metadata from the user grid. For snapshots, the initial state is
        therefore used as a reference and only the computed magnetisation is
        copied from the tommos output.

        Args:
            filename: VTU state file.

        Returns:
            Complete unstructured grid in SI units.

        Raises:
            RuntimeError: If the VTU data is missing, invalid, or incompatible
                with the initial grid.
        """
        state = _read_unstructured_grid(filename)
        _validate_magnetisation(state, filename)
        if pathlib.Path(filename) == self._m0_path:
            return state

        initial_state = _read_unstructured_grid(self._m0_path)
        initial_state.point_data["m"] = _magnetization_without_airbox(initial_state, state)
        return initial_state


def _read_unstructured_grid(filename: str | os.PathLike) -> pv.UnstructuredGrid:
    try:
        state = pv.read(filename)
    except Exception as error:
        raise RuntimeError(f"Cannot read tommos state '{filename}'.") from error
    if not isinstance(state, pv.UnstructuredGrid):
        raise RuntimeError(f"tommos state '{filename}' is not an UnstructuredGrid.")
    return state


def _validate_magnetisation(state: pv.UnstructuredGrid, filename: str | os.PathLike) -> None:
    if "m" not in state.point_data:
        raise RuntimeError(f"tommos state '{filename}' does not contain point data 'm'.")
    magnetisation = np.asarray(state.point_data["m"])
    if magnetisation.shape != (state.n_points, 3):
        raise RuntimeError(
            f"tommos state '{filename}' has magnetisation shape {magnetisation.shape}; expected {(state.n_points, 3)}."
        )
    if not np.all(np.isfinite(magnetisation)):
        raise RuntimeError(f"tommos state '{filename}' contains non-finite magnetisation values.")
