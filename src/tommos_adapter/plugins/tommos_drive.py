"""Read tommos drives with micromagneticdata."""

from __future__ import annotations

import copy
import os
import pathlib
import re

import micromagneticdata as mdata
import numpy as np
import pyvista as pv
import ubermagtable as ut

from tommos_adapter._mesh import _magnetization_without_airbox
from tommos_adapter.plugins.read_table import table_from_file


class TommosDrive(mdata.Drive):
    """Read tommos snapshots as PyVista unstructured grids.

    ``table`` contains one row per available snapshot, in the same order as
    indexing and iteration. ``full_table`` retains every computed field step,
    including steps without a snapshot when using ``mstep``.
    """

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
        if value is not None and value not in self.full_table.data.columns:
            raise ValueError(f"Column value={value!r} does not exist in data.")
        self._x = "B_hysteresis" if value is None else value
        for name in ("_table", "_full_table"):
            if (table := getattr(self, name, None)) is not None:
                # Slices and callbacks may share a Table object with their source.
                table = copy.copy(table)
                table.x = self._x
                setattr(self, name, table)

    @mdata.Drive.use_cache.setter
    def use_cache(self, value: bool) -> None:
        """Clear both table caches when disabling caching."""
        mdata.Drive.use_cache.fset(self, value)
        if not value:
            self._full_table = None

    @property
    def full_table(self) -> ut.Table:
        """Complete scalar output of the source drive, unaffected by slicing.

        Repeated ``config`` values identify steps which did not save a new
        snapshot. They remain here to preserve the full hysteresis curve.
        """
        if self.use_cache and getattr(self, "_full_table", None) is not None:
            return self._full_table
        table = table_from_file(self._table_path, x=getattr(self, "_x", None))
        if self.use_cache:
            self._full_table = table
        return table

    @property
    def table(self) -> ut.Table:
        """Scalar rows corresponding to the available magnetisation snapshots."""
        if self.use_cache and self._table is not None:
            return self._table

        table = copy.copy(self.full_table)
        # tommos increments config only when saving a snapshot. Later rows with
        # the same config describe unsaved states, so only the first row matches.
        rows = table.data.drop_duplicates("config", keep="first").set_index("config", drop=False)
        configurations = [_snapshot_configuration(filename) for filename in self._step_files]
        missing = sorted(set(configurations) - set(rows.index))
        if missing:
            raise RuntimeError(f"tommos snapshots have no table rows for configurations {missing}.")
        table.data = rows.loc[configurations].reset_index(drop=True)
        if self.use_cache:
            self._table = table
        return table

    @property
    def _step_files(self) -> list[pathlib.Path]:
        # An empty slice has a cached empty table; do not rediscover all files.
        if self.use_cache and (self._table is not None or self._step_file_list):
            return self._step_file_list
        files = sorted(self._step_file_glob, key=_snapshot_configuration)
        if self.use_cache:
            self._step_file_list = files
        return files

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
        result = initial_state.copy(deep=True)
        result.point_data["m"] = _magnetization_without_airbox(initial_state, state)
        return result


def _snapshot_configuration(filename: pathlib.Path) -> int:
    match = re.fullmatch(r"state_cfg(\d+)_.*\.vtu", filename.name)
    if match is None:
        raise RuntimeError(f"Cannot determine tommos snapshot configuration from '{filename}'.")
    return int(match.group(1))


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
