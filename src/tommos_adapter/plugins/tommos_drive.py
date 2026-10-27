"""Read TOMMOS drives with micromagneticdata."""

from __future__ import annotations

import os

import micromagneticdata as mdata
import numpy as np
import pyvista as pv

from tommos_adapter import scripts


class TommosDrive(mdata.Drive):
    """TOMMOS-specific drive returning PyVista unstructured grids."""

    @mdata.AbstractDrive.x.setter
    def x(self, value: str | None) -> None:
        """Set the independent table variable.

        Args:
            value: Table column name. Defaults to ``B_hysteresis``.

        Raises:
            ValueError: If the requested column does not exist.
        """
        if value is None:
            self._x = "B_hysteresis"
            return

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
        """Return the TOMMOS P2 configuration."""
        return (self.drive_path / f"{self.name}.p2").read_text(encoding="utf-8")

    def _read_state(self, filename: str | os.PathLike, *, initial: bool = False) -> pv.UnstructuredGrid:
        """Read a complete state in SI coordinates.

        TOMMOS snapshots store calculator-scaled coordinates and do not retain
        all metadata from the user grid. For snapshots, the initial state is
        therefore used as a reference and only the computed magnetisation is
        copied from the TOMMOS output.

        Args:
            filename: VTU state file.
            initial: Whether the file is the initial state.

        Returns:
            Complete unstructured grid in SI units.

        Raises:
            RuntimeError: If the VTU data is missing, invalid, or incompatible
                with the initial grid.
        """
        state = _read_unstructured_grid(filename)
        _validate_magnetisation(state, filename)
        if initial:
            return state

        initial_state = _read_unstructured_grid(self._m0_path)
        if not _matching_mesh(initial_state, state):
            raise RuntimeError(f"Mesh in TOMMOS output '{filename}' does not match m0.vtu.")

        result = initial_state.copy(deep=True)
        result.point_data["m"] = np.asarray(state.point_data["m"]).copy()
        return result


def _read_unstructured_grid(filename: str | os.PathLike) -> pv.UnstructuredGrid:
    try:
        state = pv.read(filename)
    except Exception as error:
        raise RuntimeError(f"Cannot read TOMMOS state '{filename}'.") from error
    if not isinstance(state, pv.UnstructuredGrid):
        raise RuntimeError(f"TOMMOS state '{filename}' is not an UnstructuredGrid.")
    return state


def _validate_magnetisation(state: pv.UnstructuredGrid, filename: str | os.PathLike) -> None:
    if "m" not in state.point_data:
        raise RuntimeError(f"TOMMOS state '{filename}' does not contain point data 'm'.")
    magnetisation = np.asarray(state.point_data["m"])
    if magnetisation.shape != (state.n_points, 3):
        raise RuntimeError(
            f"TOMMOS state '{filename}' has magnetisation shape {magnetisation.shape}; expected {(state.n_points, 3)}."
        )
    if not np.all(np.isfinite(magnetisation)):
        raise RuntimeError(f"TOMMOS state '{filename}' contains non-finite magnetisation values.")


def _matching_mesh(reference: pv.UnstructuredGrid, state: pv.UnstructuredGrid) -> bool:
    return (
        state.n_points == reference.n_points
        and state.n_cells == reference.n_cells
        and np.allclose(
            state.points * scripts.DEFAULT_MESH_UNIT,
            reference.points,
            rtol=1e-7,
            atol=scripts.DEFAULT_MESH_UNIT * 1e-5,
        )
        and np.array_equal(state.celltypes, reference.celltypes)
        and np.array_equal(state.cell_connectivity, reference.cell_connectivity)
    )
