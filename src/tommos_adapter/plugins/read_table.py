"""Read TOMMOS tabular output."""

from __future__ import annotations

import os

import mammos_entity as me
import ubermagtable as ut

_COLUMN_RENAMES = {
    "B_ext_T": "B_hysteresis",
    "J_par_T": "J_par",
}


def table_from_file(
    filename: str | os.PathLike,
    /,
    x: str | None = None,
    rename: bool = True,
) -> ut.Table:
    """Read a MaMMoS CSV file as an Ubermag table.

    See the
    `TOMMOS output documentation <https://github.com/MaMMoS-project/tommos#5-output>`_
    for details on the tabular output. Units are taken from the file metadata
    rather than hard-coded, and additional columns are preserved. With
    ``rename=True``, known TOMMOS column names are converted to Ubermag-compatible
    names.

    Args:
        filename: MaMMoS CSV file produced by TOMMOS.
        x: Independent-variable column name.
        rename: Whether to apply known Ubermag column names.

    Returns:
        Table containing all CSV columns, units, and MaMMoS metadata.
    """
    collection = me.from_csv(filename)
    data = collection.to_dataframe()
    metadata = collection.metadata()

    if rename:
        data = data.rename(columns=_COLUMN_RENAMES)
        metadata = {_COLUMN_RENAMES.get(name, name): values for name, values in metadata.items()}

    units = {name: values.get("unit", "") for name, values in metadata.items()}
    attributes = {
        "mammos": {
            "description": collection.description,
            "metadata": metadata,
        }
    }
    return ut.Table(data=data, units=units, x=x, attributes=attributes)
