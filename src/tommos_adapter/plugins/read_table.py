"""Read tommos tabular output."""

from __future__ import annotations

import os
import types

import mammos_entity as me
import pandas as pd
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
    `tommos output documentation <https://github.com/MaMMoS-project/tommos#5-output>`_
    for details on the tabular output. With ``rename=True``, known tommos column
    names are converted to shorter names.

    Args:
        filename: MaMMoS CSV file produced by tommos.
        x: Independent-variable column name.
        rename: Whether to apply shorter names to known tommos columns.

    Returns:
        Table containing all CSV columns, units, and MaMMoS metadata.
    """
    collection = me.from_csv(filename)
    mh_file = list(filename.parent.glob("*.mh"))[0]
    df = pd.read_csv(mh_file, sep=" ", names=["B", "J", "mx", "my", "mz", "E"], comment="#")
    collection.mx = me.Entity(
        ontology_label="RatioQuantity",
        value=df.mx,
        description="x component of the spatially averaged normalized magnetization (Mx/Ms)",
    )
    collection.my = me.Entity(
        ontology_label="RatioQuantity",
        value=df.my,
        description="y component of the spatially averaged normalized magnetization (My/Ms)",
    )
    collection.mz = me.Entity(
        ontology_label="RatioQuantity",
        value=df.mz,
        description="z component of the spatially averaged normalized magnetization (Mz/Ms)",
    )
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
    table = ut.Table(data=data, units=units, x=x, attributes=attributes)
    table.to_entity_collection = types.MethodType(to_entity_collection, table)
    return table


def to_entity_collection(self) -> me.EntityCollection:
    """Table method to return an entity collection.

    Returns:
        Entity collection of all the relevant entities from the simulation output.
    """
    return me.EntityCollection.from_dataframe(
        self.data, self.attributes["mammos"]["metadata"], description=self.attributes["mammos"]["description"]
    )
