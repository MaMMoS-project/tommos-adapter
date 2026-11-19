"""Test convenience functions for creating and reading meshes."""

import numpy as np
import pytest

import tommos_adapter as ta


@pytest.mark.parametrize(
    "geom, extent, h, out_name, additional_mesher_arguments",
    [("box", [1, 2, 3], 1, "box.vtu", "--backend grid"), ("ellipsoid", [4, 5, 6], 2, "ellip", "--n 3")],
)
def test_create_mesh_call_tommos(tmp_path, geom, extent, h, out_name, additional_mesher_arguments):
    """Test call to tommos via `create_mesh` with possible different arguments."""
    ta.create_mesh(
        geom, extent, h=h, out_name=tmp_path / out_name, additional_mesher_arguments=additional_mesher_arguments
    )
    mesh_path = (tmp_path / out_name).with_suffix(".vtu")
    assert mesh_path.is_file()


def test_create_mesh_wrong_geom():
    with pytest.raises(ValueError, match="Argument `geom` must be one of"):
        ta.create_mesh(geom="not_supported", extent=(1, 1, 1), h=1, out_name="test")


@pytest.mark.parametrize("extent", ["string", (1, 2), (5, 4, 3 + 2j), (1, 4, -1), (10, np.nan, 3)])
def test_create_mesh_wrong_extent(extent):
    with pytest.raises(ValueError, match="Argument `extent` must be an array-like of length 3 of finite positive"):
        ta.create_mesh(geom="box", extent=extent, h=1, out_name="test")


@pytest.mark.parametrize("h", ["string", (1, 2), 3 + 2j, np.nan, np.inf])
def test_create_mesh_wrong_h(h):
    with pytest.raises(ValueError, match="Argument `h` must be a positive finite real number"):
        ta.create_mesh(geom="box", extent=(20, 20, 20), h=h, out_name="test")


@pytest.mark.parametrize("additional_mesher_arguments", ["--nonexistent-variable error", "--nonexistent-flag"])
def test_create_mesh_wrong_additional_mesher_arguments(additional_mesher_arguments):
    """Check that not recognized additional arguments for tommos mesh raise an error."""
    with pytest.raises(
        RuntimeError, match=f"tommos mesh: error: unrecognized arguments: {additional_mesher_arguments}"
    ):
        ta.create_mesh(
            geom="box",
            extent=(20, 20, 20),
            h=1,
            out_name="test",
            additional_mesher_arguments=additional_mesher_arguments,
        )


def test_create_mesh_process_fails():
    """Check that stderr of failing process is printed."""
    with pytest.raises(RuntimeError, match=r"tommos mesh: error: argument --geom: invalid choice"):
        ta.create_mesh(geom="box", extent=(1, 1, 1), h=1, out_name="test", additional_mesher_arguments="--geom error")


@pytest.mark.parametrize("mesh_unit_length, mat_id, mat_id_name_mapping", [(1e-6, 3, None), (1e-9, 1, {1: "region_1"})])
def test_read_mesh_possible_args(mesh_cube_20, mesh_unit_length, mat_id, mat_id_name_mapping):
    """Test `read_mesh` with different possible arguments."""
    mesh_cube_20.cell_data["mat_id"] = np.full(mesh_cube_20.n_cells, mat_id, dtype=np.int32)
    mesh_cube_20 = ta.read_mesh(
        mesh_cube_20, mesh_unit_length=mesh_unit_length, mat_id_name_mapping=mat_id_name_mapping
    )
    assert mesh_cube_20.field_data["mesh_unit_length"] == mesh_unit_length
    np.testing.assert_equal(mesh_cube_20.cell_data["mat_id"], mat_id)
    np.testing.assert_equal(mesh_cube_20.field_data["mat_ids"], [mat_id])
    expected_mat_names = [f"grain_{mat_id}"] if mat_id_name_mapping is None else list(mat_id_name_mapping.values())
    assert all(mesh_cube_20.field_data["mat_names"] == expected_mat_names)


def test_read_mesh_wrong_mat_id(mesh_cube_20):
    """We modify the cell data `mat_id` in the fixture mesh to check that different dtypes are not accepted."""
    mesh_cube_20.cell_data["mat_id"] = np.ones(mesh_cube_20.n_cells, np.float64)
    with pytest.raises(ValueError, match="Cell data 'mat_id' should be an array of integers"):
        ta.read_mesh(mesh_cube_20)


def test_no_mat_id_warning(mesh_cube_20):
    del mesh_cube_20.cell_data["mat_id"]
    with pytest.warns(UserWarning, match="Given mesh does not have a 'mat_id' field data"):
        ta.read_mesh(mesh_cube_20)


def test_read_mesh_multigrain(mesh_cube_20):
    """We simulate a multigrain material by modifying a cell to have a different `mat_id`."""
    mesh_cube_20.cell_data["mat_id"][0] = mesh_cube_20.cell_data["mat_id"][0] + 1
    mesh_cube_20 = ta.read_mesh(mesh_cube_20)
    assert all(mesh_cube_20.field_data["mat_names"] == ["grain_1", "grain_2"])


@pytest.mark.parametrize("mesh_unit_length", ["string", np.array(1e-9), -1, 0, np.nan])
def test_read_mesh_wrong_mesh_unit_length(mesh_cube_20, mesh_unit_length):
    """Test failing values of `mesh_unit_length`."""
    with pytest.raises(ValueError, match="`mesh_unit_length` must be a positive finite real number"):
        ta.read_mesh(mesh_cube_20, mesh_unit_length=mesh_unit_length)


@pytest.mark.parametrize("mat_id_name_mapping", [{2: "grain_2"}, {"1": "grain_1"}, {1: "grain_1", 2: "grain_2"}])
def test_read_mesh_wrong_mat_id_name_mapping(mesh_cube_20, mat_id_name_mapping):
    """Test failing values of `mat_id_name_mapping`."""
    with pytest.raises(
        ValueError, match="Inconsistency between detected 'mat_ids' field data and given `mat_id_name_mapping`"
    ):
        ta.read_mesh(mesh_cube_20, mat_id_name_mapping=mat_id_name_mapping)


@pytest.mark.parametrize("geom, n_grains", [("poly", 7), ("poly_gb", 4)])
def test_create_mesh_multigrain(tmp_path, geom, n_grains):
    """Test multigrain geometries.

    For "poly_gb" we expect a grain boundary to appear in the `mat_names` field data.
    """
    mesh = ta.create_mesh(
        geom=geom, extent=[10, 10, 10], h=2, out_name=tmp_path / "cube", additional_mesher_arguments=f"--n {n_grains}"
    )
    assert len(mesh.field_data["mat_ids"]) == n_grains + (geom == "poly_gb")
    expected_mat_names = [f"grain_{i}" for i in range(1, n_grains + 1)]
    if geom == "poly_gb":
        expected_mat_names.append("grain_boundary")
    assert all(expected_mat_names == mesh.field_data["mat_names"])


def test_create_mesh_multigrain_mat_id_name_mapping(tmp_path):
    mapping = {1: "grain_1", 2: "material_2", 3: "3", 4: 4}
    mesh = ta.create_mesh(
        geom="poly",
        extent=[10, 10, 10],
        h=2,
        out_name=tmp_path / "cube",
        additional_mesher_arguments="--n 4",
        mat_id_name_mapping=mapping,
    )
    assert all(mesh.field_data["mat_names"] == list(mapping.values()))
