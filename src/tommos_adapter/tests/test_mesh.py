"""Test convenience functions for mesh creation.

For each mesh geometry, we will check that:
- all the mesh points are inside the geometry
- the mesh extremes lay on the geometry boundary
- the mesh volume (evaluated as sum of the volume of all tetrahedrons) are as expected.
"""

import numpy as np
import pytest

import tommos_adapter as ta


@pytest.mark.parametrize("geom", ["non_supported"])
def test_create_mesh_wrong_geom(geom):
    with pytest.raises(ValueError, match="Argument `geom` must be"):
        ta.create_mesh(geom=geom, extent=(1, 1, 1), h=1, out_name="test")


@pytest.mark.parametrize("extent", ["string", (1, 2), (5, 4, 3 + 2j), (1, 4, -1), (10, np.nan, 3)])
def test_create_mesh_wrong_extent(extent):
    with pytest.raises(ValueError, match="Argument `extent` must be"):
        ta.create_mesh(geom="box", extent=extent, h=1, out_name="test")


@pytest.mark.parametrize("h", ["string", (1, 2), 3 + 2j, np.nan, np.inf])
def test_create_mesh_wrong_h(h):
    with pytest.raises(ValueError, match="Argument `h` must be"):
        ta.create_mesh(geom="box", extent=(20, 20, 20), h=h, out_name="test")


def _eval_volume_mesh(mesh):
    """Calculation of tetrahedral mesh volume.

    For each tetrahedron defined from the points {p0, p1, p2, p3}, the volume
    can be evaluated as `V = 1/6 * det(p1-p0, p2-p0, p3-p0)`.
    The determinant can also be written as
    `det(p1-p0, p2-p0, p3-p0) = (p1-p0) · [(p2-p0) × (p3-p0)]`.
    """
    connectivity = mesh.cell_connectivity.reshape(-1, 4)
    p0 = mesh.points[connectivity[:, 0]]
    p1 = mesh.points[connectivity[:, 1]]
    p2 = mesh.points[connectivity[:, 2]]
    p3 = mesh.points[connectivity[:, 3]]
    det = np.einsum("ij,ij", p1 - p0, np.cross(p2 - p0, p3 - p0))
    vols = det / 6
    return vols


@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (5, 5, 5),
        (5, 5, 10),
        (5, 10, 10),
        (5, 10, 7),
    ],
)
def test_mesh_box(tmp_path, monkeypatch, Lx, Ly, Lz):
    """Test volume of mesh 'box'.

    Points (Px, Py, Pz) satisfy:
    - `|Px| <= Lx/2`.
    - `|Py| <= Ly/2`.
    - `|Pz| <= Lz/2`.

    Volume is `Lx * Ly * Lz`.
    """
    monkeypatch.chdir(tmp_path)
    mesh = ta.create_mesh(geom="box", extent=(Lx, Ly, Lz), h=2, out_name="box")

    # all mesh points are inside the box
    assert np.all(np.abs(mesh.points[:, 0]) <= Lx / 2)
    assert np.all(np.abs(mesh.points[:, 1]) <= Ly / 2)
    assert np.all(np.abs(mesh.points[:, 2]) <= Lz / 2)

    # mesh extremes equate the box side lengths
    assert np.isclose(mesh.points[:, 0].max(), Lx / 2)
    assert np.isclose(mesh.points[:, 1].max(), Ly / 2)
    assert np.isclose(mesh.points[:, 2].max(), Lz / 2)
    assert np.isclose(mesh.points[:, 0].min(), -Lx / 2)
    assert np.isclose(mesh.points[:, 1].min(), -Ly / 2)
    assert np.isclose(mesh.points[:, 2].min(), -Lz / 2)

    # evaluate volume
    volume_from_mesh = _eval_volume_mesh(mesh)
    expected_volume = Lx * Ly * Lz
    assert np.isclose(volume_from_mesh, expected_volume, rtol=0.1)


@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (10, 10, 10),
        (10, 10, 20),
        (10, 20, 20),
        (10, 20, 15),
    ],
)
def test_mesh_ellipsoid(tmp_path, monkeypatch, Lx, Ly, Lz):
    """Test volume of mesh 'ellipsoid'.

    Points (Px, Py, Pz) satisfy: `(Px/a)^2 + (Py/b)^2 + (Pz/c))^2 <= 1`.

    Semiaxes (a, b, c) as defined as:
    - `a = (Lx + Ly) / 4`,
    - `b = (Lx + Ly) / 4`,
    - `c = Lz / 4`.

    Volume is `a * b * c * π * 4 / 3`.
    """
    monkeypatch.chdir(tmp_path)
    mesh = ta.create_mesh(geom="ellipsoid", extent=(Lx, Ly, Lz), h=2, out_name="ellipsoid")

    # all mesh points are inside the ellipsoid
    a = (Lx + Ly) / 4
    b = (Lx + Ly) / 4
    c = Lz / 2
    for point in mesh.points:
        _p = point / (a, b, c)
        assert (_p.dot(_p)) <= 1 + 0.1  # 10% relative error

    # mesh extremes are on the ellipsoid surfaces (1% rel error on curves)
    assert np.isclose(mesh.points[:, 0].max(), a, rtol=1e-2)
    assert np.isclose(mesh.points[:, 1].max(), b, rtol=1e-2)
    assert np.isclose(mesh.points[:, 2].max(), c, rtol=1e-2)
    assert np.isclose(mesh.points[:, 0].min(), -a, rtol=1e-2)
    assert np.isclose(mesh.points[:, 1].min(), -b, rtol=1e-2)
    assert np.isclose(mesh.points[:, 2].min(), -c, rtol=1e-2)

    # evaluate volume
    volume_from_mesh = _eval_volume_mesh(mesh)
    expected_volume = a * b * c * np.pi * 4 / 3
    assert np.isclose(volume_from_mesh, expected_volume, rtol=0.1)


@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (5, 5, 5),
        (5, 5, 10),
        (5, 10, 10),
        (5, 10, 7),
    ],
)
def test_mesh_eye(tmp_path, monkeypatch, Lx, Ly, Lz):
    """Test volume of mesh 'eye'.

    Lx is the full eye length.
    Ly is the eye half-height.
    Lz is the polygon width.

    Points (Px, Py, Pz) satisfy:
    - `Py <= - 2 * Ly / Lx^2 * Px^2 + Ly / 2`.
    - `|Pz| <= Lz/2`.

    Volume is `Lx * Ly * Lz * 2 / 3`.
    """
    monkeypatch.chdir(tmp_path)
    mesh = ta.create_mesh(geom="eye", extent=(Lx, Ly, Lz), h=2, out_name="eye")

    # all mesh points are inside the eye prism
    for point in mesh.points:
        eye_border = Ly / 2 - 2 * Ly / (Lx * Lx) * point[0] * point[0]
        assert point[1] <= eye_border * 1.1  # allow 10% relative error
        assert abs(point[2]) <= Lz / 2

    # mesh extremes equate the eye borders (1% rel error on curves)
    assert np.isclose(mesh.points[:, 0].max(), Lx / 2, rtol=1e-2)
    assert np.isclose(mesh.points[:, 1].max(), Ly / 2, rtol=1e-2)
    assert np.isclose(mesh.points[:, 2].max(), Lz / 2)
    assert np.isclose(mesh.points[:, 0].min(), -Lx / 2, rtol=1e-2)
    assert np.isclose(mesh.points[:, 1].min(), -Ly / 2, rtol=1e-2)
    assert np.isclose(mesh.points[:, 2].min(), -Lz / 2)

    # evaluate volume
    volume_from_mesh = _eval_volume_mesh(mesh)
    expected_volume = Lx * Ly * Lz * 2 / 3
    assert np.isclose(volume_from_mesh, expected_volume, rtol=0.1)


@pytest.mark.parametrize(
    "Lx, Ly, Lz",
    [
        (5, 5, 5),
        (5, 5, 10),
        (5, 10, 10),
        (5, 10, 7),
    ],
)
def test_mesh_elliptic_cylinder(tmp_path, monkeypatch, Lx, Ly, Lz):
    """Test volume of mesh 'elliptic_cylinder'.

    Points (Px, Py, Pz) satisfy:
    - `(Px/(Lx/2))^2 + (Py/(Ly/2))^2 <= 1`.
    - `|Pz| <= Lz/2`.

    Volume is `π * (Lx/2) * (Ly/2) * Lz`.
    """
    monkeypatch.chdir(tmp_path)
    mesh = ta.create_mesh(geom="elliptic_cylinder", extent=(Lx, Ly, Lz), h=2, out_name="elliptic_cylinder")

    # all mesh points are inside the cylinder
    for point in mesh.points:
        _xy = point[:2] / (Lx / 2, Ly / 2)
        assert (_xy.dot(_xy)) <= 1 + 1e-5
        assert abs(point[2]) <= Lz / 2

    # mesh extremes equate the cylinder borders (1% rel error on curves)
    assert np.isclose(mesh.points[:, 0].max(), Lx / 2, rtol=1e-2)
    assert np.isclose(mesh.points[:, 1].max(), Ly / 2, rtol=1e-2)
    assert np.isclose(mesh.points[:, 2].max(), Lz / 2)
    assert np.isclose(mesh.points[:, 0].min(), -Lx / 2, rtol=1e-2)
    assert np.isclose(mesh.points[:, 1].min(), -Ly / 2, rtol=1e-2)
    assert np.isclose(mesh.points[:, 2].min(), -Lz / 2)

    # evaluate volume
    volume_from_mesh = _eval_volume_mesh(mesh)
    expected_volume = Lx * Ly * Lz * np.pi / 4
    assert np.isclose(volume_from_mesh, expected_volume, rtol=0.1)
