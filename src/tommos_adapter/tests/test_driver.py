import tommos_adapter as ta


def test_hysteresis_drive(system, mesh_path, krn_path, p2_path, tmp_path):
    """Test hysteresis drive."""
    hd = ta.HysteresisDriver(mesh_path=mesh_path, krn_path=krn_path, p2_path=p2_path)
    hd.drive(system, dirname=tmp_path)
