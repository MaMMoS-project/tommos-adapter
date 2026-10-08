# import tommos_adapter as ta


# def test_hysteresis_drive(mesh, tmp_path):
#     """Test hysteresis drive."""
#     system = mm.System(name="test_drive")
#     state = pv.read(mesh["vtu"])
#     hd = ta.HysteresisDriver()
#     hd.drive(system, dirname=tmp_path)
#     # TODO: create generic tests
