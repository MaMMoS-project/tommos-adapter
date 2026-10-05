import tommos_adapter


def test_version():
    """Check that __version__ exists and is a string."""
    assert isinstance(tommos_adapter.__version__, str)
