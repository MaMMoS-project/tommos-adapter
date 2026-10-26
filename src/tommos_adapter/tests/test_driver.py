"""Test abstract driver."""

import pytest

from tommos_adapter.tommos_driver import Driver


def test_abstract_instantiation():
    """Test failure of abstract instantiation."""
    with pytest.raises(TypeError):
        Driver()
