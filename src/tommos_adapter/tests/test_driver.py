"""Test abstract driver."""

import pytest

from tommos_adapter.tommos_driver import Driver


def test_abstract_instantiation():
    """Test failure of abstract instantiation."""
    with pytest.raises(TypeError):
        Driver()


def test_allowed_attributes():
    """Test allowed attributes of all drivers."""
    assert Driver._allowed_attributes == ["max_iter", "tol_fun", "eps_a"]
