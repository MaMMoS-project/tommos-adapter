"""Test scripts submodule."""

import numpy as np
import pytest


def test_save_npz():
    pass  # TODO: write


@pytest.mark.parametrize(
    "Js,A,K,theta,phi",
    [
        (1, 2, 3, np.pi / 4, np.pi / 3),
        (1.76, 1e-12, 4e5, 0, 0),
    ],
)
def test_write_krn(Js, A, K, theta, phi):
    pass  # TODO: write


def test_write_p2():
    pass  # TODO: write
