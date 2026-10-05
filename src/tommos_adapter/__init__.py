"""This subpackage provides an Ubermag adapter package for `tommos`."""

import importlib.metadata

from . import scripts as scripts
from .tommos_driver import HysteresisDriver as HysteresisDriver
from .tommos_runner import TommosRunner as TommosRunner

__version__ = importlib.metadata.version(__package__)
