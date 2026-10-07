"""This subpackage provides an Ubermag adapter package for `tommos`."""

import importlib.metadata

from . import scripts as scripts

__version__ = importlib.metadata.version(__package__)
