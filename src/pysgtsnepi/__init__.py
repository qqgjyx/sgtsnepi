"""PySGtSNEpi: Pure Python implementation of SG-t-SNE-Π."""

from importlib.metadata import version

from pysgtsnepi.api import sgtsnepi
from pysgtsnepi.estimator import SGtSNEpi

__version__ = version("pysgtsnepi")

__all__ = ["SGtSNEpi", "__version__", "sgtsnepi"]
