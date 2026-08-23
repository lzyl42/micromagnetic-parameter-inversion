"""Micromagnetic parameter inversion package.

Inverse identification of micromagnetic material parameters (Gilbert damping
``alpha`` and uniaxial anisotropy ``Ku``) from multi-excitation magnetization
dynamics generated with MuMax3, using MLP / 1D CNN / Temporal Transformer
models.
"""

from micromagnetic_parameter_inversion.paths import data_root, output_root
from micromagnetic_parameter_inversion.runtime import get_runtime_info, select_device, smoke_test

__all__ = [
    "data_root",
    "get_runtime_info",
    "output_root",
    "select_device",
    "smoke_test",
]

__version__ = "0.1.0"
