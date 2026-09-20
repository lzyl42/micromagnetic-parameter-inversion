"""Neural-network model subpackage.

Contains the MLP and 1D CNN inverse-regression models (see
``micromagnetic_parameter_inversion.models.mlp`` / ``.cnn1d``). This package
only imports and re-exports; it has no side effects and does not import
training/data modules here, avoiding circular dependencies.
"""

from micromagnetic_parameter_inversion.models.cnn1d import CNN1DRegressor
from micromagnetic_parameter_inversion.models.mlp import MLPRegressor

__all__ = ["CNN1DRegressor", "MLPRegressor"]
