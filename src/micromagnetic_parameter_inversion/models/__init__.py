"""Neural-network model subpackage (exports MLPRegressor only).

Training/data modules are not imported here, avoiding circular dependencies.
"""

from micromagnetic_parameter_inversion.models.mlp import MLPRegressor

__all__ = ["MLPRegressor"]
