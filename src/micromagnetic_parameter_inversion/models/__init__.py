"""神经网络模型子包。

当前仅包含 MLP 反演回归模型（见
``micromagnetic_parameter_inversion.models.mlp``）。本包只做导入与
导出，无任何副作用；不在此引入训练/数据模块，避免环形依赖。
"""

from micromagnetic_parameter_inversion.models.mlp import MLPRegressor

__all__ = ["MLPRegressor"]
