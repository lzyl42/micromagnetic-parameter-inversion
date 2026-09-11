"""神经网络模型子包（仅导出 MLPRegressor）。

不在此引入训练/数据模块，避免环形依赖。
"""

from micromagnetic_parameter_inversion.models.mlp import MLPRegressor

__all__ = ["MLPRegressor"]
