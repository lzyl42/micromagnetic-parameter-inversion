"""神经网络模型子包。

当前仅包含 MLP 反演回归模型（见
``micromagnetic_parameter_inversion.models.mlp``）。本包只做导入与
导出，无任何副作用；不在此引入训练/数据模块，避免环形依赖。
"""

from micromagnetic_parameter_inversion.models.mlp import MLPRegressor

# TODO(CNN1D-P2): 待 CNN1DRegressor 实现并评审通过后，在此导出并加入
# __all__（届时同样只做导入/导出、无副作用）。本轮骨架不导入，
# 保持现有导出与导入行为不变。
__all__ = ["MLPRegressor"]
