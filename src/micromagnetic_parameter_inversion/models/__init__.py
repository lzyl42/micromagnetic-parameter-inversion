"""神经网络模型子包。

包含 MLP、1D CNN 与 Temporal Transformer 反演回归模型（见
``.mlp`` / ``.cnn1d`` / ``.transformer``）。本包只做
导入与导出，无任何副作用；不在此引入训练/数据模块，避免环形依赖。
"""

from micromagnetic_parameter_inversion.models.cnn1d import CNN1DRegressor
from micromagnetic_parameter_inversion.models.mlp import MLPRegressor
from micromagnetic_parameter_inversion.models.transformer import TemporalTransformerRegressor

__all__ = ["CNN1DRegressor", "MLPRegressor", "TemporalTransformerRegressor"]
