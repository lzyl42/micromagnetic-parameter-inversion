"""models.mlp 单元测试：CPU 合成张量，最小 forward/backward 验证。

覆盖：``[N,P,T,3]`` 展平 ``D = P*T*3``、输出
``[N,2]`` 且 batch 维保留、hidden_dims 配置生效、构造/前向形状校验、
参数梯度有限、``input_shape``/``hidden_dims`` 字段保留（checkpoint
结构字段来源）与 state_dict 往返一致。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from micromagnetic_parameter_inversion.models import MLPRegressor


def _make_model(
    n_pulse: int = 2,
    n_time: int = 5,
    hidden_dims: tuple[int, ...] = (8, 4),
) -> MLPRegressor:
    torch.manual_seed(42)
    return MLPRegressor(input_shape=(n_pulse, n_time, 3), hidden_dims=hidden_dims)


def test_forward_output_shape_and_batch_preserved() -> None:
    model = _make_model()
    assert model(torch.randn(3, 2, 5, 3)).shape == (3, 2)
    assert model(torch.randn(1, 2, 5, 3)).shape == (1, 2)


def test_flatten_dimension_is_product_and_batch_dim_kept() -> None:
    model = _make_model(n_pulse=2, n_time=5, hidden_dims=(7,))
    first_linear = model.network[1]
    assert isinstance(first_linear, nn.Linear)
    assert first_linear.in_features == 2 * 5 * 3  # D = P*T*3
    assert first_linear.out_features == 7
    last_linear = model.network[3]
    assert isinstance(last_linear, nn.Linear)
    assert last_linear.in_features == 7
    assert last_linear.out_features == 2
    # Flatten(start_dim=1) 保留 batch 维：[N,P,T,3] → [N,2]。
    x = torch.randn(4, 2, 5, 3)
    assert model(x).shape == (4, 2)


def test_hidden_dims_configurable() -> None:
    model = _make_model(hidden_dims=(16, 8, 4))
    # Flatten + 3×(Linear, ReLU) + 末层 Linear = 8 个子模块。
    assert len(model.network) == 8
    linears = [m for m in model.network if isinstance(m, nn.Linear)]
    assert [m.in_features for m in linears] == [2 * 5 * 3, 16, 8, 4]
    assert linears[-1].out_features == 2


def test_forward_rejects_wrong_shape() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="input_shape"):
        model(torch.randn(3, 2, 6, 3))  # T 与 input_shape 不一致
    with pytest.raises(ValueError, match="input_shape"):
        model(torch.randn(3, 2, 5))  # 维度不足
    with pytest.raises(ValueError, match="input_shape"):
        model(torch.randn(2, 2, 5, 4))  # 通道维不一致


def test_constructor_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="通道"):
        MLPRegressor(input_shape=(2, 5, 4))
    with pytest.raises(ValueError, match="正"):
        MLPRegressor(input_shape=(0, 5, 3))
    with pytest.raises(ValueError, match="正"):
        MLPRegressor(input_shape=(2, 5, 3), hidden_dims=(8, 0, 4))


def test_backward_gradients_are_finite() -> None:
    model = _make_model()
    x = torch.randn(4, 2, 5, 3)
    target = torch.randn(4, 2)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_structure_fields_preserved_for_checkpoint() -> None:
    model = MLPRegressor(input_shape=(3, 7, 3), hidden_dims=(9,))
    assert model.input_shape == (3, 7, 3)
    assert model.hidden_dims == (9,)


def test_state_dict_roundtrip_gives_identical_outputs() -> None:
    model = _make_model()
    clone = MLPRegressor(input_shape=model.input_shape, hidden_dims=model.hidden_dims)
    clone.load_state_dict(model.state_dict())
    x = torch.randn(2, 2, 5, 3)
    assert torch.allclose(model(x), clone(x))
