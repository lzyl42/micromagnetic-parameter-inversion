"""Temporal Transformer 的 CPU 单元验证；不运行研究训练。"""

from typing import Any

import pytest
import torch
from torch import Tensor, nn

from micromagnetic_parameter_inversion.models import TemporalTransformerRegressor


def _model(**overrides: Any) -> TemporalTransformerRegressor:
    kwargs: dict[str, Any] = dict(
        input_shape=(2, 7, 3),
        d_model=8,
        nhead=2,
        num_layers=2,
        dim_feedforward=16,
        dropout=0.1,
        head_hidden_dims=(4,),
    )
    kwargs.update(overrides)
    return TemporalTransformerRegressor(**kwargs)


def test_token_layout_and_noncontiguous_input() -> None:
    model = _model().eval()
    x = torch.empty(2, 2, 3, 7).transpose(2, 3)
    for b in range(2):
        for p in range(2):
            for t in range(7):
                for c in range(3):
                    x[b, p, t, c] = b * 1000 + p * 100 + t * 10 + c
    seen: list[Tensor] = []

    def capture(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        seen.append(args[0].detach().clone())

    handle = model.input_projection.register_forward_pre_hook(capture)
    with torch.no_grad():
        actual = model(x)
        expected = model(x.contiguous())
    handle.remove()
    assert not x.is_contiguous()
    golden = torch.tensor(
        [
            [
                [b * 1000 + p * 100 + t * 10 + c for p in range(2) for c in range(3)]
                for t in range(7)
            ]
            for b in range(2)
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(seen[0], golden, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", [(2, 7, 3), (2, 1, 7, 3), (2, 2, 8, 3), (2, 2, 7, 2)])
def test_forward_rejects_wrong_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="input_shape"):
        _model()(torch.zeros(shape))


def test_positional_encoding_algorithm_broadcast_and_buffer() -> None:
    model = _model().eval()
    pe = model.positional_encoding
    assert pe.shape == (1, 7, 8)
    assert not pe.requires_grad
    assert "positional_encoding" in dict(model.named_buffers())
    assert "positional_encoding" not in dict(model.named_parameters())
    assert "positional_encoding" not in model.state_dict()
    golden = torch.empty(7, 8)
    for t in range(7):
        for i in range(4):
            angle = torch.tensor(t / 10000 ** (2 * i / 8))
            golden[t, 2 * i] = angle.sin()
            golden[t, 2 * i + 1] = angle.cos()
    torch.testing.assert_close(pe[0], golden)
    with torch.no_grad():
        model.input_projection.weight.zero_()
        model.input_projection.bias.zero_()
    seen: list[Tensor] = []

    def capture(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        seen.append(args[0].detach())

    handle = model.encoder.register_forward_pre_hook(capture)
    model(torch.zeros(3, 2, 7, 3))
    handle.remove()
    torch.testing.assert_close(seen[0], pe.expand(3, -1, -1))
    assert model.double().positional_encoding.dtype == torch.float64


def test_time_permutation_sensitivity_comes_from_positions() -> None:
    torch.manual_seed(81)
    model = _model(dropout=0.0).eval()
    x = torch.randn(3, 2, 7, 3)
    permutation = torch.tensor([6, 0, 4, 2, 1, 5, 3])
    with torch.no_grad():
        assert not torch.allclose(model(x), model(x[:, :, permutation]), atol=1e-6)
        model.positional_encoding.zero_()
        torch.testing.assert_close(model(x), model(x[:, :, permutation]), rtol=1e-5, atol=1e-6)


def test_finite_forward_backward_and_eval_batch_consistency() -> None:
    torch.manual_seed(13)
    model = _model()
    x = torch.randn(3, 2, 7, 3, requires_grad=True)
    y = model(x)
    assert y.shape == (3, 2) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    model.eval()
    with torch.no_grad():
        together = model(x)
        apart = torch.cat([model(row.unsqueeze(0)) for row in x])
        torch.testing.assert_close(together, apart, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(together, model(x), rtol=0, atol=0)
    assert all(not module.training for module in model.modules())


def test_seed_reproducibility_and_independent_encoder_initialization() -> None:
    torch.manual_seed(27)
    first = _model()
    torch.manual_seed(27)
    second = _model()
    for key, tensor in first.state_dict().items():
        torch.testing.assert_close(tensor, second.state_dict()[key], rtol=0, atol=0)
    torch.manual_seed(28)
    different_seed = _model()
    assert not torch.equal(first.input_projection.weight, different_seed.input_projection.weight)
    one, two = first.encoder.layers
    assert isinstance(one, nn.TransformerEncoderLayer)
    assert isinstance(two, nn.TransformerEncoderLayer)
    assert one.self_attn.in_proj_weight is not None
    assert two.self_attn.in_proj_weight is not None
    for (name, p1), (_, p2) in zip(one.named_parameters(), two.named_parameters(), strict=True):
        assert p1.data_ptr() != p2.data_ptr()
        if p1.ndim > 1:
            assert not torch.equal(p1, p2), name
        elif name.endswith("weight"):
            assert torch.equal(p1, torch.ones_like(p1))
        else:
            assert torch.equal(p1, torch.zeros_like(p1))
    assert not torch.equal(one.self_attn.in_proj_weight, two.self_attn.in_proj_weight)


def test_main_structure_parameter_count_and_modules() -> None:
    model = _model(
        input_shape=(1, 401, 3), d_model=64, nhead=4, dim_feedforward=128, head_hidden_dims=(32,)
    )
    assert sum(p.numel() for p in model.parameters()) == 69474
    assert isinstance(model.encoder.norm, nn.LayerNorm)
    assert model.encoder.enable_nested_tensor is False
    assert isinstance(model.head[1], nn.GELU)
    assert isinstance(model.head[-1], nn.Linear)
    for layer in model.encoder.layers:
        assert isinstance(layer, nn.TransformerEncoderLayer)
        assert layer.norm_first and layer.self_attn.batch_first
        assert layer.activation is torch.nn.functional.gelu
    assert _model(head_hidden_dims=()).head[0].out_features == 2


@pytest.mark.parametrize("field", ["d_model", "nhead", "num_layers", "dim_feedforward"])
@pytest.mark.parametrize("value", [True, 0, -1, 2.0, "2", None])
def test_strict_positive_integers(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _model(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"d_model": 7},
        {"nhead": 3},
        *[{"dropout": value} for value in [True, "0.1", None, -0.1, 1, float("nan"), float("inf")]],
        *[
            {"head_hidden_dims": value}
            for value in [None, 3, "3", (0,), (-1,), (True,), (2.0,), ("2",)]
        ],
        *[
            {"input_shape": value}
            for value in [(1, 3), (1, 3, 2), (True, 3, 3), (1, 0, 3), (1, 3.0, 3)]
        ],
    ],
)
def test_constructor_rejects_invalid_structure(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _model(**overrides)
