"""CPU unit tests for CNN1DRegressor (offline, synthetic data, no GPU / no training loop).

Covers the P2 conventions: structure/shape, the pulse-major channel encoding of
the real forward, non-contiguous input, constructor-argument validation, forward
shape validation, finite forward and gradients, state_dict round-trip, same-seed
initialization, module export, and the final linear output. All values are
**unit-test-only small numbers**; no real data is read, no GPU is touched, no
training loop is run, and no global torch state is modified (``fork_rng`` is used
to isolate when necessary).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from micromagnetic_parameter_inversion import models
from micromagnetic_parameter_inversion.models import CNN1DRegressor

# unit-test-only structure/values (not research hyperparameters, not corresponding
# to a research config under configs/).
_UNIT_SHAPE = (2, 6, 3)  # (P, T, 3)
_UNIT_CHANNELS = (3, 4)
_UNIT_KERNELS = (3, 5)
_UNIT_POOL_BINS = 2
_UNIT_HEAD = (5,)


def _bad(value: object) -> Any:
    """Hand deliberately illegal argument values to runtime validation outside the type
    checker (a small amount of cast, not abused)."""
    return cast(Any, value)


def _make_model(
    *,
    input_shape: tuple[int, int, int] = _UNIT_SHAPE,
    channels: tuple[int, ...] = _UNIT_CHANNELS,
    kernel_sizes: tuple[int, ...] = _UNIT_KERNELS,
    pool_bins: int = _UNIT_POOL_BINS,
    head_hidden_dims: tuple[int, ...] = _UNIT_HEAD,
) -> CNN1DRegressor:
    return CNN1DRegressor(
        input_shape,
        channels=channels,
        kernel_sizes=kernel_sizes,
        pool_bins=pool_bins,
        head_hidden_dims=head_hidden_dims,
    )


def _construct(**overrides: object) -> CNN1DRegressor:
    """Construct a model from legal defaults + overrides; illegal values go through ``_bad``
    to the constructor's validation."""
    kwargs: dict[str, object] = {
        "input_shape": _UNIT_SHAPE,
        "channels": _UNIT_CHANNELS,
        "kernel_sizes": _UNIT_KERNELS,
        "pool_bins": _UNIT_POOL_BINS,
        "head_hidden_dims": _UNIT_HEAD,
    }
    kwargs.update(overrides)
    return CNN1DRegressor(
        _bad(kwargs["input_shape"]),
        channels=_bad(kwargs["channels"]),
        kernel_sizes=_bad(kwargs["kernel_sizes"]),
        pool_bins=_bad(kwargs["pool_bins"]),
        head_hidden_dims=_bad(kwargs["head_hidden_dims"]),
    )


@pytest.mark.parametrize(
    ("input_shape", "head", "batch"),
    [
        ((2, 6, 3), (5,), 4),
        ((3, 8, 3), (), 2),
        ((1, 4, 3), (7, 3), 1),
    ],
    ids=["multi_pulse_single_head", "empty_head", "n1_multi_layer_head"],
)
def test_forward_shape_and_structural_attributes(
    input_shape: tuple[int, int, int], head: tuple[int, ...], batch: int
) -> None:
    model = _make_model(input_shape=input_shape, head_hidden_dims=head)
    assert model.input_shape == input_shape
    assert model.channels == _UNIT_CHANNELS
    assert model.kernel_sizes == _UNIT_KERNELS
    assert model.pool_bins == _UNIT_POOL_BINS
    assert model.head_hidden_dims == head
    assert isinstance(model.features, nn.Sequential)
    assert isinstance(model.pool, nn.AdaptiveAvgPool1d)
    assert isinstance(model.head, nn.Sequential)

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(batch, *input_shape))
    assert out.shape == (batch, 2)
    assert torch.isfinite(out).all()


def test_first_convolution_receives_pulse_major_component_minor_encoding() -> None:
    """In the real forward, the first Conv layer receives channels encoded as ``p*3 + c``
    (pulse-major).

    Expected values are built by hand with indexing/stack to avoid fully replicating
    permute+reshape in the test.
    """
    n_pulse, n_time = _UNIT_SHAPE[0], _UNIT_SHAPE[1]
    model = _make_model()
    model.eval()
    captured: dict[str, Tensor] = {}

    def _capture(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        captured["x"] = inputs[0].detach().clone()

    features = model.features
    assert isinstance(features, nn.Sequential)
    first_conv = features[0]
    assert isinstance(first_conv, nn.Conv1d)
    handle = first_conv.register_forward_pre_hook(_capture)
    try:
        # Give each (p, t, c) a unique identifiable value.
        base = torch.arange(1, n_pulse * n_time * 3 + 1, dtype=torch.float32)
        x = base.reshape(1, n_pulse, n_time, 3)
        with torch.no_grad():
            _ = model(x)
    finally:
        handle.remove()

    expected = torch.stack(
        [
            torch.stack([x[0, pulse, :, comp] for comp in range(3)], dim=0)
            for pulse in range(n_pulse)
        ],
        dim=0,
    ).reshape(3 * n_pulse, n_time)
    assert torch.equal(captured["x"][0], expected)


def test_features_preserve_time_length_and_pool_collapses_to_bins() -> None:
    model = _make_model()
    model.eval()
    shapes: dict[str, tuple[int, ...]] = {}

    def _conv_out(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        shapes["conv"] = tuple(output.shape)

    def _pool_out(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        shapes["pool"] = tuple(output.shape)

    features = model.features
    assert isinstance(features, nn.Sequential)
    first_conv = features[0]
    assert isinstance(first_conv, nn.Conv1d)
    conv_handle = first_conv.register_forward_hook(_conv_out)
    pool = model.pool
    assert isinstance(pool, nn.AdaptiveAvgPool1d)
    pool_handle = pool.register_forward_hook(_pool_out)
    try:
        with torch.no_grad():
            model(torch.randn(2, *_UNIT_SHAPE))
    finally:
        conv_handle.remove()
        pool_handle.remove()

    # stride=1 + odd kernel + padding=(k-1)//2 ⇒ the first conv output length is still T.
    assert shapes["conv"] == (2, _UNIT_CHANNELS[0], _UNIT_SHAPE[1])
    assert shapes["pool"] == (2, _UNIT_CHANNELS[-1], _UNIT_POOL_BINS)


def test_noncontiguous_input_matches_contiguous() -> None:
    model = _make_model()
    model.eval()
    big = torch.randn(2, *_UNIT_SHAPE[:2], 6)  # (N, P, T, 6)
    x = big[
        ..., :3
    ]  # slicing the last dim yields a consistent-shape (N, P, T, 3) non-contiguous tensor
    assert x.shape == (2, *_UNIT_SHAPE)
    assert not x.is_contiguous()
    with torch.no_grad():
        got = model(x)
        expected = model(x.contiguous())
    assert torch.isfinite(got).all()
    assert torch.equal(got, expected)


def test_constructor_normalizes_list_sequences_to_tuple() -> None:
    model = _construct(channels=[3, 4], kernel_sizes=[3, 5], head_hidden_dims=[5])
    assert model.channels == (3, 4)
    assert model.kernel_sizes == (3, 5)
    assert model.head_hidden_dims == (5,)


def test_pool_bins_accepts_one_and_t() -> None:
    assert _construct(pool_bins=1).pool_bins == 1
    assert _construct(pool_bins=_UNIT_SHAPE[1]).pool_bins == _UNIT_SHAPE[1]


_BAD_CTOR_CASES: tuple[tuple[str, dict[str, object]], ...] = (
    ("input_shape_ndim_2", {"input_shape": (2, 6)}),
    ("input_shape_ndim_4", {"input_shape": (2, 6, 3, 1)}),
    ("input_shape_zero", {"input_shape": (0, 6, 3)}),
    ("input_shape_negative", {"input_shape": (2, -1, 3)}),
    ("input_shape_bool", {"input_shape": (True, 6, 3)}),
    ("input_shape_float", {"input_shape": (2.0, 6, 3)}),
    ("input_shape_string_element", {"input_shape": (2, "6", 3)}),
    ("input_shape_component_not_3", {"input_shape": (2, 6, 4)}),
    ("input_shape_scalar", {"input_shape": 5}),
    ("input_shape_none", {"input_shape": None}),
    ("input_shape_string", {"input_shape": "shape"}),
    ("channels_scalar", {"channels": 3}),
    ("channels_none", {"channels": None}),
    ("channels_string", {"channels": "abc"}),
    ("channels_empty", {"channels": ()}),
    ("channels_zero_element", {"channels": (0, 4)}),
    ("channels_negative_element", {"channels": (-1, 4)}),
    ("channels_bool_element", {"channels": (True, 4)}),
    ("channels_float_element", {"channels": (3.0, 4)}),
    ("channels_string_element", {"channels": ("3", 4)}),
    ("kernel_empty", {"kernel_sizes": ()}),
    ("kernel_even", {"kernel_sizes": (2, 4)}),
    ("kernel_zero", {"kernel_sizes": (0, 5)}),
    ("kernel_bool", {"kernel_sizes": (True, 5)}),
    ("kernel_float", {"kernel_sizes": (3.0, 5)}),
    ("kernel_string_element", {"kernel_sizes": ("3", 5)}),
    ("kernel_length_mismatch", {"kernel_sizes": (3,)}),
    ("kernel_none", {"kernel_sizes": None}),
    ("kernel_scalar", {"kernel_sizes": 3}),
    ("pool_zero", {"pool_bins": 0}),
    ("pool_negative", {"pool_bins": -1}),
    ("pool_bool", {"pool_bins": True}),
    ("pool_float", {"pool_bins": 2.0}),
    ("pool_string", {"pool_bins": "2"}),
    ("pool_none", {"pool_bins": None}),
    ("pool_greater_than_t", {"pool_bins": _UNIT_SHAPE[1] + 1}),
    ("head_scalar", {"head_hidden_dims": 5}),
    ("head_none", {"head_hidden_dims": None}),
    ("head_string", {"head_hidden_dims": "5"}),
    ("head_zero_element", {"head_hidden_dims": (0,)}),
    ("head_negative_element", {"head_hidden_dims": (-1,)}),
    ("head_bool_element", {"head_hidden_dims": (True,)}),
    ("head_float_element", {"head_hidden_dims": (2.5,)}),
    ("head_string_element", {"head_hidden_dims": ("5",)}),
)


@pytest.mark.parametrize(
    ("case", "overrides"),
    _BAD_CTOR_CASES,
    ids=[case for case, _ in _BAD_CTOR_CASES],
)
def test_constructor_rejects_invalid_arguments(case: str, overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _construct(**overrides)


def test_forward_rejects_wrong_ndim_or_shape() -> None:
    model = _make_model()
    model.eval()
    n_pulse, n_time = _UNIT_SHAPE[0], _UNIT_SHAPE[1]
    bad_shapes = [
        (2, n_time, 3),  # ndim=3
        (2, n_pulse, n_time),  # ndim=3, missing the component dim
        (2, n_pulse, n_time, 3, 1),  # ndim=5
        (2, n_pulse, n_time, 4),  # component != 3
        (2, n_pulse + 1, n_time, 3),  # P mismatch
        (2, n_pulse, n_time + 1, 3),  # T mismatch
    ]
    for bad_shape in bad_shapes:
        with pytest.raises(ValueError):
            model(torch.randn(*bad_shape))


def test_forward_output_and_gradients_are_finite() -> None:
    model = _make_model()
    model.train()
    out = model(torch.randn(3, *_UNIT_SHAPE))
    assert torch.isfinite(out).all()
    (out**2).sum().backward()

    grads = [(name, param.grad) for name, param in model.named_parameters() if param.requires_grad]
    assert grads  # at least one trainable parameter
    # A gradient may legitimately be exactly 0, but must not be None/NaN/Inf.
    assert all(grad is not None and torch.isfinite(grad).all() for _, grad in grads)


def test_state_dict_roundtrip_via_torch_load(tmp_path: Path) -> None:
    model = _make_model()
    model.eval()
    probe = torch.randn(3, *_UNIT_SHAPE)
    with torch.no_grad():
        expected = model(probe)

    path = tmp_path / "cnn1d_state.pt"
    torch.save(model.state_dict(), path)
    reloaded = _make_model()
    reloaded.eval()
    reloaded.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        got = reloaded(probe)
    assert torch.equal(got, expected)


def test_same_seed_initialization_is_identical() -> None:
    with torch.random.fork_rng(devices=[]):  # isolate the global RNG, leaving no side effects
        torch.manual_seed(1234)
        first = _make_model()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        second = _make_model()

    first_state = first.state_dict()
    assert first_state
    for name, tensor in first_state.items():
        assert torch.equal(tensor, second.state_dict()[name]), name


def test_module_export() -> None:
    assert models.CNN1DRegressor is CNN1DRegressor
    assert "CNN1DRegressor" in models.__all__


def test_output_head_is_linear_without_final_activation() -> None:
    model = _make_model(head_hidden_dims=())
    model.eval()
    head = model.head
    assert isinstance(head, nn.Sequential)
    last = head[-1]
    assert isinstance(last, nn.Linear)
    assert last.bias is not None
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(-3.0)
    with torch.no_grad():
        out = model(torch.randn(5, *_UNIT_SHAPE))
    # The final layer is a linear output (no ReLU): the negative bias is not clipped off,
    # so the output is always -3.
    assert torch.allclose(out, torch.full_like(out, -3.0))
