"""Tests for strict training-config YAML loading (offline, tmp_path only)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from micromagnetic_parameter_inversion import training_config
from micromagnetic_parameter_inversion.models import CNN1DRegressor
from micromagnetic_parameter_inversion.training_config import (
    CNN1DModelConfig,
    ConfigError,
    ModelConfig,
    SplitRatios,
    load_config,
)

_MINIMAL = "dataset_name: demo_ds\nrun_name: run_001\n"

# Real checked-in templates (not duplicated fixtures): the CNN config must stay a
# valid YAML template that mirrors the MLP config and keeps its placeholders.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CNN_TEMPLATE_PATH = _PROJECT_ROOT / "configs" / "training" / "cnn1d.yaml"
_MLP_TEMPLATE_PATH = _PROJECT_ROOT / "configs" / "training" / "mlp.yaml"


def _write(tmp_path: Path, text: str, name: str = "mlp.yaml") -> Path:
    config_path = tmp_path / name
    config_path.write_text(text, encoding="utf-8")
    return config_path


def test_minimal_config_uses_defaults(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, _MINIMAL))
    assert config.dataset_name == "demo_ds"
    assert config.run_name == "run_001"
    assert config.data.pulse_order is None
    assert isinstance(config.model, ModelConfig)
    assert config.model.hidden_dims == (64, 32, 32)
    assert config.label.transform == "identity"
    assert config.preprocessing.std_eps == pytest.approx(1.0e-8)
    assert config.training.seed == 42
    assert config.training.device == "auto"
    assert config.training.weight_decay == 0.0  # zero is valid
    assert config.training.early_stopping.min_delta == 0.0  # zero is valid
    assert config.split.ratios == SplitRatios()
    assert config.split.min_per_split.train == 2
    assert config.output_dir is None


def test_full_config_roundtrip(tmp_path: Path) -> None:
    text = """
dataset_name: demo_ds
run_name: run_001
data:
  pulse_order: [p0, p1]
model:
  hidden_dims: [16, 8]
label: {transform: logalpha}
preprocessing: {std_eps: 1.0e-6}
training:
  seed: 7
  device: cpu
  batch_size: 4
  max_epochs: 3
  learning_rate: 0.01
  weight_decay: 0.0
  early_stopping: {patience: 2, min_delta: 0.0}
split:
  seed: 9
  ratios: {train: 0.5, val: 0.25, test: 0.25}
  min_per_split: {train: 1, val: 1, test: 0}
output_dir: custom/out
"""
    config = load_config(_write(tmp_path, text))
    assert config.data.pulse_order == ("p0", "p1")
    assert isinstance(config.model, ModelConfig)
    assert config.model.hidden_dims == (16, 8)
    assert config.label.transform == "logalpha"
    assert config.training.device == "cpu"
    assert config.training.batch_size == 4
    assert config.split.ratios == SplitRatios(0.5, 0.25, 0.25)
    assert config.split.min_per_split.test == 0
    assert config.output_dir == "custom/out"


def test_placeholder_values_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="placeholder"):
        load_config(_write(tmp_path, "dataset_name: PLACEHOLDER_DATASET_NAME\nrun_name: r\n"))
    with pytest.raises(ConfigError, match="placeholder"):
        load_config(_write(tmp_path, "dataset_name: ds\nrun_name: PLACEHOLDER_RUN_NAME\n"))


@pytest.mark.parametrize("bad", ["a/b", "a\\b", "..", ".", "", "../escape"])
def test_unsafe_dataset_name_rejected(tmp_path: Path, bad: str) -> None:
    text = f"dataset_name: {bad!r}\nrun_name: run_001\n" if bad else "run_name: run_001\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, text))


def test_missing_required_fields_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="dataset_name"):
        load_config(_write(tmp_path, "run_name: run_001\n"))
    with pytest.raises(ConfigError, match="run_name"):
        load_config(_write(tmp_path, "dataset_name: ds\n"))


def test_unknown_keys_rejected_at_every_level(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown fields"):
        load_config(_write(tmp_path, _MINIMAL + "surprise: 1\n"))
    with pytest.raises(ConfigError, match="training"):
        load_config(_write(tmp_path, _MINIMAL + "training: {batch_size: 8, zzz: 1}\n"))
    with pytest.raises(ConfigError, match="split.ratios"):
        load_config(_write(tmp_path, _MINIMAL + "split: {ratios: {train: 1.0, val: 0.0}}\n"))


def test_duplicate_yaml_keys_rejected(tmp_path: Path) -> None:
    text = _MINIMAL + "dataset_name: other_ds\n"
    with pytest.raises(ConfigError, match="failed to parse YAML"):
        load_config(_write(tmp_path, text))


def test_missing_section_key_in_required_mapping(tmp_path: Path) -> None:
    # A required sub-mapping field is missing inside a present section (all three ratios
    # keys required).
    text = (
        _MINIMAL
        + "split: {ratios: {train: 0.5, val: 0.25, test: 0.25}, min_per_split: {train: 1}}\n"
    )
    with pytest.raises(ConfigError, match="min_per_split"):
        load_config(_write(tmp_path, text))


def test_ratios_validation(tmp_path: Path) -> None:
    good = _MINIMAL + "split: {ratios: {train: 0.7, val: 0.15, test: 0.15}}\n"
    assert load_config(_write(tmp_path, good)).split.ratios.train == pytest.approx(0.7)
    for bad in (
        "split: {ratios: {train: -0.1, val: 0.6, test: 0.5}}\n",
        "split: {ratios: {train: 0.5, val: 0.5, test: 0.1}}\n",
        "split: {ratios: {train: .nan, val: 0.5, test: 0.5}}\n",
    ):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, _MINIMAL + bad))


def test_zero_and_negative_scalars(tmp_path: Path) -> None:
    ok = _MINIMAL + "training: {weight_decay: 0.0, early_stopping: {min_delta: 0.0}, seed: 0}\n"
    config = load_config(_write(tmp_path, ok))
    assert config.training.weight_decay == 0.0
    assert config.training.seed == 0
    for bad in (
        "training: {weight_decay: -0.1}\n",
        "training: {batch_size: 0}\n",
        "training: {batch_size: true}\n",
        "training: {learning_rate: 0}\n",
        "training: {max_epochs: -1}\n",
        "training: {early_stopping: {patience: 0}}\n",
        "preprocessing: {std_eps: 0}\n",
        "split: {min_per_split: {train: -1}}\n",
        "model: {hidden_dims: [64, 0]}\n",
        "model: {hidden_dims: []}\n",
    ):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, _MINIMAL + bad))


def test_device_and_transform_choices(tmp_path: Path) -> None:
    for device in ("auto", "cpu", "cuda"):
        text = _MINIMAL + f"training: {{device: {device}}}\n"
        assert load_config(_write(tmp_path, text)).training.device == device
    with pytest.raises(ConfigError, match="device"):
        load_config(_write(tmp_path, _MINIMAL + "training: {device: tpu}\n"))
    with pytest.raises(ConfigError, match="transform"):
        load_config(_write(tmp_path, _MINIMAL + "label: {transform: log10}\n"))


def test_pulse_order_and_output_dir(tmp_path: Path) -> None:
    ok = _MINIMAL + "data: {pulse_order: [p0, p1]}\noutput_dir: out\n"
    config = load_config(_write(tmp_path, ok))
    assert config.data.pulse_order == ("p0", "p1")
    assert config.output_dir == "out"
    for bad in (
        "data: {pulse_order: [p0, p0]}\n",
        "data: {pulse_order: []}\n",
        "data: {pulse_order: [a/b]}\n",
        "output_dir: 5\n",
        "output_dir: ''\n",
    ):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, _MINIMAL + bad))


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="failed to read"):
        load_config(tmp_path / "nope.yaml")


def test_config_mapping_roundtrip_and_bad_schema(tmp_path: Path) -> None:
    """config_to_mapping/from_mapping owner round-trip: consistent with YAML loading,
    bad schema rejected."""
    config = load_config(_write(tmp_path, _MINIMAL))
    mapping = training_config.config_to_mapping(config)
    rebuilt = training_config.config_from_mapping(mapping)
    assert rebuilt == config  # ExperimentConfig has no ndarray fields; structurally equal
    # The YAML snapshot can be reloaded by load_config and stays consistent.
    reloaded = load_config(_write(tmp_path, yaml.safe_dump(mapping, sort_keys=False)))
    assert reloaded == config
    # Bad schema: missing required key / non-mapping section / illegal field type.
    with pytest.raises(ConfigError, match="missing required key"):
        training_config.config_from_mapping({"dataset_name": "ds"})
    with pytest.raises(ConfigError, match="must be a mapping"):
        training_config.config_from_mapping(dict(mapping, training="oops"))
    with pytest.raises(ConfigError, match="illegal field type"):
        broken = dict(mapping)
        broken["training"] = dict(mapping["training"], batch_size="two")
        training_config.config_from_mapping(broken)


# --- P1: model.kind discrimination and strict CNN1D structure-field validation ---
# The CNN structure fixture uses **explicitly unit-test-only values** and represents no
# research hyperparameter nor a research config under configs/; the YAML and
# config_from_mapping paths share the same mapping sample.

_CNN_CHANNELS = (4, 8)
_CNN_KERNELS = (3, 5)
_CNN_POOL_BINS = 4
_CNN_HEAD = (8,)


def _cnn_model(**overrides: object) -> dict[str, object]:
    """A legal CNN ``model`` block sample; ``overrides`` may replace fields to build illegal
    samples."""
    block: dict[str, object] = {
        "kind": "cnn1d",
        "channels": list(_CNN_CHANNELS),
        "kernel_sizes": list(_CNN_KERNELS),
        "pool_bins": _CNN_POOL_BINS,
        "head_hidden_dims": list(_CNN_HEAD),
    }
    block.update(overrides)
    return block


def _without(field: str) -> dict[str, object]:
    """Remove a single field from the CNN sample (to build a missing-required-field sample)."""
    block = _cnn_model()
    block.pop(field, None)
    return block


def _root_mapping(model: object) -> dict[str, object]:
    """A minimal legal root mapping (the same mapping is reused by both the YAML and
    mapping paths)."""
    return {"dataset_name": "demo_ds", "run_name": "run_001", "model": model}


def _load_mapping(tmp_path: Path, mapping: object, name: str) -> training_config.ExperimentConfig:
    """mapping → YAML text → load_config (shares the same mapping object as the mapping path)."""
    return load_config(_write(tmp_path, yaml.safe_dump(mapping, sort_keys=False), name=name))


_BAD_CNN_MODELS: tuple[tuple[str, dict[str, object]], ...] = (
    ("missing_kind_fields", {"kind": "cnn1d"}),
    ("missing_channels", _without("channels")),
    ("missing_kernel_sizes", _without("kernel_sizes")),
    ("missing_pool_bins", _without("pool_bins")),
    ("missing_head", _without("head_hidden_dims")),
    ("unknown_kind", _cnn_model(kind="transformer")),
    ("kind_non_string", _cnn_model(kind=1)),
    ("kind_bool", _cnn_model(kind=True)),
    ("unknown_key", {**_cnn_model(), "bogus": 1}),
    ("no_kind_with_cnn_fields", _without("kind")),
    (
        "mlp_with_cnn_fields",
        {
            "kind": "mlp",
            "hidden_dims": [8],
            "channels": [4],
            "kernel_sizes": [3],
            "pool_bins": 4,
            "head_hidden_dims": [],
        },
    ),
    ("cnn_with_hidden_dims", {**_cnn_model(), "hidden_dims": [8]}),
    ("channels_empty", _cnn_model(channels=[])),
    ("channels_zero", _cnn_model(channels=[0])),
    ("channels_negative", _cnn_model(channels=[-1])),
    ("channels_bool", _cnn_model(channels=[True])),
    ("channels_float", _cnn_model(channels=[1.5])),
    ("channels_string", _cnn_model(channels=["4"])),
    ("channels_null", _cnn_model(channels=None)),
    ("kernel_empty", _cnn_model(kernel_sizes=[])),
    ("kernel_even", _cnn_model(kernel_sizes=[2, 4])),
    ("kernel_zero", _cnn_model(kernel_sizes=[0, 3])),
    ("kernel_negative", _cnn_model(kernel_sizes=[-3, 5])),
    ("kernel_bool", _cnn_model(kernel_sizes=[True, 5])),
    ("kernel_string", _cnn_model(kernel_sizes=["3", 5])),
    ("kernel_length_mismatch", _cnn_model(kernel_sizes=[3])),
    ("kernel_null", _cnn_model(kernel_sizes=None)),
    ("pool_zero", _cnn_model(pool_bins=0)),
    ("pool_negative", _cnn_model(pool_bins=-1)),
    ("pool_bool", _cnn_model(pool_bins=True)),
    ("pool_float", _cnn_model(pool_bins=4.5)),
    ("pool_string", _cnn_model(pool_bins="4")),
    ("pool_null", _cnn_model(pool_bins=None)),
    ("head_null", _cnn_model(head_hidden_dims=None)),
    ("head_zero_element", _cnn_model(head_hidden_dims=[0])),
    ("head_negative_element", _cnn_model(head_hidden_dims=[-1])),
    ("head_bool_element", _cnn_model(head_hidden_dims=[True])),
    ("head_float_element", _cnn_model(head_hidden_dims=[1.5])),
    ("head_string_element", _cnn_model(head_hidden_dims=["8"])),
)


def test_model_defaults_to_mlp_without_kind(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, _MINIMAL))
    assert isinstance(config.model, ModelConfig)
    assert config.model.kind == "mlp"
    assert config.model.hidden_dims == (64, 32, 32)


def test_explicit_mlp_equals_implicit_and_serializes_without_kind(tmp_path: Path) -> None:
    implicit = _load_mapping(tmp_path, _root_mapping({"hidden_dims": [16, 8]}), "implicit.yaml")
    explicit = _load_mapping(
        tmp_path, _root_mapping({"kind": "mlp", "hidden_dims": [16, 8]}), "explicit.yaml"
    )
    assert explicit == implicit
    assert isinstance(explicit.model, ModelConfig)
    assert explicit.model.kind == "mlp"
    assert explicit.model.hidden_dims == (16, 8)
    # Old serialization shape: the MLP writes no kind, only hidden_dims (old ckpt nested
    # config compatibility).
    serialized = training_config.config_to_mapping(explicit)
    assert serialized["model"] == {"hidden_dims": [16, 8]}
    assert training_config.config_from_mapping(serialized) == explicit


@pytest.mark.parametrize("head", [[], list(_CNN_HEAD)], ids=["empty_head", "nonempty_head"])
def test_cnn_config_roundtrips_yaml_and_mapping(tmp_path: Path, head: list[int]) -> None:
    model = _cnn_model(head_hidden_dims=head)
    config = _load_mapping(tmp_path, _root_mapping(model), "cnn.yaml")
    assert isinstance(config.model, CNN1DModelConfig)
    assert config.model.kind == "cnn1d"
    assert config.model.channels == _CNN_CHANNELS
    assert config.model.kernel_sizes == _CNN_KERNELS
    assert config.model.pool_bins == _CNN_POOL_BINS
    assert config.model.head_hidden_dims == tuple(head)  # an empty head is valid and stays empty

    serialized = training_config.config_to_mapping(config)
    assert serialized["model"] == {
        "kind": "cnn1d",
        "channels": list(_CNN_CHANNELS),
        "kernel_sizes": list(_CNN_KERNELS),
        "pool_bins": _CNN_POOL_BINS,
        "head_hidden_dims": list(head),
    }
    assert training_config.config_from_mapping(serialized) == config
    assert _load_mapping(tmp_path, serialized, "reload.yaml") == config


def test_pool_bins_positive_is_not_t_checked_at_config_layer(tmp_path: Path) -> None:
    # The config layer only validates positive non-bool; pool_bins <= T is validated by the
    # model layer, so an oversized value is legal here.
    config = _load_mapping(tmp_path, _root_mapping(_cnn_model(pool_bins=10**9)), "big.yaml")
    assert isinstance(config.model, CNN1DModelConfig)
    assert config.model.pool_bins == 10**9


@pytest.mark.parametrize(
    ("case", "bad_model"),
    _BAD_CNN_MODELS,
    ids=[case for case, _ in _BAD_CNN_MODELS],
)
def test_cnn_model_strict_rejection_both_paths(
    tmp_path: Path, case: str, bad_model: dict[str, object]
) -> None:
    """For the same model sample, both strict paths (YAML and config_from_mapping) must reject."""
    with pytest.raises(ConfigError):
        _load_mapping(tmp_path, _root_mapping(bad_model), f"{case}.yaml")
    with pytest.raises(ConfigError):
        training_config.config_from_mapping(_root_mapping(bad_model))


# --- checked-in CNN template: valid YAML, placeholders rejected, substitutes load ---


def _template_with_names(tmp_path: Path, template: Path, name: str) -> Path:
    """Copy a checked-in template into tmp_path with its placeholders replaced by
    legitimate dataset/run names."""
    text = template.read_text(encoding="utf-8")
    text = text.replace("PLACEHOLDER_DATASET_NAME", "demo_ds").replace(
        "PLACEHOLDER_RUN_NAME", "run_001"
    )
    return _write(tmp_path, text, name=name)


def test_checked_in_cnn_template_is_valid_yaml_and_rejects_placeholders() -> None:
    # The shipped template is a real YAML mapping (not comment-only) that intentionally
    # keeps its placeholders, so load_config refuses it until they are replaced.
    loaded = yaml.safe_load(_CNN_TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    with pytest.raises(ConfigError, match="placeholder"):
        load_config(_CNN_TEMPLATE_PATH)


def test_checked_in_cnn_template_loads_after_placeholder_substitution(tmp_path: Path) -> None:
    config = load_config(_template_with_names(tmp_path, _CNN_TEMPLATE_PATH, "cnn1d.yaml"))
    assert isinstance(config.model, CNN1DModelConfig)
    assert config.model.kind == "cnn1d"
    assert config.model.channels == (8, 16)
    assert config.model.kernel_sizes == (5, 5)
    assert config.model.pool_bins == 16
    assert config.model.head_hidden_dims == (16,)
    # Every shared field equals the MLP template (structural mirroring).
    mlp = load_config(_template_with_names(tmp_path, _MLP_TEMPLATE_PATH, "mlp.yaml"))
    assert config.data == mlp.data
    assert config.label == mlp.label
    assert config.preprocessing == mlp.preprocessing
    assert config.training == mlp.training
    assert config.split == mlp.split
    assert config.output_dir == mlp.output_dir is None


def test_checked_in_cnn_template_parameter_count(tmp_path: Path) -> None:
    # The template architecture is instantiated offline only (no data, no training) and
    # yields the parameter count reported in results/cnn.md.
    config = load_config(_template_with_names(tmp_path, _CNN_TEMPLATE_PATH, "cnn1d.yaml"))
    assert isinstance(config.model, CNN1DModelConfig)
    model = CNN1DRegressor(
        (1, 401, 3),
        channels=config.model.channels,
        kernel_sizes=config.model.kernel_sizes,
        pool_bins=config.model.pool_bins,
        head_hidden_dims=config.model.head_hidden_dims,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 4930
