"""Tests for strict training-config YAML loading (offline, tmp_path only)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from micromagnetic_parameter_inversion import training_config
from micromagnetic_parameter_inversion.training_config import ConfigError, SplitRatios, load_config

_MINIMAL = "dataset_name: demo_ds\nrun_name: run_001\n"


def _write(tmp_path: Path, text: str, name: str = "mlp.yaml") -> Path:
    config_path = tmp_path / name
    config_path.write_text(text, encoding="utf-8")
    return config_path


def test_minimal_config_uses_defaults(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, _MINIMAL))
    assert config.dataset_name == "demo_ds"
    assert config.run_name == "run_001"
    assert config.data.pulse_order is None
    assert config.model.hidden_dims == (64, 32, 32)
    assert config.label.transform == "identity"
    assert config.preprocessing.std_eps == pytest.approx(1.0e-8)
    assert config.training.seed == 42
    assert config.training.device == "auto"
    assert config.training.weight_decay == 0.0  # 零值合法
    assert config.training.early_stopping.min_delta == 0.0  # 零值合法
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
    assert config.model.hidden_dims == (16, 8)
    assert config.label.transform == "logalpha"
    assert config.training.device == "cpu"
    assert config.training.batch_size == 4
    assert config.split.ratios == SplitRatios(0.5, 0.25, 0.25)
    assert config.split.min_per_split.test == 0
    assert config.output_dir == "custom/out"


def test_placeholder_values_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="占位符"):
        load_config(_write(tmp_path, "dataset_name: PLACEHOLDER_DATASET_NAME\nrun_name: r\n"))
    with pytest.raises(ConfigError, match="占位符"):
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
    with pytest.raises(ConfigError, match="未知字段"):
        load_config(_write(tmp_path, _MINIMAL + "surprise: 1\n"))
    with pytest.raises(ConfigError, match="training"):
        load_config(_write(tmp_path, _MINIMAL + "training: {batch_size: 8, zzz: 1}\n"))
    with pytest.raises(ConfigError, match="split.ratios"):
        load_config(_write(tmp_path, _MINIMAL + "split: {ratios: {train: 1.0, val: 0.0}}\n"))


def test_duplicate_yaml_keys_rejected(tmp_path: Path) -> None:
    text = _MINIMAL + "dataset_name: other_ds\n"
    with pytest.raises(ConfigError, match="解析失败"):
        load_config(_write(tmp_path, text))


def test_missing_section_key_in_required_mapping(tmp_path: Path) -> None:
    # 出现的节内必填子映射字段缺失（ratios 三键必须齐全）。
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
    with pytest.raises(ConfigError, match="读取失败"):
        load_config(tmp_path / "nope.yaml")


def test_config_mapping_roundtrip_and_bad_schema(tmp_path: Path) -> None:
    """config_to_mapping/from_mapping 往返：与 YAML 加载一致、坏 schema 拒绝。"""
    config = load_config(_write(tmp_path, _MINIMAL))
    mapping = training_config.config_to_mapping(config)
    rebuilt = training_config.config_from_mapping(mapping)
    assert rebuilt == config  # ExperimentConfig 无 ndarray 字段，结构相等
    # YAML 快照可被 load_config 重新加载且一致
    reloaded = load_config(_write(tmp_path, yaml.safe_dump(mapping, sort_keys=False)))
    assert reloaded == config
    # 坏 schema：缺必填键 / 节非映射 / 字段类型非法
    with pytest.raises(ConfigError, match="缺失必填键"):
        training_config.config_from_mapping({"dataset_name": "ds"})
    with pytest.raises(ConfigError, match="必须为映射"):
        training_config.config_from_mapping(dict(mapping, training="oops"))
    with pytest.raises(ConfigError, match="字段类型非法"):
        broken = dict(mapping)
        broken["training"] = dict(mapping["training"], batch_size="two")
        training_config.config_from_mapping(broken)
