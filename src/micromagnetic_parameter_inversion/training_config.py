"""训练/评估实验配置：frozen dataclass + 严格 YAML 加载/校验。

字段与 ``configs/training/mlp.yaml`` 一一对应；``split`` 块仅被 prepare
脚本使用。``load_config`` 是唯一校验边界：所有层级严格 schema（拒绝
未知/缺失字段），dataset_name/run_name 必填、须为安全单路径段且不得保留
占位符；零值（weight_decay=0、min_delta=0）合法，校验通过后流程假定配置
合法。

校验 helper 自 ``mumax3_config`` 复用，schema 违反统一抛 ConfigError。

依赖方向：本模块为叶子（标准库 + yaml + mumax3_config 校验器）；被
training_data / preprocessing / training / evaluation 单向引用。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from micromagnetic_parameter_inversion.mumax3_config import (
    ConfigError,
    _fail,
    _require_mapping,
    _require_non_negative_int,
    _require_non_negative_number,
    _require_positive_int,
    _require_positive_number,
    _require_safe_path_segment,
    _UniqueKeyLoader,
)

# 占位值：configs/training/mlp.yaml 示例使用；load_config 拒绝仍保留占位符
# 的配置（dataset_name/run_name 运行前由用户替换）。
PLACEHOLDER_DATASET_NAME = "PLACEHOLDER_DATASET_NAME"
PLACEHOLDER_RUN_NAME = "PLACEHOLDER_RUN_NAME"

# ckpt schema 版本号（checkpoint 由 training 模块读写）。
CKPT_FORMAT_VERSION = 1

type LabelTransform = Literal["identity", "logalpha"]

# 激活函数在 ckpt 中显式留档（模型结构显式字段）；固定 ReLU。
type ActivationName = Literal["relu"]

_TOP_LEVEL_KEYS = frozenset(
    {
        "dataset_name",
        "run_name",
        "data",
        "model",
        "label",
        "preprocessing",
        "training",
        "split",
        "output_dir",
    }
)
_DATA_KEYS = frozenset({"pulse_order"})
_MODEL_KEYS = frozenset({"hidden_dims"})
_LABEL_KEYS = frozenset({"transform"})
_PREPROCESSING_KEYS = frozenset({"std_eps"})
_TRAINING_KEYS = frozenset(
    {
        "seed",
        "device",
        "batch_size",
        "max_epochs",
        "learning_rate",
        "weight_decay",
        "early_stopping",
    }
)
_EARLY_STOPPING_KEYS = frozenset({"patience", "min_delta"})
_SPLIT_KEYS = frozenset({"seed", "ratios", "min_per_split"})
_RATIO_KEYS = frozenset({"train", "val", "test"})

# device 取值与 runtime.select_device 的语义一致。
_DEVICES = frozenset({"auto", "cpu", "cuda"})
_TRANSFORMS = frozenset({"identity", "logalpha"})
# ratios 求和容差：容忍手写小数的浮点舍入。
_RATIO_TOLERANCE = 1e-9


@dataclass(frozen=True)
class DataConfig:
    """``data`` 块：pulse 顺序来源。"""

    # None = 从所选首份 config.yaml 快照冻结并记录进 dataset_meta；
    # 非 None = 显式 pulse_id 列表（须为协议集合的无重复排列）。
    pulse_order: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ModelConfig:
    """``model`` 块：网络结构（[64,32,32] 时参数量 = ``64·D + 3266``）。"""

    hidden_dims: tuple[int, ...] = (64, 32, 32)


@dataclass(frozen=True)
class LabelConfig:
    """``label`` 块：标签变换（不等于 alpha 空间采样设计，两者独立）。"""

    transform: LabelTransform = "identity"  # logalpha 仅对 alpha 取 log10，须 alpha > 0


@dataclass(frozen=True)
class PreprocessingConfig:
    """``preprocessing`` 块：标准化超参（统计量本身仅 train 组拟合）。"""

    std_eps: float = 1.0e-8  # x/y 统一：std <= eps → 除数取 1，并记录零方差位置


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """``training.early_stopping``：min_delta 只作用于停止判定，不改 best 规则。"""

    patience: int = 50
    min_delta: float = 0.0  # 0 合法：任何非改善 epoch 都计入 patience


@dataclass(frozen=True)
class TrainingParams:
    """``training`` 块：优化与运行超参（初始工程候选，非已验证科研参数）。"""

    seed: int = 42  # 与 configs/base.yaml 一致
    device: str = "auto"  # auto|cpu|cuda，交 runtime.select_device
    batch_size: int = 32
    max_epochs: int = 500
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0  # 0 合法（无 L2 正则）
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()


@dataclass(frozen=True)
class SplitRatios:
    """``split.ratios``：三者须全部非负有限且和为 1（容差 1e-9）。"""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15


@dataclass(frozen=True)
class SplitMinCounts:
    """``split.min_per_split``：任一组数量不足 → prepare 写样本前报错退出。"""

    train: int = 2
    val: int = 1
    test: int = 1


@dataclass(frozen=True)
class SplitConfig:
    """``split`` 块：仅 prepare 脚本使用（算法见 training_data.make_split）。"""

    seed: int = 42
    ratios: SplitRatios = SplitRatios()
    min_per_split: SplitMinCounts = SplitMinCounts()


@dataclass(frozen=True)
class ExperimentConfig:
    """实验配置聚合根（一份 ``configs/training/mlp.yaml`` 的内存表示）。"""

    # 必填（不可为 null/空，运行前替换占位值）；run_name 对应的输出目录
    # 已存在则拒绝启动，不隐式覆盖。
    dataset_name: str
    run_name: str
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    label: LabelConfig = LabelConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    training: TrainingParams = TrainingParams()
    split: SplitConfig = SplitConfig()
    # None = output_root()/training/mlp/<dataset_name>/<run_name>。
    output_dir: str | None = None


def _check_keys(
    mapping: dict[Any, Any], field: str, known: frozenset[str], required: frozenset[str]
) -> None:
    """严格 schema：拒绝未知字段；仅 ``required`` 子集为必填（其余可缺省）。"""
    unknown = sorted((key for key in mapping if key not in known), key=repr)
    if unknown:
        _fail(field, f"未知字段 {unknown!r}；允许的字段 {sorted(known)}")
    missing = sorted(required.difference(mapping))
    if missing:
        _fail(field, f"缺失必填字段 {missing!r}")


def _optional_section(
    root: dict[Any, Any], field: str, known: frozenset[str]
) -> dict[Any, Any] | None:
    """可选配置节：缺省 → None（调用方回退 dataclass 默认值）。"""
    if field not in root:
        return None
    raw = _require_mapping(root[field], field)
    _check_keys(raw, field, known, frozenset())
    return raw


def _parse_pulse_order(value: object, field: str) -> tuple[str, ...] | None:
    """pulse_order：null → None；否则非空、元素为安全路径段且无重复。"""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        _fail(field, f"必须为 null 或非空 pulse_id 列表 (got {value!r})")
    ids = tuple(_require_safe_path_segment(item, f"{field}[{i}]") for i, item in enumerate(value))
    if len(set(ids)) != len(ids):
        _fail(field, f"重复的 pulse_id (got {value!r})")
    return ids


def _parse_hidden_dims(value: object, field: str) -> tuple[int, ...]:
    """hidden_dims：非空的正整数序列。"""
    if not isinstance(value, list) or not value:
        _fail(field, f"必须为非空整数列表 (got {value!r})")
    return tuple(_require_positive_int(item, f"{field}[{i}]") for i, item in enumerate(value))


def _parse_ratios(value: object, field: str) -> SplitRatios:
    """ratios：非负有限数，和为 1（容差内）。"""
    raw = _require_mapping(value, field)
    _check_keys(raw, field, _RATIO_KEYS, _RATIO_KEYS)
    values = {
        name: _require_non_negative_number(raw[name], f"{field}.{name}")
        for name in ("train", "val", "test")
    }
    total = sum(values.values())
    if abs(total - 1.0) > _RATIO_TOLERANCE:
        _fail(field, f"三项之和必须为 1（容差 {_RATIO_TOLERANCE}）(got {total!r})")
    return SplitRatios(train=values["train"], val=values["val"], test=values["test"])


def _parse_min_per_split(value: object, field: str) -> SplitMinCounts:
    """min_per_split：非负整数（允许 0 = 该组可为空）。"""
    raw = _require_mapping(value, field)
    _check_keys(raw, field, _RATIO_KEYS, _RATIO_KEYS)
    return SplitMinCounts(
        train=_require_non_negative_int(raw["train"], f"{field}.train"),
        val=_require_non_negative_int(raw["val"], f"{field}.val"),
        test=_require_non_negative_int(raw["test"], f"{field}.test"),
    )


def _parse_data(root: dict[Any, Any]) -> DataConfig:
    raw = _optional_section(root, "data", _DATA_KEYS)
    if raw is None:
        return DataConfig()
    kwargs: dict[str, Any] = {}
    if "pulse_order" in raw:
        kwargs["pulse_order"] = _parse_pulse_order(raw["pulse_order"], "data.pulse_order")
    return DataConfig(**kwargs)


def _parse_model(root: dict[Any, Any]) -> ModelConfig:
    raw = _optional_section(root, "model", _MODEL_KEYS)
    if raw is None:
        return ModelConfig()
    kwargs: dict[str, Any] = {}
    if "hidden_dims" in raw:
        kwargs["hidden_dims"] = _parse_hidden_dims(raw["hidden_dims"], "model.hidden_dims")
    return ModelConfig(**kwargs)


def _parse_label(root: dict[Any, Any]) -> LabelConfig:
    raw = _optional_section(root, "label", _LABEL_KEYS)
    if raw is None or "transform" not in raw:
        return LabelConfig()
    transform = raw["transform"]
    if transform not in _TRANSFORMS:
        _fail("label.transform", f"必须为 {sorted(_TRANSFORMS)} (got {transform!r})")
    return LabelConfig(transform=transform)


def _parse_preprocessing(root: dict[Any, Any]) -> PreprocessingConfig:
    raw = _optional_section(root, "preprocessing", _PREPROCESSING_KEYS)
    if raw is None or "std_eps" not in raw:
        return PreprocessingConfig()
    return PreprocessingConfig(
        std_eps=_require_positive_number(raw["std_eps"], "preprocessing.std_eps")
    )


def _parse_early_stopping(raw: dict[Any, Any]) -> EarlyStoppingConfig:
    kwargs: dict[str, Any] = {}
    if "patience" in raw:
        kwargs["patience"] = _require_positive_int(
            raw["patience"], "training.early_stopping.patience"
        )
    if "min_delta" in raw:
        kwargs["min_delta"] = _require_non_negative_number(
            raw["min_delta"], "training.early_stopping.min_delta"
        )
    return EarlyStoppingConfig(**kwargs)


def _parse_training(root: dict[Any, Any]) -> TrainingParams:
    raw = _optional_section(root, "training", _TRAINING_KEYS)
    if raw is None:
        return TrainingParams()
    kwargs: dict[str, Any] = {}
    if "seed" in raw:
        kwargs["seed"] = _require_non_negative_int(raw["seed"], "training.seed")
    if "device" in raw:
        if raw["device"] not in _DEVICES:
            _fail("training.device", f"必须为 {sorted(_DEVICES)} (got {raw['device']!r})")
        kwargs["device"] = raw["device"]
    if "batch_size" in raw:
        kwargs["batch_size"] = _require_positive_int(raw["batch_size"], "training.batch_size")
    if "max_epochs" in raw:
        kwargs["max_epochs"] = _require_positive_int(raw["max_epochs"], "training.max_epochs")
    if "learning_rate" in raw:
        kwargs["learning_rate"] = _require_positive_number(
            raw["learning_rate"], "training.learning_rate"
        )
    if "weight_decay" in raw:
        kwargs["weight_decay"] = _require_non_negative_number(
            raw["weight_decay"], "training.weight_decay"
        )
    if "early_stopping" in raw:
        es_raw = _require_mapping(raw["early_stopping"], "training.early_stopping")
        _check_keys(es_raw, "training.early_stopping", _EARLY_STOPPING_KEYS, frozenset())
        kwargs["early_stopping"] = _parse_early_stopping(es_raw)
    return TrainingParams(**kwargs)


def _parse_split(root: dict[Any, Any]) -> SplitConfig:
    raw = _optional_section(root, "split", _SPLIT_KEYS)
    if raw is None:
        return SplitConfig()
    kwargs: dict[str, Any] = {}
    if "seed" in raw:
        kwargs["seed"] = _require_non_negative_int(raw["seed"], "split.seed")
    if "ratios" in raw:
        kwargs["ratios"] = _parse_ratios(raw["ratios"], "split.ratios")
    if "min_per_split" in raw:
        kwargs["min_per_split"] = _parse_min_per_split(raw["min_per_split"], "split.min_per_split")
    return SplitConfig(**kwargs)


def _parse_output_dir(root: dict[Any, Any]) -> str | None:
    if "output_dir" not in root:
        return None
    value = root["output_dir"]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        _fail("output_dir", f"必须为 null 或非空字符串 (got {value!r})")
    return value


def load_config(path: Path) -> ExperimentConfig:
    """读取 YAML、执行唯一入口校验并构造 ExperimentConfig。

    校验边界（此后流程假定配置合法）：所有层级严格 schema（未知字段与
    重复 YAML 键拒绝）；dataset_name/run_name 必填、安全单路径段、不得
    保留占位符；device ∈ {auto,cpu,cuda}（语义同
    runtime.select_device）；ratios 非负有限且和为 1（容差 1e-9）；
    weight_decay/min_delta/min_per_split 允许 0；可选节缺省时使用
    dataclass 默认值。

    Raises:
        ConfigError: 消息含字段路径（如 training.batch_size）。
    """
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: YAML 解析失败: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: 读取失败: {exc}") from exc

    root = _require_mapping(raw, str(path))
    _check_keys(root, str(path), _TOP_LEVEL_KEYS, frozenset({"dataset_name", "run_name"}))

    dataset_name = _require_safe_path_segment(root["dataset_name"], "dataset_name")
    if dataset_name == PLACEHOLDER_DATASET_NAME:
        _fail("dataset_name", f"占位符须在运行前替换 (got {dataset_name!r})")
    run_name = _require_safe_path_segment(root["run_name"], "run_name")
    if run_name == PLACEHOLDER_RUN_NAME:
        _fail("run_name", f"占位符须在运行前替换 (got {run_name!r})")

    return ExperimentConfig(
        dataset_name=dataset_name,
        run_name=run_name,
        data=_parse_data(root),
        model=_parse_model(root),
        label=_parse_label(root),
        preprocessing=_parse_preprocessing(root),
        training=_parse_training(root),
        split=_parse_split(root),
        output_dir=_parse_output_dir(root),
    )


def config_to_mapping(config: ExperimentConfig) -> dict[str, Any]:
    """ExperimentConfig → 嵌套纯字典（本模块维护的映射 schema）。

    形状与 ``configs/training/mlp.yaml`` 一致（可被 ``load_config`` 重新
    加载），也作为 ckpt ``config`` 副本的落盘形态；容器均为
    primitives/list/dict（tuple → list），无 dataclass/numpy 对象。
    """
    return {
        "dataset_name": config.dataset_name,
        "run_name": config.run_name,
        "data": {
            "pulse_order": (
                list(config.data.pulse_order) if config.data.pulse_order is not None else None
            )
        },
        "model": {"hidden_dims": list(config.model.hidden_dims)},
        "label": {"transform": config.label.transform},
        "preprocessing": {"std_eps": config.preprocessing.std_eps},
        "training": {
            "seed": config.training.seed,
            "device": config.training.device,
            "batch_size": config.training.batch_size,
            "max_epochs": config.training.max_epochs,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "early_stopping": {
                "patience": config.training.early_stopping.patience,
                "min_delta": config.training.early_stopping.min_delta,
            },
        },
        "split": {
            "seed": config.split.seed,
            "ratios": {
                "train": config.split.ratios.train,
                "val": config.split.ratios.val,
                "test": config.split.ratios.test,
            },
            "min_per_split": {
                "train": config.split.min_per_split.train,
                "val": config.split.min_per_split.val,
                "test": config.split.min_per_split.test,
            },
        },
        "output_dir": config.output_dir,
    }


def _mapping_section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """可选配置节：缺省 → 空映射（调用方回退 dataclass 默认值）；非映射即报错。"""
    value = mapping.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        _fail(f"config.{key}", f"必须为映射或 null (got {type(value)!r})")
    return value


def config_from_mapping(mapping: Mapping[str, Any]) -> ExperimentConfig:
    """嵌套纯字典 → ExperimentConfig（ckpt ``config`` 副本重建的唯一入口）。

    与 ``config_to_mapping`` 对称；缺省节/键回退 dataclass 默认值（ckpt
    副本总是完整）。YAML 严格加载仍走 ``load_config``（占位符/ratios
    求和等校验属 YAML 边界，不在此重复）。

    Raises:
        ConfigError: 非映射、缺失 dataset_name/run_name、节非映射或字段
            类型非法。
    """
    if not isinstance(mapping, Mapping):
        _fail("config", f"必须为映射 (got {type(mapping)!r})")
    for key in ("dataset_name", "run_name"):
        if key not in mapping:
            _fail("config", f"缺失必填键 {key!r}")
    data_raw = _mapping_section(mapping, "data")
    model_raw = _mapping_section(mapping, "model")
    label_raw = _mapping_section(mapping, "label")
    prep_raw = _mapping_section(mapping, "preprocessing")
    training_raw = _mapping_section(mapping, "training")
    es_raw = _mapping_section(training_raw, "early_stopping")
    split_raw = _mapping_section(mapping, "split")
    ratios_raw = _mapping_section(split_raw, "ratios")
    mins_raw = _mapping_section(split_raw, "min_per_split")
    training_defaults = TrainingParams()
    split_defaults = SplitConfig()
    pulse_order = data_raw.get("pulse_order")
    try:
        return ExperimentConfig(
            dataset_name=str(mapping["dataset_name"]),
            run_name=str(mapping["run_name"]),
            data=DataConfig(
                pulse_order=(None if pulse_order is None else tuple(str(p) for p in pulse_order))
            ),
            model=ModelConfig(
                hidden_dims=tuple(
                    int(d) for d in model_raw.get("hidden_dims") or ModelConfig().hidden_dims
                )
            ),
            label=LabelConfig(
                transform=cast(
                    LabelTransform, str(label_raw.get("transform", LabelConfig().transform))
                )
            ),
            preprocessing=PreprocessingConfig(
                std_eps=float(prep_raw.get("std_eps", PreprocessingConfig().std_eps))
            ),
            training=TrainingParams(
                seed=int(training_raw.get("seed", training_defaults.seed)),
                device=str(training_raw.get("device", training_defaults.device)),
                batch_size=int(training_raw.get("batch_size", training_defaults.batch_size)),
                max_epochs=int(training_raw.get("max_epochs", training_defaults.max_epochs)),
                learning_rate=float(
                    training_raw.get("learning_rate", training_defaults.learning_rate)
                ),
                weight_decay=float(
                    training_raw.get("weight_decay", training_defaults.weight_decay)
                ),
                early_stopping=EarlyStoppingConfig(
                    patience=int(es_raw.get("patience", EarlyStoppingConfig().patience)),
                    min_delta=float(es_raw.get("min_delta", EarlyStoppingConfig().min_delta)),
                ),
            ),
            split=SplitConfig(
                seed=int(split_raw.get("seed", split_defaults.seed)),
                ratios=SplitRatios(
                    train=float(ratios_raw.get("train", split_defaults.ratios.train)),
                    val=float(ratios_raw.get("val", split_defaults.ratios.val)),
                    test=float(ratios_raw.get("test", split_defaults.ratios.test)),
                ),
                min_per_split=SplitMinCounts(
                    train=int(mins_raw.get("train", split_defaults.min_per_split.train)),
                    val=int(mins_raw.get("val", split_defaults.min_per_split.val)),
                    test=int(mins_raw.get("test", split_defaults.min_per_split.test)),
                ),
            ),
            output_dir=(
                str(mapping["output_dir"]) if mapping.get("output_dir") is not None else None
            ),
        )
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        _fail("config", f"字段类型非法: {exc}")
