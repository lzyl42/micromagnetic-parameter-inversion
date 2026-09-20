"""Training/evaluation experiment config: frozen dataclasses + strict YAML loading/validation.

Fields correspond one-to-one with ``configs/training/mlp.yaml`` (plus the CNN
model schema); the ``split`` block is used only by the prepare script.
``load_config`` is the single validation boundary: every level uses a strict
schema (unknown/missing fields rejected), dataset_name/run_name are required,
must be safe single path segments, and must not keep placeholders; zero values
(weight_decay=0, min_delta=0) are valid, and after validation passes the
pipeline assumes the config is valid and does not re-defend.

The validation helpers (``_fail``/``_require_*``/``_UniqueKeyLoader``) are
reused from ``mumax3_config``, keeping the same strict style and error type
(ConfigError); schema violations uniformly raise ConfigError.

Dependency direction: this module is a leaf (stdlib + yaml + mumax3_config
validators); it is referenced one-way by training_data / preprocessing /
training / evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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

# Placeholder values: used by the configs/training/mlp.yaml example; load_config
# rejects configs that still keep placeholders (dataset_name/run_name are
# replaced by the user before running).
PLACEHOLDER_DATASET_NAME = "PLACEHOLDER_DATASET_NAME"
PLACEHOLDER_RUN_NAME = "PLACEHOLDER_RUN_NAME"

# ckpt schema version (checkpoints are read/written by the training module).
CKPT_FORMAT_VERSION = 1

type LabelTransform = Literal["identity", "logalpha"]

# The activation function is recorded explicitly in the ckpt (explicit model
# structure field); fixed ReLU for the first version.
type ActivationName = Literal["relu"]

# Model kind declared by the training entry (``training.run`` expected_kind).
type ModelKind = Literal["mlp", "cnn1d"]

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
# model block: kind values and per-kind mutually exclusive field sets
# (unknown/cross-kind fields are always rejected).
_MODEL_KINDS = frozenset({"mlp", "cnn1d"})
_MODEL_KEYS_MLP = frozenset({"kind", "hidden_dims"})
_MODEL_KEYS_CNN1D = frozenset({"kind", "channels", "kernel_sizes", "pool_bins", "head_hidden_dims"})
_CNN1D_REQUIRED = frozenset({"channels", "kernel_sizes", "pool_bins", "head_hidden_dims"})
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

# device values match the semantics of runtime.select_device.
_DEVICES = frozenset({"auto", "cpu", "cuda"})
_TRANSFORMS = frozenset({"identity", "logalpha"})
# Tolerance for the ratios sum: tolerates floating-point rounding of hand-written decimals.
_RATIO_TOLERANCE = 1e-9


@dataclass(frozen=True)
class DataConfig:
    """``data`` block: source of the pulse order."""

    # None = frozen from the first selected config.yaml snapshot and recorded in
    # dataset_meta; non-None = explicit pulse_id list (must be a duplicate-free
    # permutation of the protocol set).
    pulse_order: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ModelConfig:
    """``model`` block: MLP network architecture (with [64,32,32] the parameter count is
    ``64·D + 3266``).
    """

    hidden_dims: tuple[int, ...] = (64, 32, 32)
    # Fixed kind marker; init=False keeps the old positional construction
    # (ModelConfig()) and field order unchanged.
    kind: Literal["mlp"] = field(default="mlp", init=False)


@dataclass(frozen=True)
class CNN1DModelConfig:
    """``model`` block: 1D CNN network architecture.

    All four fields are required (``head_hidden_dims`` may be an empty tuple);
    ``kind`` is fixed to ``"cnn1d"``. ``pool_bins <= T`` is validated by the
    model layer; the config layer does not read data.
    """

    channels: tuple[int, ...]
    kernel_sizes: tuple[int, ...]
    pool_bins: int
    head_hidden_dims: tuple[int, ...]
    kind: Literal["cnn1d"] = field(default="cnn1d", init=False)


@dataclass(frozen=True)
class LabelConfig:
    """``label`` block: label transform (distinct from the alpha-space sampling design;
    the two are independent).
    """

    transform: LabelTransform = "identity"  # logalpha takes log10 of alpha only; requires alpha > 0


@dataclass(frozen=True)
class PreprocessingConfig:
    """``preprocessing`` block: normalization hyperparameters (statistics are fit on
    the train split only).
    """

    std_eps: float = (
        1.0e-8  # Uniform for x/y: std <= eps → divisor 1, and zero-variance positions are recorded
    )


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """``training.early_stopping``: min_delta affects only the stop decision, not the best rule."""

    patience: int = 50
    min_delta: float = 0.0  # 0 is valid: every non-improving epoch counts toward patience


@dataclass(frozen=True)
class TrainingParams:
    """``training`` block: optimization and runtime hyperparameters (initial
    engineering candidates, not validated research parameters)."""

    seed: int = 42  # consistent with configs/base.yaml
    device: str = "auto"  # auto|cpu|cuda, passed to runtime.select_device
    batch_size: int = 32
    max_epochs: int = 500
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0  # 0 is valid (no L2 regularization)
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()


@dataclass(frozen=True)
class SplitRatios:
    """``split.ratios``: all three must be non-negative and finite and sum to 1 (tolerance 1e-9)."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15


@dataclass(frozen=True)
class SplitMinCounts:
    """``split.min_per_split``: if any group is short → prepare fails before
    writing samples."""

    train: int = 2
    val: int = 1
    test: int = 1


@dataclass(frozen=True)
class SplitConfig:
    """``split`` block: used only by the prepare script (algorithm in training_data.make_split)."""

    seed: int = 42
    ratios: SplitRatios = SplitRatios()
    min_per_split: SplitMinCounts = SplitMinCounts()


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment config aggregate root (in-memory representation of one
    ``configs/training/mlp.yaml``).
    """

    # Required (not null/empty; replace placeholder values before running); if the
    # output directory for run_name already exists, startup is refused rather than
    # overwriting implicitly.
    dataset_name: str
    run_name: str
    data: DataConfig = DataConfig()
    model: ModelConfig | CNN1DModelConfig = ModelConfig()
    label: LabelConfig = LabelConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    training: TrainingParams = TrainingParams()
    split: SplitConfig = SplitConfig()
    # None = output_root()/training/mlp/<dataset_name>/<run_name>.
    output_dir: str | None = None


def _check_keys(
    mapping: dict[Any, Any], field: str, known: frozenset[str], required: frozenset[str]
) -> None:
    """Strict schema: reject unknown fields; only the ``required`` subset is mandatory
    (the rest may be omitted).
    """
    unknown = sorted((key for key in mapping if key not in known), key=repr)
    if unknown:
        _fail(field, f"unknown fields {unknown!r}; allowed fields {sorted(known)}")
    missing = sorted(required.difference(mapping))
    if missing:
        _fail(field, f"missing required fields {missing!r}")


def _optional_section(
    root: dict[Any, Any], field: str, known: frozenset[str]
) -> dict[Any, Any] | None:
    """Optional config section: absent → None (caller falls back to dataclass defaults)."""
    if field not in root:
        return None
    raw = _require_mapping(root[field], field)
    _check_keys(raw, field, known, frozenset())
    return raw


def _parse_pulse_order(value: object, field: str) -> tuple[str, ...] | None:
    """pulse_order: null → None; otherwise non-empty with safe path segments and no duplicates."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        _fail(field, f"must be null or a non-empty pulse_id list (got {value!r})")
    ids = tuple(_require_safe_path_segment(item, f"{field}[{i}]") for i, item in enumerate(value))
    if len(set(ids)) != len(ids):
        _fail(field, f"duplicate pulse_id (got {value!r})")
    return ids


def _parse_hidden_dims(value: object, field: str) -> tuple[int, ...]:
    """hidden_dims: non-empty sequence of positive integers."""
    if not isinstance(value, list) or not value:
        _fail(field, f"must be a non-empty integer list (got {value!r})")
    return tuple(_require_positive_int(item, f"{field}[{i}]") for i, item in enumerate(value))


def _parse_int_sequence(value: object, field: str, *, allow_empty: bool) -> tuple[int, ...]:
    """Integer sequence: elements are positive integers (rejects bool/float/string);
    the empty sequence is allowed on demand.
    """
    if not isinstance(value, list):
        _fail(field, f"must be an integer list (got {value!r})")
    if not value and not allow_empty:
        _fail(field, f"must be a non-empty integer list (got {value!r})")
    return tuple(_require_positive_int(item, f"{field}[{i}]") for i, item in enumerate(value))


def _parse_model_value(value: object, field: str) -> ModelConfig | CNN1DModelConfig:
    """Strict ``model`` block parsing (shared by YAML and config_from_mapping).

    - ``kind`` absent is equivalent to ``"mlp"`` (only the old MLP field
      ``hidden_dims`` is valid);
    - ``kind="cnn1d"`` requires all four fields: ``channels``/``kernel_sizes``
      are non-empty positive integers with matching layer counts and all-odd
      kernels, ``pool_bins`` is a positive integer, and ``head_hidden_dims`` is
      provided explicitly (may be empty) with each element a positive integer;
    - kind fields are strictly mutually exclusive, unknown fields are rejected;
      ``kind`` and sequence elements both reject bool.
    """
    raw = _require_mapping(value, field)  # null/non-mapping → error (no relaxation)
    kind = raw.get("kind", "mlp")
    if isinstance(kind, bool) or not isinstance(kind, str) or kind not in _MODEL_KINDS:
        _fail(f"{field}.kind", f"must be one of {sorted(_MODEL_KINDS)} (got {kind!r})")
    if kind == "cnn1d":
        _check_keys(raw, field, _MODEL_KEYS_CNN1D, _CNN1D_REQUIRED)
        channels = _parse_int_sequence(raw["channels"], f"{field}.channels", allow_empty=False)
        kernel_sizes = _parse_int_sequence(
            raw["kernel_sizes"], f"{field}.kernel_sizes", allow_empty=False
        )
        if len(kernel_sizes) != len(channels):
            _fail(
                f"{field}.kernel_sizes",
                f"layer count must match channels (got {len(kernel_sizes)} vs {len(channels)})",
            )
        if any(kernel % 2 == 0 for kernel in kernel_sizes):
            _fail(f"{field}.kernel_sizes", f"kernels must be odd (got {list(kernel_sizes)!r})")
        return CNN1DModelConfig(
            channels=channels,
            kernel_sizes=kernel_sizes,
            pool_bins=_require_positive_int(raw["pool_bins"], f"{field}.pool_bins"),
            head_hidden_dims=_parse_int_sequence(
                raw["head_hidden_dims"], f"{field}.head_hidden_dims", allow_empty=True
            ),
        )
    _check_keys(raw, field, _MODEL_KEYS_MLP, frozenset())
    kwargs: dict[str, Any] = {}
    if "hidden_dims" in raw:
        kwargs["hidden_dims"] = _parse_hidden_dims(raw["hidden_dims"], f"{field}.hidden_dims")
    return ModelConfig(**kwargs)


def _parse_ratios(value: object, field: str) -> SplitRatios:
    """ratios: non-negative finite numbers summing to 1 (within tolerance)."""
    raw = _require_mapping(value, field)
    _check_keys(raw, field, _RATIO_KEYS, _RATIO_KEYS)
    values = {
        name: _require_non_negative_number(raw[name], f"{field}.{name}")
        for name in ("train", "val", "test")
    }
    total = sum(values.values())
    if abs(total - 1.0) > _RATIO_TOLERANCE:
        _fail(
            field, f"the three terms must sum to 1 (tolerance {_RATIO_TOLERANCE}) (got {total!r})"
        )
    return SplitRatios(train=values["train"], val=values["val"], test=values["test"])


def _parse_min_per_split(value: object, field: str) -> SplitMinCounts:
    """min_per_split: non-negative integers (0 allowed = the group may be empty)."""
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


def _parse_model(root: dict[Any, Any]) -> ModelConfig | CNN1DModelConfig:
    """``model`` section: absent → MLP default; present → shared strict parser (null rejected)."""
    if "model" not in root:
        return ModelConfig()
    return _parse_model_value(root["model"], "model")


def _parse_label(root: dict[Any, Any]) -> LabelConfig:
    raw = _optional_section(root, "label", _LABEL_KEYS)
    if raw is None or "transform" not in raw:
        return LabelConfig()
    transform = raw["transform"]
    if transform not in _TRANSFORMS:
        _fail("label.transform", f"must be one of {sorted(_TRANSFORMS)} (got {transform!r})")
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
            _fail("training.device", f"must be one of {sorted(_DEVICES)} (got {raw['device']!r})")
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
        _fail("output_dir", f"must be null or a non-empty string (got {value!r})")
    return value


def load_config(path: Path) -> ExperimentConfig:
    """Read YAML, run the single-entry validation, and build an ExperimentConfig.

    Validation boundary (the pipeline assumes a valid config afterwards): every
    level uses a strict schema (unknown fields and duplicate YAML keys rejected);
    dataset_name/run_name are required, safe single path segments, and must not
    keep placeholders; data.pulse_order is null or a duplicate-free pulse_id
    list; model.kind ∈ {mlp,cnn1d} and is validated strictly per kind (MLP
    hidden_dims non-empty positive integers; CNN channels/kernel_sizes non-empty
    positive integers with matching layer counts and odd kernels, pool_bins a
    positive integer, head_hidden_dims explicitly present and possibly empty);
    label.transform takes a legal value; std_eps/learning_rate/batch_size/
    max_epochs/patience are positive, weight_decay/min_delta non-negative (0 is
    valid); device ∈ {auto,cpu,cuda} (same semantics as runtime.select_device);
    ratios are non-negative, finite, and sum to 1 (tolerance 1e-9);
    min_per_split are non-negative integers; output_dir is null or a non-empty
    string. Absent optional sections use dataclass defaults.

    Raises:
        ConfigError: messages include the field path (e.g. training.batch_size).
    """
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: failed to parse YAML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: failed to read: {exc}") from exc

    root = _require_mapping(raw, str(path))
    _check_keys(root, str(path), _TOP_LEVEL_KEYS, frozenset({"dataset_name", "run_name"}))

    dataset_name = _require_safe_path_segment(root["dataset_name"], "dataset_name")
    if dataset_name == PLACEHOLDER_DATASET_NAME:
        _fail("dataset_name", f"placeholder must be replaced before running (got {dataset_name!r})")
    run_name = _require_safe_path_segment(root["run_name"], "run_name")
    if run_name == PLACEHOLDER_RUN_NAME:
        _fail("run_name", f"placeholder must be replaced before running (got {run_name!r})")

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


def _model_to_mapping(model: ModelConfig | CNN1DModelConfig) -> dict[str, Any]:
    """model → plain dict: MLP keeps the old layout (no kind, for old ckpt nested
    config compatibility); CNN emits kind + the four fields (reloadable by
    ``config_from_mapping``).
    """
    if isinstance(model, CNN1DModelConfig):
        return {
            "kind": model.kind,
            "channels": list(model.channels),
            "kernel_sizes": list(model.kernel_sizes),
            "pool_bins": model.pool_bins,
            "head_hidden_dims": list(model.head_hidden_dims),
        }
    return {"hidden_dims": list(model.hidden_dims)}


def config_to_mapping(config: ExperimentConfig) -> dict[str, Any]:
    """ExperimentConfig → nested plain dict (mapping schema maintained by this module).

    The shape matches ``configs/training/mlp.yaml`` (reloadable by
    ``load_config``) and is also the on-disk form of the ckpt ``config`` copy;
    containers are all primitives/list/dict (tuple → list), with no dataclass or
    numpy objects.
    """
    return {
        "dataset_name": config.dataset_name,
        "run_name": config.run_name,
        "data": {
            "pulse_order": (
                list(config.data.pulse_order) if config.data.pulse_order is not None else None
            )
        },
        "model": _model_to_mapping(config.model),
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
    """Optional config section: absent → empty mapping (caller falls back to
    dataclass defaults); a non-mapping raises."""
    value = mapping.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        _fail(f"config.{key}", f"must be a mapping or null (got {type(value)!r})")
    return value


def config_from_mapping(mapping: Mapping[str, Any]) -> ExperimentConfig:
    """Nested plain dict → ExperimentConfig (single entry point for rebuilding the
    ckpt ``config`` copy).

    Symmetric with ``config_to_mapping``; absent sections/keys fall back to
    dataclass defaults (the ckpt copy is always complete). Strict YAML loading
    still goes through ``load_config`` (placeholder/ratios-sum checks belong to
    the YAML boundary and are not repeated here).

    Raises:
        ConfigError: non-mapping, missing dataset_name/run_name, a non-mapping
            section, or an illegal field type.
    """
    if not isinstance(mapping, Mapping):
        _fail("config", f"must be a mapping (got {type(mapping)!r})")
    for key in ("dataset_name", "run_name"):
        if key not in mapping:
            _fail("config", f"missing required key {key!r}")
    data_raw = _mapping_section(mapping, "data")
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
            model=(
                ModelConfig()
                if "model" not in mapping
                else _parse_model_value(mapping["model"], "config.model")
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
        _fail("config", f"illegal field type: {exc}")
