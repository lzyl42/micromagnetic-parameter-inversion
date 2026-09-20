"""Training loop and checkpoint I/O.

Responsibilities:

- seed entry: ``train_model`` calls ``set_seed`` **before constructing the
  model** (the training seed is owned solely by train_model); DataLoader shuffle
  uses an explicitly seeded ``torch.Generator`` (``make_data_generator``).
- training: Adam + MSE in the standardized label space (sample-count-weighted
  aggregation); batches are transformed per state on CPU into float32 numpy and
  then sent to the device; ``num_workers=0``; non-finite loss/grad/pred stops
  immediately without saving bad weights (best/final only hold weights from
  completed epochs); early stopping uses an independent reference (updated only
  when the improvement exceeds ``min_delta``) and is kept separate from the
  absolute best (updated on any strictly lower value).
- checkpoint: ``save_checkpoint``/``load_checkpoint`` are **MLP-only**, and
  ``CKPT_FORMAT_VERSION`` with the existing payload/schema is unchanged. The
  CNN uses an independent ``CNNCheckpoint``/``save_cnn_checkpoint``/
  ``load_cnn_checkpoint`` (``CNN_CKPT_FORMAT_VERSION`` + ``model_kind="cnn1d"``);
  ``load_any_checkpoint`` reads once safely and routes on the explicit
  ``model_kind``, with an absent kind accepting only the legal old MLP layout,
  and unknown/corrupt payloads always raising ``TrainingError`` without
  fallback. On the CNN save side any normal Tensor device is accepted (not
  restricted to CPU) and normalized with ``detach().to("cpu")`` before writing;
  the load side requires CPU. The on-disk dict contains only
  primitives/list/dict + CPU Tensors (no dataclass/numpy objects); an existing
  file is never overwritten; reading uses the explicit safe mode
  ``torch.load(weights_only=True, map_location="cpu")``.
- data flow: ``train_model`` returns ``TrainingResult`` (best/final checkpoint +
  per-epoch history); ``run(config_path, expected_kind=...)`` is the shared
  orchestration for ``train_mlp.py``/``train_cnn1d.py``, handling load_config
  (once), early kind rejection (before reading data/creating directories),
  train-only fitting, and the on-disk writes of best.pt/final.pt/metrics.json;
  checkpoint I/O stays in this module. Per-kind writing dispatches through
  ``save_model_checkpoint`` (CNN keeps an independent schema, never written as
  MLP).

Dependency direction: training_config / training_data / preprocessing / runtime →
(this module) → evaluation; this module must not import evaluation (the
evaluation side calls back into ``load_checkpoint`` / ``build_model``).
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from micromagnetic_parameter_inversion import (
    paths,
    preprocessing,
    runtime,
    training_config,
    training_data,
)
from micromagnetic_parameter_inversion.models.cnn1d import CNN1DRegressor
from micromagnetic_parameter_inversion.models.mlp import MLPRegressor
from micromagnetic_parameter_inversion.preprocessing import (
    PreprocessingError,
    PreprocessingState,
    transform_x,
    transform_y,
)
from micromagnetic_parameter_inversion.training_config import (
    CKPT_FORMAT_VERSION,
    ActivationName,
    CNN1DModelConfig,
    ConfigError,
    ExperimentConfig,
    ModelKind,
    _require_positive_int,
    load_config,
)
from micromagnetic_parameter_inversion.training_data import (
    InputContract,
    SampleItem,
    TrajectoryDataset,
)

# ckpt_format_version comes from training_config.CKPT_FORMAT_VERSION.
type StateDict = Mapping[str, Tensor]

# Stop reason: normal cap / early stopping / numerical failure (the ckpt format
# version is unchanged; stop metadata lives only in TrainingResult and
# metrics.json, never in the ckpt).
type StopReason = Literal["max_epochs", "early_stopping", "numerical_failure"]

_ACTIVATIONS = ("relu",)  # fixed ReLU for the first version (Checkpoint.activation records it)
_N_OUTPUTS = 2  # number of standardized label columns (alpha, ku_j_per_m3)
_N_CHANNELS = 3  # number of magnetization components (mx, my, mz)

# Independent CNN checkpoint format version (unrelated to the MLP CKPT_FORMAT_VERSION).
CNN_CKPT_FORMAT_VERSION = 1
# CNN top-level explicit structure fields (restore authority; the nested config
# model structure is recorded only).
_CNN_STRUCTURE_FIELDS = ("channels", "kernel_sizes", "pool_bins", "head_hidden_dims")
_MODEL_COMPONENT_ORDER = ("mx", "my", "mz")


class TrainingError(RuntimeError):
    """Training/ckpt contract violation (non-finite training state, format version
    mismatch, corrupted fields/shapes).

    ``epoch``/``detail``: on numerical failure, ``train_model`` carries the
    triggering epoch (1-based) and the exact reason so the CLI can write
    failure-state metrics; other violations may omit them.
    """

    def __init__(
        self, message: str, *, epoch: int | None = None, detail: str | None = None
    ) -> None:
        super().__init__(message)
        self.epoch = epoch
        self.detail = detail


@dataclass(frozen=True)
class EpochMetrics:
    """Per-epoch standardized MSE record (source of the metrics.json history).

    ``train_loss`` is the **online accumulated** batch loss of the epoch
    weighted by sample count (the accumulated value at epoch end); ``val_loss``
    is the sample-weighted MSE over the whole validation set; both are in the
    standardized label space.
    """

    epoch: int  # 1-based
    train_loss: float  # online batch-weighted mean standardized MSE
    val_loss: (
        float  # val-set weighted mean standardized MSE (criterion for best and early stopping)
    )


# MLP-only checkpoint; see CNNCheckpoint below (fully independent schema, no migration).
@dataclass(frozen=True, eq=False)
class Checkpoint:
    """Schema of best.pt / final.pt (standalone inference from ckpt + npz only).

    Contains a nested ``InputContract`` (with ndarray fields) and uses
    ``eq=False`` to avoid element-wise comparison ambiguity. Evaluation-side
    uses: ``hidden_dims``/``activation``/``contract`` restore the model
    structure and input contract (without the current YAML); ``preprocessing``
    restores the x/y transforms (without refitting); ``split_sha256`` binds and
    validates against the split copy stored in the run; and
    ``dataset_meta_relpath``/``dataset_meta_sha256`` locate and verify
    dataset_meta anchored at ``data_root()/samples/<dataset_name>/``. ``config``
    is recorded only and is not a restore source.
    """

    ckpt_format_version: int  # schema version at write time
    model_state_dict: StateDict  # model weights (on disk: a dict of CPU Tensors)
    hidden_dims: tuple[
        int, ...
    ]  # explicit model structure field (not only buried in the config copy)
    activation: ActivationName  # activation recorded explicitly (fixed "relu")
    contract: InputContract  # input contract: pulse order, T, component order, t_s
    preprocessing: (
        PreprocessingState  # preprocessing state: x [P,1,3] mean, effective scale, label, y stats
    )
    seed: int  # training.seed
    config: ExperimentConfig  # effective config copy (recorded only)
    dataset_meta_relpath: str  # path relative to the anchor data_root()/samples/<dataset>/
    dataset_meta_sha256: str  # dataset_meta content fingerprint (lightweight provenance)
    split_sha256: str  # sha256 of the run's split copy (split binding)
    best_val_loss: float | None  # always set in best.pt; may be None in final.pt
    git_sha: str | None = None  # recorded when available; independent of the dirty flag
    git_dirty: bool | None = None  # whether the worktree has uncommitted changes (when available)
    torch_version: str = ""
    numpy_version: str = ""


@dataclass(frozen=True, eq=False)
class CNNCheckpoint:
    """Independent CNN checkpoint schema (same field names/semantics as ``Checkpoint``,
    only the structure fields differ).

    Fully independent from ``Checkpoint``: the **top-level explicit structure
    fields** are the restore authority and the nested ``config`` is recorded
    only (it must not override the structure); ``model_kind="cnn1d"`` appears
    only in the payload and is not an extra dataclass argument. The evaluation
    side's ``load_state_dict(strict=True)`` handles key/shape matching; this
    module does not pre-construct models (avoiding RNG side effects).
    """

    ckpt_format_version: int  # CNN schema version at write time (CNN_CKPT_FORMAT_VERSION)
    model_state_dict: StateDict  # model weights (on disk: a dict of CPU Tensors)
    channels: tuple[int, ...]  # explicit structure: output channel count of each Conv1d layer
    kernel_sizes: tuple[int, ...]  # explicit structure: kernel length of each layer (odd)
    pool_bins: int  # explicit structure: AdaptiveAvgPool1d output bin count
    head_hidden_dims: tuple[
        int, ...
    ]  # explicit structure: regression head hidden widths (may be empty)
    activation: ActivationName  # activation recorded explicitly (fixed "relu")
    contract: InputContract  # input contract: pulse order, T, component order, t_s
    preprocessing: PreprocessingState  # preprocessing state (train-only fit)
    seed: int  # training.seed
    config: (
        ExperimentConfig  # effective config copy (recorded only; model must be CNN1DModelConfig)
    )
    dataset_meta_relpath: str  # path relative to the anchor data_root()/samples/<dataset>/
    dataset_meta_sha256: str  # dataset_meta content fingerprint (lightweight provenance)
    split_sha256: str  # sha256 of the run's split copy (split binding)
    best_val_loss: float | None  # always set in best.pt; may be None in final.pt
    git_sha: str | None = None  # recorded when available; independent of the dirty flag
    git_dirty: bool | None = None  # whether the worktree has uncommitted changes (when available)
    torch_version: str = ""
    numpy_version: str = ""


# Explicit model-kind routing result (return type of load_any_checkpoint).
type ModelCheckpoint = Checkpoint | CNNCheckpoint


@dataclass(frozen=True)
class TrainingResult:
    """Return value of ``train_model``; the entry script orchestrates artifact writes from it.

    ``best_checkpoint``/``final_checkpoint`` are ``ModelCheckpoint`` (MLP
    ``Checkpoint`` or CNN ``CNNCheckpoint``, determined by the ``config.model``
    kind, with different schemas): ``best_checkpoint`` → best.pt (best_val_loss
    required, the absolute best weights); ``final_checkpoint`` → final.pt
    (weights of the last completed epoch, best_val_loss is None); ``history`` →
    per-epoch MSE for metrics.json. ``stop_reason``/``stop_epoch``/``detail``
    record the stop state (the ckpt format version is unchanged and stop
    metadata lives only in TrainingResult and metrics.json):

    - ``max_epochs``: the limit was reached, ``stop_epoch`` = max_epochs;
    - ``early_stopping``: patience triggered, ``stop_epoch`` = the triggering epoch;
    - ``numerical_failure``: non-finite loss/grad/pred/aggregate; ``detail``
      carries the exact reason; best/final then hold the weights of the last
      **complete and finite** epoch, and the CLI must exit non-zero and write
      failure-state metrics (without printing training complete).
    """

    best_checkpoint: ModelCheckpoint
    final_checkpoint: ModelCheckpoint
    history: tuple[EpochMetrics, ...]
    stop_reason: StopReason
    stop_epoch: int  # the epoch that triggered the stop (1-based)
    detail: str | None = None  # exact reason for numerical failure etc.; None on normal stop


def set_seed(seed: int) -> None:
    """Seed entry point: called before model construction (prerequisite for
    reproducible weight initialization).

    Sets ``torch.manual_seed(seed)`` and the numpy global seed (``seed`` comes
    from ``training.seed`` and is non-negative; the numpy side takes it modulo
    2^32). No cross-hardware bitwise equivalence is promised; reproducibility
    holds on the same machine and versions.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)


def make_data_generator(seed: int) -> torch.Generator:
    """Explicitly seeded generator for DataLoader (reproducible shuffle; CPU generator)."""
    return torch.Generator().manual_seed(seed)


# MLP factory (signature/behavior unchanged); the CNN uses build_cnn_model below.
def build_model(contract: InputContract, hidden_dims: tuple[int, ...]) -> MLPRegressor:
    """Build the MLP from the input contract and hidden widths (activation fixed to
    ReLU, see MLPRegressor).

    ``contract.input_shape == (P, T, 3)`` determines the flattened dimension
    ``D = P*T*3``; hidden_dims defaults to ``(64, 32, 32)``. The evaluation side
    restores the structure with ``build_model(ckpt.contract, ckpt.hidden_dims)``.
    """
    return MLPRegressor(input_shape=contract.input_shape, hidden_dims=tuple(hidden_dims))


def build_cnn_model(
    contract: InputContract,
    *,
    channels: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    pool_bins: int,
    head_hidden_dims: tuple[int, ...],
) -> CNN1DRegressor:
    """Build a CNN1DRegressor from the input contract and explicit structure (no default
    hyperparameters).

    First strictly validates the contract semantics (component order, pulse
    order, T/C/t_s consistency with strictly increasing t_s), then delegates to
    the model constructor for structure validation (illegal structure raises
    ``ValueError`` from ``CNN1DRegressor``).

    Raises:
        TrainingError: the contract is illegal.
    """
    _validate_cnn_contract(contract, "build_cnn_model")
    return CNN1DRegressor(
        input_shape=contract.input_shape,
        channels=tuple(channels),
        kernel_sizes=tuple(kernel_sizes),
        pool_bins=pool_bins,
        head_hidden_dims=tuple(head_hidden_dims),
    )


def _batch_to_device(
    batch: tuple[Tensor, Tensor, object], state: PreprocessingState, device: str
) -> tuple[Tensor, Tensor]:
    """raw CPU batch → transformed per state on CPU to float32 numpy → device tensors."""
    x, y, _psids = batch
    x_dev = torch.from_numpy(transform_x(state, x.numpy())).to(device)
    y_dev = torch.from_numpy(transform_y(state, y.numpy())).to(device)
    return x_dev, y_dev


def run_validation(
    model: nn.Module,
    loader: DataLoader[SampleItem],
    state: PreprocessingState,
    device: str,
) -> float:
    """Compute the standardized MSE over the given DataLoader (sample-weighted aggregate).

    Items are raw unnormalized ``SampleItem`` values and transforms are applied
    on CPU per ``state``. The model is temporarily switched to eval mode and
    restored afterwards. Non-finite predictions/squared errors →
    TrainingError (the caller stops training on it and saves no bad weights).
    """
    was_training = model.training
    model.eval()
    total_se = 0.0
    total_n = 0
    try:
        with torch.no_grad():
            for batch in loader:
                x, y = _batch_to_device(batch, state, device)
                pred = model(x)
                if not bool(torch.isfinite(pred).all()):
                    raise TrainingError(
                        "validation prediction contains non-finite values (NaN/Inf)"
                    )
                se = (pred - y) ** 2
                if not bool(torch.isfinite(se).all()):
                    raise TrainingError("validation loss contains non-finite values (NaN/Inf)")
                # float32 elements are finite but a float32 sum may overflow
                # (e.g. several 1e38): aggregate in float64.
                total_se += float(se.double().sum().item())
                total_n += int(x.shape[0])
    finally:
        if was_training:
            model.train()
    if total_n == 0:
        raise TrainingError("validation set is empty, cannot compute val MSE")
    aggregate = total_se / (total_n * _N_OUTPUTS)
    if not math.isfinite(aggregate):
        raise TrainingError("validation aggregate loss is non-finite (NaN/Inf)")
    return aggregate


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[SampleItem],
    optimizer: torch.optim.Optimizer,
    state: PreprocessingState,
    device: str,
) -> float | None:
    """Train one epoch; returns the online accumulated weighted mean MSE, or None on
    non-finite state (stop).

    Order: zero_grad → forward (stop on non-finite pred) → batch mean loss (stop
    on non-finite) → backward (stop on non-finite grad, no step) → step → at
    epoch end check the weights are still finite.
    """
    model.train()
    total_se = 0.0
    total_n = 0
    for batch in loader:
        x, y = _batch_to_device(batch, state, device)
        optimizer.zero_grad()
        pred = model(x)
        if not bool(torch.isfinite(pred).all()):
            return None
        se = (pred - y) ** 2
        loss = se.mean()
        if not bool(torch.isfinite(loss)):
            return None
        loss.backward()
        if any(
            param.grad is not None and not bool(torch.isfinite(param.grad).all())
            for param in model.parameters()
        ):
            return None  # exploding/non-finite gradient: no step, avoid bad weights
        optimizer.step()
        # float32 elements are finite but a float32 sum may overflow: aggregate in float64.
        total_se += float(se.detach().double().sum().item())
        total_n += int(x.shape[0])
        if any(not bool(torch.isfinite(param).all()) for param in model.parameters()):
            return None
    if total_n == 0:
        return None
    aggregate = total_se / (total_n * _N_OUTPUTS)
    return aggregate if math.isfinite(aggregate) else None


def _clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """CPU deep copy of state_dict (later training no longer affects saved weights)."""
    return {name: value.detach().to("cpu").clone() for name, value in model.state_dict().items()}


def _git_info() -> tuple[str | None, bool | None]:
    """Repository git SHA and dirty flag (read-only subprocess limited to the repo
    path; failure → None).
    """
    repo_root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    if not sha:
        return None, None
    return sha, bool(status.strip())


# The shared training orchestration is ``run`` below (same module as train_model,
# no mutual import); the entry scripts only declare expected_kind and forward.
def train_model(
    config: ExperimentConfig,
    train_set: TrajectoryDataset,
    val_set: TrajectoryDataset,
    state: PreprocessingState,
    contract: InputContract,
    split_sha256: str,
    *,
    dataset_meta_relpath: str = "dataset_meta.yaml",
    dataset_meta_sha256: str | None = None,
) -> TrainingResult:
    """Main training routine.

    Args:
        config: experiment config (seed/device/hyperparameters/early stopping).
        train_set/val_set: Dataset of raw unnormalized samples (test never enters training).
        state: preprocessing state fitted on the train split only.
        contract: input contract (frozen by the caller from the first train sample).
        split_sha256: fingerprint of the run's split copy (split binding).
        dataset_meta_relpath: meta path relative to the anchor
            ``data_root()/samples/<dataset>/`` (default "dataset_meta.yaml").
        dataset_meta_sha256: actual dataset_meta file fingerprint; None → "".

    Orchestration: ``set_seed`` (before model construction) → build the model by
    the ``config.model`` kind (``ModelConfig`` → ``build_model``;
    ``CNN1DModelConfig`` → ``build_cnn_model``) → ``runtime.select_device`` →
    DataLoader (train shuffle=True + seeded generator; num_workers=0) → Adam →
    per epoch ``_train_one_epoch`` + ``run_validation`` → best (absolute minimum,
    updated only on a strictly lower value, CPU clone) and early stopping
    (independent reference, updated only when the improvement exceeds min_delta)
    advance separately → non-finite loss/grad/pred/aggregate stops immediately
    without saving bad weights (the reason is recorded in
    ``stop_reason``/``stop_epoch``/``detail`` and not swallowed) → returns
    ``TrainingResult`` (the on-disk writes are orchestrated by ``run``).

    Raises:
        TrainingError: training stopped due to numerical failure at the first
            epoch (no weights can be saved); the exception carries
            ``epoch``/``detail``; or an illegal model structure (ValueError is
            converted to a domain TrainingError).
    """
    model_config = config.model
    set_seed(config.training.seed)  # before model construction: reproducible weight init
    model: nn.Module
    try:
        if isinstance(model_config, CNN1DModelConfig):
            model = build_cnn_model(
                contract,
                channels=tuple(model_config.channels),
                kernel_sizes=tuple(model_config.kernel_sizes),
                pool_bins=int(model_config.pool_bins),
                head_hidden_dims=tuple(model_config.head_hidden_dims),
            )
        else:
            model = build_model(contract, model_config.hidden_dims)
    except (
        ValueError
    ) as exc:  # model-layer structure validation failed (e.g. pool_bins > T) → domain error
        raise TrainingError(f"illegal model structure: {exc}") from exc
    device = runtime.select_device(config.training.device)
    model.to(device)
    train_loader: DataLoader[SampleItem] = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=make_data_generator(config.training.seed),
        num_workers=0,
    )
    val_loader: DataLoader[SampleItem] = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    history: list[EpochMetrics] = []
    best_val = math.inf
    best_state: dict[str, Tensor] | None = None
    final_state: dict[str, Tensor] | None = None
    stop_reference = (
        math.inf
    )  # independent early-stopping reference (separate from the absolute best)
    epochs_without_improvement = 0
    stop_reason: StopReason | None = None
    stop_epoch = 0
    failure_detail: str | None = None

    for epoch in range(1, config.training.max_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, state, device)
        if train_loss is None:
            stop_reason, stop_epoch = "numerical_failure", epoch
            failure_detail = "training-side loss/gradient/weights are non-finite"
            break  # keep the existing best, save no bad weights
        try:
            val_loss = run_validation(model, val_loader, state, device)
        except TrainingError as exc:
            stop_reason, stop_epoch, failure_detail = "numerical_failure", epoch, str(exc)
            break  # validation-side non-finite: reason propagates via detail, not swallowed
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            stop_reason, stop_epoch = "numerical_failure", epoch
            failure_detail = "aggregate loss is non-finite (NaN/Inf)"
            break  # non-finite aggregate loss: keep best/final from the previous
            # complete epoch; this epoch writes no history and saves no weights
        history.append(EpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
        final_state = _clone_state_dict(model)  # weights of the last actually completed epoch
        if val_loss < best_val:  # absolute best: update only on a strictly lower value
            best_val = val_loss
            best_state = final_state
        if stop_reference - val_loss > config.training.early_stopping.min_delta:
            stop_reference = val_loss  # advance the reference only on a significant improvement
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.training.early_stopping.patience:
            stop_reason, stop_epoch = "early_stopping", epoch
            break

    if stop_reason is None:
        stop_reason, stop_epoch = "max_epochs", config.training.max_epochs

    if not history or best_state is None or final_state is None:
        raise TrainingError(
            "training stopped due to numerical failure at the first epoch; "
            "no weights can be saved"
            f" (epoch={stop_epoch}, detail={failure_detail})",
            epoch=stop_epoch,
            detail=failure_detail,
        )

    git_sha, git_dirty = _git_info()
    common_meta: dict[str, Any] = dict(
        activation="relu",
        contract=contract,
        preprocessing=state,
        seed=config.training.seed,
        config=config,
        dataset_meta_relpath=dataset_meta_relpath,
        dataset_meta_sha256=dataset_meta_sha256 if dataset_meta_sha256 is not None else "",
        split_sha256=split_sha256,
        git_sha=git_sha,
        git_dirty=git_dirty,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
    )
    best_checkpoint: ModelCheckpoint
    if isinstance(model_config, CNN1DModelConfig):
        best_checkpoint = CNNCheckpoint(
            ckpt_format_version=CNN_CKPT_FORMAT_VERSION,
            model_state_dict=best_state,
            channels=tuple(model_config.channels),
            kernel_sizes=tuple(model_config.kernel_sizes),
            pool_bins=int(model_config.pool_bins),
            head_hidden_dims=tuple(model_config.head_hidden_dims),
            best_val_loss=best_val,
            **common_meta,
        )
    else:
        best_checkpoint = Checkpoint(
            ckpt_format_version=CKPT_FORMAT_VERSION,
            model_state_dict=best_state,
            hidden_dims=tuple(model_config.hidden_dims),
            best_val_loss=best_val,
            **common_meta,
        )
    final_checkpoint = replace(best_checkpoint, model_state_dict=final_state, best_val_loss=None)
    return TrainingResult(
        best_checkpoint=best_checkpoint,
        final_checkpoint=final_checkpoint,
        history=tuple(history),
        stop_reason=stop_reason,
        stop_epoch=stop_epoch,
        detail=failure_detail,
    )


# --- Shared training orchestration: the single implementation for train_mlp.py / train_cnn1d.py ---

_EXPECTED_KINDS: frozenset[str] = frozenset({"mlp", "cnn1d"})


def _resolve_expected_kind(expected_kind: object) -> ModelKind:
    """Model kind declared by the entry: only ``"mlp"``/``"cnn1d"`` are accepted, otherwise
    ConfigError."""
    if not isinstance(expected_kind, str) or expected_kind not in _EXPECTED_KINDS:
        raise ConfigError(f"invalid expected_kind: {expected_kind!r} (must be 'mlp' or 'cnn1d')")
    return cast(ModelKind, expected_kind)


def _freeze_contract(train_set: TrajectoryDataset) -> InputContract:
    """Freeze the input contract from the first **cached** sample (no disk access);
    remaining train members are checked in cache.
    """
    first = train_set.samples[0]
    contract = InputContract(
        pulse_order=first.pulse_ids,
        n_time_steps=int(first.x.shape[1]),
        t_s=first.t_s,
    )
    train_set.validate_contract(contract)
    return contract


def _metrics_payload(
    history: Sequence[EpochMetrics],
    best_val_loss: float | None,
    stop_reason: str,
    stop_epoch: int,
    detail: str | None,
) -> dict[str, Any]:
    """metrics.json document: history + best val + stop state (no test evaluation; finite
    values)."""
    return {
        "history": [
            {"epoch": m.epoch, "train_loss": m.train_loss, "val_loss": m.val_loss} for m in history
        ],
        "best_val_loss": best_val_loss,
        "best_epoch": max(history, key=lambda m: -m.val_loss).epoch if history else None,
        "stop_reason": stop_reason,
        "stop_epoch": stop_epoch,
        "detail": detail,
    }


def _write_metrics(path: Path, result: TrainingResult) -> None:
    """Write metrics.json: per-epoch history + best val + stop state (finite JSON values)."""
    payload = _metrics_payload(
        result.history,
        result.best_checkpoint.best_val_loss,
        result.stop_reason,
        result.stop_epoch,
        result.detail,
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _write_failure_metrics(path: Path, error: TrainingError) -> None:
    """Failure-state metrics for a first-epoch numerical failure (no checkpoint to keep)."""
    payload = _metrics_payload(
        [],
        None,
        "numerical_failure",
        error.epoch if error.epoch is not None else 0,
        error.detail if error.detail is not None else str(error),
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def run(config_path: Path, *, expected_kind: ModelKind) -> Path:
    """Shared training orchestration: config → samples/split → train_model → artifacts,
    returning the run directory.

    The same implementation regardless of entry; ``expected_kind`` declares the
    model kind this entry requires:

    1. ``load_config(config_path)`` **once**; then immediately validate that
       ``expected_kind`` is legal and ``config.model.kind`` matches — **before**
       calling ``paths.data_root()``, reading samples, or creating the output
       directory; a mismatch raises ``ConfigError``;
    2. output directory: when ``config.output_dir`` is non-null use
       ``Path(...)`` verbatim (relative paths resolve against the current working
       directory); otherwise ``output_root()/training/<kind>/<dataset>/<run_name>``;
       an existing directory raises ``FileExistsError`` (before training);
    3. same frozen split + dataset_meta location (``data_root()/samples/<dataset>/``);
       construct only train/val (test is not loaded and takes no part in model selection);
    4. ``_freeze_contract`` + ``preprocessing.fit`` (**train split only**; this
       run fits its own statistics and reads no existing ckpt/preprocessing stats);
    5. after the raw split copy bytes + SHA and the dataset_meta SHA, call ``train_model``;
    6. the run directory is created after training returns, then in order write the
       split copy (raw bytes), ``config_resolved.yaml``, ``preprocessing.yaml``,
       ``metrics.json``, ``best.pt``/``final.pt`` (via ``save_model_checkpoint``
       dispatched by kind).

    Failure semantics (preserving the original MLP convention): a first-epoch
    numerical failure (``TrainingError`` carrying ``epoch``) → create the run
    directory, write only the failure-state ``metrics.json``, then re-raise;
    other domain errors (kind/structure/data) re-raise before the directory is
    created, leaving no artifacts. A ``numerical_failure`` stop after training
    returned (with valid best/final snapshots) → artifacts have been written,
    then raise ``TrainingError`` (the caller exits non-zero and prints no
    completion).

    Raises:
        ConfigError: illegal config, illegal ``expected_kind``, or a kind mismatch.
        training_data.DataError / PreprocessingError: sample/split/contract issues.
        FileExistsError: the run directory already exists.
        TrainingError: illegal model structure or a training numerical failure.
    """
    config = load_config(config_path)
    kind = _resolve_expected_kind(expected_kind)
    if config.model.kind != kind:
        raise ConfigError(
            f"training entry requires model.kind={kind!r}, but the config is "
            f"{config.model.kind!r} ({config_path})"
        )

    samples_dir = paths.data_root() / "samples" / config.dataset_name
    if not samples_dir.is_dir():
        raise training_data.DataError(f"sample directory does not exist: {samples_dir}")
    meta = training_data.load_dataset_meta(samples_dir)
    split = training_data.load_split(samples_dir, meta)
    # The training contract requires non-empty train/val (test may be empty: neither
    # trained nor evaluated).
    if not split.train:
        raise training_data.DataError(
            f"split.train is empty: training requires at least 1 train member "
            f"(split copy at {samples_dir / 'split.yaml'})"
        )
    if not split.val:
        raise training_data.DataError(
            f"split.val is empty: early stopping and best selection require a val member "
            f"(split copy at {samples_dir / 'split.yaml'})"
        )

    output_dir = (
        Path(config.output_dir)
        if config.output_dir is not None
        else paths.output_root() / "training" / kind / config.dataset_name / config.run_name
    )
    if output_dir.exists():
        raise FileExistsError(f"run directory already exists, refusing to overwrite: {output_dir}")

    # Construct only train/val; test never enters tuning/model selection and is not loaded.
    # Each npz is read exactly once: train loads without a contract and is cached, then
    # the contract frozen from the first cached sample is checked in cache; val is
    # validated during its single read.
    train_set = TrajectoryDataset(samples_dir, meta, split.train)
    contract = _freeze_contract(train_set)
    val_set = TrajectoryDataset(samples_dir, meta, split.val, contract=contract)
    state = preprocessing.fit(train_set, config.preprocessing, config.label)

    split_source = samples_dir / "split.yaml"
    split_bytes = split_source.read_bytes()
    split_sha256 = training_data.sha256_file(split_source)
    meta_source = samples_dir / "dataset_meta.yaml"
    meta_sha256 = training_data.sha256_file(meta_source)

    try:
        result = train_model(
            config,
            train_set,
            val_set,
            state,
            contract,
            split_sha256,
            dataset_meta_relpath=meta_source.name,
            dataset_meta_sha256=meta_sha256,
        )
    except TrainingError as exc:
        # Only a first-epoch numerical failure (with epoch) has no weights to keep →
        # write failure-state metrics; structure/contract errors (epoch is None) re-raise
        # before the directory is created, leaving no artifacts.
        if exc.epoch is not None:
            output_dir.mkdir(parents=True)
            _write_failure_metrics(output_dir / "metrics.json", exc)
        raise

    # The run directory is created after training returns; later failures may leave artifacts.
    output_dir.mkdir(parents=True)
    (output_dir / "split.yaml").write_bytes(split_bytes)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            training_config.config_to_mapping(config), sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
    (output_dir / "preprocessing.yaml").write_text(
        yaml.safe_dump(preprocessing.state_to_mapping(state), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _write_metrics(output_dir / "metrics.json", result)
    save_model_checkpoint(output_dir / "best.pt", result.best_checkpoint)
    save_model_checkpoint(output_dir / "final.pt", result.final_checkpoint)

    if result.stop_reason == "numerical_failure":
        # A valid snapshot (previous complete epoch) was saved, but this run counts as a
        # failure: non-zero exit and no training-complete message.
        raise TrainingError(
            f"training stopped due to numerical failure at epoch {result.stop_epoch}: "
            f"{result.detail}",
            epoch=result.stop_epoch,
            detail=result.detail,
        )

    print(f"training complete: {output_dir}")
    print(f"  epochs={len(result.history)}; best_val_loss={result.best_checkpoint.best_val_loss!r}")
    print(f"  stop: {result.stop_reason} @ epoch {result.stop_epoch}")
    return output_dir


def save_checkpoint(path: Path, ckpt: Checkpoint) -> None:
    """Serialize a Checkpoint with ``torch.save``; an existing file → FileExistsError.

    The on-disk dict contains only Python primitives/list/dict + CPU Tensors
    (dataclasses and numpy objects are always expanded/converted, see
    ``_checkpoint_to_dict``); weight tensors are detached + moved to CPU +
    cloned before saving. Runtime kind guard: a non-``Checkpoint`` is rejected
    (``TrainingError``) before mkdir/write, so the MLP schema is never written
    by mistake.
    """
    if not isinstance(ckpt, Checkpoint):
        raise TrainingError(
            f"save_checkpoint requires an MLP Checkpoint (got {type(ckpt)!r}) ({path})"
        )
    if path.exists():
        raise FileExistsError(f"checkpoint already exists, refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_to_dict(ckpt), path)


def save_model_checkpoint(path: Path, ckpt: ModelCheckpoint) -> None:
    """Dispatch to the independent save function by explicit checkpoint kind (never
    write CNN as MLP).

    ``CNNCheckpoint`` → ``save_cnn_checkpoint`` (independent ``model_kind="cnn1d"``
    schema); otherwise → ``save_checkpoint`` (which carries its own runtime kind
    guard, rejecting wrong kinds before mkdir).
    """
    if isinstance(ckpt, CNNCheckpoint):
        save_cnn_checkpoint(path, ckpt)
        return
    save_checkpoint(path, ckpt)


def _validate_cnn_checkpoint_for_save(ckpt: CNNCheckpoint, path: Path) -> None:
    """Pre-save validation: obvious errors (kind/version/structure/config type) are
    rejected before mkdir/write."""
    if not isinstance(ckpt, CNNCheckpoint):
        raise TrainingError(
            f"save_cnn_checkpoint requires a CNNCheckpoint (got {type(ckpt)!r}) ({path})"
        )
    _require_ckpt_version(ckpt.ckpt_format_version, CNN_CKPT_FORMAT_VERSION, path)
    if ckpt.activation not in _ACTIVATIONS:
        raise TrainingError(f"unknown activation function {ckpt.activation!r} ({path})")
    _validate_cnn_contract(ckpt.contract, path)
    _cnn_structure_from_values(
        ckpt.channels,
        ckpt.kernel_sizes,
        ckpt.pool_bins,
        ckpt.head_hidden_dims,
        ckpt.contract.n_time_steps,
        path,
    )
    if not isinstance(ckpt.config.model, CNN1DModelConfig):
        raise TrainingError(f"CNN checkpoint config.model must be CNN1DModelConfig ({path})")
    # Nested config round-trip: illegal numbers (e.g. batch_size=inf raising
    # OverflowError via int) become TrainingError before mkdir/write; no bad data is written.
    try:
        training_config.config_from_mapping(training_config.config_to_mapping(ckpt.config))
    except (ConfigError, OverflowError, TypeError, ValueError) as exc:
        raise TrainingError(f"CNN checkpoint config cannot round-trip ({path}): {exc}") from exc
    # The save side only validates structure (any normal Tensor device/requires_grad is
    # allowed); before writing everything is moved to CPU and detached, see
    # _cnn_checkpoint_to_dict.
    _cnn_state_dict_structure(ckpt.model_state_dict, path)
    _validate_cnn_preprocessing(ckpt.preprocessing, path)
    seed = ckpt.seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TrainingError(f"seed must be a non-negative integer (got {seed!r}) ({path})")
    best_val_loss = ckpt.best_val_loss
    if best_val_loss is not None and (
        isinstance(best_val_loss, bool)
        or not isinstance(best_val_loss, (int, float))
        or not math.isfinite(float(best_val_loss))
    ):
        raise TrainingError(
            f"best_val_loss must be null or a finite number (got {best_val_loss!r}) ({path})"
        )


def save_cnn_checkpoint(path: Path, ckpt: CNNCheckpoint) -> None:
    """Serialize a CNNCheckpoint with ``torch.save``; an existing file → FileExistsError.

    Obvious errors (version/structure/config/state_dict) are rejected before
    mkdir/write; the on-disk dict contains only primitives/list/dict + CPU
    Tensors (weights detached + CPU + cloned).
    """
    if path.exists():
        raise FileExistsError(f"checkpoint already exists, refusing to overwrite: {path}")
    _validate_cnn_checkpoint_for_save(ckpt, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cnn_checkpoint_to_dict(ckpt), path)


def _read_checkpoint_payload(path: Path) -> dict[str, Any]:
    """Single safe read of a checkpoint payload (weights_only, CPU, must be a dict)."""
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch.load exception types vary across versions; converge them here
        raise TrainingError(f"failed to load checkpoint ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingError(
            f"checkpoint payload must be a mapping (got {type(payload)!r}) ({path})"
        )
    return payload


def _require_ckpt_version(value: object, expected: int, context: object) -> int:
    """Format version: a non-bool positive integer that must equal expected, otherwise
    TrainingError."""
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise TrainingError(
            f"incompatible ckpt_format_version: {value!r} != {expected} ({context})"
        )
    return value


_MLP_CKPT_REQUIRED_KEYS = frozenset(
    {
        "ckpt_format_version",
        "model_state_dict",
        "hidden_dims",
        "activation",
        "contract",
        "preprocessing",
        "seed",
        "config",
        "dataset_meta_relpath",
        "dataset_meta_sha256",
        "split_sha256",
        "best_val_loss",
    }
)


def _mlp_checkpoint_from_payload(payload: dict[str, Any], path: Path) -> Checkpoint:
    """MLP payload → Checkpoint (original parsing body; only moved + a kind format guard).

    Rejects any ``model_kind`` field and CNN structure fields: the MLP ckpt does
    not use a kind key, so their presence is a kind mismatch.
    """
    if "model_kind" in payload:
        raise TrainingError(f"MLP checkpoint does not accept a model_kind field ({path})")
    if any(field in payload for field in _CNN_STRUCTURE_FIELDS):
        raise TrainingError(f"MLP checkpoint does not accept CNN structure fields ({path})")
    missing = sorted(_MLP_CKPT_REQUIRED_KEYS.difference(payload))
    if missing:
        raise TrainingError(f"checkpoint is missing keys {missing} ({path})")
    if payload["ckpt_format_version"] != CKPT_FORMAT_VERSION:
        raise TrainingError(
            f"incompatible ckpt_format_version: {payload['ckpt_format_version']!r} "
            f"!= {CKPT_FORMAT_VERSION} ({path})"
        )
    if payload["activation"] not in _ACTIVATIONS:
        raise TrainingError(f"unknown activation function {payload['activation']!r} ({path})")

    state_raw = payload["model_state_dict"]
    if not isinstance(state_raw, Mapping) or not state_raw:
        raise TrainingError(f"model_state_dict must be a non-empty mapping ({path})")
    state_dict = {
        str(name): value for name, value in state_raw.items() if isinstance(value, Tensor)
    }
    if len(state_dict) != len(state_raw):
        raise TrainingError(f"model_state_dict contains non-tensor entries ({path})")
    if any(value.device.type != "cpu" for value in state_dict.values()):
        raise TrainingError(f"model_state_dict contains non-CPU tensors ({path})")

    contract = _contract_from_dict(payload["contract"], path)
    preprocessing = _preprocessing_from_dict(
        payload["preprocessing"], len(contract.pulse_order), path
    )
    config = _config_from_dict(payload["config"], path)
    best_val_loss = payload["best_val_loss"]
    if best_val_loss is not None and not isinstance(best_val_loss, (int, float)):
        raise TrainingError(f"illegal best_val_loss type ({path})")
    return Checkpoint(
        ckpt_format_version=int(payload["ckpt_format_version"]),
        model_state_dict=state_dict,
        hidden_dims=tuple(int(dim) for dim in payload["hidden_dims"]),
        activation=payload["activation"],
        contract=contract,
        preprocessing=preprocessing,
        seed=int(payload["seed"]),
        config=config,
        dataset_meta_relpath=str(payload["dataset_meta_relpath"]),
        dataset_meta_sha256=str(payload["dataset_meta_sha256"]),
        split_sha256=str(payload["split_sha256"]),
        best_val_loss=float(best_val_loss) if best_val_loss is not None else None,
        git_sha=payload.get("git_sha"),
        git_dirty=payload.get("git_dirty"),
        torch_version=str(payload.get("torch_version", "")),
        numpy_version=str(payload.get("numpy_version", "")),
    )


def load_checkpoint(path: Path) -> Checkpoint:
    """Safely read an **MLP** Checkpoint (the evaluation-side MLP restore entry).

    Rejects files carrying ``model_kind`` or CNN structure fields (kind
    mismatch); for CNN use ``load_cnn_checkpoint``, and for automatic routing use
    ``load_any_checkpoint``.

    Raises:
        FileNotFoundError: the file does not exist.
        TrainingError: format version mismatch, missing keys, or corrupted
            shapes/types.
    """
    return _mlp_checkpoint_from_payload(_read_checkpoint_payload(path), path)


def load_cnn_checkpoint(path: Path) -> CNNCheckpoint:
    """Safely read a **CNN** checkpoint (requires an explicit ``model_kind="cnn1d"``).

    Does not accept MLP ckpts, unknown kinds, or a missing kind; corruption
    always raises ``TrainingError`` with no fallback.
    """
    return _cnn_checkpoint_from_payload(_read_checkpoint_payload(path), path)


def load_any_checkpoint(path: Path) -> ModelCheckpoint:
    """Read once safely, then route on the explicit ``model_kind``: ``cnn1d``→CNN;
    absent kind→only the legal old MLP layout.

    Any other explicit kind (including ``"mlp"``), unknown/null/bad values raise
    ``TrainingError``; a missing kind that still carries any CNN structure field
    is likewise rejected, with no guessing and no fallback.
    """
    payload = _read_checkpoint_payload(path)
    if "model_kind" in payload:
        if payload["model_kind"] == "cnn1d":
            return _cnn_checkpoint_from_payload(payload, path)
        raise TrainingError(f"unsupported model_kind: {payload['model_kind']!r} ({path})")
    if any(field in payload for field in _CNN_STRUCTURE_FIELDS):
        raise TrainingError(
            f"missing model_kind but CNN structure fields are present; refusing to read as "
            f"MLP ({path})"
        )
    return _mlp_checkpoint_from_payload(payload, path)


def _contract_from_dict(raw: Any, path: Path) -> InputContract:
    """Contract rebuild + shape validation (t_s length == T, channel count always 3,
    P matches pulse order).
    """
    if not isinstance(raw, Mapping):
        raise TrainingError(f"contract must be a mapping ({path})")
    pulse_order = tuple(str(p) for p in raw["pulse_order"])
    n_time_steps = int(raw["n_time_steps"])
    n_channels = int(raw["n_channels"])
    t_s = np.asarray(raw["t_s"], dtype=np.float64)
    if n_time_steps < 1 or len(pulse_order) < 1:
        raise TrainingError(f"contract P/T must be positive ({path})")
    if n_channels != 3:
        raise TrainingError(f"contract channel count is always 3 (got {n_channels}) ({path})")
    if t_s.shape != (n_time_steps,):
        raise TrainingError(
            f"contract t_s shape {t_s.shape} does not match T={n_time_steps} ({path})"
        )
    return InputContract(
        pulse_order=pulse_order,
        n_time_steps=n_time_steps,
        t_s=t_s,
        n_channels=n_channels,
        component_order=tuple(str(c) for c in raw.get("component_order") or ("mx", "my", "mz")),
    )


def _preprocessing_from_dict(raw: Any, n_pulse: int, path: Path) -> PreprocessingState:
    """ckpt preprocessing state rebuild: delegates to
    preprocessing.state_from_mapping (the owner) and cross-checks P against the
    contract; PreprocessingError is wrapped as TrainingError at the ckpt boundary.
    """
    try:
        state = preprocessing.state_from_mapping(raw)
    except PreprocessingError as exc:
        raise TrainingError(f"corrupted preprocessing state ({path}): {exc}") from exc
    if state.x_stats.mean.shape[0] != n_pulse:
        raise TrainingError(
            f"preprocessing P={state.x_stats.mean.shape[0]} does not match contract "
            f"P={n_pulse} ({path})"
        )
    return state


def _validate_cnn_preprocessing(state: PreprocessingState, path: Path) -> None:
    """CNN preprocessing numeric bounds (CNN boundary only; the old MLP schema/loader is unchanged).

    mean (x/y) must be finite; the effective scale (x/y std) must be finite and
    strictly > 0. Zero-variance positions are written as 1.0 by the fitting side,
    so 1.0 is legal; 0/negative/inf/NaN are always rejected (no bad data is written).
    """
    x_mean = np.asarray(state.x_stats.mean, dtype=np.float64)
    x_std = np.asarray(state.x_stats.std, dtype=np.float64)
    y_mean = np.asarray(state.y_stats.y_mean, dtype=np.float64)
    y_std = np.asarray(state.y_stats.y_std, dtype=np.float64)
    if not bool(np.isfinite(x_mean).all()):
        raise TrainingError(f"preprocessing x mean contains non-finite values ({path})")
    if not bool(np.isfinite(x_std).all()) or bool((x_std <= 0.0).any()):
        raise TrainingError(f"preprocessing x effective scale must be finite and positive ({path})")
    if not bool(np.isfinite(y_mean).all()):
        raise TrainingError(f"preprocessing y mean contains non-finite values ({path})")
    if not bool(np.isfinite(y_std).all()) or bool((y_std <= 0.0).any()):
        raise TrainingError(f"preprocessing y effective scale must be finite and positive ({path})")


def _config_from_dict(raw: Any, path: Path) -> ExperimentConfig:
    """ckpt config copy rebuild: delegates to
    training_config.config_from_mapping (the owner); ConfigError is wrapped as
    TrainingError at the ckpt boundary (without depending back on evaluation, and
    without duplicating the mapping schema in training).
    """
    try:
        return training_config.config_from_mapping(raw)
    except (ConfigError, OverflowError) as exc:
        raise TrainingError(f"corrupted config copy ({path}): {exc}") from exc


def _cnn_state_dict_structure(raw: Any, path: Path) -> dict[str, Tensor]:
    """CNN state_dict structure validation (non-empty mapping, str keys, Tensor values);
    no device restriction.

    Used on the save side: any normal Tensor device (e.g. CUDA/requires_grad) is
    allowed; before writing everything is normalized by
    ``_cnn_checkpoint_to_dict`` with ``detach().to("cpu").clone()``.
    """
    if not isinstance(raw, Mapping) or not raw:
        raise TrainingError(f"model_state_dict must be a non-empty mapping ({path})")
    state_dict: dict[str, Tensor] = {}
    for name, value in raw.items():
        if not isinstance(name, str):
            raise TrainingError(
                f"model_state_dict keys must be strings (got {type(name)!r}) ({path})"
            )
        if not isinstance(value, Tensor):
            raise TrainingError(f"model_state_dict contains non-tensor entries ({path})")
        state_dict[name] = value
    return state_dict


def _cnn_state_dict_from_payload(raw: Any, path: Path) -> dict[str, Tensor]:
    """CNN load boundary: structure validation + must be CPU Tensors (corruption → uniform
    TrainingError)."""
    state_dict = _cnn_state_dict_structure(raw, path)
    for value in state_dict.values():
        if value.device.type != "cpu":
            raise TrainingError(f"model_state_dict contains non-CPU tensors ({path})")
    return state_dict


def _cnn_positive_int(value: object, field: str) -> int:
    """Positive integer (rejects bool/float/string); ConfigError is uniformly wrapped as
    TrainingError."""
    try:
        return _require_positive_int(value, field)
    except ConfigError as exc:
        raise TrainingError(str(exc)) from exc


def _cnn_int_sequence(value: object, field: str, *, allow_empty: bool) -> tuple[int, ...]:
    """CNN structure integer sequence: accepts tuple/list, rejects bool/float/string and
    non-positive elements."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TrainingError(f"{field}: must be an integer sequence (got {value!r})")
    sequence = tuple(value)
    if not sequence and not allow_empty:
        raise TrainingError(f"{field}: must be a non-empty integer sequence (got {value!r})")
    for index, item in enumerate(sequence):
        _cnn_positive_int(item, f"{field}[{index}]")
    return sequence


def _cnn_structure_from_values(
    channels: object,
    kernel_sizes: object,
    pool_bins: object,
    head_hidden_dims: object,
    n_time_steps: int,
    context: object,
) -> tuple[tuple[int, ...], tuple[int, ...], int, tuple[int, ...]]:
    """Strict CNN structure validation (equivalent to the config-layer spec: rejects
    bool/float/string smuggling)."""
    normalized_channels = _cnn_int_sequence(channels, f"{context}: channels", allow_empty=False)
    normalized_kernels = _cnn_int_sequence(
        kernel_sizes, f"{context}: kernel_sizes", allow_empty=False
    )
    normalized_pool_bins = _cnn_positive_int(pool_bins, f"{context}: pool_bins")
    normalized_head = _cnn_int_sequence(
        head_hidden_dims, f"{context}: head_hidden_dims", allow_empty=True
    )
    if len(normalized_kernels) != len(normalized_channels):
        raise TrainingError(
            f"kernel_sizes layer count must match channels "
            f"(got {len(normalized_kernels)} vs {len(normalized_channels)}) ({context})"
        )
    if any(kernel % 2 == 0 for kernel in normalized_kernels):
        raise TrainingError(
            f"kernel_sizes must all be odd (got {list(normalized_kernels)!r}) ({context})"
        )
    if normalized_pool_bins > n_time_steps:
        raise TrainingError(
            f"pool_bins must be <= T={n_time_steps} (got {normalized_pool_bins}) ({context})"
        )
    return normalized_channels, normalized_kernels, normalized_pool_bins, normalized_head


def _validate_cnn_contract(contract: InputContract, context: object) -> None:
    """CNN contract semantic validation: component order, pulse order, T/C/t_s consistency with
    strictly increasing t_s."""
    if not isinstance(contract, InputContract):
        raise TrainingError(
            f"contract must be an InputContract (got {type(contract)!r}) ({context})"
        )
    if tuple(contract.component_order) != _MODEL_COMPONENT_ORDER:
        raise TrainingError(
            f"contract component order must be {_MODEL_COMPONENT_ORDER} "
            f"(got {contract.component_order!r}) ({context})"
        )
    pulse_order = tuple(contract.pulse_order)
    if (
        not pulse_order
        or any(not isinstance(p, str) or not p for p in pulse_order)
        or len(set(pulse_order)) != len(pulse_order)
    ):
        raise TrainingError(
            f"contract pulse_order must be a non-empty sequence of unique non-empty strings "
            f"({context})"
        )
    n_time = contract.n_time_steps
    if isinstance(n_time, bool) or not isinstance(n_time, int) or n_time <= 0:
        raise TrainingError(
            f"contract n_time_steps must be a positive integer (got {n_time!r}) ({context})"
        )
    if contract.n_channels != _N_CHANNELS:
        raise TrainingError(
            f"contract channel count must be {_N_CHANNELS} (got {contract.n_channels}) ({context})"
        )
    t_s = np.asarray(contract.t_s, dtype=np.float64)
    if t_s.shape != (n_time,):
        raise TrainingError(f"contract t_s shape {t_s.shape} does not match T={n_time} ({context})")
    if not bool(np.isfinite(t_s).all()):
        raise TrainingError(f"contract t_s contains non-finite values ({context})")
    if n_time > 1 and not bool(np.all(np.diff(t_s) > 0)):
        raise TrainingError(f"contract t_s must be strictly increasing ({context})")


def _cnn_contract_from_dict(raw: Any, path: Path) -> InputContract:
    """Strict CNN contract rebuild (bool/float posing as T/C rejected; t_s length
    consistent, finite, strictly increasing)."""
    if not isinstance(raw, Mapping):
        raise TrainingError(f"contract must be a mapping ({path})")
    missing = sorted(
        {"pulse_order", "n_time_steps", "n_channels", "t_s", "component_order"}.difference(raw)
    )
    if missing:
        raise TrainingError(f"contract is missing keys {missing} ({path})")
    pulse_raw = raw["pulse_order"]
    if isinstance(pulse_raw, (str, bytes)) or not isinstance(pulse_raw, (list, tuple)):
        raise TrainingError(f"contract pulse_order must be a sequence ({path})")
    pulse_order = tuple(pulse_raw)
    if (
        not pulse_order
        or any(not isinstance(p, str) or not p for p in pulse_order)
        or len(set(pulse_order)) != len(pulse_order)
    ):
        raise TrainingError(
            f"contract pulse_order must be a non-empty sequence of unique non-empty strings "
            f"({path})"
        )
    try:
        n_time_steps = _require_positive_int(raw["n_time_steps"], f"contract.n_time_steps ({path})")
        n_channels = _require_positive_int(raw["n_channels"], f"contract.n_channels ({path})")
    except ConfigError as exc:
        raise TrainingError(str(exc)) from exc
    if n_channels != _N_CHANNELS:
        raise TrainingError(
            f"contract channel count must be {_N_CHANNELS} (got {n_channels}) ({path})"
        )
    component_raw = raw["component_order"]
    if isinstance(component_raw, (str, bytes)) or not isinstance(component_raw, (list, tuple)):
        raise TrainingError(f"contract component_order must be a sequence ({path})")
    component_order = tuple(component_raw)
    if component_order != _MODEL_COMPONENT_ORDER:
        raise TrainingError(
            f"contract component order must be {_MODEL_COMPONENT_ORDER} "
            f"(got {component_order!r}) ({path})"
        )
    t_raw = raw["t_s"]
    if isinstance(t_raw, (str, bytes)) or not isinstance(t_raw, (list, tuple, np.ndarray)):
        raise TrainingError(f"contract t_s must be a sequence ({path})")
    try:
        t_s = np.asarray(t_raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"contract t_s cannot be parsed as float64 ({path}): {exc}") from exc
    if t_s.ndim != 1 or t_s.shape != (n_time_steps,):
        raise TrainingError(
            f"contract t_s shape {t_s.shape} does not match T={n_time_steps} ({path})"
        )
    if not bool(np.isfinite(t_s).all()):
        raise TrainingError(f"contract t_s contains non-finite values ({path})")
    if n_time_steps > 1 and not bool(np.all(np.diff(t_s) > 0)):
        raise TrainingError(f"contract t_s must be strictly increasing ({path})")
    return InputContract(
        pulse_order=pulse_order,
        n_time_steps=n_time_steps,
        t_s=t_s,
        n_channels=n_channels,
        component_order=component_order,
    )


_CNN_CKPT_REQUIRED_KEYS = frozenset(
    {
        "ckpt_format_version",
        "model_kind",
        "model_state_dict",
        "channels",
        "kernel_sizes",
        "pool_bins",
        "head_hidden_dims",
        "activation",
        "contract",
        "preprocessing",
        "seed",
        "config",
        "dataset_meta_relpath",
        "dataset_meta_sha256",
        "split_sha256",
        "best_val_loss",
    }
)


def _cnn_checkpoint_from_payload(payload: dict[str, Any], path: Path) -> CNNCheckpoint:
    """CNN payload → CNNCheckpoint (strict kind/version/structure/contract; config.model
    must be CNN)."""
    missing = sorted(_CNN_CKPT_REQUIRED_KEYS.difference(payload))
    if missing:
        raise TrainingError(f"CNN checkpoint is missing keys {missing} ({path})")
    if payload["model_kind"] != "cnn1d":
        raise TrainingError(f"model_kind must be 'cnn1d' (got {payload['model_kind']!r}) ({path})")
    if "hidden_dims" in payload:
        raise TrainingError(f"CNN checkpoint does not accept a hidden_dims field ({path})")
    _require_ckpt_version(payload["ckpt_format_version"], CNN_CKPT_FORMAT_VERSION, path)
    if payload["activation"] not in _ACTIVATIONS:
        raise TrainingError(f"unknown activation function {payload['activation']!r} ({path})")
    state_dict = _cnn_state_dict_from_payload(payload["model_state_dict"], path)
    contract = _cnn_contract_from_dict(payload["contract"], path)
    channels, kernel_sizes, pool_bins, head_hidden_dims = _cnn_structure_from_values(
        payload["channels"],
        payload["kernel_sizes"],
        payload["pool_bins"],
        payload["head_hidden_dims"],
        contract.n_time_steps,
        path,
    )
    preprocessing = _preprocessing_from_dict(
        payload["preprocessing"], len(contract.pulse_order), path
    )
    _validate_cnn_preprocessing(preprocessing, path)
    config = _config_from_dict(payload["config"], path)
    if not isinstance(config.model, CNN1DModelConfig):
        raise TrainingError(f"CNN checkpoint config.model must be CNN1DModelConfig ({path})")
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TrainingError(f"seed must be a non-negative integer (got {seed!r}) ({path})")
    best_val_loss = payload["best_val_loss"]
    if best_val_loss is not None and (
        isinstance(best_val_loss, bool)
        or not isinstance(best_val_loss, (int, float))
        or not math.isfinite(float(best_val_loss))
    ):
        raise TrainingError(
            f"best_val_loss must be null or a finite number (got {best_val_loss!r}) ({path})"
        )
    return CNNCheckpoint(
        ckpt_format_version=CNN_CKPT_FORMAT_VERSION,
        model_state_dict=state_dict,
        channels=channels,
        kernel_sizes=kernel_sizes,
        pool_bins=pool_bins,
        head_hidden_dims=head_hidden_dims,
        activation=payload["activation"],
        contract=contract,
        preprocessing=preprocessing,
        seed=seed,
        config=config,
        dataset_meta_relpath=str(payload["dataset_meta_relpath"]),
        dataset_meta_sha256=str(payload["dataset_meta_sha256"]),
        split_sha256=str(payload["split_sha256"]),
        best_val_loss=float(best_val_loss) if best_val_loss is not None else None,
        git_sha=payload.get("git_sha"),
        git_dirty=payload.get("git_dirty"),
        torch_version=str(payload.get("torch_version", "")),
        numpy_version=str(payload.get("numpy_version", "")),
    )


def _contract_to_dict(contract: InputContract) -> dict[str, Any]:
    """InputContract → plain dict (shared by MLP/CNN on-disk; schema unchanged)."""
    return {
        "pulse_order": [str(p) for p in contract.pulse_order],
        "n_time_steps": int(contract.n_time_steps),
        "t_s": np.asarray(contract.t_s, dtype=np.float64).tolist(),
        "n_channels": int(contract.n_channels),
        "component_order": [str(c) for c in contract.component_order],
    }


def _checkpoint_to_dict(ckpt: Checkpoint) -> dict[str, Any]:
    """Checkpoint → on-disk dict for torch.save (primitives/list/dict + CPU Tensors only)."""
    return {
        "ckpt_format_version": int(ckpt.ckpt_format_version),
        "model_state_dict": {
            name: value.detach().to("cpu").clone() for name, value in ckpt.model_state_dict.items()
        },
        "hidden_dims": [int(dim) for dim in ckpt.hidden_dims],
        "activation": str(ckpt.activation),
        "contract": _contract_to_dict(ckpt.contract),
        "preprocessing": preprocessing.state_to_mapping(ckpt.preprocessing),
        "seed": int(ckpt.seed),
        "config": training_config.config_to_mapping(ckpt.config),
        "dataset_meta_relpath": str(ckpt.dataset_meta_relpath),
        "dataset_meta_sha256": str(ckpt.dataset_meta_sha256),
        "split_sha256": str(ckpt.split_sha256),
        "best_val_loss": (float(ckpt.best_val_loss) if ckpt.best_val_loss is not None else None),
        "git_sha": ckpt.git_sha,
        "git_dirty": ckpt.git_dirty,
        "torch_version": str(ckpt.torch_version),
        "numpy_version": str(ckpt.numpy_version),
    }


def _cnn_checkpoint_to_dict(ckpt: CNNCheckpoint) -> dict[str, Any]:
    """CNNCheckpoint → on-disk dict for torch.save (primitives/list/dict + CPU Tensors only)."""
    return {
        "ckpt_format_version": int(ckpt.ckpt_format_version),
        "model_kind": "cnn1d",
        "model_state_dict": {
            name: value.detach().to("cpu").clone() for name, value in ckpt.model_state_dict.items()
        },
        "channels": [int(c) for c in ckpt.channels],
        "kernel_sizes": [int(k) for k in ckpt.kernel_sizes],
        "pool_bins": int(ckpt.pool_bins),
        "head_hidden_dims": [int(h) for h in ckpt.head_hidden_dims],
        "activation": str(ckpt.activation),
        "contract": _contract_to_dict(ckpt.contract),
        "preprocessing": preprocessing.state_to_mapping(ckpt.preprocessing),
        "seed": int(ckpt.seed),
        "config": training_config.config_to_mapping(ckpt.config),
        "dataset_meta_relpath": str(ckpt.dataset_meta_relpath),
        "dataset_meta_sha256": str(ckpt.dataset_meta_sha256),
        "split_sha256": str(ckpt.split_sha256),
        "best_val_loss": (float(ckpt.best_val_loss) if ckpt.best_val_loss is not None else None),
        "git_sha": ckpt.git_sha,
        "git_dirty": ckpt.git_dirty,
        "torch_version": str(ckpt.torch_version),
        "numpy_version": str(ckpt.numpy_version),
    }
