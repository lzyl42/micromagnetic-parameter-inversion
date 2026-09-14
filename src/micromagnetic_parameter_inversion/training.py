"""Training loop and checkpoint I/O.

- seed: ``train_model`` calls ``set_seed`` before constructing the model;
  DataLoader shuffle uses an explicitly seeded ``torch.Generator``.
- Training: Adam + MSE in the normalized label space (weighted by sample
  count); non-finite loss/grad/pred stops immediately without saving bad
  weights (best/final only contain weights from completed epochs); early
  stopping uses an independent reference (updated only when the improvement
  exceeds ``min_delta``) and is kept separate from the absolute best (updated
  on any strictly lower value).
- checkpoint: only Python primitives/list/dict + CPU Tensors are saved, and an
  existing file is never overwritten; ``load_checkpoint`` reads safely with
  ``weights_only=True`` and rebuilds the dataclass after validating the format
  version and key field/contract shapes.
- Data flow: ``train_model`` returns ``TrainingResult``; the on-disk writes of
  best.pt/final.pt/metrics.json are orchestrated by ``train_mlp.py``.

Dependency direction: training_config / training_data / preprocessing / runtime →
(this module) → evaluation; this module must not import evaluation (the
evaluation side calls back into ``load_checkpoint`` / ``build_model``).
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from micromagnetic_parameter_inversion import preprocessing, runtime, training_config
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
    ConfigError,
    ExperimentConfig,
)
from micromagnetic_parameter_inversion.training_data import (
    InputContract,
    SampleItem,
    TrajectoryDataset,
)

# ckpt_format_version comes from training_config.CKPT_FORMAT_VERSION.
type StateDict = Mapping[str, Tensor]

# Stop reason: max epochs / early stopping / numerical failure (the ckpt format
# version is unchanged; stop metadata lives only in TrainingResult and
# metrics.json, never in the ckpt).
type StopReason = Literal["max_epochs", "early_stopping", "numerical_failure"]

_ACTIVATIONS = ("relu",)  # fixed ReLU (Checkpoint.activation records it explicitly)
_N_OUTPUTS = 2  # number of normalized label columns (alpha, ku_j_per_m3)


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
    """Normalized MSE record for one epoch (source of metrics.json history).

    ``train_loss`` is the **online accumulated** batch loss of the epoch
    weighted by sample count (the accumulated value at epoch end);
    ``val_loss`` is the sample-weighted MSE over the whole validation set;
    both are in the normalized label space.
    """

    epoch: int  # 1-based
    train_loss: float  # online batch-weighted mean normalized MSE
    val_loss: float  # val-set weighted mean normalized MSE (criterion for best and early stopping)


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


@dataclass(frozen=True)
class TrainingResult:
    """Return value of ``train_model``; the entry script orchestrates artifact writes from it.

    ``best_checkpoint`` → best.pt (best_val_loss required, the absolute best
    weights); ``final_checkpoint`` → final.pt (weights of the last completed
    epoch, best_val_loss is None); ``history`` → per-epoch MSE for metrics.json.
    ``stop_reason``/``stop_epoch``/``detail`` record the stop state (the ckpt
    format version is unchanged and stop metadata lives only in TrainingResult
    and metrics.json):

    - ``max_epochs``: the limit was reached, ``stop_epoch`` = max_epochs;
    - ``early_stopping``: patience triggered, ``stop_epoch`` = the triggering epoch;
    - ``numerical_failure``: non-finite loss/grad/pred/aggregate; ``detail``
      carries the exact reason; best/final then hold the weights of the last
      **complete and finite** epoch, and the CLI must exit non-zero and write
      failure-state metrics (without printing training complete).
    """

    best_checkpoint: Checkpoint
    final_checkpoint: Checkpoint
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


def build_model(contract: InputContract, hidden_dims: tuple[int, ...]) -> MLPRegressor:
    """Build the MLP from the input contract and hidden widths (activation fixed to
    ReLU, see MLPRegressor).

    ``contract.input_shape == (P, T, 3)`` determines the flattened dimension
    ``D = P*T*3``; hidden_dims defaults to ``(64, 32, 32)``. The evaluation side
    restores the structure with ``build_model(ckpt.contract, ckpt.hidden_dims)``.
    """
    return MLPRegressor(input_shape=contract.input_shape, hidden_dims=tuple(hidden_dims))


def _batch_to_device(
    batch: tuple[Tensor, Tensor, object], state: PreprocessingState, device: str
) -> tuple[Tensor, Tensor]:
    """raw CPU batch → float32 numpy transformed per state on CPU → device tensors."""
    x, y, _psids = batch
    x_dev = torch.from_numpy(transform_x(state, x.numpy())).to(device)
    y_dev = torch.from_numpy(transform_y(state, y.numpy())).to(device)
    return x_dev, y_dev


def run_validation(
    model: MLPRegressor,
    loader: DataLoader[SampleItem],
    state: PreprocessingState,
    device: str,
) -> float:
    """Compute the normalized MSE over the given DataLoader (sample-weighted aggregate).

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
                        "validation predictions contain non-finite values (NaN/Inf)"
                    )
                se = (pred - y) ** 2
                if not bool(torch.isfinite(se).all()):
                    raise TrainingError("validation loss contains non-finite values (NaN/Inf)")
                # float32 elements are finite but a float32 sum may overflow (e.g.
                # many 1e38 values): aggregates always promote to float64.
                total_se += float(se.double().sum().item())
                total_n += int(x.shape[0])
    finally:
        if was_training:
            model.train()
    if total_n == 0:
        raise TrainingError("validation set is empty; cannot compute val MSE")
    aggregate = total_se / (total_n * _N_OUTPUTS)
    if not math.isfinite(aggregate):
        raise TrainingError("validation aggregate loss is non-finite (NaN/Inf)")
    return aggregate


def _train_one_epoch(
    model: MLPRegressor,
    loader: DataLoader[SampleItem],
    optimizer: torch.optim.Optimizer,
    state: PreprocessingState,
    device: str,
) -> float | None:
    """Train a single epoch; returns the online accumulated weighted mean MSE, or
    None for a non-finite state (stop).

    Order: zero_grad → forward (stop if pred is non-finite) → batch mean loss
    (stop if non-finite) → backward (stop if gradients are non-finite, no step)
    → step → check at epoch end that the weights are still finite.
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
            return None  # exploding/non-finite gradients: no step, avoiding bad weights
        optimizer.step()
        # float32 elements are finite but a float32 sum may overflow: promote the
        # aggregate to float64.
        total_se += float(se.detach().double().sum().item())
        total_n += int(x.shape[0])
        if any(not bool(torch.isfinite(param).all()) for param in model.parameters()):
            return None
    if total_n == 0:
        return None
    aggregate = total_se / (total_n * _N_OUTPUTS)
    return aggregate if math.isfinite(aggregate) else None


def _clone_state_dict(model: MLPRegressor) -> dict[str, Tensor]:
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

    ``set_seed`` runs before model construction; best (absolute minimum, updated
    only on a strictly lower value) and the early stopping reference (updated
    only when the improvement exceeds min_delta) advance separately; non-finite
    loss/grad/pred/aggregate stops immediately without saving bad weights, and
    the reason is recorded in ``stop_reason``/``stop_epoch``/``detail``.

    Raises:
        TrainingError: training stopped due to numerical failure at the first
            epoch (no weights can be saved); the exception carries
            ``epoch``/``detail``.
    """
    set_seed(config.training.seed)  # before model construction: reproducible weight initialization
    model = build_model(contract, config.model.hidden_dims)
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
            failure_detail = "training loss/gradient/weights are non-finite"
            break  # keep the existing best, save no bad weights
        try:
            val_loss = run_validation(model, val_loader, state, device)
        except TrainingError as exc:
            stop_reason, stop_epoch, failure_detail = "numerical_failure", epoch, str(exc)
            break  # non-finite on the validation side: reason propagates via detail, not swallowed
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
    common: dict[str, Any] = dict(
        ckpt_format_version=CKPT_FORMAT_VERSION,
        hidden_dims=tuple(config.model.hidden_dims),
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
    best_checkpoint = Checkpoint(model_state_dict=best_state, best_val_loss=best_val, **common)
    final_checkpoint = replace(best_checkpoint, model_state_dict=final_state, best_val_loss=None)
    return TrainingResult(
        best_checkpoint=best_checkpoint,
        final_checkpoint=final_checkpoint,
        history=tuple(history),
        stop_reason=stop_reason,
        stop_epoch=stop_epoch,
        detail=failure_detail,
    )


def save_checkpoint(path: Path, ckpt: Checkpoint) -> None:
    """Serialize a Checkpoint with ``torch.save``; an existing file → FileExistsError.

    The on-disk dict contains only Python primitives/list/dict + CPU Tensors
    (dataclasses and numpy objects are always expanded/converted, see
    ``_checkpoint_to_dict``); weight tensors are detached + moved to CPU +
    cloned before saving.
    """
    if path.exists():
        raise FileExistsError(f"checkpoint already exists, refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_to_dict(ckpt), path)


def load_checkpoint(path: Path) -> Checkpoint:
    """Safely read a Checkpoint (the only entry point for restoring the contract
    on the evaluation side).

    Uses ``torch.load(weights_only=True, map_location="cpu")`` in explicit safe
    mode; validates the format version, required keys, and key field/contract
    shapes (t_s length, [P,1,3]/[2] statistic shapes, state_dict as a dict of
    CPU tensors) before rebuilding the nested dataclasses.

    Raises:
        FileNotFoundError: file does not exist.
        TrainingError: format version mismatch, missing keys, or corrupted
            shapes/types.
    """
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
    required = {
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
    missing = sorted(required.difference(payload))
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
    """Rebuild preprocessing state from the ckpt: delegates to
    preprocessing.state_from_mapping and cross-checks P against the contract;
    PreprocessingError is wrapped as TrainingError at the ckpt boundary.
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


def _config_from_dict(raw: Any, path: Path) -> ExperimentConfig:
    """Rebuild the config copy from the ckpt: delegates to
    training_config.config_from_mapping and wraps ConfigError → TrainingError
    here (without depending back on evaluation).
    """
    try:
        return training_config.config_from_mapping(raw)
    except ConfigError as exc:
        raise TrainingError(f"corrupted config copy ({path}): {exc}") from exc


def _checkpoint_to_dict(ckpt: Checkpoint) -> dict[str, Any]:
    """Checkpoint → on-disk dict for torch.save (primitives/list/dict + CPU Tensors only)."""
    return {
        "ckpt_format_version": int(ckpt.ckpt_format_version),
        "model_state_dict": {
            name: value.detach().to("cpu").clone() for name, value in ckpt.model_state_dict.items()
        },
        "hidden_dims": [int(dim) for dim in ckpt.hidden_dims],
        "activation": str(ckpt.activation),
        "contract": {
            "pulse_order": [str(p) for p in ckpt.contract.pulse_order],
            "n_time_steps": int(ckpt.contract.n_time_steps),
            "t_s": np.asarray(ckpt.contract.t_s, dtype=np.float64).tolist(),
            "n_channels": int(ckpt.contract.n_channels),
            "component_order": [str(c) for c in ckpt.contract.component_order],
        },
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
