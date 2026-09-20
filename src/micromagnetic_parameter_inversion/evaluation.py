"""Evaluation: run/ckpt-located normal evaluate and physical-unit metrics.

Locating: ``--run <run_dir> [--checkpoint]``; the network structure,
preprocessing, and label transform are **all restored from the checkpoint**
(not from the current YAML). Binding checks (any mismatch raises
EvaluationError):

- split: the ``split.yaml`` copy stored inside the run (the only authority),
  whose sha256 is checked against ``ckpt.split_sha256``; the copy goes through
  ``training_data.load_split(…, split_path=…)``, the same load boundary
  (mutual-exclusion/union/npz-existence against the real sample directory) — so
  a re-split ``split.yaml`` in the samples directory does not affect evaluation;
- dataset_meta: ``ckpt.dataset_meta_relpath`` under the anchor
  ``data_root()/samples/<ckpt.config.dataset_name>/`` (after resolve it must
  stay inside the anchor, preventing escape), sha256 checked against
  ``ckpt.dataset_meta_sha256``, and dataset_name cross-checked against the ckpt
  config; the psid→Ku table divides the main domain/control.
- provenance: ``EvaluationReport.provenance`` (``EvaluationProvenance``) records
  the ckpt **actually loaded** — path (the result of
  ``resolve_checkpoint_path``, best/final/external path as a verbatim string),
  file-bytes sha256 (one read, not a second ``torch.load``), and the
  ``split_sha256`` from that ckpt (already checked against the run copy). The
  ``provenance`` key of ``test_metrics.json`` is generated from this: the JSON
  source always corresponds to the actually loaded model rather than only the
  default path; the same split across runs is legal and simply recorded
  faithfully without introducing new restrictions.

Metrics (physical units, alpha/Ku reported separately): first version reports
only MAE/RMSE; the main domain excludes the Ku = 0 control, which is reported
separately and not counted in the main domain; each subset outputs its sample
count ``n``, and an empty subset (n=0) has ``None`` metrics (JSON null, not
NaN/0); no re-splitting to pad metrics. Predictions/inputs containing
non-finite values are explicitly rejected (PreprocessingError/EvaluationError),
not treated as an empty subset.

``run_evaluation`` is pure computation (no disk writes) and returns
``(EvaluationReport, rows)``; the writes of test_metrics.json /
test_predictions.csv are orchestrated by ``evaluate_model.py`` (via
``export_test_predictions``, refusing overwrite, LF line endings, one row per
psid).

Dependency direction: a single safe read of the ckpt via
``training.load_any_checkpoint`` (explicit ``model_kind`` routing, no fallback),
then model reconstruction by checkpoint kind via ``training.build_model`` /
``training.build_cnn_model`` (the CNN uses the ckpt top-level structure fields,
not the nested config; training must not import this module); the split copy
goes through ``training_data.load_split(…, split_path=…)`` and dataset_meta
through ``training_data.load_dataset_meta(…, meta_path=…)``, the same load
boundary (this module only does SHA/escape/dataset_name checks); metric
computation lives in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from micromagnetic_parameter_inversion import paths, preprocessing, training, training_data


class EvaluationError(ValueError):
    """Evaluation contract violation (split/meta SHA mismatch, meta path escape,
    npz input contract mismatch, non-finite metric input, etc.)."""


@dataclass(frozen=True)
class SubsetMetrics:
    """Result for a single metric subset (main domain or control).

    Empty subset (n=0): ``n`` is reported as 0 and all four metrics are
    ``None`` (serialized as JSON null; NaN or 0 padding is not allowed).
    """

    n: int  # subset sample count (always reported)
    mae_alpha: float | None  # physical-unit MAE (alpha)
    rmse_alpha: float | None  # physical-unit RMSE (alpha)
    mae_ku: float | None  # physical-unit MAE (Ku, J/m^3)
    rmse_ku: float | None  # physical-unit RMSE (Ku, J/m^3)


@dataclass(frozen=True)
class EvaluationProvenance:
    """Evaluation source record (source of the ``provenance`` key in test_metrics.json).

    All fields describe the checkpoint **actually loaded**, not a default-path
    assumption:

    - ``checkpoint_path``: the verbatim result string of
      ``resolve_checkpoint_path`` (best/final/external path; not resolved, i.e.
      the path actually opened);
    - ``checkpoint_sha256``: sha256 of that ckpt file's bytes (one read;
      independent of ``torch.load`` deserialization, not a second load);
    - ``split_sha256``: the ``split_sha256`` from the actual ckpt (provenance is
      generated only after it has been checked against the run split copy).
    """

    checkpoint_path: str
    checkpoint_sha256: str
    split_sha256: str


@dataclass(frozen=True)
class EvaluationReport:
    """Test-set evaluation report (source of test_metrics.json)."""

    main: SubsetMetrics  # main domain: excludes the Ku = 0 control
    control: SubsetMetrics  # Ku = 0 physical control, reported separately; excluded from main
    provenance: EvaluationProvenance  # record of the actually loaded ckpt


@dataclass(frozen=True)
class PredictionRow:
    """One row of test_predictions.csv (one row per parameter set, no pulse_id column)."""

    parameter_set_id: str
    split: str  # normal evaluate only evaluates the run split copy's test members
    alpha_true: float
    alpha_pred: float
    ku_true: float
    ku_pred: float


def compute_subset_metrics(
    y_true: Sequence[Sequence[float]], y_pred: Sequence[Sequence[float]]
) -> SubsetMetrics:
    """Physical-unit MAE/RMSE (alpha/Ku reported separately, no cross-column merge, no re-split).

    Args:
        y_true/y_pred: ``[n, 2]`` sequences (physical units, column order alpha/ku).

    Returns:
        ``n == 0`` → ``n=0`` and all four metrics are ``None`` (JSON null, neither NaN nor 0).

    Raises:
        EvaluationError: the two sequences have different lengths, their shape is
            not ``[n,2]``, or they contain non-finite values (NaN/Inf explicitly
            rejected, not treated as an empty subset).
    """
    true_list = list(y_true)
    pred_list = list(y_pred)
    if len(true_list) != len(pred_list):
        raise EvaluationError(
            f"y_true/y_pred length mismatch: {len(true_list)} vs {len(pred_list)}"
        )
    if not true_list:
        return SubsetMetrics(n=0, mae_alpha=None, rmse_alpha=None, mae_ku=None, rmse_ku=None)
    true = np.asarray(true_list, dtype=np.float64)
    pred = np.asarray(pred_list, dtype=np.float64)
    if true.ndim != 2 or true.shape != pred.shape or true.shape[1] != 2:
        raise EvaluationError(
            f"y_true/y_pred shape must be [n,2] and consistent (got {true.shape} vs {pred.shape})"
        )
    if not (np.all(np.isfinite(true)) and np.all(np.isfinite(pred))):
        raise EvaluationError(
            "metric input contains non-finite values (NaN/Inf); explicitly rejected, not treated "
            "as an empty subset"
        )
    diff = pred - true
    mae = np.abs(diff).mean(axis=0)
    rmse = np.sqrt((diff**2).mean(axis=0))
    return SubsetMetrics(
        n=int(true.shape[0]),
        mae_alpha=float(mae[0]),
        rmse_alpha=float(rmse[0]),
        mae_ku=float(mae[1]),
        rmse_ku=float(rmse[1]),
    )


def resolve_checkpoint_path(run_dir: Path, checkpoint_path: Path | None = None) -> Path:
    """Locate the run checkpoint: default ``<run_dir>/best.pt`` (pure path logic)."""
    return checkpoint_path if checkpoint_path is not None else run_dir / "best.pt"


def run_evaluation(
    run_dir: Path,
    checkpoint_path: Path | None = None,
) -> tuple[EvaluationReport, tuple[PredictionRow, ...]]:
    """Normal evaluate main flow: returns ``(EvaluationReport, rows)`` (pure computation,
    no writes).

    Args:
        run_dir: training run directory (source for locating the split copy / ckpt).
        checkpoint_path: default ``<run_dir>/best.pt``; whether explicit or not,
            the ckpt split SHA must match the run copy.

    Orchestration:

    1. ``ckpt = training.load_any_checkpoint(...)``; route on the explicit
       ``model_kind`` (MLP/CNN, no fallback), with structure/preprocessing/label
       all from the ckpt (not the current YAML);
    2. check the run split copy sha256 against ``ckpt.split_sha256``; then build
       ``EvaluationProvenance`` (actual ckpt path, file-bytes sha256 read once,
       and the split_sha256 from that ckpt);
    3. locate dataset_meta by anchor + relpath (escape prevention) + SHA check +
       dataset_name cross-check;
    4. ``training_data.load_split(samples_dir, meta, split_path=copy)``: the same
       load boundary (npz existence against the real sample directory);
    5. batch the test members by ``ckpt.config.training.batch_size``: CPU, eval,
       ``no_grad`` forward, ``load_sample`` contract validation (x shape/pulse_ids
       order/t_s), and ``inverse_transform_y`` to restore physical units;
    6. divide main domain/control by the meta psid→Ku table (Ku = 0 → control),
       each with ``compute_subset_metrics``;
    7. assemble the ``PredictionRow`` sequence (the writes are orchestrated by
       ``evaluate_model.py``).

    Raises:
        EvaluationError: split/meta binding mismatch, path escape, illegal metric
            input, or checkpoint weights not matching the restored model
            structure (keys/shapes).
        DataError/PreprocessingError/TrainingError: npz contract, non-finite
            inverse transform, or ckpt load failure (re-raised as-is; the caller
            presents them uniformly).
    """
    run_dir = Path(run_dir)
    ckpt_path = resolve_checkpoint_path(run_dir, checkpoint_path)
    ckpt = training.load_any_checkpoint(ckpt_path)

    samples_dir = paths.data_root() / "samples" / ckpt.config.dataset_name
    split_copy = run_dir / "split.yaml"
    if not split_copy.is_file():
        raise EvaluationError(f"run is missing the split.yaml copy: {split_copy}")
    if training_data.sha256_file(split_copy) != ckpt.split_sha256:
        raise EvaluationError(
            f"run split copy does not match ckpt.split_sha256 (run binding broken): {split_copy}"
        )
    provenance = EvaluationProvenance(
        checkpoint_path=str(ckpt_path),
        checkpoint_sha256=training_data.sha256_file(ckpt_path),
        split_sha256=ckpt.split_sha256,
    )

    meta_path = _resolve_meta_path(ckpt)
    meta = training_data.load_dataset_meta(samples_dir, meta_path=meta_path)
    if meta.dataset_name != ckpt.config.dataset_name:
        raise EvaluationError(
            f"dataset_meta.dataset_name {meta.dataset_name!r} does not match ckpt config "
            f"{ckpt.config.dataset_name!r} ({meta_path})"
        )
    split = training_data.load_split(samples_dir, meta, split_path=split_copy)

    rows = _evaluate_test_rows(ckpt, samples_dir, meta, split)
    report = _build_report(rows, meta, provenance)
    return report, tuple(rows)


def _resolve_meta_path(ckpt: training.ModelCheckpoint) -> Path:
    """dataset_meta anchor location: after resolve it must stay inside the anchor
    (escape prevention) + SHA check.

    The anchor is derived from ``data_root()/samples/<ckpt.config.dataset_name>/``
    (the same source as ``run_evaluation``'s ``samples_dir``), so no extra
    argument is needed.
    """
    anchor = (paths.data_root() / "samples" / ckpt.config.dataset_name).resolve()
    candidate = (anchor / ckpt.dataset_meta_relpath).resolve()
    try:
        candidate.relative_to(anchor)
    except ValueError:
        raise EvaluationError(
            f"dataset_meta_relpath escapes the anchor directory {anchor}: "
            f"{ckpt.dataset_meta_relpath!r}"
        ) from None
    if not candidate.is_file():
        raise EvaluationError(f"dataset_meta does not exist: {candidate}")
    if training_data.sha256_file(candidate) != ckpt.dataset_meta_sha256:
        raise EvaluationError(f"dataset_meta SHA does not match ckpt: {candidate}")
    return candidate


def _evaluate_test_rows(
    ckpt: training.ModelCheckpoint,
    samples_dir: Path,
    meta: training_data.DatasetMeta,
    split: training_data.SplitDefinition,
) -> list[PredictionRow]:
    """Batched inference over the test members (CPU/eval/no_grad) → physical-unit
    PredictionRow list.

    The model structure is restored by checkpoint kind: the CNN uses the ``ckpt``
    top-level explicit structure fields (not the nested config), while the MLP
    reuses the existing ``hidden_dims``; a ``load_state_dict(strict=True)``
    key/shape mismatch is uniformly converged to ``EvaluationError`` (no
    fallback, no partial report).
    """
    if isinstance(ckpt, training.CNNCheckpoint):
        model = training.build_cnn_model(
            ckpt.contract,
            channels=ckpt.channels,
            kernel_sizes=ckpt.kernel_sizes,
            pool_bins=ckpt.pool_bins,
            head_hidden_dims=ckpt.head_hidden_dims,
        )
    else:
        model = training.build_model(ckpt.contract, ckpt.hidden_dims)
    try:
        model.load_state_dict(dict(ckpt.model_state_dict), strict=True)
    except RuntimeError as exc:
        raise EvaluationError(
            f"checkpoint weights do not match the restored model structure (keys/shapes), "
            f"refusing to evaluate: {exc}"
        ) from exc
    model.to("cpu")
    model.eval()
    batch_size = int(ckpt.config.training.batch_size)
    if batch_size < 1:
        raise EvaluationError(
            f"ckpt.config.training.batch_size must be positive (got {batch_size})"
        )

    rows: list[PredictionRow] = []
    for start in range(0, len(split.test), batch_size):
        chunk = split.test[start : start + batch_size]
        samples = [
            training_data.load_sample(samples_dir, psid, ckpt.contract)  # contract validation
            for psid in chunk
        ]
        x_raw = np.stack([sample.x for sample in samples])  # [n,P,T,3] float32
        x_norm = preprocessing.transform_x(ckpt.preprocessing, x_raw)
        with torch.no_grad():
            pred_norm = model(torch.from_numpy(x_norm))  # [n,2] standardized space
        y_physical = preprocessing.inverse_transform_y(ckpt.preprocessing, pred_norm.numpy())
        for sample, pred in zip(samples, y_physical, strict=True):
            rows.append(
                PredictionRow(
                    parameter_set_id=sample.parameter_set_id,
                    split="test",
                    alpha_true=float(sample.y[0]),
                    alpha_pred=float(pred[0]),
                    ku_true=float(sample.y[1]),
                    ku_pred=float(pred[1]),
                )
            )
    return rows


def _build_report(
    rows: Sequence[PredictionRow],
    meta: training_data.DatasetMeta,
    provenance: EvaluationProvenance,
) -> EvaluationReport:
    """Divide main domain/control by the meta psid→Ku table (Ku = 0 → control) and
    compute metrics separately."""
    main_rows: list[list[float]] = []
    main_pred: list[list[float]] = []
    control_rows: list[list[float]] = []
    control_pred: list[list[float]] = []
    for row in rows:
        if row.parameter_set_id not in meta.labels:
            raise EvaluationError(
                f"psid is not in the dataset_meta psid→Ku table: {row.parameter_set_id!r}"
            )
        ku_true_from_meta = meta.labels[row.parameter_set_id][1]
        if ku_true_from_meta == 0.0:
            control_rows.append([row.alpha_true, row.ku_true])
            control_pred.append([row.alpha_pred, row.ku_pred])
        else:
            main_rows.append([row.alpha_true, row.ku_true])
            main_pred.append([row.alpha_pred, row.ku_pred])
    return EvaluationReport(
        main=compute_subset_metrics(main_rows, main_pred),
        control=compute_subset_metrics(control_rows, control_pred),
        provenance=provenance,
    )


def export_test_predictions(path: Path, rows: Sequence[PredictionRow]) -> None:
    """Write test_predictions.csv (LF line endings, one row per psid; existing → FileExistsError).

    Fixed column order: ``parameter_set_id, split, alpha_true, alpha_pred,
    ku_true, ku_pred``; floats are written at full precision via repr.
    """
    if path.exists():
        raise FileExistsError(f"evaluation export already exists, refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "parameter_set_id,split,alpha_true,alpha_pred,ku_true,ku_pred"
    lines = [header]
    lines += [
        f"{row.parameter_set_id},{row.split},{row.alpha_true!r},{row.alpha_pred!r},"
        f"{row.ku_true!r},{row.ku_pred!r}"
        for row in rows
    ]
    # Explicit "\n" join + newline="\n": guarantees LF line endings across platforms.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
