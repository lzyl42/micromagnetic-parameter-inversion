"""Evaluation: run/ckpt-located standard evaluate with physical-unit metrics.

Location: ``--run <run_dir> [--checkpoint]``; model structure, preprocessing,
and label transform are all **restored from the checkpoint** (never from the
current YAML). Binding checks (any mismatch raises EvaluationError):

- split: the ``split.yaml`` copy stored in the run (the only authority), whose
  sha256 is checked against ``ckpt.split_sha256``; the copy goes through the
  same loading boundary as ``training_data.load_split(…, split_path=…)``
  (disjointness/union/npz existence against the real samples directory) — a
  re-split split.yaml in the samples directory does not affect evaluation;
- dataset_meta: the ``ckpt.dataset_meta_relpath`` under the anchor
  ``data_root()/samples/<ckpt.config.dataset_name>/`` (forced to stay inside
  the anchor after resolve, preventing escape); sha256 is checked against
  ``ckpt.dataset_meta_sha256`` and dataset_name is cross-checked with the ckpt
  config; the psid→Ku table splits the main domain from control.
- provenance: ``EvaluationReport.provenance`` (``EvaluationProvenance``)
  records the ckpt that was **actually loaded** — path (the result of
  ``resolve_checkpoint_path``: best/final/external path as an exact string),
  file-byte sha256 (single read, not a second ``torch.load``), and the
  ``split_sha256`` from that ckpt (already verified against the run copy).
  The ``test_metrics.json`` ``provenance`` key is generated from this: the
  JSON source always corresponds to the model actually loaded, not just the
  default path; the same split across runs is legal — sources are recorded
  faithfully without introducing new restrictions.

Metrics (physical units, alpha/Ku reported separately): MAE/RMSE only; the
main domain excludes Ku = 0 controls, which are reported separately and not
counted in the main domain; each subset outputs its sample count ``n``, and an
empty subset (n=0) has ``None`` metrics (JSON null, not NaN/0); no re-splitting
to pad metrics. Non-finite predictions/inputs are explicitly rejected
(PreprocessingError/EvaluationError) and never treated as an empty subset.

``run_evaluation`` is pure computation (no disk writes) and returns
``(EvaluationReport, rows)``; writing test_metrics.json / test_predictions.csv
is orchestrated by ``evaluate_model.py`` (via ``export_test_predictions``:
refusing to overwrite, LF line endings, one row per psid).

Dependency direction: ckpt reading and model rebuild go through
``training.load_checkpoint``/``build_model`` (training must not import this
module back); the split copy goes through
``training_data.load_split(…, split_path=…)`` and dataset_meta through
``training_data.load_dataset_meta(…, meta_path=…)``, sharing the same loading
boundary (this module only cross-checks SHA/escape/dataset_name); metric
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
    npz input contract mismatch, non-finite metric inputs, etc.)."""


@dataclass(frozen=True)
class SubsetMetrics:
    """Result for a single metric subset (main domain or control).

    Empty subset (n=0): ``n`` is reported faithfully as 0 and all four metrics
    are ``None`` (serialized as JSON null; NaN or padding zeros are not allowed).
    """

    n: int  # number of samples in the subset (always emitted)
    mae_alpha: float | None  # MAE in physical units (alpha)
    rmse_alpha: float | None  # RMSE in physical units (alpha)
    mae_ku: float | None  # MAE in physical units (Ku, J/m^3)
    rmse_ku: float | None  # RMSE in physical units (Ku, J/m^3)


@dataclass(frozen=True)
class EvaluationProvenance:
    """Evaluation provenance record (source of the ``provenance`` key in test_metrics.json).

    All fields describe the checkpoint that was **actually loaded**, not a
    default-path assumption:

    - ``checkpoint_path``: the exact string result of ``resolve_checkpoint_path``
      (best/final/external path; not resolved, i.e. the path actually opened);
    - ``checkpoint_sha256``: sha256 of the ckpt file bytes (single read;
      independent of ``torch.load`` deserialization and not a second load);
    - ``split_sha256``: the ``split_sha256`` from the actual ckpt (provenance is
      generated only after verifying it against the run's split copy).
    """

    checkpoint_path: str
    checkpoint_sha256: str
    split_sha256: str


@dataclass(frozen=True)
class EvaluationReport:
    """Test-set evaluation report (source of test_metrics.json)."""

    main: SubsetMetrics  # main domain: excludes Ku = 0 controls
    control: SubsetMetrics  # Ku = 0 controls, reported separately, excluded from main domain
    provenance: EvaluationProvenance  # provenance of the ckpt actually loaded


@dataclass(frozen=True)
class PredictionRow:
    """One row of test_predictions.csv (one row per parameter set, no pulse_id column)."""

    parameter_set_id: str
    split: str  # standard evaluate scores only the test split from the run's split copy
    alpha_true: float
    alpha_pred: float
    ku_true: float
    ku_pred: float


def compute_subset_metrics(
    y_true: Sequence[Sequence[float]], y_pred: Sequence[Sequence[float]]
) -> SubsetMetrics:
    """Physical-unit MAE/RMSE (alpha/Ku reported separately; no cross-column merge, no
    re-splitting).

    Args:
        y_true/y_pred: ``[n, 2]`` sequences (physical units, column order alpha/ku).

    Returns:
        ``n == 0`` → ``n=0`` and all four metrics are ``None`` (JSON null, not NaN and not 0).

    Raises:
        EvaluationError: the two sequences differ in length, a shape is not
            ``[n,2]``, or non-finite values are present (NaN/Inf explicitly
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
            f"y_true/y_pred shape must be [n,2] and match (got {true.shape} vs {pred.shape})"
        )
    if not (np.all(np.isfinite(true)) and np.all(np.isfinite(pred))):
        raise EvaluationError(
            "metric inputs contain non-finite values (NaN/Inf); explicitly rejected, "
            "not treated as an empty subset"
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
    """Locate the run checkpoint: defaults to ``<run_dir>/best.pt`` (pure path logic)."""
    return checkpoint_path if checkpoint_path is not None else run_dir / "best.pt"


def run_evaluation(
    run_dir: Path,
    checkpoint_path: Path | None = None,
) -> tuple[EvaluationReport, tuple[PredictionRow, ...]]:
    """Standard evaluate main routine: returns ``(EvaluationReport, rows)`` (pure
    computation, no disk writes).

    Args:
        run_dir: training run directory (source for locating the split copy / ckpt).
        checkpoint_path: defaults to ``<run_dir>/best.pt``; explicit or not, the
            ckpt's split SHA must match the run copy.

    All structure/preprocessing/label information comes from the ckpt (never the
    current YAML); provenance is generated after verifying the run split copy's
    SHA against ``ckpt.split_sha256``; dataset_meta is located by anchor +
    relpath (escape-protected) and checked for SHA and dataset_name; test
    members go through ``load_sample`` contract validation, batched CPU
    inference, and ``inverse_transform_y`` to restore physical units, with the
    main domain/control split taken from the meta psid→Ku table.

    Raises:
        EvaluationError: split/meta binding mismatch, path escape, illegal metric input.
        DataError/PreprocessingError/TrainingError: npz contract, non-finite
            inverse transform, or ckpt load failure (propagated as-is and
            presented uniformly by the caller).
    """
    run_dir = Path(run_dir)
    ckpt_path = resolve_checkpoint_path(run_dir, checkpoint_path)
    ckpt = training.load_checkpoint(ckpt_path)

    samples_dir = paths.data_root() / "samples" / ckpt.config.dataset_name
    split_copy = run_dir / "split.yaml"
    if not split_copy.is_file():
        raise EvaluationError(f"run is missing the split.yaml copy: {split_copy}")
    if training_data.sha256_file(split_copy) != ckpt.split_sha256:
        raise EvaluationError(
            f"run split copy mismatch with ckpt.split_sha256 (run binding broken): {split_copy}"
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
            f"dataset_meta.dataset_name {meta.dataset_name!r} mismatch with the ckpt config "
            f"{ckpt.config.dataset_name!r} ({meta_path})"
        )
    split = training_data.load_split(samples_dir, meta, split_path=split_copy)

    rows = _evaluate_test_rows(ckpt, samples_dir, meta, split)
    report = _build_report(rows, meta, provenance)
    return report, tuple(rows)


def _resolve_meta_path(ckpt: training.Checkpoint) -> Path:
    """dataset_meta anchor location: forced to stay inside the anchor after resolve
    (escape-protected) + SHA check.

    The anchor is derived from ``data_root()/samples/<ckpt.config.dataset_name>/``
    (same source as ``run_evaluation``'s ``samples_dir``), so no extra argument
    is needed.
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
        raise EvaluationError(f"dataset_meta SHA mismatch with the ckpt: {candidate}")
    return candidate


def _evaluate_test_rows(
    ckpt: training.Checkpoint,
    samples_dir: Path,
    meta: training_data.DatasetMeta,
    split: training_data.SplitDefinition,
) -> list[PredictionRow]:
    """Batched inference over test members (CPU/eval/no_grad) → list of physical-unit
    PredictionRows.
    """
    model = training.build_model(ckpt.contract, ckpt.hidden_dims)
    model.load_state_dict(dict(ckpt.model_state_dict))
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
            pred_norm = model(torch.from_numpy(x_norm))  # [n,2] normalized space
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
    """Split main domain/control via the meta psid→Ku table (Ku = 0 → control) and
    compute metrics separately.
    """
    main_rows: list[list[float]] = []
    main_pred: list[list[float]] = []
    control_rows: list[list[float]] = []
    control_pred: list[list[float]] = []
    for row in rows:
        if row.parameter_set_id not in meta.labels:
            raise EvaluationError(
                f"psid not present in the dataset_meta psid→Ku table: {row.parameter_set_id!r}"
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
    """Write test_predictions.csv (LF line endings, one row per psid; existing file →
    FileExistsError).

    Fixed column order: ``parameter_set_id, split, alpha_true, alpha_pred,
    ku_true, ku_pred``; floats are written at full repr precision.
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
    # explicit "\n" join + newline="\n": guarantees LF line endings across platforms
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
