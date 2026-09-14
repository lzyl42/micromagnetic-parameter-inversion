#!/usr/bin/env python3
"""evaluate_model.py — run/ckpt-located standard evaluate.

``--run`` locates the training run; ``--checkpoint`` is optional (default
``<run>/best.pt``; explicit or not, the ckpt must be SHA-bound to the run's
split copy). The current training config is not required: structure/
preprocessing/label are all restored from the checkpoint.

Writes test_metrics.json (main/control + provenance: path of the actually
loaded ckpt / its file sha256 / the split_sha256 from that ckpt) and
test_predictions.csv (via ``export_test_predictions``: LF, one row per psid);
if artifacts already exist, overwrite is refused before any computation
(TOCTOU re-check before writing).

Error handling: known contract errors (EvaluationError/DataError/
PreprocessingError/TrainingError/ConfigError/FileExistsError/
FileNotFoundError) are reported to stderr and return exit code 2; all other
exceptions propagate (bugs).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from micromagnetic_parameter_inversion import evaluation, preprocessing, training, training_data
from micromagnetic_parameter_inversion.evaluation import EvaluationReport
from micromagnetic_parameter_inversion.training_config import ConfigError


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser: required --run, optional --checkpoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained run on its bound test split and export "
            "physical-unit metrics (split SHA-bound to the checkpoint)."
        ),
    )
    parser.add_argument(
        "--run",
        required=True,
        type=pathlib.Path,
        help="Path to the training run directory (its split copy binds the evaluation).",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=None,
        help="Path to the checkpoint file (default: <run>/best.pt; split SHA must match).",
    )
    return parser


def _subset_to_dict(subset: evaluation.SubsetMetrics) -> dict[str, Any]:
    """SubsetMetrics → JSON mapping (metrics are null for an empty subset)."""
    return {
        "n": subset.n,
        "mae_alpha": subset.mae_alpha,
        "rmse_alpha": subset.rmse_alpha,
        "mae_ku": subset.mae_ku,
        "rmse_ku": subset.rmse_ku,
    }


def _metrics_document(report: EvaluationReport) -> dict[str, Any]:
    """EvaluationReport → test_metrics.json document.

    Keys: ``main``/``control`` (n and MAE/RMSE; metrics are null for an empty
    subset) + ``provenance`` (path of the actually loaded ckpt / its file
    sha256 / the split_sha256 from that ckpt — the JSON source always
    corresponds to the model actually loaded).
    """
    return {
        "main": _subset_to_dict(report.main),
        "control": _subset_to_dict(report.control),
        "provenance": {
            "checkpoint_path": report.provenance.checkpoint_path,
            "checkpoint_sha256": report.provenance.checkpoint_sha256,
            "split_sha256": report.provenance.split_sha256,
        },
    }


def _precheck(artifacts: Sequence[Path]) -> None:
    """Artifact pre-check: any existing path → FileExistsError.

    Old evaluations are never overwritten.
    """
    existing = [path for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(
            f"evaluation artifacts already exist, refusing to overwrite: {existing}"
        )


def run(run_dir: Path, checkpoint_path: Path | None = None) -> Path:
    """Run the standard evaluate and write artifacts into the run directory, returning it.

    Artifacts: ``test_metrics.json`` (n and MAE/RMSE for main/control) and
    ``test_predictions.csv`` (one row per psid). Both are written only once.
    """
    run_dir = Path(run_dir)
    metrics_path = run_dir / "test_metrics.json"
    csv_path = run_dir / "test_predictions.csv"
    _precheck((metrics_path, csv_path))
    report, rows = evaluation.run_evaluation(run_dir, checkpoint_path)
    _precheck((metrics_path, csv_path))  # must not appear during evaluation (TOCTOU re-check)
    metrics_path.write_text(
        json.dumps(_metrics_document(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    evaluation.export_test_predictions(csv_path, rows)
    print(f"evaluation complete: {run_dir}")
    print(f"  test n: main={report.main.n}, control={report.control.n}")
    print(f"  artifacts: {metrics_path.name}, {csv_path.name}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run evaluation; known contract errors are reported and return 2."""
    args = _build_arg_parser().parse_args(argv)
    try:
        run(args.run, args.checkpoint)
    except (
        ConfigError,
        training_data.DataError,
        preprocessing.PreprocessingError,
        training.TrainingError,
        evaluation.EvaluationError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
