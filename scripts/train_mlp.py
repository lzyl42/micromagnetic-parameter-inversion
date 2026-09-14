#!/usr/bin/env python3
"""train_mlp.py — config → samples/split → training → artifacts.

If the output directory (``config.output_dir`` or
``output_root()/training/mlp/<dataset_name>/<run_name>``) already exists, it
is refused before any training action. Only train/val are constructed (test
never participates in tuning/model selection and is not loaded); the contract
is frozen from the first train sample and cross-checked; the split copy is
saved byte-for-byte from raw and SHA-bound to the ckpt; preprocessing is fit
on the train split only. After successful training, the split.yaml copy,
config_resolved.yaml, preprocessing.yaml, metrics.json (without test
evaluation), and best.pt/final.pt are written in order (refusing overwrite).

Error handling: known contract errors (ConfigError/DataError/
PreprocessingError/FileExistsError/TrainingError) are reported to stderr and
return exit code 2; all other exceptions propagate (bugs).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from micromagnetic_parameter_inversion import (
    paths,
    preprocessing,
    training,
    training_config,
    training_data,
)
from micromagnetic_parameter_inversion.training_config import ConfigError, load_config
from micromagnetic_parameter_inversion.training_data import (
    DataError,
    DatasetMeta,
    InputContract,
    TrajectoryDataset,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser: only the required --config is registered."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the MLP inverse-regression model from prepared samples "
            "(CPU-friendly; test split is never touched)."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Path to the training YAML config (e.g. configs/training/mlp.yaml).",
    )
    return parser


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
    history: Sequence[training.EpochMetrics],
    best_val_loss: float | None,
    stop_reason: str,
    stop_epoch: int,
    detail: str | None,
) -> dict[str, Any]:
    """metrics.json document: history + best val + stop state (no test evaluation;
    finite values only).
    """
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


def _write_metrics(path: Path, result: training.TrainingResult) -> None:
    """Write metrics.json (payload described in ``_metrics_payload``)."""
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


def _write_failure_metrics(path: Path, error: training.TrainingError) -> None:
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


def run(config_path: Path) -> Path:
    """Run the full training flow and return the run directory."""
    config = load_config(config_path)
    samples_dir = paths.data_root() / "samples" / config.dataset_name
    if not samples_dir.is_dir():
        raise DataError(f"sample directory does not exist: {samples_dir}")
    meta: DatasetMeta = training_data.load_dataset_meta(samples_dir)
    split = training_data.load_split(samples_dir, meta)
    # The training contract requires non-empty train/val (test may be empty: neither
    # trained nor evaluated).
    if not split.train:
        raise DataError(
            f"split.train is empty: training requires at least 1 train member "
            f"(split copy at {samples_dir / 'split.yaml'})"
        )
    if not split.val:
        raise DataError(
            f"split.val is empty: early stopping and best selection require a val member "
            f"(split copy at {samples_dir / 'split.yaml'})"
        )

    output_dir = (
        Path(config.output_dir)
        if config.output_dir is not None
        else paths.output_root() / "training" / "mlp" / config.dataset_name / config.run_name
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
        result = training.train_model(
            config,
            train_set,
            val_set,
            state,
            contract,
            split_sha256,
            dataset_meta_relpath=meta_source.name,
            dataset_meta_sha256=meta_sha256,
        )
    except training.TrainingError as exc:
        # First-epoch numerical failure: no weights to keep → failure-state metrics only.
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
    training.save_checkpoint(output_dir / "best.pt", result.best_checkpoint)
    training.save_checkpoint(output_dir / "final.pt", result.final_checkpoint)

    if result.stop_reason == "numerical_failure":
        # A valid snapshot (previous complete epoch) was saved, but this run counts as a
        # failure: non-zero exit and no training-complete message.
        raise training.TrainingError(
            f"training stopped due to numerical failure at epoch "
            f"{result.stop_epoch}: {result.detail}",
            epoch=result.stop_epoch,
            detail=result.detail,
        )

    print(f"training complete: {output_dir}")
    print(f"  epochs={len(result.history)}; best_val_loss={result.best_checkpoint.best_val_loss!r}")
    print(f"  stop: {result.stop_reason} @ epoch {result.stop_epoch}")
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run training; known contract errors are reported and return 2."""
    args = _build_arg_parser().parse_args(argv)
    try:
        run(args.config)
    except (
        ConfigError,
        preprocessing.PreprocessingError,
        training_data.DataError,
        training.TrainingError,
        FileExistsError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
