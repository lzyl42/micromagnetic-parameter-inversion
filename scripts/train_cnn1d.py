#!/usr/bin/env python3
"""train_cnn1d.py — CNN1D training entry (implemented; a thin wrapper over ``training.run``).

Usage: only ``--config <training YAML>`` (no other switches). This entry
declares ``expected_kind="cnn1d"``, i.e. it requires ``model.kind == "cnn1d"``;
an ``mlp`` config raises ``ConfigError`` before loading data or creating
directories.

It shares the ``training.run`` orchestration with ``train_mlp.py`` and uses the
**same sample directory and frozen split**, but fully does **not reuse** the
MLP weights/checkpoint/preprocessing statistics/artifacts: every run performs
its own train-only fit and writes artifacts to an independent directory
``output_root()/training/cnn1d/<dataset_name>/<run_name>`` (``output_dir`` is
used verbatim when it overrides), so it does not interfere with the MLP outputs.
Test is still evaluated only in ``evaluate_model.py``; training and model
selection use train/val only.

Error handling: known contract errors (ConfigError/DataError/
PreprocessingError/FileExistsError/TrainingError) are reported to stderr and
return exit code 2; all other exceptions propagate (bugs).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence
from pathlib import Path

from micromagnetic_parameter_inversion import preprocessing, training, training_data
from micromagnetic_parameter_inversion.training_config import ConfigError


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser: only the required --config is registered."""
    parser = argparse.ArgumentParser(
        prog="train_cnn1d.py",
        description=(
            "Train the CNN1D inverse-regression model from prepared samples "
            "(CPU-friendly; test split is never touched)."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Path to the training YAML config (e.g. configs/training/cnn1d.yaml).",
    )
    return parser


def run(config_path: Path) -> Path:
    """Run CNN1D training (a thin wrapper over ``training.run``) and return the run directory."""
    return training.run(config_path, expected_kind="cnn1d")


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
