#!/usr/bin/env python3
"""Transformer training CLI; kind mismatch is rejected before reading data or writing outputs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from micromagnetic_parameter_inversion import preprocessing, training, training_data
from micromagnetic_parameter_inversion.training_config import ConfigError


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_transformer.py",
        description="Train the Temporal Transformer from prepared train/val samples.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Training YAML config path")
    return parser


def run(config_path: Path) -> Path:
    """Use shared orchestration with an independent Transformer run directory."""
    return training.run(config_path, expected_kind="transformer")


def main(argv: Sequence[str] | None = None) -> int:
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
