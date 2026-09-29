#!/usr/bin/env python3
"""Forward re-simulation validation for one checkpoint + one raw parameter group.

``--checkpoint`` selects the model (structure/contract/preprocessing are all restored
from the ckpt), ``--sample-dir`` points to one raw parameter group directory
(config.yaml/index.csv/per-pulse trajectory.csv), and ``--output`` is required and
refuses to overwrite an existing directory.

Flow: predict (alpha, Ku) → replace only alpha/Ku/dataset_name in the original config
(``<source name>_forward_<uuid short code>``) and write forward_config.yaml → re-run
Relax plus per-pulse simulations through ``mumax3_pipeline.run_parameter_set``
(default layout ``data_root()/raw/<forward dataset_name>/<parameter_set_id>/``; the
real simulation directory is written into metrics.json and printed) → compare against
the observed trajectories and plot.

``--output`` directory products: prediction.json, forward_config.yaml, metrics.json,
comparisons/<pulse_id>.csv, figures/<pulse_id>.png|svg; the raw simulation data stays
in the pipeline's default data_root()/raw dataset directory (an independent
dataset_name; observed data is never overwritten). With ``--dry-run`` only
prediction.json and forward_config.yaml are written and MuMax3 is not run.

Known contract errors are reported to stderr and exit with code 2; a real simulation is
a long task and requires user approval.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

from micromagnetic_parameter_inversion import forward_validation
from micromagnetic_parameter_inversion.mumax3_config import ConfigError
from micromagnetic_parameter_inversion.preprocessing import PreprocessingError
from micromagnetic_parameter_inversion.training import TrainingError
from micromagnetic_parameter_inversion.training_data import DataError


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser: required --checkpoint/--sample-dir/--output, optional --dry-run."""
    parser = argparse.ArgumentParser(
        description=(
            "Forward validation: predict (alpha, Ku) from one raw parameter group, "
            "re-run the same MuMax3 excitations into a fresh raw dataset, and compare."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=pathlib.Path,
        help="Path to the checkpoint file (structure/contract/preprocessing come from it).",
    )
    parser.add_argument(
        "--sample-dir",
        required=True,
        type=pathlib.Path,
        help="Path to one raw parameter group directory (config.yaml/index.csv/trajectories).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help=(
            "New report directory (prediction.json/forward_config.yaml/metrics.json/"
            "comparisons/figures); existing directories are refused."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write prediction.json and forward_config.yaml; do not run MuMax3.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run forward validation.

    Known contract errors are reported and return 2.
    """
    args = _build_arg_parser().parse_args(argv)
    try:
        forward_validation.run_forward_validation(
            args.checkpoint, args.sample_dir, args.output, dry_run=args.dry_run
        )
    except (
        ConfigError,
        DataError,
        PreprocessingError,
        TrainingError,
        forward_validation.ForwardValidationError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
