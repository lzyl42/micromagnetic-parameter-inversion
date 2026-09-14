#!/usr/bin/env python3
"""prepare_training_samples.py — raw → data/samples/<dataset>/.

Read the training YAML config (placeholder dataset_name/run_name rejected) and
resolve ``--parameter-set-ids``: an explicit psid list or ``all`` (expands only
subdirectories under the selected dataset directory, never scans across
datasets; the two cannot be mixed). Then training_data freezes the protocol
snapshot, reads samples group by group, decides the split, and pre-checks
before writing to refuse overwrite (no transactions). Generated dataset
subdirectories are not committed to Git.

Schema value/field violations raise ConfigError; layout/membership violations
raise DataError; main catches them, prints a friendly error to stderr, and
returns exit code 2 without a traceback.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from micromagnetic_parameter_inversion import paths, training_data
from micromagnetic_parameter_inversion.training_config import ConfigError, load_config
from micromagnetic_parameter_inversion.training_data import DataError, DatasetMeta, Sample


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser: --config and --parameter-set-ids are required."""
    parser = argparse.ArgumentParser(
        description=(
            "Normalize MuMax3 raw outputs into per-parameter-set npz samples "
            "under data/samples/<dataset_name>/."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Path to the training YAML config (e.g. configs/training/mlp.yaml).",
    )
    parser.add_argument(
        "--parameter-set-ids",
        required=True,
        nargs="+",
        metavar="ID",
        help="Explicit parameter set IDs, or the literal 'all' (dataset-local only).",
    )
    return parser


def _require_psid(name: str) -> str:
    """CLI-side psid pre-check: non-empty safe single path segment (detailed validation
    in training_data).
    """
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise DataError(f"illegal parameter_set_id: {name!r}")
    return name


def _resolve_psids(
    raw_dataset_dir: pathlib.Path, parameter_set_ids: Sequence[str]
) -> tuple[str, ...]:
    """Resolve the selected psids: an explicit list or ``all`` (expanded within one
    dataset directory only).
    """
    if len(parameter_set_ids) == 1 and parameter_set_ids[0] == "all":
        if not raw_dataset_dir.is_dir():
            _fail_dir(raw_dataset_dir)
        psids = sorted(entry.name for entry in raw_dataset_dir.iterdir() if entry.is_dir())
        if not psids:
            raise DataError(
                f"dataset directory contains no parameter set subdirectories: {raw_dataset_dir}"
            )
        return tuple(psids)
    if "all" in parameter_set_ids:
        raise DataError("'all' cannot be mixed with explicit parameter_set_id values")
    psids = tuple(_require_psid(name) for name in parameter_set_ids)
    if len(set(psids)) != len(psids):
        raise DataError(f"duplicate parameter_set_id: {list(psids)}")
    for psid in psids:
        if not (raw_dataset_dir / psid).is_dir():
            raise DataError(f"parameter set directory does not exist: {raw_dataset_dir / psid}")
    return psids


def _fail_dir(raw_dataset_dir: pathlib.Path) -> None:
    raise DataError(f"dataset raw directory does not exist: {raw_dataset_dir}")


def _collect_samples(
    raw_dataset_dir: pathlib.Path,
    psids: Sequence[str],
    protocol: training_data.ProtocolSummary,
    dataset_name: str,
) -> tuple[list[Sample], DatasetMeta]:
    """Read samples group by group and assemble dataset_meta (with sha256/source
    path/time provenance).
    """
    samples: list[Sample] = []
    labels: dict[str, tuple[float, float]] = {}
    config_sha256: dict[str, str] = {}
    for psid in psids:
        config_path = raw_dataset_dir / psid / "config.yaml"
        if not config_path.is_file():
            raise DataError(f"config.yaml snapshot missing: {config_path}")
        config_sha256[psid] = training_data.sha256_file(config_path)
        sample, group_labels = training_data.read_parameter_group(
            raw_dataset_dir, psid, protocol.pulse_order, protocol.n_time_steps
        )
        samples.append(sample)
        labels[psid] = group_labels
    meta = DatasetMeta(
        dataset_name=dataset_name,
        pulse_order=protocol.pulse_order,
        n_time_steps=protocol.n_time_steps,
        labels=labels,
        members=tuple(psids),
        protocol={
            "sample_interval_s": protocol.sample_interval_s,
            "pulses": {pid: dict(fields) for pid, fields in protocol.pulses.items()},
        },
        config_sha256=config_sha256,
        source_index_relpaths={psid: f"{psid}/index.csv" for psid in psids},
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    return samples, meta


def run(config_path: pathlib.Path, parameter_set_ids: Sequence[str]) -> pathlib.Path:
    """Run the full prepare flow and return the output directory."""
    config = load_config(config_path)
    raw_dataset_dir = paths.data_root() / "raw" / config.dataset_name
    if not raw_dataset_dir.is_dir():
        _fail_dir(raw_dataset_dir)
    psids = tuple(sorted(_resolve_psids(raw_dataset_dir, parameter_set_ids)))
    protocol = training_data.load_protocol_snapshot(raw_dataset_dir, psids, config.data.pulse_order)
    samples, meta = _collect_samples(raw_dataset_dir, psids, protocol, config.dataset_name)
    split = training_data.make_split(psids, config.split)
    samples_dir = paths.data_root() / "samples" / config.dataset_name
    training_data.write_prepared_dataset(samples_dir, samples, meta, split)
    print(f"prepare complete: {samples_dir}")
    print(
        f"  members: {len(psids)} groups; split train/val/test = "
        f"{len(split.train)}/{len(split.val)}/{len(split.test)}"
    )
    print(f"  pulse order: {', '.join(protocol.pulse_order)}; T = {protocol.n_time_steps}")
    return samples_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run prepare; known contract errors are reported and return 2."""
    args = _build_arg_parser().parse_args(argv)
    try:
        run(args.config, args.parameter_set_ids)
    except (ConfigError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
