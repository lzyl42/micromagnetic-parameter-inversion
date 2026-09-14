#!/usr/bin/env python3
"""generate_dataset.py — generate experiment YAMLs and run MuMax3 simulations with
bounded concurrency.

Current preset: Protocol B (fixed discrete benchmark) Sobol 1024 main-domain
point set (dataset cofeb_protocol_b_a2_sobol1024_v1: ES12, 40×20×4, 10 ps × 401
points = 0..4 ns after field cutoff, pulse_A2 only), defined in
FIXED_CONFIGS / PARAMETERS. Before running, confirm that the target output set
directory data/raw/cofeb_protocol_b_a2_sobol1024_v1/<parameter_set_id>/ does
not exist; if it exists, do not run (the pipeline raises FileExistsError and
existing data must not be deleted).

Each fixed config × parameter combination builds one experiment YAML (schema
strictly matching configs/experiments/mumax3_simulation.yaml), written to
artifacts/generated_configs/<dataset_name>/ and run immediately. Concurrency
is controlled solely by the script constant MAX_WORKERS; all parameters are
never queued at once. On failure or Ctrl-C, new submissions stop and
already-submitted tasks are allowed to finish naturally (child processes are
not killed); the first original exception is re-raised at the end. There is
no resume/cleanup/skip-existing support. PARAMETERS is generated with a fixed
seed and does not read external parameter files; data from different protocols
must not be mixed.
"""

from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import yaml
from scipy.stats import qmc

from micromagnetic_parameter_inversion import mumax3_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Single concurrency setting: at most MAX_WORKERS in-flight simulations per fixed config.
MAX_WORKERS = 2

# The single fixed config for Protocol B (fixed discrete benchmark), matching
# the fixed fields of configs/experiments/mumax3_simulation.yaml. Changing any
# fixed protocol field (material/geometry/recording/numerics/pulses) requires a
# new dataset_name; data from different protocols must not be mixed.
FIXED_CONFIGS: list[dict[str, Any]] = [
    {
        "dataset_name": "cofeb_protocol_b_a2_sobol1024_v1",
        "material": {
            "ms_a_per_m": 1.25e6,  # A/m
            "aex_j_per_m": 15e-12,  # J/m
            "anisotropy_axis": [1.0, 0.0, 0.0],  # easy axis +x
        },
        "geometry": {
            "size_m": [100e-9, 50e-9, 2e-9],  # true triaxial ellipsoid full diameters [m]
            "cells": [40, 20, 4],
        },
        "initial_m": [1.0, 0.0, 0.0],  # +x
        "recording": {
            "sample_interval_s": 10e-12,  # 10 ps
            "sample_count": 401,  # 0..4 ns after field cutoff
        },
        "numerics": {
            "edge_smooth": 12,
            "solver": 5,
            "max_err": 1.0e-5,
            "max_dt_s": 1.0e-11,
            "gamma_ll_rad_per_t_s": 1.7595e11,
            "relax_torque_threshold_t": -1.0,  # upstream default
        },
        "pulses": [
            {
                "pulse_id": "pulse_A2",
                "b_ext_amplitude_mT": 2.0,
                "direction": [0.0, 1.0, 0.0],  # y
                "duration_s": 50e-12,  # 50 ps
            },
        ],
    },
]

# Sobol 1024 points, scramble=True, rng=42; alpha log-sampled [0.004, 0.020],
# Ku linearly sampled [2000, 30000] J/m^3.
PARAMETERS: list[dict[str, float]] = [
    {
        "alpha": 0.004 * 5 ** float(u),
        "ku_j_per_m3": 2000.0 + 28000.0 * float(v),
    }
    for u, v in qmc.Sobol(d=2, scramble=True, rng=42).random_base2(m=10)
]

_LOG_LOCK = threading.Lock()


def _log(message: str) -> None:
    """Thread-safe single-line stdout logging: timestamp prefix, flattened newlines,
    immediate flush.
    """
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    line = line.replace("\r", " ").replace("\n", " ")
    with _LOG_LOCK:
        print(line, flush=True)


def build_config(fixed_config: dict[str, Any], parameter: dict[str, float]) -> dict[str, Any]:
    """Deep-copy the fixed config and inject (alpha, Ku) into material; top-level
    constants are not modified.
    """
    config = copy.deepcopy(fixed_config)
    config["material"]["alpha"] = parameter["alpha"]
    config["material"]["ku_j_per_m3"] = parameter["ku_j_per_m3"]
    return config


def _config_filename(index: int, alpha: float, ku_j_per_m3: float) -> str:
    """Simple index + alpha/Ku filename (directories are separated by dataset_name, so
    no protocol index is needed).
    """
    return f"{index:02d}_alpha{alpha:.6f}_ku{ku_j_per_m3:g}.yaml"


def _task_prefix(dataset_name: str, index: int, total: int, parameter: dict[str, float]) -> str:
    """Per-task log prefix: dataset, group index/total, and alpha/Ku."""
    return (
        f"[{dataset_name}] group {index}/{total} "
        f"alpha={parameter['alpha']:.6f} ku={parameter['ku_j_per_m3']:g}"
    )


def _write_config_yaml(config: dict[str, Any], config_path: Path) -> None:
    """Write the experiment YAML (overwrites an existing target)."""
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )


def run_simulation(config_path: Path) -> None:
    """Run one experiment config (blocks until done); failures propagate (thin wrapper
    for easy replacement/mock).
    """
    mumax3_pipeline.run_parameter_set(config_path)


def _run_task(task: tuple[str, dict[str, Any], Path]) -> None:
    """Per-task helper: write the config, log start, run the simulation, log
    completion/failure and elapsed time.

    Config writing is inside the try block: write failures also go through the
    failure log and are collected via the future's exception.
    """
    prefix, config, config_path = task
    started = time.perf_counter()
    try:
        _write_config_yaml(config, config_path)
        _log(f"{prefix} config written: {config_path.name}")
        _log(f"{prefix} simulation started: {config_path.name}")
        run_simulation(config_path)
    except Exception as exc:
        duration = time.perf_counter() - started
        _log(f"{prefix} simulation failed elapsed={duration:.3f}s {type(exc).__name__}: {exc}")
        raise
    duration = time.perf_counter() - started
    _log(f"{prefix} simulation completed elapsed={duration:.3f}s")


def _validate_max_workers(max_workers: int) -> int:
    """Validate the concurrency limit as a positive integer; bool/non-int/non-positive
    values are rejected.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise TypeError(
            f"max_workers must be an int, got {type(max_workers).__name__}: {max_workers!r}"
        )
    if max_workers <= 0:
        raise ValueError(f"max_workers must be a positive integer, got {max_workers}")
    return max_workers


def _run_parameter_group(
    fixed_config: dict[str, Any], parameters: list[dict[str, float]], max_workers: int
) -> None:
    """Run one fixed config: write index-stable YAMLs and run simulations with bounded concurrency.

    At most max_workers futures are submitted and unfinished at any time; the
    whole done batch is recorded before deciding whether to submit more, new
    submissions stop after a failure, and the first original exception is
    re-raised; Ctrl-C likewise does not kill submitted child processes.
    """
    dataset_name = str(fixed_config["dataset_name"])
    output_dir = PROJECT_ROOT / "artifacts" / "generated_configs" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine original-order indices and filenames before scheduling (stable indices).
    tasks: list[tuple[str, dict[str, Any], Path]] = []
    for index, parameter in enumerate(parameters, start=1):
        config = build_config(fixed_config, parameter)
        config_path = output_dir / _config_filename(
            index, parameter["alpha"], parameter["ku_j_per_m3"]
        )
        prefix = _task_prefix(dataset_name, index, len(parameters), parameter)
        tasks.append((prefix, config, config_path))

    _log(f"[{dataset_name}] start: total={len(tasks)} concurrency={max_workers}")

    first_error: Exception | None = None
    completed = 0
    failed = 0
    submitted = 0
    pending_interrupt: KeyboardInterrupt | None = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: set[Future[None]] = set()
        next_index = 0

        def submit_ready() -> None:
            """Top up in-flight tasks to max_workers; no new submissions after a failure is seen."""
            nonlocal next_index, submitted
            while len(futures) < max_workers and next_index < len(tasks) and first_error is None:
                task = tasks[next_index]
                next_index += 1
                futures.add(executor.submit(_run_task, task))
                submitted += 1

        def collect(done: set[Future[None]]) -> None:
            """Collect the whole done batch and record results and cumulative progress;
            extra submissions are decided by the caller."""
            nonlocal completed, failed, first_error
            for future in done:
                futures.discard(future)
                try:
                    future.result()
                except Exception as exc:
                    failed += 1
                    if first_error is None:
                        first_error = exc
                else:
                    completed += 1
            _log(
                f"[{dataset_name}] progress: completed={completed} failed={failed} "
                f"not_started={len(tasks) - submitted}"
            )

        try:
            submit_ready()
            while futures:
                done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                collect(
                    done
                )  # process the whole done batch (failures included) before submitting more
                if first_error is None:
                    submit_ready()
        except KeyboardInterrupt as interrupt:
            _log(
                f"[{dataset_name}] interrupt received; stopping new submissions and "
                f"waiting for submitted tasks"
            )
            while futures:
                done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                collect(done)
            pending_interrupt = interrupt

    not_started = len(tasks) - submitted
    _log(
        f"[{dataset_name}] summary: total={len(tasks)} completed={completed} "
        f"failed={failed} not_started={not_started}"
    )

    if first_error is not None:
        raise first_error
    if pending_interrupt is not None:
        raise pending_interrupt


def generate_dataset(
    fixed_configs: list[dict[str, Any]],
    parameters: list[dict[str, float]],
    max_workers: int = MAX_WORKERS,
) -> None:
    """For each fixed config × parameter, write YAMLs and run simulations with bounded concurrency.

    max_workers defaults to the script constant MAX_WORKERS; illegal values are
    rejected before any file is written.
    """
    workers = _validate_max_workers(max_workers)
    for fixed_config in fixed_configs:
        _run_parameter_group(fixed_config, parameters, workers)


def main() -> None:
    generate_dataset(FIXED_CONFIGS, PARAMETERS, max_workers=MAX_WORKERS)


if __name__ == "__main__":
    main()
