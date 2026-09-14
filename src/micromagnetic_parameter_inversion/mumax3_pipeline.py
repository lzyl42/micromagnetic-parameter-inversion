"""MuMax3 simulation pipeline.

Parameter-set validation -> equilibrium (once) -> per-pulse simulation ->
trajectory/index export.

dataset_name is a naming convention for the fixed protocol (material/geometry/
simulation); changing the protocol requires a new dataset_name; parameter_set_id
represents only alpha+Ku (the split key). Protocol consistency across
parameter_set_ids within the same dataset is not verified automatically.

Any step failure propagates immediately and stops the run; partial run files are
kept for manual diagnosis (the temporary index.csv.tmp is cleaned up), with no
automatic recovery. MuMax3 is always invoked through external.run_mumax3 (subprocess
argument list, shell=True forbidden).
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Final

from micromagnetic_parameter_inversion import external
from micromagnetic_parameter_inversion.mumax3_config import load_config, parameter_set_id
from micromagnetic_parameter_inversion.mumax3_results import parse_table, write_trajectory_csv
from micromagnetic_parameter_inversion.mumax3_script import (
    render_equilibrium_script,
    render_simulation_script,
)
from micromagnetic_parameter_inversion.paths import PROJECT_ROOT, data_root

# Template originals are read-only and fixed at PROJECT_ROOT/simulations/mumax3.
_TEMPLATE_DIR: Final[Path] = PROJECT_ROOT / "simulations" / "mumax3"
_EQUILIBRIUM_TEMPLATE_NAME: Final[str] = "equilibrium.mx3.in"
_SIMULATION_TEMPLATE_NAME: Final[str] = "simulation.mx3.in"

# Fixed output-layout names (not in YAML).
_CONFIG_SNAPSHOT_NAME: Final[str] = "config.yaml"
_INDEX_NAME: Final[str] = "index.csv"
_RUN_LOG_NAME: Final[str] = "run.log"
_EQUILIBRIUM_DIR_NAME: Final[str] = "equilibrium"
_RUNS_DIR_NAME: Final[str] = "runs"
_EQUILIBRIUM_SCRIPT_NAME: Final[str] = "equilibrium.mx3"
_SIMULATION_SCRIPT_NAME: Final[str] = "simulation.mx3"
_EQUILIBRIUM_OUT_DIR_NAME: Final[str] = "equilibrium.out"
_SIMULATION_OUT_DIR_NAME: Final[str] = "simulation.out"

# Outputs contractually required to exist and be non-empty after each successful
# MuMax3 run (inside the .out/ directory); native log.txt/references.bib are not
# required.
_EQUILIBRIUM_REQUIRED_OUTPUTS: Final[tuple[str, ...]] = ("equilibrium.ovf",)
_SIMULATION_REQUIRED_OUTPUTS: Final[tuple[str, ...]] = (
    "table.txt",
    "m_t0.ovf",
    "m_tfinal.ovf",
)

# Minimal fixed index.csv fields.
_INDEX_FIELDS: Final[tuple[str, ...]] = (
    "parameter_set_id",
    "pulse_id",
    "alpha",
    "ku_j_per_m3",
    "b_ext_x_T",
    "b_ext_y_T",
    "b_ext_z_T",
    "pulse_duration_s",
    "trajectory_path",
)


def _write_text_deterministic(path: Path, text: str) -> None:
    """Write UTF-8 with forced LF newlines for cross-platform byte determinism."""
    path.write_text(text, encoding="utf-8", newline="\n")


def _format_float(value: float) -> str:
    """Deterministic float format: CPython shortest round-trip repr.

    Stable across platforms under IEEE 754.
    """
    return repr(float(value))


def _ensure_trailing_newline(text: str) -> str:
    """Append one newline to non-empty text missing a trailing newline.

    This guarantees following section headers start on a new line.
    """
    return text if not text or text.endswith("\n") else text + "\n"


def _run_and_log(workdir: Path, script_name: str) -> None:
    """Run the script through external.run_mumax3 and write run.log deterministically.

    run.log records the script name, returncode, and the captured stdout/stderr
    verbatim (a newline is inserted at the stdout / "--- stderr ---" boundary when the
    trailing newline is missing); a non-zero returncode raises a RuntimeError carrying
    the script path and returncode (no retry).
    """
    script_path = workdir / script_name
    completed = external.run_mumax3(script_path, workdir=workdir)
    log_text = (
        f"# script: {script_name}\n"
        f"# returncode: {completed.returncode}\n"
        "--- stdout ---\n"
        f"{_ensure_trailing_newline(completed.stdout)}"
        "--- stderr ---\n"
        f"{completed.stderr}"
    )
    _write_text_deterministic(workdir / _RUN_LOG_NAME, log_text)
    if completed.returncode != 0:
        raise RuntimeError(
            f"MuMax3 execution failed: {script_path} (workdir={workdir}, "
            f"returncode={completed.returncode}); see {workdir / _RUN_LOG_NAME}"
        )


def _require_output_files(base_dir: Path, names: tuple[str, ...], context: str) -> None:
    """Check that contract-required output files exist and are non-empty.

    A file qualifies when is_file() and st_size > 0.
    """
    missing_or_empty: list[str] = []
    for name in names:
        path = base_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            missing_or_empty.append(name)
    if missing_or_empty:
        raise FileNotFoundError(
            f"{context}: missing or empty contract-required output files "
            f"{missing_or_empty} (directory {base_dir})"
        )


def _write_index_csv(index_path: Path, rows: list[list[str]]) -> None:
    """Atomically write index.csv: minimal fixed fields, deterministic row order.

    One row per pulse. Write index.csv.tmp in the same directory first, then
    Path.replace to the final name; on write/replace failure remove the tmp file and
    re-raise, leaving the final index absent.
    """
    tmp_path = index_path.with_name(index_path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(_INDEX_FIELDS)
            writer.writerows(rows)
        tmp_path.replace(index_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def run_parameter_set(config_path: Path) -> Path:
    """Run all simulations for one parameter set and return its output directory.

    The output root is ``data_root()/raw/<dataset_name>/<parameter_set_id>/``;
    raises ``FileExistsError`` if it already exists; equilibrium runs exactly once;
    each pulse gets an independent working directory; index.csv is written atomically
    only after all pulses succeed.
    """
    config = load_config(config_path)
    set_id = parameter_set_id(config.material.alpha, config.material.ku_j_per_m3)

    # Refuse if it already exists (no implicit overwrite, no cleanup of a failed run).
    out_dir = data_root() / "raw" / config.dataset_name / set_id
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists, refusing to overwrite: {out_dir}")

    # Templates are read-only; read them before creating any directory, so a missing
    # template does not leave a half-built output directory behind.
    equilibrium_template = (_TEMPLATE_DIR / _EQUILIBRIUM_TEMPLATE_NAME).read_text(encoding="utf-8")
    simulation_template = (_TEMPLATE_DIR / _SIMULATION_TEMPLATE_NAME).read_text(encoding="utf-8")

    equilibrium_dir = out_dir / _EQUILIBRIUM_DIR_NAME
    runs_dir = out_dir / _RUNS_DIR_NAME
    equilibrium_out_dir = equilibrium_dir / _EQUILIBRIUM_OUT_DIR_NAME

    out_dir.mkdir(parents=True)
    equilibrium_dir.mkdir()
    runs_dir.mkdir()

    # Config snapshot: copied byte-for-byte, not reformatted or rewritten.
    shutil.copyfile(config_path, out_dir / _CONFIG_SNAPSHOT_NAME)

    # equilibrium is rendered/executed only once, producing the equilibrium.ovf
    # shared by all pulses.
    _write_text_deterministic(
        equilibrium_dir / _EQUILIBRIUM_SCRIPT_NAME,
        render_equilibrium_script(config, equilibrium_template),
    )
    _run_and_log(equilibrium_dir, _EQUILIBRIUM_SCRIPT_NAME)
    _require_output_files(equilibrium_out_dir, _EQUILIBRIUM_REQUIRED_OUTPUTS, "equilibrium run")

    # Independent working directory per pulse; index rows are collected here, in
    # config pulse order.
    index_rows: list[list[str]] = []
    for pulse in config.pulses:
        pulse_dir = runs_dir / pulse.pulse_id
        pulse_out_dir = pulse_dir / _SIMULATION_OUT_DIR_NAME
        pulse_dir.mkdir()
        _write_text_deterministic(
            pulse_dir / _SIMULATION_SCRIPT_NAME,
            render_simulation_script(config, pulse, simulation_template),
        )
        _run_and_log(pulse_dir, _SIMULATION_SCRIPT_NAME)
        _require_output_files(
            pulse_out_dir, _SIMULATION_REQUIRED_OUTPUTS, f"pulse {pulse.pulse_id!r} run"
        )

        rows = parse_table(
            pulse_out_dir / "table.txt",
            pulse_duration_s=pulse.duration_s,
            sample_interval_s=config.recording.sample_interval_s,
            sample_count=config.recording.sample_count,
        )
        trajectory_csv = pulse_dir / "trajectory.csv"
        write_trajectory_csv(rows, trajectory_csv)

        b_ext = [pulse.b_ext_amplitude_t * axis for axis in pulse.direction]
        index_rows.append(
            [
                set_id,
                pulse.pulse_id,
                _format_float(config.material.alpha),
                _format_float(config.material.ku_j_per_m3),
                _format_float(b_ext[0]),
                _format_float(b_ext[1]),
                _format_float(b_ext[2]),
                _format_float(pulse.duration_s),
                trajectory_csv.relative_to(out_dir).as_posix(),
            ]
        )

    # After all pulses succeed, atomically write the final index.csv; failed runs are
    # kept for diagnosis.
    _write_index_csv(out_dir / _INDEX_NAME, index_rows)
    return out_dir
