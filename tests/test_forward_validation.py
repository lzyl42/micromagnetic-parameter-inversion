"""Offline CPU tests for forward re-simulation validation (script + module).

All inputs are synthetic raw layouts written under ``tmp_path`` with
``MICROMAG_DATA_ROOT`` / ``MICROMAG_OUTPUT_ROOT`` redirected; output directories
and the report directory are also ``tmp_path``-scoped. MuMax3 is always faked by
monkeypatching the pipeline call site, so no real MuMax3/GPU/dataset/checkpoint
artifact is read or written; the repository's local ``.npz``/``.pt`` files are
never touched. Checkpoints are tiny synthetic ones for the three model kinds
(zeroed weights give a deterministic, known prediction equal to the y mean).
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from numpy.typing import NDArray
from torch import nn

from micromagnetic_parameter_inversion import (
    external,
    forward_validation,
    mumax3_pipeline,
    paths,
    preprocessing,
    training,
    training_config,
    training_data,
)
from micromagnetic_parameter_inversion.mumax3_config import load_config, parameter_set_id

_SCRIPT = paths.PROJECT_ROOT / "scripts" / "validate_forward.py"

# Test-only protocol constants (not research parameters).
_DATASET = "ds_obs"
_PSID = "grp0001"
_PULSES = ("pulse_a", "pulse_b")
_N_STEPS = 5
_INTERVAL_S = 1.0e-9
_TRUE_ALPHA = 0.02
_TRUE_KU = 1.2e5
_PRED_ALPHA = 0.015
_PRED_KU = 4.2e4
_FORWARD_M = (0.6, 0.3, 0.74)

_CJK = re.compile("[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")
_DATASET_FORWARD_RE = re.compile(r"^ds_obs_forward_[0-9a-f]{8}$")

_CSV_HEADER = [
    "sample_index",
    "t_s_s",
    "observed_mx",
    "observed_my",
    "observed_mz",
    "forward_mx",
    "forward_my",
    "forward_mz",
    "residual_mx",
    "residual_my",
    "residual_mz",
]


@dataclass(frozen=True)
class _RawGroup:
    sample_dir: Path
    times: NDArray[np.float64]
    observed: dict[str, NDArray[np.float64]]


def _load_script() -> Any:
    """Load scripts/validate_forward.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("validate_forward", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Redirect data/output roots to temporary directories (the only roots used)."""
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    data_root.mkdir()
    out_root.mkdir()
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(data_root))
    monkeypatch.setenv(paths.OUTPUT_ROOT_ENV, str(out_root))
    return data_root, out_root


# ---------------------------------------------------------------------------
# Synthetic raw layout / checkpoint fixtures
# ---------------------------------------------------------------------------


def _observed_trajectory(pulse_index: int, n_steps: int = _N_STEPS) -> NDArray[np.float64]:
    """Deterministic observed magnetization [T, 3]; pulses differ in m_x.

    Values keep |m| <= 1 (physically plausible); no QC is performed by the reader,
    but the forward fake must also satisfy the pipeline's bounds.
    """
    k = np.arange(n_steps, dtype=np.float64)
    return np.stack([0.5 + 0.05 * pulse_index + 0.01 * k, 0.2 + 0.01 * k, 0.8 - 0.01 * k], axis=1)


def _write_trajectory_csv(path: Path, times: NDArray[np.float64], m: NDArray[np.float64]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["sample_index,t_s,m_x,m_y,m_z"]
    for index in range(len(times)):
        values = (float(index), float(times[index]), *(float(v) for v in m[index]))
        lines.append(",".join(repr(value) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _simulation_config_mapping(
    dataset: str, *, alpha: float = _TRUE_ALPHA, ku: float = _TRUE_KU
) -> dict[str, Any]:
    """Full valid simulation config (all required sections) for the raw fixture."""
    return {
        "dataset_name": dataset,
        "material": {
            "ms_a_per_m": 8.0e5,
            "aex_j_per_m": 1.3e-11,
            "alpha": alpha,
            "ku_j_per_m3": ku,
            "anisotropy_axis": [0.0, 0.0, 1.0],
        },
        "geometry": {"size_m": [160.0e-9, 80.0e-9, 3.0e-9], "cells": [8, 4, 1]},
        "initial_m": [1.0, 0.0, 0.0],
        "recording": {"sample_interval_s": _INTERVAL_S, "sample_count": _N_STEPS},
        "numerics": {
            "edge_smooth": 1,
            "solver": 6,
            "max_err": 1.0e-6,
            "max_dt_s": 5.0e-13,
            "gamma_ll_rad_per_t_s": 1.76e11,
            "relax_torque_threshold_t": -1.0,
        },
        "pulses": [
            {
                "pulse_id": pulse_id,
                "b_ext_amplitude_mT": 50.0 - index,
                "direction": [0.0, 0.0, 1.0],
                "duration_s": 2.0e-9 - 0.1e-9 * index,
            }
            for index, pulse_id in enumerate(_PULSES)
        ],
    }


def _write_raw_group(
    data_root: Path,
    dataset: str = _DATASET,
    *,
    psid: str = _PSID,
    interval_s: float = _INTERVAL_S,
) -> _RawGroup:
    """Synthesize one raw parameter group: config.yaml + index.csv + per-pulse CSVs.

    Time columns use ``repr(i * interval_s)`` so they round-trip exactly (the
    pipeline writes the same grid via ``.17g``).
    """
    sample_dir = data_root / "raw" / dataset / psid
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "config.yaml").write_text(
        yaml.safe_dump(_simulation_config_mapping(dataset), sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    times = np.arange(_N_STEPS, dtype=np.float64) * interval_s
    observed: dict[str, NDArray[np.float64]] = {}
    index_rows: list[str] = []
    for pulse_index, pulse_id in enumerate(_PULSES):
        m = _observed_trajectory(pulse_index)
        observed[pulse_id] = m
        relative = f"runs/{pulse_id}/trajectory.csv"
        _write_trajectory_csv(sample_dir / relative, times, m)
        index_rows.append(
            ",".join(
                (
                    psid,
                    pulse_id,
                    repr(_TRUE_ALPHA),
                    repr(_TRUE_KU),
                    "0.001",
                    "0",
                    "0",
                    "1e-09",
                    relative,
                )
            )
        )
    (sample_dir / "index.csv").write_text(
        ",".join(training_data.INDEX_COLUMNS) + "\n" + "\n".join(index_rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return _RawGroup(sample_dir=sample_dir, times=times, observed=observed)


def _contract_for(raw: _RawGroup) -> training_data.InputContract:
    return training_data.InputContract(
        pulse_order=_PULSES, n_time_steps=_N_STEPS, t_s=raw.times.copy()
    )


def _preprocessing_state() -> preprocessing.PreprocessingState:
    """Identity labels with mean exactly at the deterministic prediction.

    Zeroed model weights give a model output of 0, so the physical prediction is
    exactly ``y_mean`` (float32 after the module's final cast).
    """
    return preprocessing.PreprocessingState(
        x_stats=preprocessing.InputStats(
            mean=np.zeros((len(_PULSES), 1, 3)),
            std=np.ones((len(_PULSES), 1, 3)),
            eps=1.0e-8,
            zero_variance_positions=(),
        ),
        y_stats=preprocessing.LabelState(
            transform="identity",
            y_mean=np.array([_PRED_ALPHA, _PRED_KU], dtype=np.float64),
            y_std=np.ones(2),
            eps=1.0e-8,
            zero_variance_outputs=(),
        ),
    )


def _zeroed_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return {name: value.detach().to("cpu").clone() for name, value in model.state_dict().items()}


def _model_config_mapping(kind: str) -> dict[str, Any]:
    if kind == "mlp":
        return {"hidden_dims": [4]}
    if kind == "cnn1d":
        return {
            "kind": "cnn1d",
            "channels": [2],
            "kernel_sizes": [3],
            "pool_bins": 2,
            "head_hidden_dims": [4],
        }
    return {
        "kind": "transformer",
        "d_model": 8,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "head_hidden_dims": [4],
    }


def _make_checkpoint(
    kind: str,
    path: Path,
    contract: training_data.InputContract,
    state: preprocessing.PreprocessingState,
) -> Path:
    """Build → zero → wrap a tiny synthetic checkpoint of the requested kind."""
    config = training_config.config_from_mapping(
        {"dataset_name": "ds_ckpt", "run_name": f"run_{kind}", "model": _model_config_mapping(kind)}
    )
    if kind == "mlp":
        model = training.build_model(contract, (4,))
        ckpt: training.ModelCheckpoint = training.Checkpoint(
            ckpt_format_version=training_config.CKPT_FORMAT_VERSION,
            model_state_dict=_zeroed_state_dict(model),
            hidden_dims=(4,),
            activation="relu",
            contract=contract,
            preprocessing=state,
            seed=7,
            config=config,
            dataset_meta_relpath="dataset_meta.yaml",
            dataset_meta_sha256="0" * 64,
            split_sha256="1" * 64,
            best_val_loss=0.25,
        )
    elif kind == "cnn1d":
        model = training.build_cnn_model(
            contract, channels=(2,), kernel_sizes=(3,), pool_bins=2, head_hidden_dims=(4,)
        )
        ckpt = training.CNNCheckpoint(
            ckpt_format_version=training.CNN_CKPT_FORMAT_VERSION,
            model_state_dict=_zeroed_state_dict(model),
            channels=(2,),
            kernel_sizes=(3,),
            pool_bins=2,
            head_hidden_dims=(4,),
            activation="relu",
            contract=contract,
            preprocessing=state,
            seed=7,
            config=config,
            dataset_meta_relpath="dataset_meta.yaml",
            dataset_meta_sha256="0" * 64,
            split_sha256="1" * 64,
            best_val_loss=0.25,
        )
    else:
        model = training.build_transformer_model(
            contract,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            head_hidden_dims=(4,),
        )
        ckpt = training.TransformerCheckpoint(
            ckpt_format_version=training.TRANSFORMER_CKPT_FORMAT_VERSION,
            model_state_dict=_zeroed_state_dict(model),
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            head_hidden_dims=(4,),
            contract=contract,
            preprocessing=state,
            seed=7,
            config=config,
            dataset_meta_relpath="dataset_meta.yaml",
            dataset_meta_sha256="0" * 64,
            split_sha256="1" * 64,
            best_val_loss=0.25,
        )
    training.save_model_checkpoint(path, ckpt)
    return path


def _expected_prediction() -> tuple[float, float]:
    """float32-cast physical prediction (the module's final cast boundary)."""
    return float(np.float32(_PRED_ALPHA)), float(np.float32(_PRED_KU))


# ---------------------------------------------------------------------------
# Fake MuMax3 pipeline (never launches MuMax3)
# ---------------------------------------------------------------------------

_RUN_CALL_RE = re.compile(r"(?m)^[ \t]*Run\(([^)]+)\)")
_LOOP_RE = re.compile(r"for i := 1; i < (\d+);")


def _sampling_from_script(script_text: str) -> tuple[int, float, float]:
    """Extract (sample_count, interval_s, duration_s) from the rendered script."""
    run_args = _RUN_CALL_RE.findall(script_text)
    assert len(run_args) == 2, run_args
    loop = _LOOP_RE.search(script_text)
    assert loop is not None, script_text
    return int(loop.group(1)), float(run_args[1]), float(run_args[0])


@dataclass
class _FakeMumax:
    runs: list[Path] = field(default_factory=list)


def _install_fake_mumax(monkeypatch: pytest.MonkeyPatch) -> _FakeMumax:
    """Fake muMax3 execution at the pipeline call site; writes contract outputs.

    Forward trajectories are the constant ``_FORWARD_M`` on an exact
    ``duration_s + i*interval_s`` grid, so the pipeline parse contract and the
    forward/observed time-axis check see the same sampling protocol.
    """
    fake = _FakeMumax()

    def fake_run(
        script: str | Path, *, workdir: str | Path | None = None, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        script_path = Path(script)
        run_dir = Path(workdir) if workdir is not None else Path()
        fake.runs.append(run_dir)
        is_equilibrium = script_path.name == "equilibrium.mx3"
        out_dir = run_dir / ("equilibrium.out" if is_equilibrium else "simulation.out")
        out_dir.mkdir(parents=True)
        if is_equilibrium:
            (out_dir / "equilibrium.ovf").write_bytes(b"test-only equilibrium.ovf\n")
        else:
            count, interval_s, duration_s = _sampling_from_script(
                script_path.read_text(encoding="utf-8")
            )
            lines = ["# t (s)\tmx ()\tmy ()\tmz ()"]
            for index in range(count):
                values = (duration_s + index * interval_s, *_FORWARD_M)
                lines.append("\t".join(repr(value) for value in values))
            (out_dir / "table.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
            )
            (out_dir / "m_t0.ovf").write_bytes(b"test-only m_t0.ovf\n")
            (out_dir / "m_tfinal.ovf").write_bytes(b"test-only m_tfinal.ovf\n")
        return subprocess.CompletedProcess([str(script_path)], 0, "fake mumax stdout\n", "")

    monkeypatch.setattr(mumax3_pipeline.external, "run_mumax3", fake_run)
    return fake


def _reference_metrics(
    observed: NDArray[np.float64], forward: NDArray[np.float64]
) -> dict[str, float]:
    """Independent reimplementation of the per-pulse metrics definitions."""
    diff = np.asarray(forward, dtype=np.float64) - np.asarray(observed, dtype=np.float64)
    component = np.sqrt(np.mean(diff**2, axis=0))
    return {
        "rmse_mx": float(component[0]),
        "rmse_my": float(component[1]),
        "rmse_mz": float(component[2]),
        "rmse_vector": float(np.sqrt(np.mean(np.sum(diff**2, axis=1)))),
        "max_vector_error": float(np.max(np.linalg.norm(diff, axis=1))),
    }


# ---------------------------------------------------------------------------
# Sample reading: valid layout and contract/mismatch guards
# ---------------------------------------------------------------------------


def test_read_sample_valid_layout(env: tuple[Path, Path]) -> None:
    data_root, _ = env
    raw = _write_raw_group(data_root)
    contract = _contract_for(raw)

    sample = forward_validation.read_sample(raw.sample_dir, contract)

    assert sample.labels == (_TRUE_ALPHA, _TRUE_KU)
    assert sample.x.dtype == np.float32 and sample.x.shape == (len(_PULSES), _N_STEPS, 3)
    expected_x = np.stack([raw.observed[pulse] for pulse in _PULSES]).astype(np.float32)
    np.testing.assert_array_equal(sample.x, expected_x)
    assert set(sample.times) == set(_PULSES) and set(sample.trajectories) == set(_PULSES)
    for pulse_id in _PULSES:
        assert sample.times[pulse_id].dtype == np.float64
        assert sample.trajectories[pulse_id].dtype == np.float64
        np.testing.assert_array_equal(sample.times[pulse_id], raw.times)
        np.testing.assert_array_equal(sample.trajectories[pulse_id], raw.observed[pulse_id])


def _drop_last_index_row(sample_dir: Path) -> None:
    lines = (sample_dir / "index.csv").read_text(encoding="utf-8").splitlines()
    (sample_dir / "index.csv").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")


def _append_extra_index_row(sample_dir: Path) -> None:
    with (sample_dir / "index.csv").open("a", encoding="utf-8") as handle:
        handle.write(
            f"{_PSID},pulse_c,{_TRUE_ALPHA!r},{_TRUE_KU!r},0.001,0,0,1e-09,"
            f"runs/pulse_a/trajectory.csv\n"
        )


def _truncate_trajectory(sample_dir: Path) -> None:
    path = sample_dir / "runs" / "pulse_a" / "trajectory.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_drop_last_index_row, "missing pulse"),
        (_append_extra_index_row, "extra pulse"),
        (_truncate_trajectory, "shape mismatch"),
    ],
)
def test_read_sample_rejects_pulse_set_or_shape_mismatch(
    env: tuple[Path, Path], mutate: Any, match: str
) -> None:
    data_root, _ = env
    raw = _write_raw_group(data_root)
    contract = _contract_for(raw)
    mutate(raw.sample_dir)

    with pytest.raises(training_data.DataError, match=match):
        forward_validation.read_sample(raw.sample_dir, contract)


def test_read_sample_rejects_shifted_time_grid(env: tuple[Path, Path]) -> None:
    data_root, _ = env
    raw = _write_raw_group(data_root)
    contract = _contract_for(raw)
    # Same shape, shifted sampling protocol: must not be fed to the model.
    shifted = raw.times + 5.0e-10
    _write_trajectory_csv(
        raw.sample_dir / "runs" / "pulse_a" / "trajectory.csv", shifted, raw.observed["pulse_a"]
    )

    with pytest.raises(forward_validation.ForwardValidationError, match="time axis"):
        forward_validation.read_sample(raw.sample_dir, contract)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_predict_parameter_set_matches_independent_numpy_reference(
    env: tuple[Path, Path], tmp_path: Path
) -> None:
    data_root, _ = env
    raw = _write_raw_group(data_root)
    contract = _contract_for(raw)
    state = _preprocessing_state()
    config = training_config.config_from_mapping(
        {"dataset_name": "ds_ckpt", "run_name": "run_ref", "model": {"hidden_dims": [4]}}
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = training.build_model(contract, (4,))
    ckpt = training.Checkpoint(
        ckpt_format_version=training_config.CKPT_FORMAT_VERSION,
        model_state_dict={name: value.clone() for name, value in model.state_dict().items()},
        hidden_dims=(4,),
        activation="relu",
        contract=contract,
        preprocessing=state,
        seed=7,
        config=config,
        dataset_meta_relpath="dataset_meta.yaml",
        dataset_meta_sha256="0" * 64,
        split_sha256="1" * 64,
        best_val_loss=0.25,
    )

    x = np.stack([raw.observed[pulse] for pulse in _PULSES]).astype(np.float32)
    got = forward_validation.predict_parameter_set(model, ckpt, x)

    # Independent reference: explicit flatten + Linear/ReLU/Linear numpy math.
    state_dict = ckpt.model_state_dict
    weight_1 = state_dict["network.1.weight"].detach().numpy()
    bias_1 = state_dict["network.1.bias"].detach().numpy()
    weight_2 = state_dict["network.3.weight"].detach().numpy()
    bias_2 = state_dict["network.3.bias"].detach().numpy()
    x_norm = preprocessing.transform_x(state, x[None])
    hidden = np.maximum(x_norm.reshape(1, -1) @ weight_1.T + bias_1, 0.0)
    physical = (hidden @ weight_2.T + bias_2) * state.y_stats.y_std + state.y_stats.y_mean
    expected = np.asarray(physical[0], dtype=np.float32).astype(np.float64)
    np.testing.assert_allclose(got, expected, rtol=1.0e-4, atol=1.0e-6)

    # The prediction flows through x (not a constant): a different x gives a different y.
    x_other = (x * 0.5).astype(np.float32)
    assert not np.allclose(
        got, forward_validation.predict_parameter_set(model, ckpt, x_other), atol=1.0e-8
    )


# ---------------------------------------------------------------------------
# Dry-run: prediction.json + protocol-preserving forward_config.yaml
# ---------------------------------------------------------------------------


def test_dry_run_writes_prediction_and_protocol_preserving_config(
    env: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, _ = env
    raw = _write_raw_group(data_root)
    ckpt_path = _make_checkpoint(
        "mlp", tmp_path / "best.pt", _contract_for(raw), _preprocessing_state()
    )
    output_dir = tmp_path / "report"

    class _FakeUuid:
        hex = "0123456789abcdef"

    monkeypatch.setattr(forward_validation.uuid, "uuid4", lambda: _FakeUuid())

    assert (
        forward_validation.run_forward_validation(
            ckpt_path, raw.sample_dir, output_dir, dry_run=True
        )
        == output_dir
    )

    alpha_pred, ku_pred = _expected_prediction()
    document = json.loads((output_dir / "prediction.json").read_text(encoding="utf-8"))
    assert set(document) == {
        "alpha_true",
        "ku_j_per_m3_true",
        "alpha_pred",
        "ku_j_per_m3_pred",
        "alpha_abs_error",
        "ku_j_per_m3_abs_error",
    }
    assert document["alpha_true"] == _TRUE_ALPHA and document["ku_j_per_m3_true"] == _TRUE_KU
    assert document["alpha_pred"] == alpha_pred and document["ku_j_per_m3_pred"] == ku_pred
    assert document["alpha_abs_error"] == abs(alpha_pred - _TRUE_ALPHA)
    assert document["ku_j_per_m3_abs_error"] == abs(ku_pred - _TRUE_KU)

    original = yaml.safe_load((raw.sample_dir / "config.yaml").read_text(encoding="utf-8"))
    forward = yaml.safe_load((output_dir / "forward_config.yaml").read_text(encoding="utf-8"))
    assert forward["dataset_name"] == f"{_DATASET}_forward_01234567"
    expected = copy.deepcopy(original)
    expected["dataset_name"] = forward["dataset_name"]
    expected["material"]["alpha"] = alpha_pred
    expected["material"]["ku_j_per_m3"] = ku_pred
    assert forward == expected

    # Reloaded configs: only the three target fields differ from the observed protocol.
    original_config = load_config(raw.sample_dir / "config.yaml")
    forward_config = load_config(output_dir / "forward_config.yaml")
    assert forward_config.dataset_name == f"{_DATASET}_forward_01234567"
    assert forward_config == replace(
        original_config,
        dataset_name=forward_config.dataset_name,
        material=replace(original_config.material, alpha=alpha_pred, ku_j_per_m3=ku_pred),
    )

    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "comparisons").exists()
    assert not (output_dir / "figures").exists()
    assert sorted(p.name for p in (data_root / "raw").iterdir()) == [_DATASET]


@pytest.mark.parametrize("kind", ["mlp", "cnn1d", "transformer"])
def test_dry_run_cli_supports_all_three_model_kinds(
    env: tuple[Path, Path], tmp_path: Path, kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root, _ = env
    raw = _write_raw_group(data_root)
    ckpt_path = _make_checkpoint(
        kind, tmp_path / "best.pt", _contract_for(raw), _preprocessing_state()
    )
    output_dir = tmp_path / "report"

    exit_code = _load_script().main(
        [
            "--checkpoint",
            str(ckpt_path),
            "--sample-dir",
            str(raw.sample_dir),
            "--output",
            str(output_dir),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert not _CJK.search(captured.out + captured.err)
    assert "dry-run complete" in captured.out
    alpha_pred, ku_pred = _expected_prediction()
    document = json.loads((output_dir / "prediction.json").read_text(encoding="utf-8"))
    assert document["alpha_pred"] == alpha_pred
    assert document["ku_j_per_m3_pred"] == ku_pred
    assert (output_dir / "forward_config.yaml").is_file()
    assert not (output_dir / "metrics.json").exists()


# ---------------------------------------------------------------------------
# Full offline run: real pipeline + faked MuMax3 execution
# ---------------------------------------------------------------------------


def test_full_run_with_fake_mumax3_end_to_end(
    env: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root, _ = env
    raw = _write_raw_group(data_root)
    ckpt_path = _make_checkpoint(
        "mlp", tmp_path / "best.pt", _contract_for(raw), _preprocessing_state()
    )
    output_dir = tmp_path / "report"
    untouched = {
        path.relative_to(raw.sample_dir): path.read_bytes()
        for path in sorted(raw.sample_dir.rglob("*"))
        if path.is_file()
    }
    fake = _install_fake_mumax(monkeypatch)

    exit_code = _load_script().main(
        [
            "--checkpoint",
            str(ckpt_path),
            "--sample-dir",
            str(raw.sample_dir),
            "--output",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert not _CJK.search(captured.out + captured.err)
    assert "forward simulation directory" in captured.out
    assert "forward validation complete" in captured.out
    # One equilibrium run plus one run per pulse, all through the faked call site.
    assert [run_dir.name for run_dir in fake.runs] == ["equilibrium", "pulse_a", "pulse_b"]

    alpha_pred, ku_pred = _expected_prediction()
    forward = yaml.safe_load((output_dir / "forward_config.yaml").read_text(encoding="utf-8"))
    assert forward["material"]["alpha"] == alpha_pred
    assert forward["material"]["ku_j_per_m3"] == ku_pred
    assert _DATASET_FORWARD_RE.match(forward["dataset_name"])

    # The real pipeline wrote the (fake) raw simulation dataset under data_root/raw.
    simulation_dir = (
        data_root
        / "raw"
        / forward["dataset_name"]
        / parameter_set_id(forward["material"]["alpha"], forward["material"]["ku_j_per_m3"])
    )
    assert (simulation_dir / "config.yaml").read_bytes() == (
        output_dir / "forward_config.yaml"
    ).read_bytes()
    assert (simulation_dir / "index.csv").is_file()
    assert (simulation_dir / "equilibrium" / "equilibrium.out" / "equilibrium.ovf").is_file()
    assert (simulation_dir / "equilibrium" / "run.log").is_file()
    for pulse_id in _PULSES:
        forward_csv = simulation_dir / "runs" / pulse_id / "trajectory.csv"
        assert forward_csv.is_file()
        forward_m = np.loadtxt(forward_csv, delimiter=",", skiprows=1, usecols=(2, 3, 4), ndmin=2)
        np.testing.assert_array_equal(forward_m, np.tile(_FORWARD_M, (_N_STEPS, 1)))
    assert sorted(p.name for p in (data_root / "raw").iterdir()) == sorted(
        [_DATASET, forward["dataset_name"]]
    )
    assert paths.data_root() == data_root.resolve()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["simulation_dir"] == str(simulation_dir)
    assert set(metrics["pulses"]) == set(_PULSES)
    forward_array = np.tile(_FORWARD_M, (_N_STEPS, 1))
    for pulse_id in _PULSES:
        expected_metrics = _reference_metrics(raw.observed[pulse_id], forward_array)
        assert set(metrics["pulses"][pulse_id]) == set(expected_metrics)
        for key, value in expected_metrics.items():
            assert metrics["pulses"][pulse_id][key] == pytest.approx(value, rel=1.0e-12, abs=0.0)

    for pulse_id in _PULSES:
        with (output_dir / "comparisons" / f"{pulse_id}.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == _CSV_HEADER and len(rows) == _N_STEPS + 1
        observed = raw.observed[pulse_id]
        for index in range(_N_STEPS):
            row = rows[index + 1]
            assert int(row[0]) == index
            assert float(row[1]) == raw.times[index]
            assert [float(value) for value in row[2:5]] == [float(v) for v in observed[index]]
            assert [float(value) for value in row[5:8]] == list(_FORWARD_M)
            assert [float(value) for value in row[8:11]] == [
                forward_value - observed_value
                for forward_value, observed_value in zip(_FORWARD_M, observed[index], strict=True)
            ]
        assert (output_dir / "figures" / f"{pulse_id}.png").is_file()
        assert (output_dir / "figures" / f"{pulse_id}.svg").is_file()

    # Observed raw data is never overwritten by the forward dataset.
    assert untouched == {
        path.relative_to(raw.sample_dir): path.read_bytes()
        for path in sorted(raw.sample_dir.rglob("*"))
        if path.is_file()
    }


def test_full_run_rejects_forward_time_axis_mismatch(
    env: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, _ = env
    raw = _write_raw_group(data_root)
    ckpt_path = _make_checkpoint(
        "mlp", tmp_path / "best.pt", _contract_for(raw), _preprocessing_state()
    )
    # A same-shape forward product with a different sampling protocol.
    fake_forward = _write_raw_group(data_root, "ds_fake_forward", interval_s=2.0 * _INTERVAL_S)
    monkeypatch.setattr(
        forward_validation, "run_forward_simulation", lambda config_path: fake_forward.sample_dir
    )
    output_dir = tmp_path / "report"

    with pytest.raises(forward_validation.ForwardValidationError, match="time axis"):
        forward_validation.run_forward_validation(ckpt_path, raw.sample_dir, output_dir)

    assert (output_dir / "prediction.json").is_file()
    assert (output_dir / "forward_config.yaml").is_file()
    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "comparisons").exists()


def test_missing_mumax3_reports_friendly_error_and_keeps_scene(
    env: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root, _ = env
    raw = _write_raw_group(data_root)
    ckpt_path = _make_checkpoint(
        "mlp", tmp_path / "best.pt", _contract_for(raw), _preprocessing_state()
    )
    output_dir = tmp_path / "report"
    monkeypatch.delenv("MUMAX3_BIN", raising=False)
    monkeypatch.setattr(external, "find_mumax3", lambda: None)

    exit_code = _load_script().main(
        [
            "--checkpoint",
            str(ckpt_path),
            "--sample-dir",
            str(raw.sample_dir),
            "--output",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert not _CJK.search(captured.out + captured.err)
    assert "MuMax3 not found" in captured.err
    assert (output_dir / "prediction.json").is_file()
    assert (output_dir / "forward_config.yaml").is_file()
    assert not (output_dir / "metrics.json").exists()
    # Partial pipeline state is kept for diagnosis (no cleanup, no false completion).
    forward = yaml.safe_load((output_dir / "forward_config.yaml").read_text(encoding="utf-8"))
    partial = data_root / "raw" / forward["dataset_name"]
    assert partial.is_dir()
    assert "forward validation complete" not in captured.out


# ---------------------------------------------------------------------------
# Refusal / guard paths
# ---------------------------------------------------------------------------


def test_output_dir_refusal_and_dangling_symlink(env: tuple[Path, Path], tmp_path: Path) -> None:
    data_root, _ = env
    raw = _write_raw_group(data_root)
    ckpt_path = _make_checkpoint(
        "mlp", tmp_path / "best.pt", _contract_for(raw), _preprocessing_state()
    )
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        forward_validation.run_forward_validation(ckpt_path, raw.sample_dir, existing, dry_run=True)
    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert not (existing / "prediction.json").exists()

    dangling = tmp_path / "dangling"
    os.symlink(tmp_path / "no_such_target", dangling)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        forward_validation.run_forward_validation(ckpt_path, raw.sample_dir, dangling, dry_run=True)


def test_cli_refusal_exit_codes(
    env: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root, _ = env
    raw = _write_raw_group(data_root)
    ckpt_path = _make_checkpoint(
        "mlp", tmp_path / "best.pt", _contract_for(raw), _preprocessing_state()
    )
    script = _load_script()
    existing = tmp_path / "existing"
    existing.mkdir()

    exit_code = script.main(
        [
            "--checkpoint",
            str(ckpt_path),
            "--sample-dir",
            str(raw.sample_dir),
            "--output",
            str(existing),
            "--dry-run",
        ]
    )
    assert exit_code == 2
    assert "refusing to overwrite" in capsys.readouterr().err

    missing = tmp_path / "missing.pt"
    exit_code = script.main(
        [
            "--checkpoint",
            str(missing),
            "--sample-dir",
            str(raw.sample_dir),
            "--output",
            str(tmp_path / "report2"),
            "--dry-run",
        ]
    )
    assert exit_code == 2
    assert "checkpoint does not exist" in capsys.readouterr().err


def test_restore_model_rejects_weight_structure_mismatch(
    env: tuple[Path, Path], tmp_path: Path
) -> None:
    data_root, _ = env
    raw = _write_raw_group(data_root)
    contract = _contract_for(raw)
    state = _preprocessing_state()
    # Weights from a 5-wide first layer, checkpoint restores a 4-wide structure.
    model = training.build_model(contract, (5,))
    config = training_config.config_from_mapping(
        {"dataset_name": "ds_ckpt", "run_name": "run_bad", "model": {"hidden_dims": [4]}}
    )
    ckpt = training.Checkpoint(
        ckpt_format_version=training_config.CKPT_FORMAT_VERSION,
        model_state_dict=_zeroed_state_dict(model),
        hidden_dims=(4,),
        activation="relu",
        contract=contract,
        preprocessing=state,
        seed=7,
        config=config,
        dataset_meta_relpath="dataset_meta.yaml",
        dataset_meta_sha256="0" * 64,
        split_sha256="1" * 64,
        best_val_loss=0.25,
    )
    ckpt_path = tmp_path / "mismatch.pt"
    training.save_checkpoint(ckpt_path, ckpt)

    with pytest.raises(forward_validation.ForwardValidationError, match="weights"):
        forward_validation.run_forward_validation(
            ckpt_path, raw.sample_dir, tmp_path / "report", dry_run=True
        )


def test_report_writers_refuse_overwrite(tmp_path: Path) -> None:
    times = np.arange(2, dtype=np.float64)
    observed = np.zeros((2, 3))
    forward = np.ones((2, 3))
    prediction_path = tmp_path / "prediction.json"
    forward_validation.write_prediction_json(
        prediction_path, labels=(0.01, 1.0e4), prediction=np.array([0.02, 2.0e4])
    )
    with pytest.raises(FileExistsError):
        forward_validation.write_prediction_json(
            prediction_path, labels=(0.01, 1.0e4), prediction=np.array([0.02, 2.0e4])
        )

    csv_path = tmp_path / "comparisons" / "pulse_a.csv"
    forward_validation.write_comparison_csv(csv_path, times, observed, forward)
    csv_bytes = csv_path.read_bytes()
    with pytest.raises(FileExistsError):
        forward_validation.write_comparison_csv(csv_path, times, observed, forward)
    assert csv_path.read_bytes() == csv_bytes

    metrics_path = tmp_path / "metrics.json"
    forward_validation.write_metrics_json(
        metrics_path, {"pulse_a": {"rmse_vector": 0.5}}, simulation_dir=tmp_path / "sim"
    )
    with pytest.raises(FileExistsError):
        forward_validation.write_metrics_json(
            metrics_path, {"pulse_a": {"rmse_vector": 0.5}}, simulation_dir=tmp_path / "sim"
        )


def test_write_forward_config_refuses_overwrite(env: tuple[Path, Path], tmp_path: Path) -> None:
    data_root, _ = env
    raw = _write_raw_group(data_root)
    target = tmp_path / "forward_config.yaml"
    forward_validation.write_forward_config(
        raw.sample_dir / "config.yaml", target, alpha=0.01, ku_j_per_m3=1.0e4
    )
    assert load_config(target).material.alpha == 0.01

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        forward_validation.write_forward_config(
            raw.sample_dir / "config.yaml", target, alpha=0.02, ku_j_per_m3=2.0e4
        )
    assert load_config(target).material.alpha == 0.01


# ---------------------------------------------------------------------------
# Metrics and figure contracts
# ---------------------------------------------------------------------------


def test_trajectory_metrics_exact_values() -> None:
    observed = np.zeros((2, 3))
    forward = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    metrics = forward_validation.trajectory_metrics(observed, forward)
    assert metrics == pytest.approx(
        {
            "rmse_mx": np.sqrt(4.5),
            "rmse_my": np.sqrt(8.0),
            "rmse_mz": 0.0,
            "rmse_vector": np.sqrt(12.5),
            "max_vector_error": 4.0,
        },
        rel=0.0,
        abs=1.0e-15,
    )

    constant = forward_validation.trajectory_metrics(np.ones((3, 3)), np.ones((3, 3)))
    assert constant == {
        "rmse_mx": 0.0,
        "rmse_my": 0.0,
        "rmse_mz": 0.0,
        "rmse_vector": 0.0,
        "max_vector_error": 0.0,
    }


def test_plot_pulse_comparison_figure_contract_and_overwrite(tmp_path: Path) -> None:
    times = np.arange(_N_STEPS, dtype=np.float64) * _INTERVAL_S
    observed = _observed_trajectory(0)
    observed_before = observed.copy()
    forward = np.tile(np.asarray(_FORWARD_M, dtype=np.float64), (_N_STEPS, 1))
    base = tmp_path / "figures" / "pulse_a"

    png_path, svg_path = forward_validation.plot_pulse_comparison(
        times,
        observed,
        forward,
        pulse_id="pulse_a",
        output_base=base,
        context={"model_kind": "mlp", "alpha_pred": 0.01, "ku_pred": 1.0e4},
    )

    assert png_path == Path(f"{base}.png") and svg_path == Path(f"{base}.svg")
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg_text = svg_path.read_text(encoding="utf-8")
    assert svg_text.startswith("<?xml") and "<svg" in svg_text
    ET.fromstring(svg_text)
    np.testing.assert_array_equal(observed, observed_before)  # inputs are never modified

    png_before = png_path.read_bytes()
    with pytest.raises(FileExistsError):
        forward_validation.plot_pulse_comparison(
            times, observed, forward, pulse_id="pulse_a", output_base=base
        )
    assert png_path.read_bytes() == png_before

    # An existing SVG alone is refused before any PNG is created.
    base_b = tmp_path / "figures" / "pulse_b"
    Path(f"{base_b}.svg").write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError):
        forward_validation.plot_pulse_comparison(
            times, observed, forward, pulse_id="pulse_b", output_base=base_b
        )
    assert not Path(f"{base_b}.png").exists()
