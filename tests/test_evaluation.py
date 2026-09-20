"""Tests for the evaluation CLI, split/meta binding and exports (offline, CPU).

End-to-end: a real CLI training run of 1 epoch (tmp synthetic samples) → evaluation;
supplemented by unit checks of compute_subset_metrics/export. Fixtures are
self-contained (no cross-test-module import); device is always cpu; no
GPU/MuMax3/real data is touched.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from micromagnetic_parameter_inversion import (
    evaluation,
    paths,
    preprocessing,
    training,
    training_config,
    training_data,
)
from micromagnetic_parameter_inversion.evaluation import (
    EvaluationError,
    PredictionRow,
    compute_subset_metrics,
    export_test_predictions,
)
from micromagnetic_parameter_inversion.training_config import (
    SplitConfig,
    SplitMinCounts,
    SplitRatios,
)

_TRAIN_SCRIPT = paths.PROJECT_ROOT / "scripts" / "train_mlp.py"
_EVAL_SCRIPT = paths.PROJECT_ROOT / "scripts" / "evaluate_model.py"


def _load_script(path: Path, name: str) -> Any:
    """Load a scripts/ module by path (scripts/ is not a package; self-contained, no
    cross-test import)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Redirect MICROMAG_DATA_ROOT / MICROMAG_OUTPUT_ROOT to temporary directories."""
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    data_root.mkdir()
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(data_root))
    monkeypatch.setenv(paths.OUTPUT_ROOT_ENV, str(out_root))
    return data_root, out_root


def _write_prepared_samples(
    data_root: Path,
    dataset: str,
    *,
    n_groups: int = 6,
    n_pulses: int = 2,
    n_steps: int = 8,
    with_zero_ku: bool = False,
    seed: int = 0,
) -> Path:
    """Synthesize npz + dataset_meta.yaml + split.yaml directly (isomorphic to the training
    test but self-contained)."""
    rng = np.random.default_rng(seed)
    psids = tuple(f"ps{i:04d}" for i in range(n_groups))
    pulses = tuple(f"p{j}" for j in range(n_pulses))
    samples: list[training_data.Sample] = []
    labels: dict[str, tuple[float, float]] = {}
    for i, psid in enumerate(psids):
        x = rng.normal(size=(n_pulses, n_steps, 3)).astype(np.float32)
        t_s = np.arange(n_steps, dtype=np.float64) * 1.0e-9
        alpha = 0.01 * (i + 1)
        ku = 0.0 if (with_zero_ku and i == 0) else 1.0e4 * (i + 1)
        samples.append(
            training_data.Sample(
                parameter_set_id=psid,
                x=x,
                y=np.array([alpha, ku], dtype=np.float32),
                t_s=t_s,
                pulse_ids=pulses,
            )
        )
        labels[psid] = (alpha, ku)
    meta = training_data.DatasetMeta(
        dataset_name=dataset,
        pulse_order=pulses,
        n_time_steps=n_steps,
        labels=labels,
        members=psids,
    )
    split = training_data.make_split(
        psids,
        SplitConfig(
            seed=5,
            ratios=SplitRatios(0.5, 0.25, 0.25),
            min_per_split=SplitMinCounts(1, 1, 1),
        ),
    )
    samples_dir = data_root / "samples" / dataset
    training_data.write_prepared_dataset(samples_dir, samples, meta, split)
    return samples_dir


def _train_config_text(dataset: str, run_name: str) -> str:
    return (
        f"dataset_name: {dataset}\n"
        f"run_name: {run_name}\n"
        "model: {hidden_dims: [8, 4]}\n"
        "training: {\n"
        "  seed: 11, device: cpu, batch_size: 2, max_epochs: 1,\n"
        "  learning_rate: 0.01, weight_decay: 0.0,\n"
        "  early_stopping: {patience: 50, min_delta: 0.0},\n"
        "}\n"
        "split: {seed: 5, ratios: {train: 0.5, val: 0.25, test: 0.25},"
        " min_per_split: {train: 1, val: 1, test: 1}}\n"
        "output_dir: null\n"
    )


@pytest.fixture()
def trained_run(env: tuple[Path, Path]) -> tuple[Path, Path]:
    """Real CLI training of 1 epoch → (run_dir, samples_dir)."""
    data_root, _ = env
    samples_dir = _write_prepared_samples(data_root, "ds_e2e", with_zero_ku=True)
    config_path = data_root / "cfg.yaml"
    config_path.write_text(_train_config_text("ds_e2e", "r1"), encoding="utf-8")
    run_dir = _load_script(_TRAIN_SCRIPT, "train_mlp_script").run(config_path)
    return run_dir, samples_dir


def test_evaluate_end_to_end(
    trained_run: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, samples_dir = trained_run
    script = _load_script(_EVAL_SCRIPT, "evaluate_model_script")
    assert script.run(run_dir) == run_dir

    metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    run_split = yaml.safe_load((run_dir / "split.yaml").read_text(encoding="utf-8"))
    meta = training_data.load_dataset_meta(samples_dir)
    test_members = run_split["test"]
    expected_control = sum(1 for psid in test_members if meta.labels[psid][1] == 0.0)
    assert metrics["main"]["n"] == len(test_members) - expected_control
    assert metrics["control"]["n"] == expected_control
    for subset in ("main", "control"):
        for key in ("mae_alpha", "rmse_alpha", "mae_ku", "rmse_ku"):
            value = metrics[subset][key]
            assert value is None or math.isfinite(value)
    # provenance: the JSON source corresponds to the actually loaded ckpt (default
    # best.pt), not a default-path assumption
    assert set(metrics) == {"main", "control", "provenance"}
    assert metrics["provenance"]["checkpoint_path"] == str(run_dir / "best.pt")
    assert metrics["provenance"]["checkpoint_sha256"] == training_data.sha256_file(
        run_dir / "best.pt"
    )
    assert metrics["provenance"]["split_sha256"] == training_data.sha256_file(
        run_dir / "split.yaml"
    )

    csv_bytes = (run_dir / "test_predictions.csv").read_bytes()
    assert b"\r" not in csv_bytes  # LF line endings
    lines = csv_bytes.decode("utf-8").splitlines()
    assert lines[0] == "parameter_set_id,split,alpha_true,alpha_pred,ku_true,ku_pred"
    assert len(lines) - 1 == len(test_members)  # one row per psid
    assert [line.split(",")[0] for line in lines[1:]] == sorted(test_members)
    assert all(line.split(",")[1] == "test" for line in lines[1:])
    assert "evaluation complete" in capsys.readouterr().out


def test_metrics_values_empty_and_nonfinite() -> None:
    metrics = compute_subset_metrics([[1.0, 2.0], [3.0, 4.0]], [[2.0, 2.0], [3.0, 6.0]])
    assert metrics.n == 2
    assert metrics.mae_alpha == pytest.approx(0.5)
    assert metrics.rmse_alpha == pytest.approx(math.sqrt(0.5))
    assert metrics.mae_ku == pytest.approx(1.0)
    assert metrics.rmse_ku == pytest.approx(math.sqrt(2.0))

    empty = compute_subset_metrics([], [])
    assert (empty.n, empty.mae_alpha, empty.rmse_alpha, empty.mae_ku, empty.rmse_ku) == (
        0,
        None,
        None,
        None,
        None,
    )

    with pytest.raises(EvaluationError, match="non-finite"):
        compute_subset_metrics([[1.0, 2.0]], [[float("nan"), 2.0]])
    with pytest.raises(EvaluationError, match="length mismatch"):
        compute_subset_metrics([[1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]])
    with pytest.raises(EvaluationError, match="shape"):
        compute_subset_metrics([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]])


def test_export_csv_lf_and_refuses_overwrite(tmp_path: Path) -> None:
    rows = (
        PredictionRow("ps0000", "test", 0.5, 0.25, 100.0, 98.0),
        PredictionRow("ps0001", "test", 0.25, 0.5, 200.0, 210.0),
    )
    target = tmp_path / "nested" / "test_predictions.csv"
    export_test_predictions(target, rows)
    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "parameter_set_id,split,alpha_true,alpha_pred,ku_true,ku_pred"
    assert lines[1] == f"ps0000,test,{0.5!r},{0.25!r},{100.0!r},{98.0!r}"
    assert target.read_bytes().count(b"\n") == 3  # header + 2 rows, LF
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_test_predictions(target, rows)


def test_split_sha_mismatch_rejected(
    trained_run: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, _ = trained_run
    split_copy = run_dir / "split.yaml"
    split_copy.write_bytes(split_copy.read_bytes() + b"\n# drifted\n")
    script = _load_script(_EVAL_SCRIPT, "evaluate_model_script")
    with pytest.raises(EvaluationError, match="does not match ckpt.split_sha256"):
        evaluation.run_evaluation(run_dir)
    # CLI explicit --checkpoint must still bind the run split: friendly error + exit code 2
    code = script.main(["--run", str(run_dir), "--checkpoint", str(run_dir / "best.pt")])
    assert code == 2
    assert "does not match ckpt.split_sha256" in capsys.readouterr().err


def test_meta_sha_mismatch_rejected(trained_run: tuple[Path, Path]) -> None:
    run_dir, samples_dir = trained_run
    meta_path = samples_dir / "dataset_meta.yaml"
    meta_path.write_text(meta_path.read_text(encoding="utf-8") + "\n# drifted\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="dataset_meta SHA"):
        evaluation.run_evaluation(run_dir)


def test_meta_relpath_escape_rejected(trained_run: tuple[Path, Path], tmp_path: Path) -> None:
    run_dir, _ = trained_run
    payload = torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu")
    payload["dataset_meta_relpath"] = "../outside.yaml"
    forged = tmp_path / "forged.pt"
    torch.save(payload, forged)
    with pytest.raises(EvaluationError, match="escapes the anchor"):
        evaluation.run_evaluation(run_dir, forged)


def test_pulse_time_contract_mismatch_rejected(trained_run: tuple[Path, Path]) -> None:
    run_dir, samples_dir = trained_run
    run_split = yaml.safe_load((run_dir / "split.yaml").read_text(encoding="utf-8"))
    victim = samples_dir / f"{run_split['test'][0]}.npz"
    with np.load(victim, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["x"] = arrays["x"][:, :-1, :]  # one fewer T step: mismatch with the ckpt contract
    arrays["t_s"] = arrays["t_s"][:-1]
    np.savez_compressed(victim, **arrays)
    with pytest.raises(training_data.DataError, match="contract"):
        evaluation.run_evaluation(run_dir)


def test_samples_resplit_but_run_split_authoritative(trained_run: tuple[Path, Path]) -> None:
    run_dir, samples_dir = trained_run
    original_test = yaml.safe_load((run_dir / "split.yaml").read_text(encoding="utf-8"))["test"]
    # the split.yaml under samples is re-cut (different seed): the run copy is still
    # the only authority
    recut = training_data.make_split(
        tuple(sorted(training_data.load_dataset_meta(samples_dir).members)),
        SplitConfig(
            seed=99,
            ratios=SplitRatios(0.34, 0.33, 0.33),
            min_per_split=SplitMinCounts(0, 0, 0),
        ),
    )
    (samples_dir / "split.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": recut.seed,
                "ratios": {
                    "train": recut.ratios.train,
                    "val": recut.ratios.val,
                    "test": recut.ratios.test,
                },
                "train": list(recut.train),
                "val": list(recut.val),
                "test": list(recut.test),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report, rows = evaluation.run_evaluation(run_dir)
    assert [row.parameter_set_id for row in rows] == sorted(original_test)
    assert report.main.n + report.control.n == len(original_test)


def test_multi_batch_tail_batch_order_and_provenance(
    env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple members across multiple batches (including a tail batch): row order strictly
    follows split.test order, main/control non-empty, provenance corresponds to the
    actually loaded ckpt."""
    data_root, _ = env
    samples_dir = _write_prepared_samples(data_root, "ds_multi", n_groups=7, with_zero_ku=True)
    # deliberately non-sorted test order, also containing the control (ps0000, Ku=0)
    test_order = ["ps0003", "ps0000", "ps0006", "ps0002", "ps0005"]
    _rewrite_split(samples_dir, ["ps0001"], ["ps0004"], test_order)
    config_path = data_root / "cfg_multi.yaml"
    config_path.write_text(_train_config_text("ds_multi", "r1"), encoding="utf-8")
    run_dir = _load_script(_TRAIN_SCRIPT, "train_mlp_script").run(config_path)

    # Detect batch boundaries: transform_x is called exactly once per batch
    batch_shapes: list[tuple[int, ...]] = []
    real_transform_x = preprocessing.transform_x

    def spy_transform_x(state: Any, x: Any) -> Any:
        batch_shapes.append(tuple(np.asarray(x).shape))
        return real_transform_x(state, x)

    monkeypatch.setattr(preprocessing, "transform_x", spy_transform_x)
    report, rows = evaluation.run_evaluation(run_dir)

    # batch_size=2, test has 5 members → 2+2+1: tail batch of 1 sample
    assert batch_shapes == [(2, 2, 8, 3), (2, 2, 8, 3), (1, 2, 8, 3)]
    # row order strictly follows split.test order (non-sorted)
    assert [row.parameter_set_id for row in rows] == test_order
    assert all(row.split == "test" for row in rows)
    # main=4 (excluding the control ps0000), control=1, both non-empty with finite metrics
    assert report.main.n == 4 and report.control.n == 1
    for subset in (report.main, report.control):
        assert subset.mae_alpha is not None and math.isfinite(subset.mae_alpha)
        assert subset.rmse_ku is not None and math.isfinite(subset.rmse_ku)
    # provenance: the actually loaded default best.pt
    assert report.provenance.checkpoint_path == str(run_dir / "best.pt")
    assert report.provenance.checkpoint_sha256 == training_data.sha256_file(run_dir / "best.pt")
    assert report.provenance.split_sha256 == training_data.sha256_file(run_dir / "split.yaml")


def test_cli_checkpoint_final_provenance_and_rowcount(
    trained_run: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """``--checkpoint final.pt``: the JSON provenance records the actual path/hash, row
    count == test member count."""
    run_dir, _ = trained_run
    script = _load_script(_EVAL_SCRIPT, "evaluate_model_script")
    final_path = run_dir / "final.pt"
    assert final_path.is_file()
    assert script.main(["--run", str(run_dir), "--checkpoint", str(final_path)]) == 0
    capsys.readouterr()
    metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"main", "control", "provenance"}
    provenance = metrics["provenance"]
    assert provenance["checkpoint_path"] == str(final_path)
    assert provenance["checkpoint_sha256"] == training_data.sha256_file(final_path)
    assert provenance["checkpoint_sha256"] != training_data.sha256_file(run_dir / "best.pt")
    assert provenance["split_sha256"] == training_data.sha256_file(run_dir / "split.yaml")
    run_split = yaml.safe_load((run_dir / "split.yaml").read_text(encoding="utf-8"))
    lines = (run_dir / "test_predictions.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) - 1 == len(run_split["test"])


def test_provenance_tracks_actually_loaded_checkpoint(
    trained_run: tuple[Path, Path], tmp_path: Path
) -> None:
    """ckpts with the same split hash but different model weights: provenance reflects the
    actually loaded file.

    The same split across runs is legal (no new restriction); after mutating one set of
    weights the split binding is unchanged, and provenance's checkpoint_sha256/path must
    follow the actually loaded ckpt.
    """
    run_dir, _ = trained_run
    report_default, _ = evaluation.run_evaluation(run_dir)
    assert report_default.provenance.checkpoint_path == str(run_dir / "best.pt")
    assert report_default.provenance.checkpoint_sha256 == training_data.sha256_file(
        run_dir / "best.pt"
    )
    assert report_default.provenance.split_sha256 == training_data.sha256_file(
        run_dir / "split.yaml"
    )

    payload = torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu")
    name = sorted(payload["model_state_dict"])[0]
    payload["model_state_dict"][name] = payload["model_state_dict"][name] + 1.0
    mutated = tmp_path / "mutated.pt"
    torch.save(payload, mutated)

    report_mutated, _ = evaluation.run_evaluation(run_dir, mutated)
    assert report_mutated.provenance.checkpoint_path == str(mutated)
    assert report_mutated.provenance.checkpoint_sha256 == training_data.sha256_file(mutated)
    assert (
        report_mutated.provenance.checkpoint_sha256 != report_default.provenance.checkpoint_sha256
    )
    # split binding unchanged: provenance.split_sha256 comes from the actual ckpt and
    # matches the run copy
    assert report_mutated.provenance.split_sha256 == report_default.provenance.split_sha256


def test_cli_second_eval_blocked(
    trained_run: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, _ = trained_run
    script = _load_script(_EVAL_SCRIPT, "evaluate_model_script")
    assert script.run(run_dir) == run_dir
    assert script.main(["--run", str(run_dir)]) == 2
    assert "refusing to overwrite" in capsys.readouterr().err


def test_main_missing_run_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = _load_script(_EVAL_SCRIPT, "evaluate_model_script").main(
        ["--run", str(tmp_path / "nope")]
    )
    assert code == 2
    assert "error:" in capsys.readouterr().err


def _rewrite_split(samples_dir: Path, train: list[str], val: list[str], test: list[str]) -> None:
    """Rewrite the split.yaml under samples (member completeness/mutual exclusion is
    guaranteed by the test data)."""
    (samples_dir / "split.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": 5,
                "ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
                "train": train,
                "val": val,
                "test": test,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_evaluate_with_empty_test_split_null_metrics(env: tuple[Path, Path]) -> None:
    data_root, _ = env
    samples_dir = _write_prepared_samples(data_root, "ds_no_test")
    meta = training_data.load_dataset_meta(samples_dir)
    psids = sorted(meta.members)
    _rewrite_split(samples_dir, psids[:4], psids[4:], [])  # empty test: legal (min allows 0)
    config_path = data_root / "cfg.yaml"
    config_path.write_text(_train_config_text("ds_no_test", "r1"), encoding="utf-8")
    run_dir = _load_script(_TRAIN_SCRIPT, "train_mlp_script").run(config_path)

    assert _load_script(_EVAL_SCRIPT, "evaluate_model_script").run(run_dir) == run_dir
    metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["main"]["n"] == 0 and metrics["control"]["n"] == 0
    assert metrics["main"]["mae_alpha"] is None and metrics["main"]["rmse_ku"] is None
    lines = (run_dir / "test_predictions.csv").read_text(encoding="utf-8").splitlines()
    assert lines == ["parameter_set_id,split,alpha_true,alpha_pred,ku_true,ku_pred"]


def test_train_rejects_empty_train_split(env: tuple[Path, Path]) -> None:
    data_root, out_root = env
    samples_dir = _write_prepared_samples(data_root, "ds_no_train")
    _rewrite_split(
        samples_dir, [], sorted(meta_psids(samples_dir)[:2]), sorted(meta_psids(samples_dir)[2:])
    )
    config_path = data_root / "cfg.yaml"
    config_path.write_text(_train_config_text("ds_no_train", "r1"), encoding="utf-8")
    with pytest.raises(training_data.DataError, match="split.train is empty"):
        _load_script(_TRAIN_SCRIPT, "train_mlp_script").run(config_path)
    assert not (out_root / "training" / "mlp" / "ds_no_train" / "r1").exists()


def meta_psids(samples_dir: Path) -> list[str]:
    """Small in-test helper: list manifest members in order."""
    return list(training_data.load_dataset_meta(samples_dir).members)


# ---------------------------------------------------------------------------
# P3: evaluation integration for the independent CNN checkpoint (CPU-synthetic; no
# training, no reading real MLP artifacts)
# ---------------------------------------------------------------------------

_CNN_CHANNELS = (3, 4)  # unit-test-only structure, not a research hyperparameter
_CNN_KERNELS = (3, 5)
_CNN_POOL_BINS = 2
_CNN_HEAD = (5,)
# Fixed split members: test contains the control ps0000 and two main-domain members
# (batch_size=2 → 2+1 batches).
_CNN_SPLIT_OVERRIDE = (["ps0001", "ps0002"], ["ps0003"], ["ps0000", "ps0004", "ps0005"])


def _cnn_config(
    dataset: str, run_name: str, *, model_overrides: dict[str, object] | None = None
) -> training_config.ExperimentConfig:
    """Synthesize a CNN config (unit-test-only structure; not via a research YAML)."""
    model: dict[str, object] = {
        "kind": "cnn1d",
        "channels": list(_CNN_CHANNELS),
        "kernel_sizes": list(_CNN_KERNELS),
        "pool_bins": _CNN_POOL_BINS,
        "head_hidden_dims": list(_CNN_HEAD),
    }
    if model_overrides is not None:
        model.update(model_overrides)
    return training_config.config_from_mapping(
        {
            "dataset_name": dataset,
            "run_name": run_name,
            "model": model,
            "training": {
                "seed": 11,
                "device": "cpu",
                "batch_size": 2,
                "max_epochs": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
            },
        }
    )


def _make_cnn_run(
    data_root: Path,
    run_dir: Path,
    dataset: str,
    *,
    split_override: tuple[list[str], list[str], list[str]] | None = _CNN_SPLIT_OVERRIDE,
    model_overrides: dict[str, object] | None = None,
) -> tuple[Path, training.CNNCheckpoint, training_config.ExperimentConfig]:
    """Synthesize a CNN run: train-only preprocessing + random-weight CNNCheckpoint (no training).

    Only this file's helpers synthesize npz/meta/split; no CNN optimization/training is
    run and no real MLP checkpoint/artifacts/statistics are read. ``preprocessing.fit``
    is fed train members only (leak prevention).
    """
    samples_dir = _write_prepared_samples(data_root, dataset, n_groups=6, with_zero_ku=True)
    if split_override is not None:
        _rewrite_split(samples_dir, *split_override)
    meta = training_data.load_dataset_meta(samples_dir)
    split = training_data.load_split(samples_dir, meta)

    config = _cnn_config(dataset, run_dir.name, model_overrides=model_overrides)
    assert isinstance(config.model, training_config.CNN1DModelConfig)
    cnn_model = config.model

    train_set = training_data.TrajectoryDataset(samples_dir, meta, split.train)
    first = train_set.samples[0]
    contract = training_data.InputContract(
        pulse_order=first.pulse_ids,
        n_time_steps=int(first.x.shape[1]),
        t_s=first.t_s,
    )
    state = preprocessing.fit(train_set, config.preprocessing, config.label)

    with torch.random.fork_rng(devices=[]):  # isolate the global RNG, leaving no side effects
        torch.manual_seed(0)
        model = training.build_cnn_model(
            contract,
            channels=cnn_model.channels,
            kernel_sizes=cnn_model.kernel_sizes,
            pool_bins=cnn_model.pool_bins,
            head_hidden_dims=cnn_model.head_hidden_dims,
        )
    state_dict = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    ckpt = training.CNNCheckpoint(
        ckpt_format_version=training.CNN_CKPT_FORMAT_VERSION,
        model_state_dict=state_dict,
        channels=cnn_model.channels,
        kernel_sizes=cnn_model.kernel_sizes,
        pool_bins=cnn_model.pool_bins,
        head_hidden_dims=cnn_model.head_hidden_dims,
        activation="relu",
        contract=contract,
        preprocessing=state,
        seed=config.training.seed,
        config=config,
        dataset_meta_relpath="dataset_meta.yaml",
        dataset_meta_sha256=training_data.sha256_file(samples_dir / "dataset_meta.yaml"),
        split_sha256=training_data.sha256_file(samples_dir / "split.yaml"),
        best_val_loss=0.5,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "split.yaml").write_bytes((samples_dir / "split.yaml").read_bytes())
    training.save_cnn_checkpoint(run_dir / "best.pt", ckpt)
    return samples_dir, ckpt, config


@pytest.fixture()
def cnn_run(
    env: tuple[Path, Path], tmp_path: Path
) -> tuple[Path, Path, training.CNNCheckpoint, training_config.ExperimentConfig]:
    """Each test gets an independent CNN run (so evaluation artifacts' "refusing to
    overwrite" do not interfere)."""
    data_root, _ = env
    run_dir = tmp_path / "run_cnn"
    samples_dir, ckpt, config = _make_cnn_run(data_root, run_dir, "ds_cnn")
    return run_dir, samples_dir, ckpt, config


def _expected_cnn_predictions(
    ckpt: training.CNNCheckpoint, samples_dir: Path, psids: tuple[str, ...]
) -> dict[str, tuple[float, float]]:
    """Independent reference: physical-unit predictions from a self-built explicit-structure
    model + ckpt preprocessing inverse transform."""
    with torch.random.fork_rng(devices=[]):
        model = training.build_cnn_model(
            ckpt.contract,
            channels=ckpt.channels,
            kernel_sizes=ckpt.kernel_sizes,
            pool_bins=ckpt.pool_bins,
            head_hidden_dims=ckpt.head_hidden_dims,
        )
    model.load_state_dict(dict(ckpt.model_state_dict), strict=True)
    model.eval()
    expected: dict[str, tuple[float, float]] = {}
    for psid in psids:
        sample = training_data.load_sample(samples_dir, psid, ckpt.contract)
        x_norm = preprocessing.transform_x(ckpt.preprocessing, sample.x[None])
        with torch.no_grad():
            pred_norm = model(torch.from_numpy(x_norm))
        physical = preprocessing.inverse_transform_y(ckpt.preprocessing, pred_norm.numpy())[0]
        expected[psid] = (float(physical[0]), float(physical[1]))
    return expected


def _corrupt_state_dict(state_dict: training.StateDict, corruption: str) -> dict[str, Any]:
    """Build a corrupted model_state_dict: missing key / extra key / wrong shape."""
    state = dict(state_dict)
    if corruption == "missing_key":
        state.pop(sorted(state)[0])
    elif corruption == "extra_key":
        state["bogus.weight"] = torch.zeros(1)
    else:  # wrong_shape: keep the element count but change the shape (size mismatch)
        name = sorted(state)[0]
        state[name] = state[name].reshape(1, -1)
    return state


def _load_ckpt_payload_for_corruption(ckpt_path: Path) -> dict[str, Any]:
    """Read a legal ckpt's on-disk payload for "corruption" rewriting (same method as
    ``_corrupt_state_dict``).

    **Deliberately bypasses ``save_cnn_checkpoint`` validation**: illegal values such as
    batch_size=inf, y_std=0, x_std=inf would be rejected early by a normal save, so here
    an existing legal ``best.pt`` payload is ``torch.load``-ed, mutated in place, then
    ``torch.save``-d to simulate real disk corruption, exercising only the load/evaluate
    boundary without relaxing or changing save behavior.
    """
    return torch.load(ckpt_path, weights_only=True, map_location="cpu")


def test_cnn_evaluate_end_to_end(cnn_run: tuple[Path, Path, training.CNNCheckpoint, Any]) -> None:
    run_dir, samples_dir, ckpt, _ = cnn_run
    report, rows = evaluation.run_evaluation(run_dir)

    meta = training_data.load_dataset_meta(samples_dir)
    run_split = yaml.safe_load((run_dir / "split.yaml").read_text(encoding="utf-8"))
    test_members = run_split["test"]
    expected_control = sum(1 for psid in test_members if meta.labels[psid][1] == 0.0)

    assert [row.parameter_set_id for row in rows] == test_members
    assert len(rows) == len(test_members)
    assert report.main.n == len(test_members) - expected_control
    assert report.control.n == expected_control
    for subset in (report.main, report.control):
        for key in ("mae_alpha", "rmse_alpha", "mae_ku", "rmse_ku"):
            value = getattr(subset, key)
            assert value is None or math.isfinite(value)

    assert report.provenance.checkpoint_path == str(run_dir / "best.pt")
    assert report.provenance.checkpoint_sha256 == training_data.sha256_file(run_dir / "best.pt")
    assert report.provenance.split_sha256 == training_data.sha256_file(run_dir / "split.yaml")

    # Physical-unit predictions match the independent reference ("self model + ckpt preprocessing
    # inverse transform"), not merely asserting existence.
    expected = _expected_cnn_predictions(ckpt, samples_dir, tuple(test_members))
    for row in rows:
        exp_alpha, exp_ku = expected[row.parameter_set_id]
        assert row.split == "test"
        assert row.alpha_true == pytest.approx(meta.labels[row.parameter_set_id][0])
        assert row.ku_true == pytest.approx(meta.labels[row.parameter_set_id][1])
        assert row.alpha_pred == pytest.approx(exp_alpha, rel=1.0e-5, abs=1.0e-6)
        assert row.ku_pred == pytest.approx(exp_ku, rel=1.0e-5, abs=1.0e-6)


def test_cnn_evaluate_script_end_to_end(
    cnn_run: tuple[Path, Path, training.CNNCheckpoint, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, _samples_dir, _ckpt, _config = cnn_run
    script = _load_script(_EVAL_SCRIPT, "evaluate_model_script")
    assert script.run(run_dir) == run_dir

    metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"main", "control", "provenance"}
    assert metrics["provenance"]["checkpoint_path"] == str(run_dir / "best.pt")
    assert metrics["provenance"]["checkpoint_sha256"] == training_data.sha256_file(
        run_dir / "best.pt"
    )
    run_split = yaml.safe_load((run_dir / "split.yaml").read_text(encoding="utf-8"))
    lines = (run_dir / "test_predictions.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) - 1 == len(run_split["test"])
    assert "evaluation complete" in capsys.readouterr().out


@pytest.mark.parametrize("case", ["split_sha", "meta_sha", "contract_t_s"])
def test_cnn_binding_tamper_rejected(
    cnn_run: tuple[Path, Path, training.CNNCheckpoint, Any], case: str
) -> None:
    run_dir, samples_dir, _ckpt, _config = cnn_run
    if case == "split_sha":
        target = run_dir / "split.yaml"
        target.write_bytes(target.read_bytes() + b"\n# drifted\n")
        with pytest.raises(EvaluationError, match="does not match ckpt.split_sha256"):
            evaluation.run_evaluation(run_dir)
    elif case == "meta_sha":
        target = samples_dir / "dataset_meta.yaml"
        target.write_text(target.read_text(encoding="utf-8") + "\n# drifted\n", encoding="utf-8")
        with pytest.raises(EvaluationError, match="dataset_meta SHA"):
            evaluation.run_evaluation(run_dir)
    else:
        run_split = yaml.safe_load((run_dir / "split.yaml").read_text(encoding="utf-8"))
        victim = samples_dir / f"{run_split['test'][0]}.npz"
        with np.load(victim, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        # Equal length, finite, strictly increasing, but different values: passes the
        # checkpoint loader and must be rejected by load_sample's full bitwise t_s
        # comparison (not just shape/length).
        arrays["t_s"] = arrays["t_s"] + np.float64(1.0e-15)
        np.savez_compressed(victim, **arrays)
        with pytest.raises((training_data.DataError, EvaluationError), match="t_s"):
            evaluation.run_evaluation(run_dir)
        assert not (run_dir / "test_metrics.json").exists()
        assert not (run_dir / "test_predictions.csv").exists()
        # The CLI rejects it equally and produces no success report.
        assert (
            _load_script(_EVAL_SCRIPT, "evaluate_model_script").main(["--run", str(run_dir)]) == 2
        )
        assert not (run_dir / "test_metrics.json").exists()
        assert not (run_dir / "test_predictions.csv").exists()


@pytest.mark.parametrize("corruption", ["missing_key", "extra_key", "wrong_shape"])
def test_cnn_corrupt_state_dict_rejected_at_evaluate(
    cnn_run: tuple[Path, Path, training.CNNCheckpoint, Any], tmp_path: Path, corruption: str
) -> None:
    run_dir, _samples_dir, ckpt, _config = cnn_run
    forged = tmp_path / f"corrupt_{corruption}.pt"
    training.save_cnn_checkpoint(
        forged,
        dataclasses.replace(
            ckpt, model_state_dict=_corrupt_state_dict(ckpt.model_state_dict, corruption)
        ),
    )
    with pytest.raises(EvaluationError):
        evaluation.run_evaluation(run_dir, forged)
    # Produces no success report / partial artifacts.
    assert not (run_dir / "test_metrics.json").exists()
    assert not (run_dir / "test_predictions.csv").exists()

    # The CLI fails equally gracefully (exit 2) and writes no artifacts.
    assert (
        _load_script(_EVAL_SCRIPT, "evaluate_model_script").main(
            ["--run", str(run_dir), "--checkpoint", str(forged)]
        )
        == 2
    )
    assert not (run_dir / "test_metrics.json").exists()
    assert not (run_dir / "test_predictions.csv").exists()


def test_cnn_recovery_uses_explicit_structure_not_nested_config(
    cnn_run: tuple[Path, Path, training.CNNCheckpoint, Any], tmp_path: Path
) -> None:
    """Same kind=cnn1d but a nested config structure differing from the top-level explicit
    fields: must restore from the explicit fields."""
    run_dir, samples_dir, ckpt, _config = cnn_run
    other_config = _cnn_config(
        ckpt.config.dataset_name,
        "other_run",
        model_overrides={
            "channels": [5, 6],
            "kernel_sizes": [3, 3],
            "pool_bins": 3,
            "head_hidden_dims": [9],
        },
    )
    forged = tmp_path / "explicit_structure.pt"
    training.save_cnn_checkpoint(forged, dataclasses.replace(ckpt, config=other_config))

    # Restoring from the nested config structure would cause a load_state_dict shape
    # mismatch; this must succeed.
    _report, rows = evaluation.run_evaluation(run_dir, forged)
    run_split = yaml.safe_load((run_dir / "split.yaml").read_text(encoding="utf-8"))
    expected = _expected_cnn_predictions(ckpt, samples_dir, tuple(run_split["test"]))
    for row in rows:
        exp_alpha, exp_ku = expected[row.parameter_set_id]
        assert row.alpha_pred == pytest.approx(exp_alpha, rel=1.0e-5, abs=1.0e-6)
        assert row.ku_pred == pytest.approx(exp_ku, rel=1.0e-5, abs=1.0e-6)


def test_cnn_overflow_config_rejected_without_leak(
    cnn_run: tuple[Path, Path, training.CNNCheckpoint, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """config.training.batch_size=inf: the CNN loader must converge to an explicit error,
    not leak OverflowError."""
    run_dir, _samples_dir, _ckpt, _config = cnn_run
    # Bypass save_cnn_checkpoint (which already rejects early correctly) and rewrite the
    # on-disk payload directly to simulate corruption.
    payload = _load_ckpt_payload_for_corruption(run_dir / "best.pt")
    payload["config"]["training"]["batch_size"] = float("inf")
    forged = tmp_path / "overflow_batch.pt"
    torch.save(payload, forged)

    code = _load_script(_EVAL_SCRIPT, "evaluate_model_script").main(
        ["--run", str(run_dir), "--checkpoint", str(forged)]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "error:" in err
    assert "OverflowError" not in err  # does not leak the raw overflow exception
    assert ("checkpoint" in err) or ("config" in err)  # explicit checkpoint/config error
    assert not (run_dir / "test_metrics.json").exists()
    assert not (run_dir / "test_predictions.csv").exists()


@pytest.mark.parametrize("bad", ["y_std_zero", "x_std_inf"])
def test_cnn_bad_preprocessing_state_rejected(
    cnn_run: tuple[Path, Path, training.CNNCheckpoint, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bad: str,
) -> None:
    """Illegal CNN normalization state (y_std=0 / x_std=inf): the CLI rejects it with no
    report artifacts."""
    run_dir, _samples_dir, _ckpt, _config = cnn_run
    # Bypass save_cnn_checkpoint (which already rejects early correctly) and rewrite the
    # on-disk payload directly to simulate corruption.
    payload = _load_ckpt_payload_for_corruption(run_dir / "best.pt")
    if bad == "y_std_zero":
        payload["preprocessing"]["y_stats"]["y_std"] = np.zeros(2, dtype=np.float64).tolist()
    else:
        x_std = np.asarray(payload["preprocessing"]["x_stats"]["std"], dtype=np.float64)
        payload["preprocessing"]["x_stats"]["std"] = np.full_like(x_std, np.inf).tolist()
    forged = tmp_path / f"bad_preprocessing_{bad}.pt"
    torch.save(payload, forged)

    code = _load_script(_EVAL_SCRIPT, "evaluate_model_script").main(
        ["--run", str(run_dir), "--checkpoint", str(forged)]
    )
    assert code == 2
    assert "error:" in capsys.readouterr().err
    assert not (run_dir / "test_metrics.json").exists()
    assert not (run_dir / "test_predictions.csv").exists()
