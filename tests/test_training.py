"""Tests for the training CLI, ckpt safety and reload parity (offline, CPU).

Synthetic npz/meta/split are written directly into tmp_path via training_data
(no raw/prepare, no cross-test fixture import); device is always cpu; no
GPU/MuMax3.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from micromagnetic_parameter_inversion import (
    evaluation,
    paths,
    preprocessing,
    training,
    training_config,
    training_data,
)
from micromagnetic_parameter_inversion.training_config import (
    CNN1DModelConfig,
    ConfigError,
    ExperimentConfig,
    ModelConfig,
    SplitConfig,
    SplitMinCounts,
    SplitRatios,
    load_config,
)

_SCRIPT_PATH = paths.PROJECT_ROOT / "scripts" / "train_mlp.py"
_CNN_SCRIPT_PATH = paths.PROJECT_ROOT / "scripts" / "train_cnn1d.py"

# Shared run (training.run, early expected_kind rejection) and the two thin entries;
# the CNN reuses only this file's synthetic-data helpers and neither reads nor
# reuses MLP weights/statistics/artifacts.


def _load_script(path: Path = _SCRIPT_PATH, name: str = "train_mlp_script") -> Any:
    """Load a scripts/ module by path (scripts/ is not a package; self-contained, no
    cross-test import)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_cnn_script() -> Any:
    """Load the train_cnn1d script module (thin entry; not MLP by default)."""
    return _load_script(_CNN_SCRIPT_PATH, "train_cnn1d_script")


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
    seed: int = 0,
) -> Path:
    """Synthesize npz + dataset_meta.yaml + split.yaml directly (skipping raw/prepare)."""
    rng = np.random.default_rng(seed)
    psids = tuple(f"ps{i:04d}" for i in range(n_groups))
    pulses = tuple(f"p{j}" for j in range(n_pulses))
    samples: list[training_data.Sample] = []
    labels: dict[str, tuple[float, float]] = {}
    for i, psid in enumerate(psids):
        x = rng.normal(size=(n_pulses, n_steps, 3)).astype(np.float32)
        t_s = np.arange(n_steps, dtype=np.float64) * 1.0e-9
        alpha = 0.01 * (i + 1)
        ku = 1.0e4 * (i + 1)
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


def _config_text(
    dataset: str,
    run_name: str,
    *,
    max_epochs: int = 2,
    patience: int = 50,
    min_delta: float = 0.0,
    seed: int = 11,
    output_dir: str | None = None,
) -> str:
    output = "null" if output_dir is None else output_dir
    return (
        f"dataset_name: {dataset}\n"
        f"run_name: {run_name}\n"
        "model: {hidden_dims: [8, 4]}\n"
        "training: {\n"
        f"  seed: {seed}, device: cpu, batch_size: 2, max_epochs: {max_epochs},\n"
        "  learning_rate: 0.01, weight_decay: 0.0,\n"
        f"  early_stopping: {{patience: {patience}, min_delta: {min_delta}}},\n"
        "}\n"
        "split: {seed: 5, ratios: {train: 0.5, val: 0.25, test: 0.25},"
        " min_per_split: {train: 1, val: 1, test: 1}}\n"
        f"output_dir: {output}\n"
    )


def _write_config(data_root: Path, text: str, name: str = "cfg.yaml") -> Path:
    config_path = data_root / name
    config_path.write_text(text, encoding="utf-8")
    return config_path


def test_run_trains_and_writes_artifacts(env: tuple[Path, Path]) -> None:
    data_root, out_root = env
    samples_dir = _write_prepared_samples(data_root, "ds1")
    config_path = _write_config(data_root, _config_text("ds1", "r1"))
    run_dir = _load_script().run(config_path)

    assert run_dir == out_root / "training" / "mlp" / "ds1" / "r1"
    for name in (
        "best.pt",
        "final.pt",
        "split.yaml",
        "config_resolved.yaml",
        "preprocessing.yaml",
        "metrics.json",
    ):
        assert (run_dir / name).is_file(), name

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["history"]) == 2
    assert metrics["stop_reason"] == "max_epochs"
    assert metrics["stop_epoch"] == 2
    assert metrics["detail"] is None
    for entry in metrics["history"]:
        assert math.isfinite(entry["train_loss"]) and math.isfinite(entry["val_loss"])
    assert metrics["best_val_loss"] == min(e["val_loss"] for e in metrics["history"])
    assert metrics["best_epoch"] in {entry["epoch"] for entry in metrics["history"]}

    # config_resolved.yaml can be reloaded by load_config with consistent fields
    resolved = load_config(run_dir / "config_resolved.yaml")
    assert (resolved.dataset_name, resolved.run_name) == ("ds1", "r1")
    assert isinstance(resolved.model, ModelConfig)
    assert resolved.model.hidden_dims == (8, 4)
    assert resolved.training.max_epochs == 2
    # the split copy equals the source split.yaml raw bytes
    assert (run_dir / "split.yaml").read_bytes() == (samples_dir / "split.yaml").read_bytes()
    prep = yaml.safe_load((run_dir / "preprocessing.yaml").read_text(encoding="utf-8"))
    assert len(prep["x_stats"]["mean"]) == 2  # [P=2, 1, 3]
    assert len(prep["x_stats"]["mean"][0][0]) == 3
    assert prep["y_stats"]["transform"] == "identity"

    # weights_only safe read
    for name in ("best.pt", "final.pt"):
        payload = torch.load(run_dir / name, weights_only=True, map_location="cpu")
        assert isinstance(payload, dict)
        assert all(isinstance(t, torch.Tensor) for t in payload["model_state_dict"].values())


def test_ckpt_roundtrip_reload_parity(env: tuple[Path, Path]) -> None:
    data_root, _ = env
    samples_dir = _write_prepared_samples(data_root, "ds2")
    config_path = _write_config(data_root, _config_text("ds2", "r1"))
    run_dir = _load_script().run(config_path)

    ckpt = training.load_checkpoint(run_dir / "best.pt")
    assert isinstance(
        ckpt, training.Checkpoint
    )  # necessary narrowing under the load_any union type
    assert ckpt.contract.input_shape == (2, 8, 3)
    assert ckpt.contract.pulse_order == ("p0", "p1")
    assert ckpt.hidden_dims == (8, 4)
    assert ckpt.activation == "relu"
    assert ckpt.best_val_loss is not None and math.isfinite(ckpt.best_val_loss)
    assert ckpt.split_sha256 == training_data.sha256_file(samples_dir / "split.yaml")
    assert ckpt.dataset_meta_relpath == "dataset_meta.yaml"
    assert ckpt.dataset_meta_sha256 == training_data.sha256_file(samples_dir / "dataset_meta.yaml")
    assert all(t.device.type == "cpu" for t in ckpt.model_state_dict.values())
    assert all(t.isfinite().all() for t in ckpt.model_state_dict.values())
    # preprocessing state roundtrip: mean/effective-scale shapes and semantics
    assert ckpt.preprocessing.x_stats.mean.shape == (2, 1, 3)
    assert ckpt.preprocessing.y_stats.y_mean.shape == (2,)

    def _predict(checkpoint: training.Checkpoint) -> torch.Tensor:
        model = training.build_model(checkpoint.contract, checkpoint.hidden_dims)
        model.load_state_dict(dict(checkpoint.model_state_dict))
        model.eval()
        probe = torch.zeros(3, *checkpoint.contract.input_shape)
        with torch.no_grad():
            return model(probe)

    # two independent loads + rebuilt models → bitwise-identical predictions
    again = training.load_checkpoint(run_dir / "best.pt")
    assert torch.equal(_predict(ckpt), _predict(again))
    # final.pt has best_val_loss None (only best has it)
    final = training.load_checkpoint(run_dir / "final.pt")
    assert final.best_val_loss is None
    assert torch.equal(_predict(final), _predict(training.load_checkpoint(run_dir / "final.pt")))


def test_save_checkpoint_refuses_overwrite(env: tuple[Path, Path]) -> None:
    data_root, _ = env
    _write_prepared_samples(data_root, "ds3")
    config_path = _write_config(data_root, _config_text("ds3", "r1"))
    run_dir = _load_script().run(config_path)
    ckpt = training.load_checkpoint(run_dir / "best.pt")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        training.save_checkpoint(run_dir / "best.pt", ckpt)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        training.save_checkpoint(run_dir / "final.pt", ckpt)


def test_load_checkpoint_rejects_corrupt(env: tuple[Path, Path], tmp_path: Path) -> None:
    data_root, _ = env
    _write_prepared_samples(data_root, "ds4")
    config_path = _write_config(data_root, _config_text("ds4", "r1"))
    run_dir = _load_script().run(config_path)
    payload = torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu")

    def _broken(mutate: Any) -> Path:
        import copy

        damaged = copy.deepcopy(payload)
        mutate(damaged)
        path = tmp_path / "broken.pt"
        torch.save(damaged, path)
        return path

    with pytest.raises(training.TrainingError, match="ckpt_format_version"):
        training.load_checkpoint(_broken(lambda p: p.__setitem__("ckpt_format_version", 999)))
    with pytest.raises(training.TrainingError, match="missing keys"):
        training.load_checkpoint(_broken(lambda p: p.pop("contract")))
    with pytest.raises(training.TrainingError, match="t_s"):
        training.load_checkpoint(
            _broken(lambda p: p["contract"].__setitem__("t_s", p["contract"]["t_s"][:-1]))
        )
    with pytest.raises(training.TrainingError, match="a mapping"):
        broken_list = tmp_path / "not_a_dict.pt"
        torch.save([1, 2, 3], broken_list)
        training.load_checkpoint(broken_list)
    with pytest.raises(training.TrainingError, match="shape"):
        training.load_checkpoint(
            _broken(lambda p: p["preprocessing"]["x_stats"].__setitem__("mean", [[0.0]]))
        )
    with pytest.raises(FileNotFoundError):
        training.load_checkpoint(tmp_path / "missing.pt")


def test_run_rejects_existing_run_dir(
    env: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    data_root, _ = env
    _write_prepared_samples(data_root, "ds5")
    config_path = _write_config(data_root, _config_text("ds5", "r1"))
    script = _load_script()
    assert script.run(config_path).is_dir()
    assert script.main(["--config", str(config_path)]) == 2
    assert "already exists" in capsys.readouterr().err


def test_earlystop_reference_independent_of_best(env: tuple[Path, Path]) -> None:
    data_root, _ = env
    _write_prepared_samples(data_root, "ds6")
    # min_delta extremely large: the reference never improves significantly, so with
    # patience=1 training must stop after the 2nd epoch
    config_path = _write_config(
        data_root, _config_text("ds6", "r1", max_epochs=10, patience=1, min_delta=1.0e6)
    )
    run_dir = _load_script().run(config_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["history"]) == 2  # early stopping triggered, max_epochs not reached
    assert metrics["stop_reason"] == "early_stopping"
    assert metrics["stop_epoch"] == 2
    # the absolute best is separate from the earlystop reference: best still takes the
    # lowest val loss overall
    assert metrics["best_val_loss"] == min(e["val_loss"] for e in metrics["history"])
    final = training.load_checkpoint(run_dir / "final.pt")
    assert final.best_val_loss is None


def test_seed_determinism_across_runs(env: tuple[Path, Path]) -> None:
    data_root, _ = env
    _write_prepared_samples(data_root, "ds7")
    script = _load_script()
    run_a = script.run(_write_config(data_root, _config_text("ds7", "r_a"), name="a.yaml"))
    run_b = script.run(_write_config(data_root, _config_text("ds7", "r_b"), name="b.yaml"))
    metrics_a = json.loads((run_a / "metrics.json").read_text(encoding="utf-8"))
    metrics_b = json.loads((run_b / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_a["history"] == metrics_b["history"]  # same seed and data → bitwise identical
    best_a = training.load_checkpoint(run_a / "best.pt")
    best_b = training.load_checkpoint(run_b / "best.pt")
    for name, tensor in best_a.model_state_dict.items():
        assert torch.equal(tensor, best_b.model_state_dict[name]), name


def test_contract_mismatch_across_members_rejected(env: tuple[Path, Path]) -> None:
    data_root, out_root = env
    samples_dir = _write_prepared_samples(data_root, "ds8")
    meta = training_data.load_dataset_meta(samples_dir)
    split = training_data.load_split(samples_dir, meta)
    victim = samples_dir / f"{split.val[0]}.npz"
    with np.load(victim, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["x"] = arrays["x"][:, :-1, :]  # one fewer T step: cross-member contract mismatch
    arrays["t_s"] = arrays["t_s"][:-1]
    np.savez_compressed(victim, **arrays)

    config_path = _write_config(data_root, _config_text("ds8", "r1"))
    with pytest.raises(training_data.DataError, match="contract"):
        _load_script().run(config_path)
    assert not (out_root / "training" / "mlp" / "ds8" / "r1").exists()  # rejected before training


class _TwoSampleSet(torch.utils.data.Dataset):
    """Minimal dataset with two samples and y always 0 (with a 1e19-bias model to
    trigger float32 sum overflow)."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return torch.zeros(1, 1, 3), torch.zeros(2), f"ps{index:04d}"


def test_run_validation_float64_aggregate_survives_float32_sum_overflow() -> None:
    """se elements are finite in float32 (1e38) but the float32 sum overflows →
    aggregation must promote to float64."""
    from torch.utils.data import DataLoader

    model = training.build_model(
        training_data.InputContract(
            pulse_order=("p0",), n_time_steps=1, t_s=np.zeros(1, dtype=np.float64)
        ),
        (4,),
    )
    first_linear = model.network[1]
    last_linear = model.network[3]
    assert isinstance(first_linear, nn.Linear) and isinstance(last_linear, nn.Linear)
    with (
        torch.no_grad()
    ):  # constant output 1e19: zero the hidden layers, leaving only the final bias
        first_linear.weight.zero_()
        first_linear.bias.zero_()
        last_linear.weight.zero_()
        last_linear.bias.fill_(1.0e19)
    state = preprocessing.PreprocessingState(
        x_stats=preprocessing.InputStats(
            mean=np.zeros((1, 1, 3)),
            std=np.ones((1, 1, 3)),
            eps=1.0e-8,
            zero_variance_positions=(),
        ),
        y_stats=preprocessing.LabelState(
            transform="identity",
            y_mean=np.zeros(2),
            y_std=np.ones(2),
            eps=1.0e-8,
            zero_variance_outputs=(),
        ),
    )
    loader: DataLoader[Any] = DataLoader(_TwoSampleSet(), batch_size=2)
    val_loss = training.run_validation(model, loader, state, "cpu")
    # 2 samples × 2 columns × (1e19)^2 = 4e38 > float32 max: the old implementation gave inf here
    assert math.isfinite(val_loss)
    assert val_loss == pytest.approx(1.0e38, rel=1.0e-3)


def test_aggregate_nonfinite_stops_and_preserves_previous_epoch(
    env: tuple[Path, Path], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject a non-finite 2nd-epoch val aggregate: stop, write no history, keep the
    1st complete-epoch weights."""
    data_root, _ = env
    _write_prepared_samples(data_root, "ds_guard")
    config_path = _write_config(data_root, _config_text("ds_guard", "r1", max_epochs=3))
    calls = {"n": 0}

    def fake_validation(model: Any, loader: Any, state: Any, device: str) -> float:
        calls["n"] += 1
        return 1.0 if calls["n"] == 1 else math.inf

    monkeypatch.setattr(training, "run_validation", fake_validation)
    script = _load_script()
    with pytest.raises(training.TrainingError) as exc_info:
        script.run(config_path)
    assert exc_info.value.epoch == 2  # the epoch (1-based) that triggered the numerical failure
    assert exc_info.value.detail is not None and "non-finite" in exc_info.value.detail

    run_dir = _run_dir(env, "ds_guard", "r1")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stop_reason"] == "numerical_failure"
    assert metrics["stop_epoch"] == 2
    assert metrics["detail"] is not None and "non-finite" in metrics["detail"]
    assert len(metrics["history"]) == 1  # 2nd-epoch aggregate non-finite: no history written
    assert metrics["best_val_loss"] == 1.0
    best = training.load_checkpoint(run_dir / "best.pt")
    final = training.load_checkpoint(run_dir / "final.pt")
    assert best.best_val_loss == 1.0 and final.best_val_loss is None
    for name, tensor in best.model_state_dict.items():
        assert torch.equal(
            tensor, final.model_state_dict[name]
        )  # both are the 1st complete-epoch weights
    # both ckpts can be safely serialized and read back
    torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu")
    torch.load(run_dir / "final.pt", weights_only=True, map_location="cpu")
    # the CLI fails non-zero and prints no training-complete message (rerun the same
    # failure with a separate run_name)
    calls["n"] = 0  # reset the injection counter: r2 reproduces the same [1.0, inf] sequence
    config_rerun = _write_config(data_root, _config_text("ds_guard", "r2", max_epochs=3))
    assert script.main(["--config", str(config_rerun)]) == 2
    captured = capsys.readouterr()
    assert "training complete" not in captured.out
    assert "numerical failure" in captured.err
    rerun_metrics = json.loads(
        (_run_dir(env, "ds_guard", "r2") / "metrics.json").read_text(encoding="utf-8")
    )
    assert rerun_metrics["stop_reason"] == "numerical_failure"
    assert rerun_metrics["stop_epoch"] == 2


def _run_dir(env: tuple[Path, Path], dataset: str, run_name: str) -> Path:
    """Small in-test helper: default output-directory layout."""
    _, out_root = env
    return out_root / "training" / "mlp" / dataset / run_name


def test_first_epoch_numerical_failure_writes_failure_metrics_no_checkpoints(
    env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """First-epoch numerical failure: TrainingError carries epoch/detail; failure metrics,
    no ckpt."""
    data_root, _ = env
    _write_prepared_samples(data_root, "ds_first_fail")
    config_path = _write_config(data_root, _config_text("ds_first_fail", "r1", max_epochs=3))

    def failing_validation(model: Any, loader: Any, state: Any, device: str) -> float:
        raise training.TrainingError("validation prediction contains non-finite values (NaN/Inf)")

    monkeypatch.setattr(training, "run_validation", failing_validation)
    script = _load_script()
    with pytest.raises(training.TrainingError) as exc_info:
        script.run(config_path)
    assert exc_info.value.epoch == 1
    assert exc_info.value.detail is not None and "non-finite" in exc_info.value.detail

    run_dir = _run_dir(env, "ds_first_fail", "r1")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stop_reason"] == "numerical_failure"
    assert metrics["stop_epoch"] == 1
    assert metrics["history"] == []
    assert metrics["best_val_loss"] is None
    assert not (run_dir / "best.pt").exists()  # no spurious checkpoint
    assert not (run_dir / "final.pt").exists()
    # CLI fails non-zero (rerun the same failure with a separate run_name)
    config_rerun = _write_config(
        data_root, _config_text("ds_first_fail", "r2", max_epochs=3), name="cfg_r2.yaml"
    )
    assert script.main(["--config", str(config_rerun)]) == 2
    rerun_metrics = json.loads(
        (_run_dir(env, "ds_first_fail", "r2") / "metrics.json").read_text(encoding="utf-8")
    )
    assert rerun_metrics["stop_reason"] == "numerical_failure"
    assert rerun_metrics["stop_epoch"] == 1


def test_best_and_final_diverge_when_val_worsens(
    env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """val worsens: best stays at the better epoch, final is the last epoch (they diverge)."""
    data_root, _ = env
    _write_prepared_samples(data_root, "ds_diverge")
    config_path = _write_config(data_root, _config_text("ds_diverge", "r1", max_epochs=2))
    vals = iter([1.0, 5.0])
    monkeypatch.setattr(training, "run_validation", lambda model, loader, state, device: next(vals))
    run_dir = _load_script().run(config_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stop_reason"] == "max_epochs"
    assert metrics["stop_epoch"] == 2
    assert metrics["detail"] is None
    best = training.load_checkpoint(run_dir / "best.pt")
    final = training.load_checkpoint(run_dir / "final.pt")
    assert best.best_val_loss == 1.0 and final.best_val_loss is None
    assert any(
        not torch.equal(b, f)
        for b, f in zip(
            best.model_state_dict.values(), final.model_state_dict.values(), strict=True
        )
    )


def test_earlystop_reference_accumulates_small_improvements(
    env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accumulated tiny improvements do not reset the reference: min_delta=0.5, patience=2
    → stop at epoch 3."""
    data_root, _ = env
    _write_prepared_samples(data_root, "ds_accumulate")
    config_path = _write_config(
        data_root,
        _config_text("ds_accumulate", "r1", max_epochs=10, patience=2, min_delta=0.5),
    )
    vals = iter([1.0, 0.8, 0.7])
    monkeypatch.setattr(training, "run_validation", lambda model, loader, state, device: next(vals))
    run_dir = _load_script().run(config_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert (
        len(metrics["history"]) == 3
    )  # 0.2/0.1 are not significant improvements → the counter accumulates
    assert metrics["stop_reason"] == "early_stopping"
    assert metrics["stop_epoch"] == 3
    assert metrics["best_val_loss"] == 0.7  # the absolute best is separate from the reference


def test_weights_finite_update_from_same_seed_init(env: tuple[Path, Path]) -> None:
    """Same-seed init vs trained weights: finite and genuinely updated (not unchanged in place)."""
    data_root, _ = env
    _write_prepared_samples(data_root, "ds_update")
    config_path = _write_config(data_root, _config_text("ds_update", "r1", max_epochs=2))
    run_dir = _load_script().run(config_path)
    ckpt = training.load_checkpoint(run_dir / "best.pt")
    training.set_seed(11)  # consistent with the config training.seed
    fresh = training.build_model(ckpt.contract, ckpt.hidden_dims)
    trained = dict(ckpt.model_state_dict)
    assert all(torch.isfinite(t).all() for t in trained.values())
    assert any(not torch.equal(trained[name], fresh.state_dict()[name]) for name in trained)


# --- P1: CNN config early kind rejection at the entry / top of train_model ---
# Hyperparameters are **explicitly unit-test-only values** and represent no research config.


def _cnn_config_text(
    dataset: str,
    run_name: str,
    *,
    channels: tuple[int, ...] = (4, 8),
    kernel_sizes: tuple[int, ...] = (3, 5),
    pool_bins: int = 4,
    head_hidden_dims: tuple[int, ...] = (8,),
    max_epochs: int = 2,
    output_dir: str | None = None,
) -> str:
    """Legal CNN config text (structure/stopping/output-dir configurable; unit-test-only
    hyperparameters)."""
    output = "null" if output_dir is None else output_dir
    return (
        f"dataset_name: {dataset}\n"
        f"run_name: {run_name}\n"
        f"model: {{kind: cnn1d, channels: {list(channels)}, "
        f"kernel_sizes: {list(kernel_sizes)}, pool_bins: {pool_bins}, "
        f"head_hidden_dims: {list(head_hidden_dims)}}}\n"
        "training: {\n"
        f"  seed: 11, device: cpu, batch_size: 2, max_epochs: {max_epochs},\n"
        "  learning_rate: 0.01, weight_decay: 0.0,\n"
        "  early_stopping: {patience: 50, min_delta: 0.0},\n"
        "}\n"
        "split: {seed: 5, ratios: {train: 0.5, val: 0.25, test: 0.25},"
        " min_per_split: {train: 1, val: 1, test: 1}}\n"
        f"output_dir: {output}\n"
    )


def _run_cnn(
    data_root: Path,
    dataset: str,
    run_name: str,
    *,
    max_epochs: int = 2,
    output_dir: str | None = None,
    channels: tuple[int, ...] = (4, 8),
    kernel_sizes: tuple[int, ...] = (3, 5),
    pool_bins: int = 4,
    head_hidden_dims: tuple[int, ...] = (8,),
) -> Path:
    """Write a CNN config + synthetic samples, train through the train_cnn1d thin entry,
    return the run directory."""
    config_path = _write_config(
        data_root,
        _cnn_config_text(
            dataset,
            run_name,
            max_epochs=max_epochs,
            output_dir=output_dir,
            channels=channels,
            kernel_sizes=kernel_sizes,
            pool_bins=pool_bins,
            head_hidden_dims=head_hidden_dims,
        ),
        name=f"{dataset}_{run_name}.yaml",
    )
    return _load_cnn_script().run(config_path)


def _cnn_run_dir(env: tuple[Path, Path], dataset: str, run_name: str) -> Path:
    """Small in-test helper: CNN default output-directory layout."""
    _, out_root = env
    return out_root / "training" / "cnn1d" / dataset / run_name


@pytest.mark.parametrize(
    ("entry", "config_text", "match"),
    [
        ("mlp", _cnn_config_text("ds_guard_cnn", "r1"), "cnn1d"),
        ("cnn1d", _config_text("ds_guard_mlp", "r1"), "mlp"),
    ],
    ids=["mlp_entry_rejects_cnn", "cnn_entry_rejects_mlp"],
)
def test_entry_kind_mismatch_early_reject_before_data_root(
    entry: str,
    config_text: str,
    match: str,
    env: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry mismatch: after a single load_config, early-reject by kind (ConfigError)
    before reading data_root / creating directories.

    The kind guard now lives in the shared ``training.run`` entry (the scripts are
    thin wrappers), so the monkeypatch target is ``training.paths``; the CLI also
    exits 2 with no output directory.
    """
    _, out_root = env
    config_path = _write_config(tmp_path, config_text, name=f"guard_{entry}.yaml")
    calls = {"data_root": 0}

    def _forbidden_data_root() -> Path:
        calls["data_root"] += 1
        raise AssertionError("data_root() must not be called before the early kind rejection")

    monkeypatch.setattr(training.paths, "data_root", _forbidden_data_root)
    script = _load_script() if entry == "mlp" else _load_cnn_script()
    with pytest.raises(ConfigError, match=match):
        script.run(config_path)
    assert calls["data_root"] == 0
    assert not (out_root / "training").exists()

    # The CLI fails equally gracefully: exit 2, the error names the expected/actual kind,
    # no output dir
    assert script.main(["--config", str(config_path)]) == 2
    assert match in capsys.readouterr().err
    assert calls["data_root"] == 0
    assert not (out_root / "training").exists()


def test_run_invalid_expected_kind_rejected_before_data_root(
    env: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An illegal expected_kind (not mlp/cnn1d) is rejected before data_root/reading
    data/creating dirs."""
    _, out_root = env
    config_path = _write_config(tmp_path, _config_text("ds_bad_kind", "r1"), name="bad.yaml")
    calls = {"data_root": 0}

    def _forbidden_data_root() -> Path:
        calls["data_root"] += 1
        raise AssertionError("data_root() must not be called before the expected_kind validation")

    monkeypatch.setattr(training.paths, "data_root", _forbidden_data_root)
    with pytest.raises((ConfigError, ValueError)):
        training.run(config_path, expected_kind=cast(Any, "transformer"))
    assert calls["data_root"] == 0
    assert not (out_root / "training").exists()


def test_train_model_supports_cnn_config(env: tuple[Path, Path]) -> None:
    """P4: train_model genuinely supports CNN (no longer rejects by kind at the top);
    returns CNNCheckpoint."""
    data_root, _ = env
    samples_dir, config = _cnn_samples_config(data_root, "ds_train_cnn")
    contract, state = _cnn_contract_state(samples_dir, config)
    meta = training_data.load_dataset_meta(samples_dir)
    split = training_data.load_split(samples_dir, meta)
    train_set = training_data.TrajectoryDataset(samples_dir, meta, split.train, contract=contract)
    val_set = training_data.TrajectoryDataset(samples_dir, meta, split.val, contract=contract)
    assert isinstance(
        training.build_cnn_model(
            contract, channels=(4, 8), kernel_sizes=(3, 5), pool_bins=4, head_hidden_dims=(8,)
        ),
        nn.Module,
    )

    result = training.train_model(
        config,
        train_set,
        val_set,
        state,
        contract,
        training_data.sha256_file(samples_dir / "split.yaml"),
    )
    assert result.history  # at least 1 complete epoch
    assert isinstance(result.best_checkpoint, training.CNNCheckpoint)
    assert isinstance(result.final_checkpoint, training.CNNCheckpoint)
    assert result.best_checkpoint.best_val_loss is not None
    assert result.final_checkpoint.best_val_loss is None


# --- P3: independent CNN checkpoint / model factory / load_any routing ---
# Data is synthesized by this file's existing helpers; preprocessing statistics are
# fit only on split.train and no MLP checkpoint/statistics are read or converted.
# Hyperparameters are unit-test-only values, not a research config.


def _cnn_samples_config(
    data_root: Path,
    dataset: str,
    *,
    channels: tuple[int, ...] = (4, 8),
    kernel_sizes: tuple[int, ...] = (3, 5),
    pool_bins: int = 4,
    head_hidden_dims: tuple[int, ...] = (8,),
) -> tuple[Path, ExperimentConfig]:
    """Synthesize npz/meta/split + a CNN config loadable by load_config."""
    samples_dir = _write_prepared_samples(data_root, dataset)
    config_path = _write_config(
        data_root,
        _cnn_config_text(
            dataset,
            "r1",
            channels=channels,
            kernel_sizes=kernel_sizes,
            pool_bins=pool_bins,
            head_hidden_dims=head_hidden_dims,
        ),
        name=f"{dataset}.yaml",
    )
    return samples_dir, load_config(config_path)


def _cnn_contract_state(
    samples_dir: Path, config: ExperimentConfig
) -> tuple[training_data.InputContract, preprocessing.PreprocessingState]:
    """train-only: freeze the contract from the real split.train and fit preprocessing
    (touching no MLP artifacts)."""
    meta = training_data.load_dataset_meta(samples_dir)
    split = training_data.load_split(samples_dir, meta)
    train_set = training_data.TrajectoryDataset(samples_dir, meta, split.train)
    first = train_set.samples[0]
    contract = training_data.InputContract(
        pulse_order=first.pulse_ids,
        n_time_steps=int(first.x.shape[1]),
        t_s=first.t_s,
    )
    train_set.validate_contract(contract)
    state = preprocessing.fit(train_set, config.preprocessing, config.label)
    return contract, state


def _build_cnn_checkpoint(
    data_root: Path,
    dataset: str,
    *,
    channels: tuple[int, ...] = (4, 8),
    kernel_sizes: tuple[int, ...] = (3, 5),
    pool_bins: int = 4,
    head_hidden_dims: tuple[int, ...] = (8,),
    config_model: CNN1DModelConfig | None = None,
) -> tuple[Path, training.CNNCheckpoint]:
    """Synthetic-data train-only fit + factory-built network → self-built CNN checkpoint
    (not an MLP conversion)."""
    samples_dir, config = _cnn_samples_config(
        data_root,
        dataset,
        channels=channels,
        kernel_sizes=kernel_sizes,
        pool_bins=pool_bins,
        head_hidden_dims=head_hidden_dims,
    )
    if config_model is not None:
        config = replace(config, model=config_model)
    contract, state = _cnn_contract_state(samples_dir, config)
    model = training.build_cnn_model(
        contract,
        channels=channels,
        kernel_sizes=kernel_sizes,
        pool_bins=pool_bins,
        head_hidden_dims=head_hidden_dims,
    )
    ckpt = training.CNNCheckpoint(
        ckpt_format_version=training.CNN_CKPT_FORMAT_VERSION,
        model_state_dict={
            name: value.detach().to("cpu").clone() for name, value in model.state_dict().items()
        },
        channels=tuple(channels),
        kernel_sizes=tuple(kernel_sizes),
        pool_bins=int(pool_bins),
        head_hidden_dims=tuple(head_hidden_dims),
        activation="relu",
        contract=contract,
        preprocessing=state,
        seed=11,
        config=config,
        dataset_meta_relpath="dataset_meta.yaml",
        dataset_meta_sha256=training_data.sha256_file(samples_dir / "dataset_meta.yaml"),
        split_sha256=training_data.sha256_file(samples_dir / "split.yaml"),
        best_val_loss=0.5,
    )
    path = data_root / f"{dataset}_cnn.pt"
    training.save_cnn_checkpoint(path, ckpt)
    return path, ckpt


def _saved_damage(
    tmp_path: Path,
    payload: Any,
    mutate: Callable[[Any], None],
    *,
    name: str = "damaged.pt",
) -> Path:
    """Deep-copy the payload, apply the mutation, then save (without mutating the shared
    object in place)."""
    damaged = copy.deepcopy(payload)
    mutate(damaged)
    path = tmp_path / name
    torch.save(damaged, path)
    return path


def _cnn_predict(ckpt: training.CNNCheckpoint) -> torch.Tensor:
    """Rebuild the CNN from the explicit structure fields and run inference on a fixed
    input (CPU)."""
    model = training.build_cnn_model(
        ckpt.contract,
        channels=ckpt.channels,
        kernel_sizes=ckpt.kernel_sizes,
        pool_bins=ckpt.pool_bins,
        head_hidden_dims=ckpt.head_hidden_dims,
    )
    model.load_state_dict(dict(ckpt.model_state_dict))
    model.eval()
    probe = torch.zeros(3, *ckpt.contract.input_shape)
    with torch.no_grad():
        return model(probe)


def test_cnn_checkpoint_roundtrip_parity(env: tuple[Path, Path]) -> None:
    """save→load: structure/weights/contract/preprocessing/metadata round-trip,
    bitwise-identical CPU inference."""
    data_root, _ = env
    path, _ = _build_cnn_checkpoint(data_root, "ds_cnn_rt")
    ckpt = training.load_cnn_checkpoint(path)

    assert isinstance(ckpt, training.CNNCheckpoint)
    assert ckpt.ckpt_format_version == training.CNN_CKPT_FORMAT_VERSION == 1
    assert (ckpt.channels, ckpt.kernel_sizes, ckpt.pool_bins, ckpt.head_hidden_dims) == (
        (4, 8),
        (3, 5),
        4,
        (8,),
    )
    assert ckpt.activation == "relu"
    assert ckpt.contract.input_shape == (2, 8, 3)
    assert ckpt.contract.pulse_order == ("p0", "p1")
    assert ckpt.contract.component_order == ("mx", "my", "mz")
    assert ckpt.preprocessing.x_stats.mean.shape == (2, 1, 3)
    assert ckpt.preprocessing.y_stats.y_mean.shape == (2,)
    assert ckpt.seed == 11
    assert ckpt.best_val_loss == pytest.approx(0.5)
    assert ckpt.dataset_meta_relpath == "dataset_meta.yaml"
    assert ckpt.split_sha256
    assert all(t.device.type == "cpu" for t in ckpt.model_state_dict.values())

    # weights_only safe read; the CNN payload has model_kind and no hidden_dims
    payload = torch.load(path, weights_only=True, map_location="cpu")
    assert payload["model_kind"] == "cnn1d"
    assert "hidden_dims" not in payload
    assert all(isinstance(t, torch.Tensor) for t in payload["model_state_dict"].values())

    again = training.load_cnn_checkpoint(path)
    assert torch.equal(_cnn_predict(ckpt), _cnn_predict(again))


def test_cnn_checkpoint_empty_head_hidden_dims_legal(env: tuple[Path, Path]) -> None:
    """head_hidden_dims may be empty (= direct linear output); round-trip and inference
    both work."""
    data_root, _ = env
    path, _ = _build_cnn_checkpoint(data_root, "ds_cnn_empty", head_hidden_dims=())
    ckpt = training.load_cnn_checkpoint(path)
    assert ckpt.head_hidden_dims == ()
    assert cast(CNN1DModelConfig, ckpt.config.model).head_hidden_dims == ()
    model = training.build_cnn_model(
        ckpt.contract,
        channels=ckpt.channels,
        kernel_sizes=ckpt.kernel_sizes,
        pool_bins=ckpt.pool_bins,
        head_hidden_dims=(),
    )
    model.load_state_dict(dict(ckpt.model_state_dict))
    with torch.no_grad():
        assert model(torch.zeros(1, 2, 8, 3)).shape == (1, 2)


def test_build_cnn_model_structure_and_contract_rejection(env: tuple[Path, Path]) -> None:
    """Factory: correct structure/shape; duplicate pulses, component order, t_s, pool>T
    are all rejected."""
    data_root, _ = env
    samples_dir, config = _cnn_samples_config(data_root, "ds_cnn_factory")
    contract, _ = _cnn_contract_state(samples_dir, config)
    model = training.build_cnn_model(
        contract,
        channels=(4, 8),
        kernel_sizes=(3, 5),
        pool_bins=4,
        head_hidden_dims=(8,),
    )
    assert model.input_shape == (2, 8, 3)
    assert (model.channels, model.kernel_sizes, model.pool_bins, model.head_hidden_dims) == (
        (4, 8),
        (3, 5),
        4,
        (8,),
    )
    with torch.no_grad():
        assert model(torch.zeros(2, 2, 8, 3)).shape == (2, 2)

    bad_contracts = {
        "duplicate_pulse": training_data.InputContract(
            pulse_order=("p0", "p0"), n_time_steps=8, t_s=np.zeros(8)
        ),
        "component_order": training_data.InputContract(
            pulse_order=("p0", "p1"),
            n_time_steps=8,
            t_s=np.zeros(8),
            component_order=("mx", "mx", "mz"),
        ),
        "non_increasing_t_s": training_data.InputContract(
            pulse_order=("p0", "p1"),
            n_time_steps=8,
            t_s=np.array([0.0, 2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        ),
        "non_finite_t_s": training_data.InputContract(
            pulse_order=("p0", "p1"),
            n_time_steps=8,
            t_s=np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0]),
        ),
        "timegrid_shape": training_data.InputContract(
            pulse_order=("p0", "p1"), n_time_steps=8, t_s=np.zeros(7)
        ),
        "channel_shape": training_data.InputContract(
            pulse_order=("p0", "p1"),
            n_time_steps=8,
            t_s=np.zeros(8),
            n_channels=2,
            component_order=("mx", "my"),
        ),
    }
    for bad in bad_contracts.values():
        with pytest.raises((training.TrainingError, ValueError)):
            training.build_cnn_model(
                bad,
                channels=(4, 8),
                kernel_sizes=(3, 5),
                pool_bins=4,
                head_hidden_dims=(8,),
            )
    with pytest.raises((training.TrainingError, ValueError)):
        training.build_cnn_model(  # pool_bins > T=8
            contract,
            channels=(4, 8),
            kernel_sizes=(3, 5),
            pool_bins=9,
            head_hidden_dims=(8,),
        )


def test_mlp_and_cnn_checkpoints_mutually_rejected(env: tuple[Path, Path]) -> None:
    """The old MLP format has no kind and rejects CNN files; the CNN loader rejects MLP files."""
    data_root, _ = env
    _write_prepared_samples(data_root, "ds_cross_mlp")
    run_dir = _load_script().run(_write_config(data_root, _config_text("ds_cross_mlp", "r1")))
    cnn_path, _ = _build_cnn_checkpoint(data_root, "ds_cross_cnn")

    with pytest.raises(training.TrainingError):
        training.load_checkpoint(cnn_path)
    with pytest.raises(training.TrainingError):
        training.load_cnn_checkpoint(run_dir / "best.pt")

    mlp_payload = torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu")
    assert "model_kind" not in mlp_payload
    assert "channels" not in mlp_payload
    assert "kind" not in mlp_payload["config"]["model"]
    cnn_payload = torch.load(cnn_path, weights_only=True, map_location="cpu")
    assert cnn_payload["model_kind"] == "cnn1d"
    assert "hidden_dims" not in cnn_payload


def test_load_any_checkpoint_routes_both_formats(env: tuple[Path, Path]) -> None:
    """load_any: old MLP → Checkpoint; CNN → CNNCheckpoint (structure from the explicit fields)."""
    data_root, _ = env
    assert training.ModelCheckpoint is not None
    _write_prepared_samples(data_root, "ds_any_mlp")
    run_dir = _load_script().run(_write_config(data_root, _config_text("ds_any_mlp", "r1")))
    cnn_path, cnn_ckpt = _build_cnn_checkpoint(data_root, "ds_any_cnn")

    mlp = training.load_any_checkpoint(run_dir / "best.pt")
    assert isinstance(mlp, training.Checkpoint)
    cnn = training.load_any_checkpoint(cnn_path)
    assert isinstance(cnn, training.CNNCheckpoint)
    assert cnn.channels == cnn_ckpt.channels
    assert torch.equal(_cnn_predict(cnn), _cnn_predict(cnn_ckpt))


def test_load_any_checkpoint_loads_each_file_once(
    env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_any calls torch.load exactly once per file, with weights_only / CPU mapping."""
    data_root, _ = env
    _write_prepared_samples(data_root, "ds_once_mlp")
    run_dir = _load_script().run(_write_config(data_root, _config_text("ds_once_mlp", "r1")))
    cnn_path, _ = _build_cnn_checkpoint(data_root, "ds_once_cnn")

    real_load = torch.load
    calls: list[dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy)
    training.load_any_checkpoint(run_dir / "best.pt")
    assert len(calls) == 1
    training.load_any_checkpoint(cnn_path)
    assert len(calls) == 2
    for kwargs in calls:
        assert kwargs.get("weights_only") is True
        assert kwargs.get("map_location") == "cpu"


@pytest.mark.parametrize("kind_value", ["mlp", None, "transformer"])
def test_load_any_rejects_non_cnn_kind_on_mlp_payload(
    env: tuple[Path, Path], tmp_path: Path, kind_value: str | None
) -> None:
    """Any explicit kind on an MLP payload (including "mlp") → rejected, no fallback to
    the old loader."""
    data_root, _ = env
    _write_prepared_samples(data_root, "ds_kind_mlp")
    run_dir = _load_script().run(_write_config(data_root, _config_text("ds_kind_mlp", "r1")))
    payload = torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu")
    bad = _saved_damage(tmp_path, payload, lambda p: p.__setitem__("model_kind", kind_value))
    with pytest.raises(training.TrainingError):
        training.load_any_checkpoint(bad)


def test_load_any_rejects_layout_and_kind_mismatch(env: tuple[Path, Path], tmp_path: Path) -> None:
    """Unknown/absent kind and layout mismatch always error: never fall back to MLP."""
    data_root, _ = env
    _write_prepared_samples(data_root, "ds_mismatch_mlp")
    run_dir = _load_script().run(_write_config(data_root, _config_text("ds_mismatch_mlp", "r1")))
    mlp_payload = torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu")
    mlp_as_cnn = _saved_damage(
        tmp_path,
        mlp_payload,
        lambda p: p.__setitem__("model_kind", "cnn1d"),
        name="mlp_as_cnn.pt",
    )
    with pytest.raises(training.TrainingError):
        training.load_any_checkpoint(mlp_as_cnn)

    cnn_path, _ = _build_cnn_checkpoint(data_root, "ds_mismatch_cnn")
    cnn_payload = torch.load(cnn_path, weights_only=True, map_location="cpu")
    cnn_no_kind = _saved_damage(
        tmp_path, cnn_payload, lambda p: p.pop("model_kind"), name="cnn_no_kind.pt"
    )
    with pytest.raises(training.TrainingError):
        training.load_any_checkpoint(cnn_no_kind)
    for kind_value in ("mlp", None, "transformer"):
        bad = _saved_damage(
            tmp_path,
            cnn_payload,
            lambda p, v=kind_value: p.__setitem__("model_kind", v),
            name=f"cnn_kind_{kind_value}.pt",
        )
        with pytest.raises(training.TrainingError):
            training.load_any_checkpoint(bad)


def test_cnn_explicit_structure_survives_nested_config(env: tuple[Path, Path]) -> None:
    """The CNN explicit structure fields are authoritative: a differing nested config.model
    structure does not override them."""
    data_root, _ = env
    nested = CNN1DModelConfig(
        channels=(16, 16), kernel_sizes=(3, 3), pool_bins=2, head_hidden_dims=()
    )
    path, _ = _build_cnn_checkpoint(data_root, "ds_cnn_authoritative", config_model=nested)
    ckpt = training.load_cnn_checkpoint(path)
    assert (ckpt.channels, ckpt.kernel_sizes, ckpt.pool_bins, ckpt.head_hidden_dims) == (
        (4, 8),
        (3, 5),
        4,
        (8,),
    )
    nested_loaded = cast(CNN1DModelConfig, ckpt.config.model)
    assert isinstance(nested_loaded, CNN1DModelConfig)
    assert nested_loaded.channels == (16, 16)
    model = training.build_cnn_model(
        ckpt.contract,
        channels=ckpt.channels,
        kernel_sizes=ckpt.kernel_sizes,
        pool_bins=ckpt.pool_bins,
        head_hidden_dims=ckpt.head_hidden_dims,
    )
    assert model.channels == (4, 8)
    model.load_state_dict(dict(ckpt.model_state_dict))


def test_save_cnn_checkpoint_refuses_overwrite(env: tuple[Path, Path]) -> None:
    data_root, _ = env
    path, ckpt = _build_cnn_checkpoint(data_root, "ds_cnn_overwrite")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        training.save_cnn_checkpoint(path, ckpt)


def _mutate_state_non_tensor(payload: Any) -> None:
    name = next(iter(payload["model_state_dict"]))
    payload["model_state_dict"][name] = [1.0, 2.0]


def _mutate_state_non_str_key(payload: Any) -> None:
    payload["model_state_dict"][1] = torch.zeros(1)


def _mutate_t_s_nan(payload: Any) -> None:
    payload["contract"]["t_s"][0] = float("nan")


def _mutate_preprocessing_p(payload: Any) -> None:
    x_stats = payload["preprocessing"]["x_stats"]
    x_stats["mean"] = [[[0.0, 0.0, 0.0]]]  # P=1, inconsistent with contract P=2
    x_stats["std"] = [[[1.0, 1.0, 1.0]]]


_CNN_CORRUPTIONS: list[tuple[str, Callable[[Any], None]]] = [
    ("version_bool", lambda p: p.__setitem__("ckpt_format_version", True)),
    ("version_unknown", lambda p: p.__setitem__("ckpt_format_version", 999)),
    ("missing_channels", lambda p: p.pop("channels")),
    ("missing_kernel_sizes", lambda p: p.pop("kernel_sizes")),
    ("missing_pool_bins", lambda p: p.pop("pool_bins")),
    ("missing_head_hidden_dims", lambda p: p.pop("head_hidden_dims")),
    ("kernel_even", lambda p: p.__setitem__("kernel_sizes", [2, 4])),
    ("kernel_layer_mismatch", lambda p: p.__setitem__("kernel_sizes", [3])),
    ("pool_bins_gt_T", lambda p: p.__setitem__("pool_bins", 9)),
    ("unknown_activation", lambda p: p.__setitem__("activation", "gelu")),
    ("state_dict_empty", lambda p: p.__setitem__("model_state_dict", {})),
    ("state_dict_non_tensor", _mutate_state_non_tensor),
    ("state_dict_non_str_key", _mutate_state_non_str_key),
    ("contract_missing_key", lambda p: p["contract"].pop("n_time_steps")),
    ("contract_wrong_T", lambda p: p["contract"].__setitem__("n_time_steps", 7)),
    ("contract_t_s_nan", _mutate_t_s_nan),
    (
        "contract_t_s_not_increasing",
        lambda p: p["contract"].__setitem__("t_s", list(reversed(p["contract"]["t_s"]))),
    ),
    (
        "contract_component_order",
        lambda p: p["contract"].__setitem__("component_order", ["mx", "mx", "mz"]),
    ),
    ("preprocessing_P_mismatch", _mutate_preprocessing_p),
    ("config_model_wrong_kind", lambda p: p["config"].__setitem__("model", {"hidden_dims": [8]})),
    ("seed_bool", lambda p: p.__setitem__("seed", True)),
    ("seed_negative", lambda p: p.__setitem__("seed", -1)),
    ("best_loss_bool", lambda p: p.__setitem__("best_val_loss", True)),
    ("best_loss_nonfinite", lambda p: p.__setitem__("best_val_loss", float("nan"))),
]


@pytest.mark.parametrize("mutate", [pytest.param(fn, id=name) for name, fn in _CNN_CORRUPTIONS])
def test_load_cnn_checkpoint_rejects_corrupt(
    env: tuple[Path, Path], tmp_path: Path, mutate: Callable[[Any], None]
) -> None:
    """A corrupt CNN payload always raises TrainingError (no KeyError/TypeError, no MLP
    fallback)."""
    data_root, _ = env
    path, _ = _build_cnn_checkpoint(data_root, "ds_cnn_corrupt")
    payload = torch.load(path, weights_only=True, map_location="cpu")
    broken = _saved_damage(tmp_path, payload, mutate)
    with pytest.raises(training.TrainingError):
        training.load_cnn_checkpoint(broken)


def test_save_cnn_checkpoint_accepts_any_tensor_and_detaches(
    env: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save does not restrict CPU: accepts requires_grad tensors and detaches before writing;
    load still requires CPU."""
    data_root, _ = env
    _, ckpt = _build_cnn_checkpoint(data_root, "ds_cnn_device")
    probe = torch.zeros(2, requires_grad=True)
    with_probe = replace(ckpt, model_state_dict={**ckpt.model_state_dict, "probe": probe})

    saved = data_root / "device_probe.pt"
    training.save_cnn_checkpoint(saved, with_probe)
    payload = torch.load(saved, weights_only=True, map_location="cpu")
    assert payload["model_state_dict"]["probe"].requires_grad is False
    assert payload["model_state_dict"]["probe"].device.type == "cpu"

    # save uses structure validation, not the load-side CPU-restricted helper: stubbing
    # the latter does not affect saving.
    def _forbidden(raw: Any, path: Path) -> Any:
        raise AssertionError("save_cnn_checkpoint must not call _cnn_state_dict_from_payload")

    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(training, "_cnn_state_dict_from_payload", _forbidden)
        saved2 = data_root / "device_probe2.pt"
        training.save_cnn_checkpoint(saved2, with_probe)
        assert saved2.is_file()

    # load-boundary guard: a genuinely non-CPU tensor (meta, constructible CPU-only;
    # CUDA itself untested).
    bad_payload = torch.load(saved, weights_only=True, map_location="cpu")
    key = next(iter(bad_payload["model_state_dict"]))
    bad_payload["model_state_dict"][key] = torch.empty(2, device="meta")
    bad_path = tmp_path / "meta_state.pt"
    torch.save(bad_payload, bad_path)
    with pytest.raises(training.TrainingError):
        training.load_cnn_checkpoint(bad_path)
    with pytest.raises(training.TrainingError):
        training.load_any_checkpoint(bad_path)


def _set_path(payload: Any, keys: tuple[Any, ...], value: Any) -> None:
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


_CNN_PREP_CORRUPTIONS: list[tuple[str, Callable[[Any], None]]] = [
    (
        "x_mean_nan",
        lambda p: _set_path(p, ("preprocessing", "x_stats", "mean", 0, 0, 0), float("nan")),
    ),
    (
        "x_mean_inf",
        lambda p: _set_path(p, ("preprocessing", "x_stats", "mean", 0, 0, 0), float("inf")),
    ),
    ("x_std_zero", lambda p: _set_path(p, ("preprocessing", "x_stats", "std", 0, 0, 0), 0.0)),
    ("x_std_negative", lambda p: _set_path(p, ("preprocessing", "x_stats", "std", 0, 0, 0), -1.0)),
    (
        "x_std_inf",
        lambda p: _set_path(p, ("preprocessing", "x_stats", "std", 0, 0, 0), float("inf")),
    ),
    (
        "y_mean_nan",
        lambda p: _set_path(p, ("preprocessing", "y_stats", "y_mean", 0), float("nan")),
    ),
    (
        "y_mean_inf",
        lambda p: _set_path(p, ("preprocessing", "y_stats", "y_mean", 0), float("inf")),
    ),
    ("y_std_zero", lambda p: _set_path(p, ("preprocessing", "y_stats", "y_std", 0), 0.0)),
    ("y_std_negative", lambda p: _set_path(p, ("preprocessing", "y_stats", "y_std", 0), -1.0)),
    ("y_std_inf", lambda p: _set_path(p, ("preprocessing", "y_stats", "y_std", 0), float("inf"))),
]


@pytest.mark.parametrize(
    "mutate", [pytest.param(fn, id=name) for name, fn in _CNN_PREP_CORRUPTIONS]
)
def test_load_cnn_rejects_bad_preprocessing_numbers(
    env: tuple[Path, Path], tmp_path: Path, mutate: Callable[[Any], None]
) -> None:
    """The CNN boundary rejects non-finite mean / non-positive or non-finite effective scale."""
    data_root, _ = env
    path, _ = _build_cnn_checkpoint(data_root, "ds_cnn_prep")
    payload = torch.load(path, weights_only=True, map_location="cpu")
    broken = _saved_damage(tmp_path, payload, mutate)
    with pytest.raises(training.TrainingError):
        training.load_cnn_checkpoint(broken)


def test_save_cnn_rejects_bad_preprocessing_before_write(env: tuple[Path, Path]) -> None:
    """The save side likewise validates before mkdir/writing: a bad effective scale is
    not written."""
    data_root, _ = env
    _, ckpt = _build_cnn_checkpoint(data_root, "ds_cnn_prep_save")
    bad_state = replace(
        ckpt.preprocessing,
        x_stats=replace(
            ckpt.preprocessing.x_stats, std=np.zeros_like(ckpt.preprocessing.x_stats.std)
        ),
    )
    target = data_root / "bad_prep.pt"
    with pytest.raises(training.TrainingError):
        training.save_cnn_checkpoint(target, replace(ckpt, preprocessing=bad_state))
    assert not target.exists()


def test_cnn_preprocessing_effective_scale_one_accepted(env: tuple[Path, Path]) -> None:
    """An effective scale of 1 at zero-variance positions is legal (save→load round-trip)."""
    data_root, _ = env
    _, ckpt = _build_cnn_checkpoint(data_root, "ds_cnn_prep_one")
    one_state = replace(
        ckpt.preprocessing,
        x_stats=replace(
            ckpt.preprocessing.x_stats, std=np.ones_like(ckpt.preprocessing.x_stats.std)
        ),
        y_stats=replace(
            ckpt.preprocessing.y_stats, y_std=np.ones_like(ckpt.preprocessing.y_stats.y_std)
        ),
    )
    path = data_root / "prep_one.pt"
    training.save_cnn_checkpoint(path, replace(ckpt, preprocessing=one_state))
    loaded = training.load_cnn_checkpoint(path)
    assert np.allclose(loaded.preprocessing.x_stats.std, 1.0)
    assert np.allclose(loaded.preprocessing.y_stats.y_std, 1.0)


def test_config_overflow_becomes_training_error(env: tuple[Path, Path], tmp_path: Path) -> None:
    """batch_size=inf: the load boundary turns OverflowError → TrainingError; the nested save
    round-trip is rejected."""
    data_root, _ = env
    _write_prepared_samples(data_root, "ds_overflow_mlp")
    run_dir = _load_script().run(_write_config(data_root, _config_text("ds_overflow_mlp", "r1")))
    mlp_payload = torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu")
    mlp_bad = _saved_damage(
        tmp_path,
        mlp_payload,
        lambda p: p["config"]["training"].__setitem__("batch_size", float("inf")),
        name="mlp_overflow.pt",
    )
    with pytest.raises(training.TrainingError):
        training.load_checkpoint(mlp_bad)
    with pytest.raises(training.TrainingError):
        training.load_any_checkpoint(mlp_bad)

    cnn_path, cnn_ckpt = _build_cnn_checkpoint(data_root, "ds_overflow_cnn")
    cnn_payload = torch.load(cnn_path, weights_only=True, map_location="cpu")
    cnn_bad = _saved_damage(
        tmp_path,
        cnn_payload,
        lambda p: p["config"]["training"].__setitem__("batch_size", float("inf")),
        name="cnn_overflow.pt",
    )
    with pytest.raises(training.TrainingError):
        training.load_cnn_checkpoint(cnn_bad)
    with pytest.raises(training.TrainingError):
        training.load_any_checkpoint(cnn_bad)

    # Nested save path: the same illegal config is rejected before mkdir/writing.
    bad_config = replace(
        cnn_ckpt.config, training=replace(cnn_ckpt.config.training, batch_size=float("inf"))
    )
    target = data_root / "overflow.pt"
    with pytest.raises(training.TrainingError):
        training.save_cnn_checkpoint(target, replace(cnn_ckpt, config=bad_config))
    assert not target.exists()


def test_cnn_contract_component_order_required_mlp_default_unchanged(
    env: tuple[Path, Path], tmp_path: Path
) -> None:
    """The independent CNN format requires an explicit component_order; the old MLP
    default remains readable."""
    data_root, _ = env
    cnn_path, _ = _build_cnn_checkpoint(data_root, "ds_comp_cnn")
    cnn_payload = torch.load(cnn_path, weights_only=True, map_location="cpu")
    assert cnn_payload["contract"]["component_order"] == ["mx", "my", "mz"]
    cnn_missing = _saved_damage(
        tmp_path,
        cnn_payload,
        lambda p: p["contract"].pop("component_order"),
        name="cnn_no_comp.pt",
    )
    with pytest.raises(training.TrainingError):
        training.load_cnn_checkpoint(cnn_missing)
    with pytest.raises(training.TrainingError):
        training.load_any_checkpoint(cnn_missing)

    _write_prepared_samples(data_root, "ds_comp_mlp")
    run_dir = _load_script().run(_write_config(data_root, _config_text("ds_comp_mlp", "r1")))
    mlp_payload = torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu")
    mlp_missing = _saved_damage(
        tmp_path,
        mlp_payload,
        lambda p: p["contract"].pop("component_order"),
        name="mlp_no_comp.pt",
    )
    mlp_ckpt = training.load_checkpoint(mlp_missing)
    assert mlp_ckpt.contract.component_order == ("mx", "my", "mz")
    assert isinstance(training.load_any_checkpoint(mlp_missing), training.Checkpoint)


# --- P4: shared training.run / CNN training entry / independent artifacts and fitting ---
# All hyperparameters are **explicitly unit-test-only values** (max_epochs 1-2, CPU,
# tiny fixture) and represent no research config; this reuses the file's existing
# synthetic-data helpers and reads no real data/GPU.


def _run_kind(kind: str, data_root: Path, dataset: str, run_name: str) -> Path:
    """Train through each kind's own thin entry (reusing the same synthetic data) and return
    the run directory."""
    if kind == "mlp":
        return _load_script().run(
            _write_config(
                data_root, _config_text(dataset, run_name), name=f"{dataset}_{run_name}.yaml"
            )
        )
    return _run_cnn(data_root, dataset, run_name)


def test_cnn_run_trains_and_writes_artifacts(env: tuple[Path, Path]) -> None:
    """CNN thin-entry run: complete artifacts, correct split raw bytes/config_resolved/
    preprocessing/metrics; load_cnn + restore can infer; the MLP loader rejects CNN;
    evaluate physical metrics are finite and member-aligned."""
    data_root, out_root = env
    dataset = "ds_cnn_e2e"
    samples_dir = _write_prepared_samples(data_root, dataset)
    run_dir = _run_cnn(data_root, dataset, "r1")

    assert run_dir == out_root / "training" / "cnn1d" / dataset / "r1"
    for name in (
        "best.pt",
        "final.pt",
        "split.yaml",
        "config_resolved.yaml",
        "preprocessing.yaml",
        "metrics.json",
    ):
        assert (run_dir / name).is_file(), name

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["history"]) == 2
    assert metrics["stop_reason"] == "max_epochs"
    assert metrics["stop_epoch"] == 2
    assert metrics["detail"] is None
    for entry in metrics["history"]:
        assert math.isfinite(entry["train_loss"]) and math.isfinite(entry["val_loss"])
    assert metrics["best_val_loss"] == min(e["val_loss"] for e in metrics["history"])

    resolved = load_config(run_dir / "config_resolved.yaml")
    assert isinstance(resolved.model, CNN1DModelConfig)
    assert resolved.model.channels == (4, 8)
    assert resolved.model.pool_bins == 4
    assert (run_dir / "split.yaml").read_bytes() == (samples_dir / "split.yaml").read_bytes()
    prep = yaml.safe_load((run_dir / "preprocessing.yaml").read_text(encoding="utf-8"))
    assert len(prep["x_stats"]["mean"]) == 2  # [P=2, 1, 3]
    assert len(prep["x_stats"]["mean"][0][0]) == 3
    assert prep["y_stats"]["transform"] == "identity"

    ckpt = training.load_cnn_checkpoint(run_dir / "best.pt")
    with pytest.raises(training.TrainingError):
        training.load_checkpoint(run_dir / "best.pt")  # the MLP loader rejects CNN
    assert ckpt.contract.input_shape == (2, 8, 3)

    def _predict(checkpoint: training.CNNCheckpoint) -> torch.Tensor:
        model = training.build_cnn_model(
            checkpoint.contract,
            channels=checkpoint.channels,
            kernel_sizes=checkpoint.kernel_sizes,
            pool_bins=checkpoint.pool_bins,
            head_hidden_dims=checkpoint.head_hidden_dims,
        )
        model.load_state_dict(dict(checkpoint.model_state_dict))
        model.eval()
        with torch.no_grad():
            return model(torch.zeros(1, *checkpoint.contract.input_shape))

    again = training.load_cnn_checkpoint(run_dir / "best.pt")
    assert torch.equal(_predict(ckpt), _predict(again))

    report, rows = evaluation.run_evaluation(run_dir)
    test_members = yaml.safe_load((run_dir / "split.yaml").read_text(encoding="utf-8"))["test"]
    assert [row.parameter_set_id for row in rows] == test_members
    assert all(row.split == "test" for row in rows)
    assert report.main.n + report.control.n == len(test_members)
    for subset in (report.main, report.control):
        for key in ("mae_alpha", "rmse_alpha", "mae_ku", "rmse_ku"):
            value = getattr(subset, key)
            assert value is None or math.isfinite(value)
    assert report.provenance.checkpoint_path == str(run_dir / "best.pt")


@pytest.mark.parametrize("kind", ["mlp", "cnn1d"])
def test_explicit_output_dir_override(kind: str, env: tuple[Path, Path]) -> None:
    """An explicit output_dir overrides the default per-kind directory; the default dir
    (training/<kind>/...) is not created."""
    data_root, out_root = env
    dataset = f"ds_out_{kind}"
    _write_prepared_samples(data_root, dataset)
    target = out_root / "custom" / f"{kind}_run"
    if kind == "mlp":
        config_path = _write_config(
            data_root, _config_text(dataset, "r1", output_dir=str(target)), name="out_mlp.yaml"
        )
        run_dir = _load_script().run(config_path)
    else:
        run_dir = _run_cnn(data_root, dataset, "r1", output_dir=str(target))
    assert run_dir == target
    assert (target / "best.pt").is_file()
    assert not (out_root / "training").exists()


def test_shared_run_parses_config_once(
    env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared run parses the config exactly once (the thin entry does not repeat
    load_config)."""
    data_root, _ = env
    dataset = "ds_parse_once"
    _write_prepared_samples(data_root, dataset)
    config_path = _write_config(data_root, _config_text(dataset, "r1"))
    real_load = training_config.load_config
    calls = {"n": 0}

    def spy(path: Path) -> ExperimentConfig:
        calls["n"] += 1
        return real_load(path)

    # Whether training.run calls it module-qualified or uses a pre-imported symbol, it
    # parses exactly once.
    monkeypatch.setattr(training_config, "load_config", spy)
    if hasattr(training, "load_config"):
        monkeypatch.setattr(training, "load_config", spy)
    run_dir = _load_script().run(config_path)
    assert calls["n"] == 1
    assert (run_dir / "best.pt").is_file()


def test_run_rejects_existing_cnn_run_dir(
    env: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    data_root, _ = env
    dataset = "ds_cnn_exists"
    _write_prepared_samples(data_root, dataset)
    config_path = _write_config(data_root, _cnn_config_text(dataset, "r1"), name="cnn_exists.yaml")
    script = _load_cnn_script()
    assert script.run(config_path).is_dir()
    assert script.main(["--config", str(config_path)]) == 2
    assert "already exists" in capsys.readouterr().err


def test_cnn_first_epoch_numerical_failure_no_checkpoints(
    env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """CNN first-epoch numerical failure: TrainingError(epoch=1); failure metrics, no
    spurious ckpt, CLI exit 2."""
    data_root, _ = env
    dataset = "ds_cnn_first_fail"
    _write_prepared_samples(data_root, dataset)
    config_path = _write_config(
        data_root, _cnn_config_text(dataset, "r1", max_epochs=3), name="cnn_ff.yaml"
    )

    def failing_validation(model: Any, loader: Any, state: Any, device: str) -> float:
        raise training.TrainingError("validation prediction contains non-finite values (NaN/Inf)")

    monkeypatch.setattr(training, "run_validation", failing_validation)
    script = _load_cnn_script()
    with pytest.raises(training.TrainingError) as exc_info:
        script.run(config_path)
    assert exc_info.value.epoch == 1
    assert exc_info.value.detail is not None and "non-finite" in exc_info.value.detail

    run_dir = _cnn_run_dir(env, dataset, "r1")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stop_reason"] == "numerical_failure"
    assert metrics["stop_epoch"] == 1
    assert metrics["history"] == []
    assert metrics["best_val_loss"] is None
    assert not (run_dir / "best.pt").exists()
    assert not (run_dir / "final.pt").exists()

    config_rerun = _write_config(
        data_root, _cnn_config_text(dataset, "r2", max_epochs=3), name="cnn_ff2.yaml"
    )
    assert script.main(["--config", str(config_rerun)]) == 2
    rerun = json.loads(
        (_cnn_run_dir(env, dataset, "r2") / "metrics.json").read_text(encoding="utf-8")
    )
    assert rerun["stop_reason"] == "numerical_failure"
    assert rerun["stop_epoch"] == 1


def test_cnn_later_numerical_failure_preserves_previous_epoch(
    env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """CNN later-epoch numerical failure: keep the last complete finite epoch's
    best/final, CLI exit 2."""
    data_root, _ = env
    dataset = "ds_cnn_late_fail"
    _write_prepared_samples(data_root, dataset)
    config_path = _write_config(
        data_root, _cnn_config_text(dataset, "r1", max_epochs=3), name="cnn_lf.yaml"
    )
    calls = {"n": 0}

    def fake_validation(model: Any, loader: Any, state: Any, device: str) -> float:
        calls["n"] += 1
        return 1.0 if calls["n"] == 1 else math.inf

    monkeypatch.setattr(training, "run_validation", fake_validation)
    script = _load_cnn_script()
    with pytest.raises(training.TrainingError) as exc_info:
        script.run(config_path)
    assert exc_info.value.epoch == 2

    run_dir = _cnn_run_dir(env, dataset, "r1")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stop_reason"] == "numerical_failure"
    assert metrics["stop_epoch"] == 2
    assert len(metrics["history"]) == 1
    assert metrics["best_val_loss"] == 1.0
    best = training.load_cnn_checkpoint(run_dir / "best.pt")
    final = training.load_cnn_checkpoint(run_dir / "final.pt")
    assert best.best_val_loss == 1.0 and final.best_val_loss is None
    for name, tensor in best.model_state_dict.items():
        assert torch.equal(
            tensor, final.model_state_dict[name]
        )  # both are the 1st complete-epoch weights

    config_rerun = _write_config(
        data_root, _cnn_config_text(dataset, "r2", max_epochs=3), name="cnn_lf2.yaml"
    )
    calls["n"] = 0
    assert script.main(["--config", str(config_rerun)]) == 2


def test_cnn_seed_determinism_across_runs(env: tuple[Path, Path]) -> None:
    """Same seed and data → CNN history and best weights are bitwise identical (CPU; not
    generalized to GPU)."""
    data_root, _ = env
    dataset = "ds_cnn_seed"
    _write_prepared_samples(data_root, dataset)
    run_a = _run_cnn(data_root, dataset, "r_a")
    run_b = _run_cnn(data_root, dataset, "r_b")
    metrics_a = json.loads((run_a / "metrics.json").read_text(encoding="utf-8"))
    metrics_b = json.loads((run_b / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_a["history"] == metrics_b["history"]
    best_a = training.load_cnn_checkpoint(run_a / "best.pt")
    best_b = training.load_cnn_checkpoint(run_b / "best.pt")
    for name, tensor in best_a.model_state_dict.items():
        assert torch.equal(tensor, best_b.model_state_dict[name]), name


def test_save_model_checkpoint_routes_by_type(env: tuple[Path, Path], tmp_path: Path) -> None:
    """save_model_checkpoint: MLP→old layout (no model_kind), CNN→independent layout;
    refuses overwrite/unknown types."""
    data_root, _ = env
    _write_prepared_samples(data_root, "ds_route_mlp")
    mlp_run = _load_script().run(_write_config(data_root, _config_text("ds_route_mlp", "r1")))
    mlp_ckpt = training.load_checkpoint(mlp_run / "best.pt")
    mlp_target = tmp_path / "routed_mlp.pt"
    training.save_model_checkpoint(mlp_target, mlp_ckpt)
    mlp_payload = torch.load(mlp_target, weights_only=True, map_location="cpu")
    assert "model_kind" not in mlp_payload and "channels" not in mlp_payload
    assert mlp_payload["hidden_dims"] == [8, 4]  # old MLP payload layout unchanged
    assert isinstance(training.load_any_checkpoint(mlp_target), training.Checkpoint)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        training.save_model_checkpoint(mlp_target, mlp_ckpt)

    _write_prepared_samples(data_root, "ds_route_cnn")
    cnn_run = _run_cnn(data_root, "ds_route_cnn", "r1")
    cnn_ckpt = training.load_cnn_checkpoint(cnn_run / "best.pt")
    cnn_target = tmp_path / "routed_cnn.pt"
    training.save_model_checkpoint(cnn_target, cnn_ckpt)
    cnn_payload = torch.load(cnn_target, weights_only=True, map_location="cpu")
    assert cnn_payload["model_kind"] == "cnn1d" and "hidden_dims" not in cnn_payload
    routed_cnn = training.load_any_checkpoint(cnn_target)
    assert isinstance(routed_cnn, training.CNNCheckpoint)
    assert routed_cnn.channels == cnn_ckpt.channels

    unsupported = tmp_path / "routed_bad.pt"
    with pytest.raises((training.TrainingError, TypeError)):
        training.save_model_checkpoint(unsupported, cast(Any, object()))
    assert not unsupported.exists()


@pytest.mark.parametrize("kind", ["mlp", "cnn1d"])
def test_preprocessing_fit_train_only_ignores_val_test(
    kind: str, env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """fit consumes only split.train members; poisoning val/test data does not change this
    run's statistics.

    MLP/CNN statistics are not required to differ (they may coincide on the same data),
    and no other run's statistics are read.
    """
    data_root, _ = env
    dataset = f"ds_fit_{kind}"
    samples_dir = _write_prepared_samples(data_root, dataset)
    meta = training_data.load_dataset_meta(samples_dir)
    split = training_data.load_split(samples_dir, meta)
    seen: list[tuple[str, ...]] = []
    real_fit = preprocessing.fit

    def spy_fit(dataset_obj: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(tuple(sorted(sample.parameter_set_id for sample in dataset_obj.samples)))
        return real_fit(dataset_obj, *args, **kwargs)

    monkeypatch.setattr(preprocessing, "fit", spy_fit)
    run_a = _run_kind(kind, data_root, dataset, "clean")
    assert seen == [tuple(sorted(split.train))]
    stats_a = yaml.safe_load((run_a / "preprocessing.yaml").read_text(encoding="utf-8"))

    for psid in (*split.val, *split.test):  # poison val/test: not part of fitting
        npz = samples_dir / f"{psid}.npz"
        with np.load(npz, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["x"] = arrays["x"] + 1000.0
        arrays["y"] = arrays["y"] + 500.0
        np.savez_compressed(npz, **arrays)

    seen.clear()
    run_b = _run_kind(kind, data_root, dataset, "poisoned")
    assert seen == [tuple(sorted(split.train))]
    stats_b = yaml.safe_load((run_b / "preprocessing.yaml").read_text(encoding="utf-8"))
    assert stats_a == stats_b
    # split frozen: the two runs' in-run copies have identical raw bytes
    assert (run_a / "split.yaml").read_bytes() == (run_b / "split.yaml").read_bytes()


def test_cnn_pool_bins_gt_T_cli_friendly_no_output(
    env: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Illegal pool_bins > T: legal at the config layer but rejected by the model layer →
    CLI exits 2 gracefully with no successful artifacts."""
    data_root, out_root = env
    dataset = "ds_cnn_pool_bad"
    _write_prepared_samples(data_root, dataset)  # T=8
    config_path = _write_config(
        data_root, _cnn_config_text(dataset, "r1", pool_bins=9), name="pool_bad.yaml"
    )
    assert _load_cnn_script().main(["--config", str(config_path)]) == 2
    err = capsys.readouterr().err
    assert "error:" in err and "pool_bins" in err
    run_dir = out_root / "training" / "cnn1d" / dataset / "r1"
    assert not (run_dir / "best.pt").exists()
    assert not (run_dir / "metrics.json").exists()


@pytest.mark.parametrize("loader", [_load_script, _load_cnn_script], ids=["mlp_entry", "cnn_entry"])
def test_main_requires_config_argument(loader: Callable[[], Any]) -> None:
    """For both entries --config is required: missing exits argparse with code 2."""
    with pytest.raises(SystemExit) as exc_info:
        loader().main([])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("kind", ["mlp", "cnn1d"])
def test_main_same_error_for_missing_samples(
    kind: str, env: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """For both entries the same DataError (missing sample dir) exits 2 gracefully and
    creates no run dir."""
    _, out_root = env
    dataset = "ds_missing_samples"
    if kind == "mlp":
        config_path = _write_config(tmp_path, _config_text(dataset, "r1"), name="miss_mlp.yaml")
        script = _load_script()
    else:
        config_path = _write_config(tmp_path, _cnn_config_text(dataset, "r1"), name="miss_cnn.yaml")
        script = _load_cnn_script()
    assert script.main(["--config", str(config_path)]) == 2
    assert "sample directory does not exist" in capsys.readouterr().err
    assert not (out_root / "training").exists()
