"""Tests for the training CLI, ckpt safety and reload parity (offline, CPU).

Synthetic npz/meta/split are written directly via training_data into tmp_path
(no raw/prepare, no cross-test fixture imports); device is always cpu; no
GPU/MuMax3.
"""

from __future__ import annotations

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
from torch import nn

from micromagnetic_parameter_inversion import paths, preprocessing, training, training_data
from micromagnetic_parameter_inversion.training_config import (
    SplitConfig,
    SplitMinCounts,
    SplitRatios,
    load_config,
)

_SCRIPT_PATH = paths.PROJECT_ROOT / "scripts" / "train_mlp.py"


def _load_script() -> Any:
    """Load the train_mlp script module by path (scripts/ is not a package; self-contained)."""
    spec = importlib.util.spec_from_file_location("train_mlp_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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
) -> str:
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
        "output_dir: null\n"
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

    # config_resolved.yaml is reloadable by load_config with identical fields
    resolved = load_config(run_dir / "config_resolved.yaml")
    assert (resolved.dataset_name, resolved.run_name) == ("ds1", "r1")
    assert resolved.model.hidden_dims == (8, 4)
    assert resolved.training.max_epochs == 2
    # split copy = raw bytes of the source split.yaml
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
    # preprocessing state roundtrip: mean/effective scale shapes and semantics
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
    # final.pt has best_val_loss None (only best.pt has it)
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
    with pytest.raises(training.TrainingError, match="mapping"):
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
    assert "refusing to overwrite" in capsys.readouterr().err


def test_earlystop_reference_independent_of_best(env: tuple[Path, Path]) -> None:
    data_root, _ = env
    _write_prepared_samples(data_root, "ds6")
    # very large min_delta: the reference never improves significantly → stop after epoch 2
    config_path = _write_config(
        data_root, _config_text("ds6", "r1", max_epochs=10, patience=1, min_delta=1.0e6)
    )
    run_dir = _load_script().run(config_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["history"]) == 2  # early stopping triggered before max_epochs
    assert metrics["stop_reason"] == "early_stopping"
    assert metrics["stop_epoch"] == 2
    # absolute best is separate from the earlystop reference: best is still the lowest val loss
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
    assert metrics_a["history"] == metrics_b["history"]  # same seed, same data → bitwise identical
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
    arrays["x"] = arrays["x"][:, :-1, :]  # one step fewer in T: contract mismatch across members
    arrays["t_s"] = arrays["t_s"][:-1]
    np.savez_compressed(victim, **arrays)

    config_path = _write_config(data_root, _config_text("ds8", "r1"))
    with pytest.raises(training_data.DataError, match="contract"):
        _load_script().run(config_path)
    assert not (out_root / "training" / "mlp" / "ds8" / "r1").exists()  # refused before training


class _TwoSampleSet(torch.utils.data.Dataset):
    """Minimal two-sample dataset with y always 0 (pairs with a 1e19-bias model for
    float32 sum overflow).
    """

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return torch.zeros(1, 1, 3), torch.zeros(2), f"ps{index:04d}"


def test_run_validation_float64_aggregate_survives_float32_sum_overflow() -> None:
    """se elements are float32-finite (1e38) but the float32 sum overflows → the
    aggregate must promote to float64.
    """
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
    with torch.no_grad():  # constant output 1e19: hidden layers zeroed, last layer bias only
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
    """Inject a non-finite val aggregate at epoch 2: stop, no history, keep first
    complete epoch weights.
    """
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
    assert exc_info.value.epoch == 2  # numerical-failure triggering epoch (1-based)
    assert exc_info.value.detail is not None and "non-finite" in exc_info.value.detail

    run_dir = _run_dir(env, "ds_guard", "r1")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stop_reason"] == "numerical_failure"
    assert metrics["stop_epoch"] == 2
    assert metrics["detail"] is not None and "non-finite" in metrics["detail"]
    assert len(metrics["history"]) == 1  # non-finite aggregate at epoch 2: no history written
    assert metrics["best_val_loss"] == 1.0
    best = training.load_checkpoint(run_dir / "best.pt")
    final = training.load_checkpoint(run_dir / "final.pt")
    assert best.best_val_loss == 1.0 and final.best_val_loss is None
    for name, tensor in best.model_state_dict.items():
        assert torch.equal(
            tensor, final.model_state_dict[name]
        )  # both hold first complete-epoch weights
    # both ckpts can be serialized and read back safely
    torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu")
    torch.load(run_dir / "final.pt", weights_only=True, map_location="cpu")
    # CLI fails non-zero and does not print training complete (rerun with a separate run_name)
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
    """Small test helper: default output directory layout."""
    _, out_root = env
    return out_root / "training" / "mlp" / dataset / run_name


def test_first_epoch_numerical_failure_writes_failure_metrics_no_checkpoints(
    env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """First-epoch numerical failure: TrainingError carries epoch/detail; failure
    metrics, no ckpt.
    """
    data_root, _ = env
    _write_prepared_samples(data_root, "ds_first_fail")
    config_path = _write_config(data_root, _config_text("ds_first_fail", "r1", max_epochs=3))

    def failing_validation(model: Any, loader: Any, state: Any, device: str) -> float:
        raise training.TrainingError("validation predictions contain non-finite values (NaN/Inf)")

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
    assert not (run_dir / "best.pt").exists()  # no fake checkpoint
    assert not (run_dir / "final.pt").exists()
    # CLI fails non-zero (rerun with a separate run_name)
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
    """val worsens: best stays at the better epoch while final is the last epoch (they diverge)."""
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
    """Accumulated small improvements do not reset the reference: min_delta=0.5,
    patience=2 → stop at epoch 3.
    """
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
    )  # 0.2/0.1 are not significant improvements → counter accumulates
    assert metrics["stop_reason"] == "early_stopping"
    assert metrics["stop_epoch"] == 3
    assert metrics["best_val_loss"] == 0.7  # absolute best is separate from the reference


def test_weights_finite_update_from_same_seed_init(env: tuple[Path, Path]) -> None:
    """Same-seed init vs trained weights: finite and actually updated (not unchanged in place)."""
    data_root, _ = env
    _write_prepared_samples(data_root, "ds_update")
    config_path = _write_config(data_root, _config_text("ds_update", "r1", max_epochs=2))
    run_dir = _load_script().run(config_path)
    ckpt = training.load_checkpoint(run_dir / "best.pt")
    training.set_seed(11)  # matches the config training.seed
    fresh = training.build_model(ckpt.contract, ckpt.hidden_dims)
    trained = dict(ckpt.model_state_dict)
    assert all(torch.isfinite(t).all() for t in trained.values())
    assert any(not torch.equal(trained[name], fresh.state_dict()[name]) for name in trained)
