"""Offline tests for scripts/generate_dataset.py (all offline; never touches real MuMax3).

The first 5 cases verify the built-in FIXED_CONFIGS × PARAMETERS and the pure
build_config behavior (zero writes); the rest are concurrency/failure-path
mock tests: monkeypatched PROJECT_ROOT=tmp_path, a fake run_simulation, and
wait/config writes stubbed as needed; all waits carry a 5s timeout. The
generator script is loaded by path (scripts/ is not a package) and never runs
the real main/real tasks.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
import threading
from concurrent.futures import ALL_COMPLETED
from pathlib import Path
from typing import Any

import pytest
import yaml
from scipy.stats import qmc

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "generate_dataset.py"


def _load_script() -> Any:
    """Load the script module by path (scripts/ is not a package); the module name is
    not __main__, so main is not executed.
    """
    spec = importlib.util.spec_from_file_location("generate_dataset_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generate_dataset = _load_script()

# The only three null injected fields in the template: stripped from both sides when
# comparing fixed fields.
_INJECTED_MATERIAL_KEYS = ("alpha", "ku_j_per_m3")


def test_script_import_without_running_main() -> None:
    """The script loads by path and the import phase does not trigger main()."""
    assert generate_dataset.__name__ == "generate_dataset_script"
    assert callable(generate_dataset.main)
    assert callable(generate_dataset.build_config)


def test_parameters_1024_keys_finite_in_domain_unique() -> None:
    """1024 points; each has only the alpha/Ku keys, all finite, inside the main domain,
    with no duplicates.
    """
    parameters = generate_dataset.PARAMETERS
    assert len(parameters) == 1024
    seen: set[tuple[float, float]] = set()
    for parameter in parameters:
        assert set(parameter) == {"alpha", "ku_j_per_m3"}
        alpha = float(parameter["alpha"])
        ku = float(parameter["ku_j_per_m3"])
        assert math.isfinite(alpha) and math.isfinite(ku)
        assert 0.004 <= alpha <= 0.020
        assert 2000.0 <= ku <= 30000.0
        pair = (alpha, ku)
        assert pair not in seen
        seen.add(pair)


def test_parameters_reproducible_with_rng42() -> None:
    """Recomputation with the same qmc Sobol rng=42 matches the built-in point set point
    by point, and the recomputation is itself reproducible.
    """
    uvs = qmc.Sobol(d=2, scramble=True, rng=42).random_base2(m=10)
    assert uvs.shape == (1024, 2)
    assert (uvs == qmc.Sobol(d=2, scramble=True, rng=42).random_base2(m=10)).all()
    for parameter, row in zip(generate_dataset.PARAMETERS, uvs, strict=True):
        assert float(parameter["alpha"]) == 0.004 * 5.0 ** float(row[0])
        assert float(parameter["ku_j_per_m3"]) == 2000.0 + 28000.0 * float(row[1])


def test_fixed_configs_match_template_fixed_fields() -> None:
    """FIXED_CONFIGS fixed fields match the repository template (only the three null
    injected fields are stripped).
    """
    template = yaml.safe_load(
        (_PROJECT_ROOT / "configs" / "experiments" / "mumax3_simulation.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert template["dataset_name"] is None
    assert template["material"]["alpha"] is None
    assert template["material"]["ku_j_per_m3"] is None
    template.pop("dataset_name")
    for key in _INJECTED_MATERIAL_KEYS:
        template["material"].pop(key)
    for fixed_config in generate_dataset.FIXED_CONFIGS:
        stripped = copy.deepcopy(fixed_config)
        stripped.pop("dataset_name")
        for key in _INJECTED_MATERIAL_KEYS:
            assert key not in stripped["material"]  # alpha/Ku are injected only by build_config
        assert stripped == template


def test_build_config_injects_parameters_without_mutating_constants() -> None:
    """build_config deep-copies and injects alpha/Ku; the FIXED_CONFIGS constants remain
    unchanged.
    """
    fixed_config = generate_dataset.FIXED_CONFIGS[0]
    snapshot = copy.deepcopy(fixed_config)
    config = generate_dataset.build_config(fixed_config, {"alpha": 0.008, "ku_j_per_m3": 12345.0})
    assert config["material"]["alpha"] == 0.008
    assert config["material"]["ku_j_per_m3"] == 12345.0
    assert config["dataset_name"] == fixed_config["dataset_name"]
    config["material"]["ms_a_per_m"] = 0.0
    assert [snapshot] == generate_dataset.FIXED_CONFIGS


# ---------------------------------------------------------------------------
# Concurrency/failure-path mock tests: PROJECT_ROOT→tmp_path, run_simulation→fake,
# never touching real MuMax3; all waits carry a 5s timeout and cannot hang.
# ---------------------------------------------------------------------------


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_run: Any) -> None:
    """Point PROJECT_ROOT at tmp_path and replace run_simulation with a fake."""
    monkeypatch.setattr(generate_dataset, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generate_dataset, "run_simulation", fake_run)


def test_max_workers_2_overlap_bounded_and_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """max_workers=2: the two tasks really overlap in flight, the peak never exceeds 2,
    and group logs are complete.
    """
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}
    two_in_flight = threading.Event()

    def fake_run(config_path: Path) -> None:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            if state["active"] >= 2:
                two_in_flight.set()
        assert two_in_flight.wait(timeout=5), "did not observe two tasks in flight"
        with lock:
            state["active"] -= 1

    _patch_runtime(monkeypatch, tmp_path, fake_run)
    generate_dataset.generate_dataset(
        generate_dataset.FIXED_CONFIGS, generate_dataset.PARAMETERS[:3], max_workers=2
    )
    assert state["peak"] == 2
    out = capsys.readouterr().out
    assert "start: total=3 concurrency=2" in out
    assert out.count("config written") == 3
    assert out.count("simulation started") == 3
    assert out.count("simulation completed elapsed=") == 3
    assert "progress: completed=" in out
    assert "summary: total=3 completed=3 failed=0 not_started=0" in out


def test_max_workers_1_serial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """max_workers=1: serial, at most 1 in flight at any time."""
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    def fake_run(config_path: Path) -> None:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        with lock:
            state["active"] -= 1

    _patch_runtime(monkeypatch, tmp_path, fake_run)
    generate_dataset.generate_dataset(
        generate_dataset.FIXED_CONFIGS, generate_dataset.PARAMETERS[:3], max_workers=1
    )
    assert state["peak"] == 1
    assert "start: total=3 concurrency=1" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [0, -1, True, False, 2.5])
def test_invalid_max_workers_rejected_before_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad: Any
) -> None:
    """Illegal max_workers (0/negative/bool/float) is rejected before any file is written."""

    def fake_run(config_path: Path) -> None:
        raise AssertionError(f"illegal max_workers must not run any task: {config_path}")

    _patch_runtime(monkeypatch, tmp_path, fake_run)
    with pytest.raises((TypeError, ValueError)):
        generate_dataset.generate_dataset(
            generate_dataset.FIXED_CONFIGS, generate_dataset.PARAMETERS[:1], max_workers=bad
        )
    assert not (tmp_path / "artifacts").exists()


def test_first_batch_failure_stops_submission_and_reraises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """First batch: one failure and one success: the third is not submitted and the
    original exception propagates; the failure log includes elapsed time.
    """
    barrier = threading.Barrier(2)
    calls: list[str] = []

    def fake_run(config_path: Path) -> None:
        calls.append(config_path.name)
        barrier.wait(timeout=5)  # ensure two tasks are in flight together
        if config_path.name.startswith("01_"):
            raise RuntimeError("boom-01")

    real_wait = generate_dataset.wait

    def wait_all(fs: Any, return_when: Any = None) -> Any:
        return real_wait(fs, return_when=ALL_COMPLETED)

    _patch_runtime(monkeypatch, tmp_path, fake_run)
    monkeypatch.setattr(generate_dataset, "wait", wait_all)
    with pytest.raises(RuntimeError, match="boom-01") as excinfo:
        generate_dataset.generate_dataset(
            generate_dataset.FIXED_CONFIGS, generate_dataset.PARAMETERS[:3], max_workers=2
        )
    assert excinfo.value.args == ("boom-01",)
    assert sorted(name[:2] for name in calls) == ["01", "02"]  # the third was not started
    out = capsys.readouterr().out
    assert "simulation failed elapsed=" in out
    assert "summary: total=3 completed=1 failed=1 not_started=1" in out


def test_keyboard_interrupt_reraised_and_stops_next_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ctrl-C: after draining, it must be re-raised and the next fixed config group
    must not start.
    """
    fixed_b = copy.deepcopy(generate_dataset.FIXED_CONFIGS[0])
    fixed_b["dataset_name"] = "cofeb_test_interrupt_next_group"
    calls: list[str] = []
    real_wait = generate_dataset.wait
    wait_calls = {"n": 0}

    def fake_run(config_path: Path) -> None:
        calls.append(config_path.name)

    def wait_interrupt_once(fs: Any, return_when: Any = None) -> Any:
        wait_calls["n"] += 1
        if wait_calls["n"] == 1:
            raise KeyboardInterrupt
        return real_wait(fs, return_when=return_when)

    _patch_runtime(monkeypatch, tmp_path, fake_run)
    monkeypatch.setattr(generate_dataset, "wait", wait_interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        generate_dataset.generate_dataset(
            [generate_dataset.FIXED_CONFIGS[0], fixed_b],
            generate_dataset.PARAMETERS[:2],
            max_workers=1,
        )
    assert len(calls) == 1  # only the first task of the first group ran
    assert "interrupt received" in capsys.readouterr().out
    assert not (
        tmp_path / "artifacts" / "generated_configs" / "cofeb_test_interrupt_next_group"
    ).exists()


def test_config_write_failure_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Config write failure: failure log and future collection still run, the original
    exception propagates, no 'config written'.
    """

    def broken_write(config: dict[str, Any], config_path: Path) -> None:
        raise OSError("disk-full")

    monkeypatch.setattr(generate_dataset, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generate_dataset, "_write_config_yaml", broken_write)
    with pytest.raises(OSError, match="disk-full"):
        generate_dataset.generate_dataset(
            generate_dataset.FIXED_CONFIGS, generate_dataset.PARAMETERS[:1], max_workers=1
        )
    out = capsys.readouterr().out
    assert "simulation failed" in out and "OSError" in out
    assert "config written" not in out
    assert "summary: total=1 completed=0 failed=1 not_started=0" in out


def test_run_simulation_delegates_and_main_passes_max_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_simulation delegates to the pipeline; main calls generate_dataset with the
    built-in constants and MAX_WORKERS.
    """
    recorded: list[Path] = []

    class _FakePipeline:
        @staticmethod
        def run_parameter_set(config_path: Path) -> None:
            recorded.append(config_path)

    monkeypatch.setattr(generate_dataset, "mumax3_pipeline", _FakePipeline)
    target = Path("fake-config.yaml")
    generate_dataset.run_simulation(target)
    assert recorded == [target]

    seen: dict[str, Any] = {}

    def fake_generate(
        fixed_configs: list[dict[str, Any]],
        parameters: list[dict[str, float]],
        max_workers: int = 2,
    ) -> None:
        seen["fixed"] = fixed_configs
        seen["params"] = parameters
        seen["workers"] = max_workers

    monkeypatch.setattr(generate_dataset, "generate_dataset", fake_generate)
    generate_dataset.main()
    assert seen["fixed"] is generate_dataset.FIXED_CONFIGS
    assert seen["params"] is generate_dataset.PARAMETERS
    assert seen["workers"] == generate_dataset.MAX_WORKERS == 2
