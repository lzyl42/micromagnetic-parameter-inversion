"""Offline tests for scripts/generate_dataset.py（全部离线，不触真实 MuMax3）.

前 5 个用例核对内置 FIXED_CONFIGS × PARAMETERS 与 build_config 纯函数行为
（零写入）；其余为并发/失败路径 mock 测试：monkeypatch PROJECT_ROOT=tmp_path、
run_simulation 假函数，wait/写配置按需打桩，所有等待带 5s 超时。生成器脚本
按路径加载（scripts/ 非包），不执行真实 main/真实任务。
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
    """按路径加载脚本模块（scripts/ 非包）；模块名非 __main__，不执行 main。"""
    spec = importlib.util.spec_from_file_location("generate_dataset_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generate_dataset = _load_script()

# 模板仅有的三处待填 null 注入字段：比较固定字段时从两侧剔除。
_INJECTED_MATERIAL_KEYS = ("alpha", "ku_j_per_m3")


def test_script_import_without_running_main() -> None:
    """脚本按路径加载成功，导入阶段不触发 main()。"""
    assert generate_dataset.__name__ == "generate_dataset_script"
    assert callable(generate_dataset.main)
    assert callable(generate_dataset.build_config)


def test_parameters_1024_keys_finite_in_domain_unique() -> None:
    """1024 点；每点仅 alpha/Ku 两键、有限、在主域内、无重复。"""
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
    """同 qmc Sobol rng=42 重算与内置点集逐点一致，且重算自身可复现。"""
    uvs = qmc.Sobol(d=2, scramble=True, rng=42).random_base2(m=10)
    assert uvs.shape == (1024, 2)
    assert (uvs == qmc.Sobol(d=2, scramble=True, rng=42).random_base2(m=10)).all()
    for parameter, row in zip(generate_dataset.PARAMETERS, uvs, strict=True):
        assert float(parameter["alpha"]) == 0.004 * 5.0 ** float(row[0])
        assert float(parameter["ku_j_per_m3"]) == 2000.0 + 28000.0 * float(row[1])


def test_fixed_configs_match_template_fixed_fields() -> None:
    """FIXED_CONFIGS 固定字段与仓库模板一致（仅剔除三处 null 注入字段）。"""
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
            assert key not in stripped["material"]  # alpha/Ku 仅由 build_config 注入
        assert stripped == template


def test_build_config_injects_parameters_without_mutating_constants() -> None:
    """build_config 深拷贝注入 alpha/Ku；FIXED_CONFIGS 常量保持不变。"""
    fixed_config = generate_dataset.FIXED_CONFIGS[0]
    snapshot = copy.deepcopy(fixed_config)
    config = generate_dataset.build_config(fixed_config, {"alpha": 0.008, "ku_j_per_m3": 12345.0})
    assert config["material"]["alpha"] == 0.008
    assert config["material"]["ku_j_per_m3"] == 12345.0
    assert config["dataset_name"] == fixed_config["dataset_name"]
    config["material"]["ms_a_per_m"] = 0.0
    assert [snapshot] == generate_dataset.FIXED_CONFIGS


# ---------------------------------------------------------------------------
# 并发/失败路径 mock 测试：PROJECT_ROOT→tmp_path、run_simulation→假函数，
# 不触真实 MuMax3；所有等待带 5s 超时，不会挂死。
# ---------------------------------------------------------------------------


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_run: Any) -> None:
    """把 PROJECT_ROOT 指到 tmp_path 并替换 run_simulation 为假函数。"""
    monkeypatch.setattr(generate_dataset, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generate_dataset, "run_simulation", fake_run)


def test_max_workers_2_overlap_bounded_and_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """max_workers=2：两任务确实重叠在途、峰值不超 2；组级日志齐全。"""
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}
    two_in_flight = threading.Event()

    def fake_run(config_path: Path) -> None:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            if state["active"] >= 2:
                two_in_flight.set()
        assert two_in_flight.wait(timeout=5), "未观察到两任务同时在途"
        with lock:
            state["active"] -= 1

    _patch_runtime(monkeypatch, tmp_path, fake_run)
    generate_dataset.generate_dataset(
        generate_dataset.FIXED_CONFIGS, generate_dataset.PARAMETERS[:3], max_workers=2
    )
    assert state["peak"] == 2
    out = capsys.readouterr().out
    assert "开始: 总数=3 并发=2" in out
    assert out.count("配置已写出") == 3
    assert out.count("模拟启动") == 3
    assert out.count("模拟完成 耗时=") == 3
    assert "进度: 已完成=" in out
    assert "汇总: 总计=3 已完成=3 已失败=0 未启动=0" in out


def test_max_workers_1_serial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """max_workers=1：串行，任意时刻最多 1 个在途。"""
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
    assert "开始: 总数=3 并发=1" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [0, -1, True, False, 2.5])
def test_invalid_max_workers_rejected_before_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad: Any
) -> None:
    """非法 max_workers（0/负/bool/float）在任何文件写出前拒绝。"""

    def fake_run(config_path: Path) -> None:
        raise AssertionError(f"非法 max_workers 不应运行任何任务: {config_path}")

    _patch_runtime(monkeypatch, tmp_path, fake_run)
    with pytest.raises((TypeError, ValueError)):
        generate_dataset.generate_dataset(
            generate_dataset.FIXED_CONFIGS, generate_dataset.PARAMETERS[:1], max_workers=bad
        )
    assert not (tmp_path / "artifacts").exists()


def test_first_batch_failure_stops_submission_and_reraises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """首批一失败一成功：不提交第 3 个，原异常上抛；失败日志含耗时。"""
    barrier = threading.Barrier(2)
    calls: list[str] = []

    def fake_run(config_path: Path) -> None:
        calls.append(config_path.name)
        barrier.wait(timeout=5)  # 保证两任务同时在途
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
    assert sorted(name[:2] for name in calls) == ["01", "02"]  # 第 3 个未启动
    out = capsys.readouterr().out
    assert "模拟失败 耗时=" in out
    assert "汇总: 总计=3 已完成=1 已失败=1 未启动=1" in out


def test_keyboard_interrupt_reraised_and_stops_next_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ctrl-C：排空后必须重抛，且不开始后续 fixed config 组。"""
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
    assert len(calls) == 1  # 仅第一组首个任务运行
    assert "收到中断" in capsys.readouterr().out
    assert not (
        tmp_path / "artifacts" / "generated_configs" / "cofeb_test_interrupt_next_group"
    ).exists()


def test_config_write_failure_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """配置写失败：走失败日志与 future 收割，原异常上抛，无「配置已写出」。"""

    def broken_write(config: dict[str, Any], config_path: Path) -> None:
        raise OSError("disk-full")

    monkeypatch.setattr(generate_dataset, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generate_dataset, "_write_config_yaml", broken_write)
    with pytest.raises(OSError, match="disk-full"):
        generate_dataset.generate_dataset(
            generate_dataset.FIXED_CONFIGS, generate_dataset.PARAMETERS[:1], max_workers=1
        )
    out = capsys.readouterr().out
    assert "模拟失败" in out and "OSError" in out
    assert "配置已写出" not in out
    assert "汇总: 总计=1 已完成=0 已失败=1 未启动=0" in out


def test_run_simulation_delegates_and_main_passes_max_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_simulation 委托 pipeline；main 用内置常量与 MAX_WORKERS 调 generate_dataset。"""
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
