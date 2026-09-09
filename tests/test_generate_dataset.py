"""Offline tests for scripts/generate_dataset.py（阶段A参数准备；零模拟零写入）.

仅加载脚本模块并核对内置 FIXED_CONFIGS × PARAMETERS 与 build_config 纯
函数行为：不调用 main()、不运行 MuMax3、不写任何文件。生成器脚本按路径
加载（scripts/ 非包）。
"""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

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
