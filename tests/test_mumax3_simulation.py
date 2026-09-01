"""MuMax3 vertical slice 骨架测试。

实现未完成：全部用例 skip（skip 后函数体不执行）；函数体保留中文 TODO
与显式失败，不静默 pass，不运行真实 MuMax3。
"""

from __future__ import annotations

import pytest

SKIP = pytest.mark.skip(reason="TODO: MuMax3 vertical slice 尚未实现")


@SKIP
def test_derive_cell_size_m() -> None:
    """cell_size[i] == size_m[i] / cells[i]（单一真值）。"""
    # TODO: 构造 GeometryConfig，断言 derive_cell_size_m 逐分量结果。
    raise NotImplementedError


@SKIP
def test_mt_to_t_conversion() -> None:
    """mt_to_t 换算因子为 1e-3。"""
    # TODO: 断言 mt_to_t(x) == x * 1e-3。
    raise NotImplementedError


@SKIP
def test_parameter_set_id_depends_only_on_alpha_and_ku() -> None:
    """parameter_set_id 只依赖 alpha 与 Ku，不含激励/几何/执行信息。"""
    # TODO: 改变无关字段身份不变；改变 alpha 或 Ku 身份改变。
    raise NotImplementedError


@SKIP
def test_equilibrium_runs_once_and_shared_by_pulses() -> None:
    """equilibrium 每个 parameter set 只渲染/执行一次，多 pulse 共用。"""
    # TODO: 桩掉 external.run_mumax3，断言 equilibrium 恰执行一次，
    #  且每个 pulse 的脚本引用同一 equilibrium.ovf。
    raise NotImplementedError


@SKIP
def test_sampling_integer_indices_fixed_row_count() -> None:
    """采样行数固定为 sample_count；索引 0 起整数；t_s=0 为关场后首点。"""
    # TODO: 合成 table -> parse_table -> 断言行数/索引/t_s 映射。
    raise NotImplementedError


@SKIP
def test_manifest_written_last() -> None:
    """manifest.json 仅在全部 pulse 成功后最后写一次。"""
    # TODO: 桩掉执行，断言 manifest 写入晚于 index.csv 且仅一次。
    raise NotImplementedError
