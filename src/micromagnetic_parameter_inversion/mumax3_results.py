"""MuMax3 table 解析与轨迹 CSV 导出（vertical slice）。

轨迹行 schema 固定为五列：sample_index, t_s, m_x, m_y, m_z。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

type TrajectoryRow = tuple[int, float, float, float, float]


def parse_table(
    table_path: Path, *, sample_interval_s: float, sample_count: int
) -> list[TrajectoryRow]:
    """解析单个 pulse 的 MuMax3 table 为固定长度轨迹行列表。"""
    # TODO: 真实 table 格式（表头/分隔符/列名）待 fixture 固化后再映射列。
    #  预期静态采样契约（待真实 single-cell MuMax3 pilot 验证）：运行脉冲
    #  -> 关场 -> 立即 TableSave（sample_index=0、t_s=0）-> 再执行
    #  sample_count-1 次 Run(sample_interval_s)+TableSave；行按采样顺序
    #  映射整数索引 0,1,...，t_s = sample_index * sample_interval_s。
    #  行数（== sample_count）与时间检查在此完成，失败直接上抛。
    raise NotImplementedError


def write_trajectory_csv(rows: Sequence[TrajectoryRow], path: Path) -> None:
    """把轨迹行写出为 CSV，表头固定 sample_index,t_s,m_x,m_y,m_z。"""
    # TODO: UTF-8 写出；数值用确定性格式；父目录由调用方保证存在。
    raise NotImplementedError
