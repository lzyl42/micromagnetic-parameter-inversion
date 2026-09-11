"""MuMax3 table 解析与轨迹 CSV 导出。

轨迹行 schema 固定为五列：sample_index, t_s, m_x, m_y, m_z。

MuMax3 3.12 默认 table（table.txt）：首行为 ``#`` 注释表头，列以 tab 分隔，
每列为 ``名称 (单位)`` 形式。本模块要求表头（去开头 ``#`` 与字段两侧空白
后）语义上严格等于且顺序固定为四列 ``t (s)``、``mx ()``、``my ()``、
``mz ()``，数据行必须恰 4 列有限数值；采样契约参数由 config 校验，此处只
校验 table 内容。任何契约违反一律 TableParseError，不做修复或截断。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

type TrajectoryRow = tuple[int, float, float, float, float]

# 固定表头：去开头 ``#`` 后按 tab 切分、剥离字段两侧空白，必须逐列相等。
_HEADER_COLUMNS = ("t (s)", "mx ()", "my ()", "mz ()")

# 时间网格容差系数（乘时间尺度）；磁化校验容差（|分量| 与 |m|^2 上界的
# 浮点余量）。两者都远小于任何物理偏差，只吸收正常浮点输出噪声。
_TIME_CHECK_TOLERANCE = 1e-6
_MAGNETIZATION_TOLERANCE = 1e-6

_CSV_HEADER = "sample_index,t_s,m_x,m_y,m_z"
_FLOAT_FORMAT = ".17g"


class TableParseError(ValueError):
    """MuMax3 table 内容违反解析契约；消息含文件/行号/列名定位。"""


def _parse_field(field: str, table_path: Path, line_number: int, column: str) -> float:
    """解析单个数值字段；非数值与非有限值（nan/inf）直接拒绝。"""
    try:
        value = float(field)
    except ValueError:
        raise TableParseError(
            f"{table_path}: 第 {line_number} 行列 {column!r} 非数值 {field!r}"
        ) from None
    if not math.isfinite(value):
        raise TableParseError(f"{table_path}: 第 {line_number} 行列 {column!r} 非有限值 {field!r}")
    return value


def parse_table(
    table_path: Path,
    *,
    pulse_duration_s: float,
    sample_interval_s: float,
    sample_count: int,
) -> list[TrajectoryRow]:
    """解析单个 pulse 的 MuMax3 table 为固定长度轨迹行列表。

    采样契约参数（pulse_duration_s/sample_interval_s/sample_count）已由
    config 校验，此处不重复业务校验。内容契约：表头语义严格等于
    _HEADER_COLUMNS（顺序固定，拒绝 ns、错误单位、缺列、额外列）；数据行
    数恰为 sample_count 且全部数值有限；原生时间满足 ``raw_t[0] ≈
    pulse_duration_s``，此后每行 ``raw_t[i] ≈ pulse_duration_s +
    i*sample_interval_s``（容差为 _TIME_CHECK_TOLERANCE 乘时间尺度，覆盖
    正常浮点输出噪声，远小于一个采样间隔，不会掩盖漏采/多采）；每行磁化
    满足 |分量| <= 1+1e-6 且 mx^2+my^2+mz^2 <= 1+1e-6。输出重锚为整数
    index 与 ``i * sample_interval_s``，不透传原生时间。表头后出现的
    ``#`` 注释行同样拒绝（可能意味着数据流重置/重复表头）。
    """
    text = table_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        raise TableParseError(f"{table_path}: 文件为空，缺少表头")
    header_line = lines[0].strip()
    if not header_line.startswith("#"):
        raise TableParseError(f"{table_path}: 第 1 行不是 '#' 表头 {lines[0]!r}")
    columns = tuple(cell.strip() for cell in header_line[1:].split("\t"))
    if columns != _HEADER_COLUMNS:
        raise TableParseError(
            f"{table_path}: 表头必须恰为且顺序等于 {_HEADER_COLUMNS}；实际 {columns}"
        )

    tolerance = _TIME_CHECK_TOLERANCE * max(abs(pulse_duration_s), abs(sample_interval_s))
    rows: list[TrajectoryRow] = []
    for offset, line in enumerate(lines[1:], start=2):
        if line.lstrip().startswith("#"):
            raise TableParseError(f"{table_path}: 第 {offset} 行出现表头后的注释行 {line!r}")
        if not line.strip():
            raise TableParseError(f"{table_path}: 第 {offset} 行为空行")
        fields = line.split("\t")
        if len(fields) != len(_HEADER_COLUMNS):
            raise TableParseError(
                f"{table_path}: 第 {offset} 行列数 {len(fields)} 不等于表头列数 "
                f"{len(_HEADER_COLUMNS)}"
            )

        raw_t = _parse_field(fields[0], table_path, offset, "t")
        m_x = _parse_field(fields[1], table_path, offset, "mx")
        m_y = _parse_field(fields[2], table_path, offset, "my")
        m_z = _parse_field(fields[3], table_path, offset, "mz")

        expected_t = pulse_duration_s + len(rows) * sample_interval_s
        if abs(raw_t - expected_t) > tolerance:
            raise TableParseError(
                f"{table_path}: 第 {offset} 行原生时间 {raw_t!r} 偏离采样网格 "
                f"{expected_t!r} 超过容差 {tolerance!r}；疑似漏采/多采"
            )
        for name, value in (("mx", m_x), ("my", m_y), ("mz", m_z)):
            if abs(value) > 1.0 + _MAGNETIZATION_TOLERANCE:
                raise TableParseError(
                    f"{table_path}: 第 {offset} 行 {name}={value!r} 超出 |分量| <= 1 容许范围"
                )
        norm_sq = m_x * m_x + m_y * m_y + m_z * m_z
        if norm_sq > 1.0 + _MAGNETIZATION_TOLERANCE:
            raise TableParseError(
                f"{table_path}: 第 {offset} 行 |m|^2 = {norm_sq!r} 超出 1 容许范围"
            )

        sample_index = len(rows)
        rows.append((sample_index, sample_index * sample_interval_s, m_x, m_y, m_z))

    if len(rows) != sample_count:
        raise TableParseError(
            f"{table_path}: 数据行数 {len(rows)} 不等于 sample_count {sample_count}"
        )
    return rows


def write_trajectory_csv(rows: Sequence[TrajectoryRow], path: Path) -> None:
    """把轨迹行写出为 CSV，表头固定 sample_index,t_s,m_x,m_y,m_z。

    UTF-8、newline=""（禁用换行翻译，统一 LF）；浮点一律 .17g 往返
    确定性格式；不创建父目录（由调用方保证存在）。
    """
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(f"{_CSV_HEADER}\n")
        for sample_index, t_s, m_x, m_y, m_z in rows:
            stream.write(
                f"{sample_index:d},"
                f"{format(t_s, _FLOAT_FORMAT)},"
                f"{format(m_x, _FLOAT_FORMAT)},"
                f"{format(m_y, _FLOAT_FORMAT)},"
                f"{format(m_z, _FLOAT_FORMAT)}\n"
            )
