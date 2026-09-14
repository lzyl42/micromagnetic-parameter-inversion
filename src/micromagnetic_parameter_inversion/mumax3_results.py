"""MuMax3 table parsing and trajectory CSV export.

The trajectory row schema is fixed at five columns: sample_index, t_s, m_x, m_y, m_z.

The MuMax3 3.12 default table (table.txt): the first line is a ``#`` comment header,
columns are tab-separated, and each column has the form ``name (unit)``. This module
requires the header (after stripping the leading ``#`` and surrounding whitespace of
each field) to be semantically strictly equal, and fixed in order, to the four
columns ``t (s)``, ``mx ()``, ``my ()``, ``mz ()``, and data rows to have exactly 4
finite numeric columns; sampling-contract parameters are validated by the config, so
only the table content is checked here. Any contract violation raises
TableParseError, with no repair or truncation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

type TrajectoryRow = tuple[int, float, float, float, float]

# Fixed header: after stripping the leading ``#``, split on tab and strip surrounding
# whitespace from each field; columns must match exactly.
_HEADER_COLUMNS = ("t (s)", "mx ()", "my ()", "mz ()")

# Time-grid tolerance factor (times the time scale); magnetization check tolerance
# (floating-point margin on the |component| and |m|^2 upper bounds). Both are far
# below any physical deviation and only absorb normal floating-point output noise.
_TIME_CHECK_TOLERANCE = 1e-6
_MAGNETIZATION_TOLERANCE = 1e-6

_CSV_HEADER = "sample_index,t_s,m_x,m_y,m_z"
_FLOAT_FORMAT = ".17g"


class TableParseError(ValueError):
    """MuMax3 table content violates the parsing contract; messages locate file/line/column."""


def _parse_field(field: str, table_path: Path, line_number: int, column: str) -> float:
    """Parse a single numeric field; non-numeric and non-finite values (nan/inf) are rejected."""
    try:
        value = float(field)
    except ValueError:
        raise TableParseError(
            f"{table_path}: line {line_number} column {column!r} is not numeric: {field!r}"
        ) from None
    if not math.isfinite(value):
        raise TableParseError(
            f"{table_path}: line {line_number} column {column!r} non-finite value {field!r}"
        )
    return value


def parse_table(
    table_path: Path,
    *,
    pulse_duration_s: float,
    sample_interval_s: float,
    sample_count: int,
) -> list[TrajectoryRow]:
    """Parse one pulse's MuMax3 table into a fixed-length list of trajectory rows.

    Sampling-contract parameters (pulse_duration_s/sample_interval_s/sample_count)
    have already been validated by the config; no business validation is repeated
    here. Content contract: the header is semantically strictly equal to
    _HEADER_COLUMNS (fixed order; rejects ns, wrong units, missing columns, and extra
    columns); the number of data rows is exactly sample_count and all values are
    finite; the native time satisfies ``raw_t[0] ≈ pulse_duration_s``, and thereafter
    each row satisfies ``raw_t[i] ≈ pulse_duration_s + i*sample_interval_s``
    (tolerance is _TIME_CHECK_TOLERANCE times the time scale, covering normal
    floating-point output noise and far below one sampling interval, so it cannot
    mask missing/extra samples); each row's magnetization satisfies
    |component| <= 1+1e-6 and mx^2+my^2+mz^2 <= 1+1e-6. Output is re-anchored to
    integer indices and ``i * sample_interval_s``; native times are not passed
    through. ``#`` comment lines appearing after the header are rejected as well
    (they may indicate a data-stream reset / repeated header).
    """
    text = table_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        raise TableParseError(f"{table_path}: file is empty, missing header")
    header_line = lines[0].strip()
    if not header_line.startswith("#"):
        raise TableParseError(f"{table_path}: line 1 is not a '#' header {lines[0]!r}")
    columns = tuple(cell.strip() for cell in header_line[1:].split("\t"))
    if columns != _HEADER_COLUMNS:
        raise TableParseError(
            f"{table_path}: header must be exactly {_HEADER_COLUMNS} in this order; got {columns}"
        )

    tolerance = _TIME_CHECK_TOLERANCE * max(abs(pulse_duration_s), abs(sample_interval_s))
    rows: list[TrajectoryRow] = []
    for offset, line in enumerate(lines[1:], start=2):
        if line.lstrip().startswith("#"):
            raise TableParseError(
                f"{table_path}: line {offset} has a comment line after the header: {line!r}"
            )
        if not line.strip():
            raise TableParseError(f"{table_path}: line {offset} is an empty line")
        fields = line.split("\t")
        if len(fields) != len(_HEADER_COLUMNS):
            raise TableParseError(
                f"{table_path}: line {offset} column count {len(fields)} does not equal "
                f"the header column count {len(_HEADER_COLUMNS)}"
            )

        raw_t = _parse_field(fields[0], table_path, offset, "t")
        m_x = _parse_field(fields[1], table_path, offset, "mx")
        m_y = _parse_field(fields[2], table_path, offset, "my")
        m_z = _parse_field(fields[3], table_path, offset, "mz")

        expected_t = pulse_duration_s + len(rows) * sample_interval_s
        if abs(raw_t - expected_t) > tolerance:
            raise TableParseError(
                f"{table_path}: line {offset} native time {raw_t!r} deviates from the "
                f"sampling grid {expected_t!r} beyond tolerance {tolerance!r}; "
                f"suspected missing/extra samples"
            )
        for name, value in (("mx", m_x), ("my", m_y), ("mz", m_z)):
            if abs(value) > 1.0 + _MAGNETIZATION_TOLERANCE:
                raise TableParseError(
                    f"{table_path}: line {offset} {name}={value!r} exceeds the "
                    f"|component| <= 1 allowance"
                )
        norm_sq = m_x * m_x + m_y * m_y + m_z * m_z
        if norm_sq > 1.0 + _MAGNETIZATION_TOLERANCE:
            raise TableParseError(
                f"{table_path}: line {offset} |m|^2 = {norm_sq!r} exceeds the 1 allowance"
            )

        sample_index = len(rows)
        rows.append((sample_index, sample_index * sample_interval_s, m_x, m_y, m_z))

    if len(rows) != sample_count:
        raise TableParseError(
            f"{table_path}: data row count {len(rows)} does not equal sample_count {sample_count}"
        )
    return rows


def write_trajectory_csv(rows: Sequence[TrajectoryRow], path: Path) -> None:
    """Write trajectory rows as CSV with the fixed header sample_index,t_s,m_x,m_y,m_z.

    UTF-8, newline="" (newline translation disabled, LF only); floats always in the
    deterministic ``.17g`` round-trip format; parent directories are not created (the
    caller guarantees they exist).
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
