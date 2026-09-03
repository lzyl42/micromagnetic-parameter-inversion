"""MuMax3 vertical slice 公共行为测试：全部离线，不依赖 GPU/MuMax3、不写 data/raw。

MuMax3 执行经 monkeypatch 替换 pipeline 调用点伪造；数值均为 test-only 占位值。
"""

from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from micromagnetic_parameter_inversion import mumax3_pipeline
from micromagnetic_parameter_inversion.mumax3_config import (
    ConfigError,
    derive_cell_size_m,
    load_config,
    parameter_set_id,
)
from micromagnetic_parameter_inversion.mumax3_results import (
    TableParseError,
    parse_table,
    write_trajectory_csv,
)
from micromagnetic_parameter_inversion.mumax3_script import (
    render_equilibrium_script,
    render_simulation_script,
)
from micromagnetic_parameter_inversion.paths import PROJECT_ROOT

# 仓库真实模板（只读）与渲染/输出契约中的固定文本。
_EQUILIBRIUM_TEMPLATE = PROJECT_ROOT / "simulations" / "mumax3" / "equilibrium.mx3.in"
_SIMULATION_TEMPLATE = PROJECT_ROOT / "simulations" / "mumax3" / "simulation.mx3.in"
_LOADFILE_LINE = 'm.LoadFile("../../equilibrium/equilibrium.out/equilibrium.ovf")'
_TRAJECTORY_HEADER = "sample_index,t_s,m_x,m_y,m_z"
_INTERVAL_S = 1.0e-12
_SAMPLE_COUNT = 5
_DURATION_S = 2.0e-9


def _test_config_dict() -> dict[str, Any]:
    """最小合法 test-only 配置：alpha=0、负 Ku、非零 mT 脉冲 + 零场脉冲。"""
    return {
        "dataset_name": "test-dataset",
        "material": {
            "ms_a_per_m": 8.0e5,
            "aex_j_per_m": 1.3e-11,
            "alpha": 0.0,
            "ku_j_per_m3": -5.0e5,
            "anisotropy_axis": [0.0, 0.0, 1.0],
        },
        "geometry": {"size_m": [160.0e-9, 80.0e-9, 3.0e-9], "cells": [32, 16, 1]},
        "initial_m": [1.0, 0.0, 0.0],
        "recording": {"sample_interval_s": _INTERVAL_S, "sample_count": _SAMPLE_COUNT},
        "pulses": [
            {
                "pulse_id": "pulse_a",
                "b_ext_amplitude_mT": 50.0,
                "direction": [0.0, 0.0, 1.0],
                "duration_s": _DURATION_S,
            },
            {
                "pulse_id": "pulse_b",
                "b_ext_amplitude_mT": 0.0,
                "direction": [1.0, 0.0, 0.0],
                "duration_s": 1.0e-9,
            },
        ],
    }


def _write_config_yaml(tmp_path: Path, mapping: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return config_path


def _write_valid_config(tmp_path: Path) -> Path:
    return _write_config_yaml(tmp_path, _test_config_dict())


def _table_text(sample_count: int, interval_s: float, duration_s: float) -> str:
    """真实 MuMax3 表头格式的合成 table；raw_t 从 duration_s 起，m 物理合法。"""
    lines = ["# t (s)\tmx ()\tmy ()\tmz ()"]
    for i in range(sample_count):
        lines.append("\t".join(repr(v) for v in (duration_s + i * interval_s, 0.8, 0.6, 0.0)))
    return "\n".join(lines) + "\n"


def _edit_field(lines: list[str], line_index: int, field_index: int, value: str) -> list[str]:
    fields = lines[line_index].split("\t")
    fields[field_index] = value
    edited = list(lines)
    edited[line_index] = "\t".join(fields)
    return edited


def _sampling_from_script(script_text: str) -> tuple[int, float, float]:
    """从渲染后的 simulation 脚本提取 (sample_count, interval_s, duration_s)。"""
    run_args = re.findall(r"(?m)^[ \t]*Run\(([^)]+)\)", script_text)
    assert len(run_args) == 2, run_args
    loop = re.search(r"for i := 1; i < (\d+);", script_text)
    assert loop is not None, script_text
    return int(loop.group(1)), float(run_args[1]), float(run_args[0])


@dataclass
class _FakeMumax:
    data_dir: Path
    runs: list[Path] = field(default_factory=list)
    simulation_returncode: int = 0
    empty_output: str | None = None


def _install_pipeline_fake(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: Any
) -> _FakeMumax:
    """替换 pipeline 的 run_mumax3/data_root 调用点并伪造契约 .out 产物。"""
    fake = _FakeMumax(data_dir=tmp_path / "data-root", **kwargs)

    def fake_run(
        script: str | Path, *, workdir: str | Path | None = None, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        script_path = Path(script)
        run_dir = Path(workdir) if workdir is not None else Path()
        fake.runs.append(run_dir)
        is_equilibrium = script_path.name == "equilibrium.mx3"
        returncode = 0 if is_equilibrium else fake.simulation_returncode
        if returncode != 0:
            return subprocess.CompletedProcess([str(script_path)], returncode, "", "boom")
        out_dir = run_dir / ("equilibrium.out" if is_equilibrium else "simulation.out")
        out_dir.mkdir(parents=True)
        if is_equilibrium:
            outputs = {"equilibrium.ovf": b"test-only equilibrium.ovf"}
        else:
            count, interval_s, duration_s = _sampling_from_script(
                script_path.read_text(encoding="utf-8")
            )
            outputs = {
                "table.txt": _table_text(count, interval_s, duration_s).encode("utf-8"),
                "m_t0.ovf": b"test-only m_t0.ovf",
                "m_tfinal.ovf": b"test-only m_tfinal.ovf",
            }
        for name, payload in outputs.items():
            (out_dir / name).write_bytes(b"" if name == fake.empty_output else payload)
        return subprocess.CompletedProcess([str(script_path)], 0, "fake mumax stdout\n", "")

    monkeypatch.setattr(mumax3_pipeline.external, "run_mumax3", fake_run)
    monkeypatch.setattr(mumax3_pipeline, "data_root", lambda: fake.data_dir)
    return fake


def _expected_out_dir(fake: _FakeMumax, config_path: Path) -> Path:
    config = load_config(config_path)
    set_id = parameter_set_id(config.material.alpha, config.material.ku_j_per_m3)
    return fake.data_dir / "raw" / config.dataset_name / set_id


def test_load_config_valid_one_shot(tmp_path: Path) -> None:
    """合法边界一次覆盖：alpha=0、负 Ku、mT->T、零场脉冲、cell size。"""
    config = load_config(_write_valid_config(tmp_path))
    assert config.material.alpha == 0.0
    assert config.material.ku_j_per_m3 == -5.0e5
    assert [pulse.pulse_id for pulse in config.pulses] == ["pulse_a", "pulse_b"]
    assert config.pulses[0].b_ext_amplitude_t == 50.0e-3
    assert config.pulses[1].b_ext_amplitude_t == 0.0
    assert derive_cell_size_m(config.geometry) == pytest.approx((5.0e-9, 5.0e-9, 3.0e-9))


def test_load_config_rejects_invalid_yaml(tmp_path: Path) -> None:
    """代表性非法 YAML（子目录隔离）：schema/取值/唯一性/路径段一律 ConfigError。"""
    cases = [
        ("unknown-field", lambda m: m["material"].update(typo=1.0)),
        ("missing-field", lambda m: m["material"].pop("alpha")),
        ("non-finite", lambda m: m["material"].update(ku_j_per_m3=float("inf"))),
        ("negative-alpha", lambda m: m["material"].update(alpha=-0.01)),
        ("negative-amplitude", lambda m: m["pulses"][0].update(b_ext_amplitude_mT=-1.0)),
        ("bad-cells", lambda m: m["geometry"].update(cells=[32, 16, 0])),
        ("non-unit-vector", lambda m: m["pulses"][1].update(direction=[1.0, 1.0, 0.0])),
        ("duplicate-id", lambda m: m["pulses"][1].update(pulse_id="pulse_a")),
        ("path-escape", lambda m: m.update(dataset_name="../escape")),
    ]
    for name, mutate in cases:
        mapping = _test_config_dict()
        mutate(mapping)
        with pytest.raises(ConfigError):
            load_config(_write_config_yaml(tmp_path / name, mapping))
    dup_path = tmp_path / "dup-key" / "config.yaml"
    dup_path.parent.mkdir(parents=True)
    dup_path.write_text(
        "dataset_name: a\nmaterial:\n  alpha: 0.1\n  alpha: 0.2\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_config(dup_path)


def test_parameter_set_id_format_and_dependence(tmp_path: Path) -> None:
    """公共格式契约：16 位小写 hex；同参数稳定、任一参数改变则改变。"""
    alpha, ku = 0.0, -5.0e5
    set_id = parameter_set_id(alpha, ku)
    assert re.fullmatch(r"[0-9a-f]{16}", set_id)
    assert parameter_set_id(alpha, ku) == set_id
    assert parameter_set_id(0.01, ku) != set_id
    assert parameter_set_id(alpha, 5.0e5) != set_id

    # 其余配置维度全变而 (alpha, Ku) 不变 -> id 不变（不受 pulse 等字段影响）。
    other = _test_config_dict()
    other["dataset_name"] = "other-dataset"
    other["material"]["ms_a_per_m"] = 7.0e5
    other["geometry"]["cells"] = [16, 8, 1]
    other["recording"]["sample_count"] = 3
    other["pulses"] = [{**other["pulses"][0], "pulse_id": "solo"}]
    variant = load_config(_write_config_yaml(tmp_path / "other", other))
    assert parameter_set_id(variant.material.alpha, variant.material.ku_j_per_m3) == set_id


def test_renderers_render_real_repo_templates(tmp_path: Path) -> None:
    """两个公共 renderer 用仓库模板完整渲染：无残留占位符，含关键命令。"""
    config = load_config(_write_valid_config(tmp_path))
    equilibrium_template = _EQUILIBRIUM_TEMPLATE.read_text(encoding="utf-8")
    simulation_template = _SIMULATION_TEMPLATE.read_text(encoding="utf-8")

    equilibrium = render_equilibrium_script(config, equilibrium_template)
    assert "{{" not in equilibrium
    assert equilibrium.count("Relax()") == 1
    assert "B_ext = vector(0, 0, 0)" in equilibrium

    for pulse in config.pulses:
        rendered = render_simulation_script(config, pulse, simulation_template)
        assert "{{" not in rendered
        assert rendered.count(_LOADFILE_LINE) == 1
        assert rendered.count("TableSave()") == 2
        assert "Relax" not in rendered
        count, interval_s, duration_s = _sampling_from_script(rendered)
        assert (count, interval_s, duration_s) == (
            config.recording.sample_count,
            config.recording.sample_interval_s,
            pulse.duration_s,
        )
        assert re.search(r"(?m)^alpha = ", rendered)
        b_ext_lines = [line for line in rendered.splitlines() if line.startswith("B_ext = vector(")]
        parsed = [
            [float(component) for component in line[len("B_ext = vector(") : -1].split(",")]
            for line in b_ext_lines
        ]
        assert parsed == [
            [pulse.b_ext_amplitude_t * axis for axis in pulse.direction],
            [0.0, 0.0, 0.0],
        ]


def test_parse_table_valid_and_csv_export(tmp_path: Path) -> None:
    """真实格式 table 正常解析：原生首时刻 = duration，输出重锚为整数网格。"""
    table_path = tmp_path / "table.txt"
    table_path.write_text(_table_text(_SAMPLE_COUNT, _INTERVAL_S, _DURATION_S), encoding="utf-8")
    lines = table_path.read_text(encoding="utf-8").splitlines()
    assert float(lines[1].split("\t")[0]) == _DURATION_S

    rows = parse_table(
        table_path,
        pulse_duration_s=_DURATION_S,
        sample_interval_s=_INTERVAL_S,
        sample_count=_SAMPLE_COUNT,
    )
    for i, row in enumerate(rows):
        assert row == (i, i * _INTERVAL_S, 0.8, 0.6, 0.0)

    trajectory_path = tmp_path / "trajectory.csv"
    write_trajectory_csv(rows, trajectory_path)
    csv_lines = trajectory_path.read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == _TRAJECTORY_HEADER
    assert len(csv_lines) == 1 + _SAMPLE_COUNT
    for i, line in enumerate(csv_lines[1:]):
        cells = line.split(",")
        assert int(cells[0]) == i
        assert [float(value) for value in cells[1:]] == list(rows[i][1:])


def test_parse_table_rejects_contract_violations(tmp_path: Path) -> None:
    """代表性契约违反（子目录隔离）：单位/列/首时刻/网格/行数/NaN/范数。"""
    lines = _table_text(_SAMPLE_COUNT, _INTERVAL_S, _DURATION_S).splitlines()
    cases = [
        ("wrong-unit", [lines[0].replace("t (s)", "t (ns)"), *lines[1:]]),
        ("missing-column", [lines[0].replace("\tmz ()", ""), *lines[1:]]),
        ("wrong-first-time", _edit_field(lines, 1, 0, "0")),
        ("wrong-grid", _edit_field(lines, 3, 0, "2.003e-09")),
        ("too-few-rows", lines[:-1]),
        ("nan-value", _edit_field(lines, 2, 1, "nan")),
        ("norm-out-of-range", _edit_field(lines, 2, 2, "1")),
    ]
    for name, corrupted in cases:
        bad_path = tmp_path / name / "table.txt"
        bad_path.parent.mkdir(parents=True)
        bad_path.write_text("\n".join(corrupted) + "\n", encoding="utf-8")
        with pytest.raises(TableParseError):
            parse_table(
                bad_path,
                pulse_duration_s=_DURATION_S,
                sample_interval_s=_INTERVAL_S,
                sample_count=_SAMPLE_COUNT,
            )


def test_pipeline_success_writes_full_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """全流程成功：equilibrium 一次且先行、共享 LoadFile、index 两行正确。"""
    fake = _install_pipeline_fake(monkeypatch, tmp_path)
    config_path = _write_valid_config(tmp_path)
    config = load_config(config_path)
    out_dir = _expected_out_dir(fake, config_path)

    returned = mumax3_pipeline.run_parameter_set(config_path)

    assert returned == out_dir
    assert fake.runs == [
        out_dir / "equilibrium",
        *(out_dir / "runs" / pulse.pulse_id for pulse in config.pulses),
    ]
    equilibrium_ovf = out_dir / "equilibrium" / "equilibrium.out" / "equilibrium.ovf"
    assert list(out_dir.rglob("equilibrium.ovf")) == [equilibrium_ovf]
    for pulse in config.pulses:
        pulse_dir = out_dir / "runs" / pulse.pulse_id
        simulation_script = (pulse_dir / "simulation.mx3").read_text(encoding="utf-8")
        assert simulation_script.count(_LOADFILE_LINE) == 1
        trajectory_lines = (pulse_dir / "trajectory.csv").read_text(encoding="utf-8").splitlines()
        assert trajectory_lines[0] == _TRAJECTORY_HEADER
        assert len(trajectory_lines) == 1 + config.recording.sample_count
    assert (out_dir / "config.yaml").read_bytes() == config_path.read_bytes()
    for run_dir in fake.runs:
        assert (run_dir / "run.log").is_file()
    assert not (out_dir / "index.csv.tmp").exists()

    with (out_dir / "index.csv").open(encoding="utf-8", newline="") as handle:
        index_rows = list(csv.reader(handle))
    assert ",".join(index_rows[0]) == (
        "parameter_set_id,pulse_id,alpha,ku_j_per_m3,"
        "b_ext_x_T,b_ext_y_T,b_ext_z_T,pulse_duration_s,trajectory_path"
    )
    assert len(index_rows) == 1 + len(config.pulses)
    for row, pulse in zip(index_rows[1:], config.pulses, strict=True):
        floats = [repr(config.material.alpha), repr(config.material.ku_j_per_m3)]
        floats += [repr(pulse.b_ext_amplitude_t * axis) for axis in pulse.direction]
        floats.append(repr(pulse.duration_s))
        expected = [out_dir.name, pulse.pulse_id, *floats, f"runs/{pulse.pulse_id}/trajectory.csv"]
        assert row == expected


def test_pipeline_stops_when_mumax_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """pulse 阶段非零返回：run.log 留存，后续 pulse 不执行，无最终 index。"""
    fake = _install_pipeline_fake(monkeypatch, tmp_path, simulation_returncode=1)
    config_path = _write_valid_config(tmp_path)
    out_dir = _expected_out_dir(fake, config_path)

    with pytest.raises(RuntimeError):
        mumax3_pipeline.run_parameter_set(config_path)

    failed_log = (out_dir / "runs" / "pulse_a" / "run.log").read_text(encoding="utf-8")
    assert "returncode: 1" in failed_log
    assert len(fake.runs) == 2
    assert not (out_dir / "index.csv").exists()


def test_pipeline_rejects_empty_core_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """任一核心产物为空均失败（子目录隔离），且不产生最终 index。"""
    for name in ("equilibrium.ovf", "table.txt", "m_t0.ovf", "m_tfinal.ovf"):
        fake = _install_pipeline_fake(monkeypatch, tmp_path / name, empty_output=name)
        config_path = _write_valid_config(tmp_path / name)
        out_dir = _expected_out_dir(fake, config_path)
        with pytest.raises(FileNotFoundError):
            mumax3_pipeline.run_parameter_set(config_path)
        assert not (out_dir / "index.csv").exists()


def test_pipeline_refuses_existing_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _install_pipeline_fake(monkeypatch, tmp_path)
    config_path = _write_valid_config(tmp_path)
    out_dir = _expected_out_dir(fake, config_path)
    out_dir.mkdir(parents=True)
    (out_dir / "sentinel.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        mumax3_pipeline.run_parameter_set(config_path)

    assert fake.runs == []
    assert (out_dir / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    assert not (out_dir / "index.csv").exists()
