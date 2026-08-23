"""Tests for environment-variable path overrides and defaults."""

from __future__ import annotations

from micromagnetic_parameter_inversion import paths


def test_defaults_fall_back_to_project_dirs(monkeypatch) -> None:
    monkeypatch.delenv(paths.DATA_ROOT_ENV, raising=False)
    monkeypatch.delenv(paths.OUTPUT_ROOT_ENV, raising=False)
    assert paths.data_root() == (paths.PROJECT_ROOT / "data").resolve()
    assert paths.output_root() == (paths.PROJECT_ROOT / "artifacts").resolve()


def test_env_override_data_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "custom-data"))
    assert paths.data_root() == (tmp_path / "custom-data").resolve()


def test_env_override_output_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(paths.OUTPUT_ROOT_ENV, str(tmp_path / "custom-out"))
    assert paths.output_root() == (tmp_path / "custom-out").resolve()


def test_env_override_with_tilde(monkeypatch, tmp_path) -> None:
    # Tilde expansion must work for cross-platform style overrides.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(paths.DATA_ROOT_ENV, "~/micromag-data")
    assert paths.data_root() == (tmp_path / "micromag-data").resolve()


def test_ensure_dirs_creates_roots(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "d"))
    monkeypatch.setenv(paths.OUTPUT_ROOT_ENV, str(tmp_path / "o"))
    data, out = paths.ensure_dirs()
    assert data.is_dir()
    assert out.is_dir()
