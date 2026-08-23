"""Tests for MuMax3 discovery and the not-installed diagnostic.

These tests must never require MuMax3 to actually be installed.
"""

from __future__ import annotations

import sys

import pytest

from micromagnetic_parameter_inversion import external


def test_default_name_is_platform_appropriate(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert external.default_mumax3_name() == "mumax3.exe"
    monkeypatch.setattr(sys, "platform", "linux")
    assert external.default_mumax3_name() == "mumax3"


def test_find_mumax3_returns_none_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("MUMAX3_BIN", raising=False)
    monkeypatch.setattr(external.shutil, "which", lambda _: None)
    assert external.find_mumax3() is None


def test_mumax3_status_reports_not_installed(monkeypatch) -> None:
    monkeypatch.delenv("MUMAX3_BIN", raising=False)
    monkeypatch.setattr(external.shutil, "which", lambda _: None)
    status = external.mumax3_status()
    assert status.available is False
    assert status.executable is None
    assert "MUMAX3_BIN" in status.message


def test_run_mumax3_raises_diagnostic_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("MUMAX3_BIN", raising=False)
    monkeypatch.setattr(external.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="MuMax3 not found"):
        external.run_mumax3("script.mx3")


def test_find_mumax3_uses_env_override(monkeypatch, tmp_path) -> None:
    fake = tmp_path / "mumax3"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("MUMAX3_BIN", str(fake))
    monkeypatch.setattr(external.shutil, "which", lambda _: None)
    assert external.find_mumax3() == str(fake.resolve())


def test_find_mumax3_prefers_env_override_on_path(monkeypatch) -> None:
    monkeypatch.setenv("MUMAX3_BIN", "/opt/custom/mumax3")
    monkeypatch.setattr(
        external.shutil,
        "which",
        lambda c: "/opt/custom/mumax3" if c == "/opt/custom/mumax3" else None,
    )
    assert external.find_mumax3() == "/opt/custom/mumax3"
