"""Tests for compute-device selection and CPU fallback.

These tests must never require a GPU or MuMax3.
"""

from __future__ import annotations

import torch

from micromagnetic_parameter_inversion import runtime


def test_select_device_cpu_forced() -> None:
    assert runtime.select_device("cpu") == "cpu"


def test_select_device_auto_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert runtime.select_device("auto") == "cpu"


def test_select_device_auto_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert runtime.select_device("auto") == "cuda"


def test_select_device_cuda_unavailable_raises(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        runtime.select_device("cuda")
    except RuntimeError as exc:
        assert "CUDA requested" in str(exc)
    else:  # pragma: no cover - failure branch
        raise AssertionError("expected RuntimeError")


def test_select_device_unknown_preference_raises() -> None:
    try:
        runtime.select_device("gpu")
    except ValueError as exc:
        assert "Unknown device preference" in str(exc)
    else:  # pragma: no cover - failure branch
        raise AssertionError("expected ValueError")


def test_smoke_test_executes_on_cpu(monkeypatch) -> None:
    # Force CPU so the smoke test is a genuine CPU fallback check.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert runtime.smoke_test() == "cpu"


def test_get_runtime_info_consistent() -> None:
    info = runtime.get_runtime_info()
    assert info.torch_version
    assert info.python_version
    assert info.device in ("cpu", "cuda")
    assert info.cuda_available == (info.cuda_device_count > 0)
    if info.cuda_available:
        assert info.cuda_device_name
        assert info.torch_cuda_version
