"""`pfnstudio run <slug> --install-deps` installs PyTorch when it is missing.

Plain `pip install pfnstudio` leaves torch out (a 1-3 GB download the
authoring commands never need), and every base model runs on it. These tests
fake torch's absence; nothing is installed.
"""

from __future__ import annotations

import importlib.util
import subprocess

import pytest
import typer
from pfnstudio import cli


def _no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "torch" else real(name, *a, **k),
    )


def test_torch_present_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    monkeypatch.setattr(subprocess, "check_call", lambda args, **k: calls.append(args))
    cli._ensure_torch(install=True, ref="cdfm-base:latest")
    assert calls == []


def test_missing_torch_with_install_deps_pip_installs_it(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    _no_torch(monkeypatch)
    monkeypatch.setattr(subprocess, "check_call", lambda args, **k: calls.append(args))
    cli._ensure_torch(install=True, ref="cdfm-base:latest")
    assert len(calls) == 1
    assert calls[0][1:4] == ["-m", "pip", "install"]
    assert calls[0][-1] == cli._TORCH_SPEC == "torch>=2.2"


def test_missing_torch_without_install_deps_stops_with_the_fix(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[list[str]] = []
    _no_torch(monkeypatch)
    monkeypatch.setattr(subprocess, "check_call", lambda args, **k: calls.append(args))
    with pytest.raises(typer.Exit) as exc:
        cli._ensure_torch(install=False, ref="cdfm-base:latest")
    assert exc.value.exit_code == 1
    assert calls == []
    out = capsys.readouterr().out
    assert "--install-deps" in out and "cdfm-base:latest" in out
