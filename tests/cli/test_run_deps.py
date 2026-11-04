"""`pfnstudio run <slug>` installs what the model needs only with a yes.

It lists PyTorch (when missing) and the adapter's unmet requirements, asks in
a terminal (Enter = yes), installs without asking under -y, and with no
terminal stops with the -y command. Nothing is installed by these tests.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import typer
from pfnstudio import cli


def _torch(monkeypatch: pytest.MonkeyPatch, *, present: bool) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: (
            (object() if present else None) if name == "torch" else real(name, *a, **k)
        ),
    )


def _pip_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "check_call", lambda args, **k: calls.append(args))
    return calls


def _tty(monkeypatch: pytest.MonkeyPatch, on: bool) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: on, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: on, raising=False)


def test_unmet_requirements_skips_installed_comments_and_options() -> None:
    lines = [
        "# a comment",
        "--extra-index-url https://example.invalid/simple",
        "typer>=0.1",  # installed with the cli
        "surely-not-installed-pkg-xyz==1.0",
        "typer>=999",  # installed, wrong version
        "never-on-this-python ; python_version < '3.0'",  # marker false
    ]
    assert cli._unmet_requirements(lines) == ["surely-not-installed-pkg-xyz==1.0", "typer>=999"]


def test_nothing_missing_asks_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _torch(monkeypatch, present=True)
    calls = _pip_calls(monkeypatch)
    (tmp_path / "requirements.txt").write_text("typer>=0.1\n")
    cli._ensure_deps(tmp_path / "requirements.txt", yes=False, ref="cdfm-base:latest")
    assert calls == []


def test_yes_installs_torch_then_the_requirements(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _torch(monkeypatch, present=False)
    calls = _pip_calls(monkeypatch)
    req = tmp_path / "requirements.txt"
    req.write_text("surely-not-installed-pkg-xyz==1.0\n")
    cli._ensure_deps(req, yes=True, ref="cdfm-base:latest")
    assert [c[1:4] for c in calls] == [["-m", "pip", "install"]] * 2
    assert calls[0][-1] == cli._TORCH_SPEC
    assert calls[1][-2:] == ["-r", str(req)]


def test_no_terminal_stops_with_the_yes_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _torch(monkeypatch, present=False)
    calls = _pip_calls(monkeypatch)
    _tty(monkeypatch, False)
    with pytest.raises(typer.Exit) as exc:
        cli._ensure_deps(None, yes=False, ref="cdfm-base:latest")
    assert exc.value.exit_code == 1 and calls == []
    out = capsys.readouterr().out
    assert "pfnstudio run cdfm-base:latest -y" in out


def test_terminal_asks_and_a_no_installs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _torch(monkeypatch, present=False)
    calls = _pip_calls(monkeypatch)
    _tty(monkeypatch, True)
    asked: list[str] = []
    monkeypatch.setattr(typer, "confirm", lambda q, default=False: asked.append(q) or False)
    with pytest.raises(typer.Exit):
        cli._ensure_deps(None, yes=False, ref="cdfm-base:latest")
    assert asked and calls == []


def test_terminal_yes_installs(monkeypatch: pytest.MonkeyPatch) -> None:
    _torch(monkeypatch, present=False)
    calls = _pip_calls(monkeypatch)
    _tty(monkeypatch, True)
    monkeypatch.setattr(typer, "confirm", lambda q, default=False: default)
    cli._ensure_deps(None, yes=False, ref="cdfm-base:latest")
    assert len(calls) == 1 and calls[0][-1] == cli._TORCH_SPEC
