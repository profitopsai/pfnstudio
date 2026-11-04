"""Unit tests for the local daemon + client — no server, no model, no network."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from pfnstudio import client, daemon

# ── mount paths / slugs ───────────────────────────────────────────────────────


def test_mount_path_builder() -> None:
    assert daemon.mount_path("cdfm-base") == "/models/cdfm-base"
    assert daemon.mount_path("timee.base_v2") == "/models/timee.base_v2"


@pytest.mark.parametrize("bad", ["", "..", "a/b", "a b", "../x", "x?y"])
def test_mount_path_rejects_unsafe_slugs(bad: str) -> None:
    with pytest.raises(ValueError):
        daemon.mount_path(bad)


# ── ref parsing (lock.parse_model_ref when present, fallback otherwise) ───────


def test_parse_ref_fallback_without_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pfnstudio.lock", None)  # force ImportError
    assert daemon.parse_ref("cdfm-base") == ("cdfm-base", None)
    assert daemon.parse_ref("cdfm-base:v1.2") == ("cdfm-base", "v1.2")
    with pytest.raises(ValueError):
        daemon.parse_ref("   ")


def test_parse_ref_delegates_to_lock_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("pfnstudio.lock")
    fake.parse_model_ref = lambda ref: ("from-lock", "tagged")  # type: ignore[attr-defined]
    fake.read_lock = lambda d: {"digest": "sha256:abc"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pfnstudio.lock", fake)
    assert daemon.parse_ref("anything:x") == ("from-lock", "tagged")
    assert daemon.read_digest(Path("/nonexistent")) == "sha256:abc"


def test_read_digest_degrades(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "pfnstudio.lock", None)
    assert daemon.read_digest(tmp_path) is None
    assert daemon.read_digest(None) is None
    fake = types.ModuleType("pfnstudio.lock")
    fake.read_lock = lambda d: None  # no lock.json in the bundle  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pfnstudio.lock", fake)
    assert daemon.read_digest(tmp_path) is None


# ── tier + manifest ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [("native", "native"), ("Adapter", "adapter"), (None, None), ("", None), ("container", None)],
)
def test_normalize_tier(raw, expected) -> None:
    assert daemon.normalize_tier(raw) == expected


def test_run_manifest_matches_run_command_trick(tmp_path: Path) -> None:
    p = daemon.write_run_manifest(tmp_path, "cdfm-base", "cdfm")
    assert p == tmp_path / "runs" / "cdfm-base.yaml"
    text = p.read_text(encoding="utf-8")
    assert "  adapter: cdfm\n" in text and "  baseAdapter: cdfm\n" in text
    import yaml

    data = yaml.safe_load(text)
    assert data["hyperparams"]["adapter"] == "cdfm" == data["hyperparams"]["baseAdapter"]


# ── registry → /ps shape ──────────────────────────────────────────────────────


def _model(slug: str, loaded_at: str, **kw) -> daemon.LoadedModel:
    return daemon.LoadedModel(
        slug=slug,
        fm_project="cdfm",
        adapter="cdfm",
        checkpoint_dir=Path("/ckpt"),
        bundle_dir=Path("/bundle"),
        manifest_path=Path("/state/runs") / f"{slug}.yaml",
        loaded_at=loaded_at,
        app=object(),
        **kw,
    )


def test_registry_ps_shape_and_order() -> None:
    reg = daemon.Registry()
    reg.add(_model("b-model", "2026-11-03T10:00:01Z", digest="sha256:bb", tier="adapter", tag="v2"))
    reg.add(_model("a-model", "2026-11-03T10:00:00Z"))
    rows = reg.ps()
    assert [r["slug"] for r in rows] == ["a-model", "b-model"]
    row = rows[1]
    assert row["fmProject"] == "cdfm" and row["digest"] == "sha256:bb"
    assert row["tier"] == "adapter" and row["tag"] == "v2"
    assert row["path"] == "/models/b-model" and row["loaded_at"] == "2026-11-03T10:00:01Z"
    assert "app" not in row  # the ASGI app never leaks into /ps
    assert reg.remove("a-model") is not None and reg.remove("a-model") is None
    assert len(reg) == 1 and reg.get("b-model") is not None


# ── host app routes (in-process ASGI; no server, no model) ───────────────────


def test_host_app_management_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("PFNSTUDIO_SERVE_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "pfnstudio.lock", None)  # test the daemon, not lock's grammar
    reg = daemon.Registry()
    reg.add(_model("a-model", "2026-11-03T10:00:00Z"))
    app = daemon.build_host_app(reg, tmp_path)
    with TestClient(app) as c:
        assert c.get("/healthz").json() == {"status": "ready", "models": 1}
        assert [r["slug"] for r in c.get("/ps").json()] == ["a-model"]
        assert c.post("/load", json={}).status_code == 400
        assert c.post("/load", json={"ref": "../evil"}).status_code == 400
        assert c.post("/unload", json={"slug": "missing"}).status_code == 404
        # an already-loaded slug is answered from the registry — nothing resolved
        monkeypatch.setattr(daemon, "load_model", lambda *a, **k: pytest.fail("must not resolve"))
        out = c.post("/load", json={"ref": "a-model:any"}).json()
        assert out["already_loaded"] is True and out["slug"] == "a-model"


def test_host_app_requires_token_when_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("PFNSTUDIO_SERVE_TOKEN", "s3cret")
    app = daemon.build_host_app(daemon.Registry(), tmp_path)
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200  # liveness stays open
        assert c.get("/ps").status_code == 401
        assert c.get("/ps", headers={"Authorization": "Bearer s3cret"}).json() == []


# ── client (pure helpers only) ────────────────────────────────────────────────


def test_client_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PFNSTUDIO_HOST", raising=False)
    assert client.base_url() == "http://127.0.0.1:11440"
    monkeypatch.setenv("PFNSTUDIO_HOST", "127.0.0.1:12345")
    assert client.base_url() == "http://127.0.0.1:12345"
    monkeypatch.setenv("PFNSTUDIO_HOST", "https://daemon.example.internal/")
    assert client.base_url() == "https://daemon.example.internal"


def test_client_bearer_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PFNSTUDIO_SERVE_TOKEN", raising=False)
    assert client._headers() == {}
    monkeypatch.setenv("PFNSTUDIO_SERVE_TOKEN", "tok")
    assert client._headers() == {"Authorization": "Bearer tok"}


def test_ensure_running_refuses_to_spawn_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFNSTUDIO_HOST", "http://daemon.example.internal:11440")
    monkeypatch.setattr(client, "is_running", lambda: False)
    monkeypatch.setattr(client, "_spawn_daemon", lambda *a: pytest.fail("must not spawn"))
    with pytest.raises(RuntimeError):
        client.ensure_running()
