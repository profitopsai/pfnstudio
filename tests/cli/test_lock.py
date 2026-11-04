"""Tests for the model lock (packages/cli/pfnstudio/lock.py). Pure Python: no
network, no torch — a fake bundle + checkpoint on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pfnstudio.lock import (
    build_lock,
    digest_of,
    parse_model_ref,
    read_lock,
    sha256_files,
    write_lock,
)


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("timee-base", ("timee-base", "latest")),
        ("timee-base:latest", ("timee-base", "latest")),
        ("cdfm-base:v0.2", ("cdfm-base", "v0.2")),
        ("  t0-alpha-base  ", ("t0-alpha-base", "latest")),
        ("a_b.c:1", ("a_b.c", "1")),
    ],
)
def test_parse_model_ref_accepts(ref: str, expected: tuple[str, str]) -> None:
    assert parse_model_ref(ref) == expected


@pytest.mark.parametrize(
    "ref",
    ["", "   ", ":latest", "slug:", "slug:a:b", "Slug", "sl ug", "-lead", "slug:T AG", "slug/x"],
)
def test_parse_model_ref_rejects(ref: str) -> None:
    with pytest.raises(ValueError):
        parse_model_ref(ref)


def test_sha256_files_is_order_independent_and_content_sensitive(tmp_path: Path) -> None:
    (tmp_path / "adapters").mkdir()
    (tmp_path / "adapters" / "a.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "model.yaml").write_text("id: m\n", encoding="utf-8")
    forward = sha256_files(tmp_path, ["adapters/a.py", "model.yaml"])
    backward = sha256_files(tmp_path, ["model.yaml", "adapters/a.py"])
    assert forward == backward
    assert len(forward) == 64
    (tmp_path / "model.yaml").write_text("id: changed\n", encoding="utf-8")
    assert sha256_files(tmp_path, ["adapters/a.py", "model.yaml"]) != forward


def test_sha256_files_distinguishes_path_from_content(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"xy")
    (tmp_path / "b").write_bytes(b"xy")
    assert sha256_files(tmp_path, ["a"]) != sha256_files(tmp_path, ["b"])


def _fake_bundle(root: Path) -> Path:
    bundle = root / "bundle" / "timee"
    (bundle / "adapters").mkdir(parents=True)
    (bundle / "adapters" / "timee.py").write_text("def load_model():\n    pass\n", encoding="utf-8")
    (bundle / "requirements.txt").write_text("timee-upstream==1.0\n", encoding="utf-8")
    (bundle / "__pycache__").mkdir()
    (bundle / "__pycache__" / "timee.cpython-312.pyc").write_bytes(b"\x00byte-code")
    return bundle


def _fake_checkpoint(root: Path, name: str = "org--timee--main--0123456789abcdef") -> Path:
    ckpt = root / "checkpoints" / name
    ckpt.mkdir(parents=True)
    (ckpt / "model.pt").write_bytes(b"fake-weights-" + b"w" * 32)
    (ckpt / "topology.json").write_text('{"d_in": 4}\n', encoding="utf-8")
    (ckpt / ".locks").mkdir()
    (ckpt / ".locks" / "x.lock").write_bytes(b"")
    return ckpt


def _lock(root: Path, bundle: Path, ckpt: Path, **over: object) -> dict:
    kw: dict = dict(
        slug="timee-base",
        tag="latest",
        fm_project="timee",
        base_adapter="timee",
        bundle_dir=bundle,
        checkpoint_dir=ckpt,
        checkpoint_ref={"model.pt": "hf://org/timee/model.pt"},
    )
    kw.update(over)
    return build_lock(**kw)


def test_build_lock_shape_and_determinism(tmp_path: Path) -> None:
    bundle = _fake_bundle(tmp_path)
    ckpt = _fake_checkpoint(tmp_path)
    first = _lock(tmp_path, bundle, ckpt)
    second = _lock(tmp_path, bundle, ckpt)

    assert first["schema"] == 1
    assert first["slug"] == "timee-base" and first["tag"] == "latest"
    assert first["fmProject"] == "timee" and first["baseAdapter"] == "timee"
    # __pycache__ is not part of the served bundle; lock.json is not yet written.
    assert first["bundle"]["files"] == 2
    assert first["checkpoint"]["ref"] == {"model.pt": "hf://org/timee/model.pt"}
    assert first["checkpoint"]["revision"] == "main"
    assert first["requirements_sha256"] is not None
    assert first["digest"].startswith("sha256:") and len(first["digest"]) == 7 + 64
    assert first["created_at"].endswith("Z")

    assert first["digest"] == second["digest"] == digest_of(first) == digest_of(second)


def test_digest_ignores_created_at_but_tracks_inputs(tmp_path: Path) -> None:
    bundle = _fake_bundle(tmp_path)
    ckpt = _fake_checkpoint(tmp_path)
    lock = _lock(tmp_path, bundle, ckpt)
    base = digest_of(lock)

    assert digest_of({**lock, "created_at": "1999-01-01T00:00:00Z"}) == base
    assert digest_of({**lock, "tag": "v2"}) != base
    assert digest_of({**lock, "checkpoint": {**lock["checkpoint"], "sha256": "0" * 64}}) != base
    assert digest_of({**lock, "requirements_sha256": None}) != base

    (ckpt / "model.pt").write_bytes(b"other-weights")
    assert _lock(tmp_path, bundle, ckpt)["digest"] != base


def test_revision_absent_for_non_hub_cache_dirs(tmp_path: Path) -> None:
    bundle = _fake_bundle(tmp_path)
    for name in ("manifest--0123456789abcdef", "url--0123456789abcdef", "plain"):
        ckpt = _fake_checkpoint(tmp_path / name, name)
        assert _lock(tmp_path, bundle, ckpt)["checkpoint"]["revision"] is None


def test_write_read_round_trip_and_lock_excluded_from_bundle_hash(tmp_path: Path) -> None:
    bundle = _fake_bundle(tmp_path)
    ckpt = _fake_checkpoint(tmp_path)
    assert read_lock(bundle) is None

    lock = _lock(tmp_path, bundle, ckpt)
    path = write_lock(bundle, lock)
    assert path == bundle / "lock.json"
    assert json.loads(path.read_text(encoding="utf-8")) == lock
    assert read_lock(bundle) == lock

    # Re-locking with lock.json present must not change the bundle hash or digest.
    again = _lock(tmp_path, bundle, ckpt)
    assert again["bundle"] == lock["bundle"]
    assert again["digest"] == lock["digest"]
