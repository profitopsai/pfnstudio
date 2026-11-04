"""Filesystem-only tests for pfnstudio.store (the local model store behind
pull / ls / rm / show). No network, no torch, no daemon: the two cache roots
are redirected into tmp_path through the same env overrides the CLI honors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pfnstudio import store

REF_A = {
    "model.pt": "hf://<your-org>/<repo-name>/v1/model.pt",
    "topology.json": "hf://<your-org>/<repo-name>/v1/topology.json",
}
REF_B = {"model.safetensors": "hf://<your-org>/<other-repo>/model.safetensors"}


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    b, c = tmp_path / "basemodels", tmp_path / "checkpoints"
    monkeypatch.setenv("PFNSTUDIO_BASEMODELS_CACHE", str(b))
    monkeypatch.setenv("PFNSTUDIO_CHECKPOINT_CACHE", str(c))
    return b, c


def _listing(slug: str, fm: str, ref: dict, **extra) -> dict:
    return {
        "type": "basemodel",
        "slug": slug,
        "fmProject": fm,
        "baseAdapter": fm,
        "title": f"{slug} title",
        "checkpointRef": ref,
        **extra,
    }


def _bundle(
    slug: str, fm: str, ref: dict, *, lock: bool = True, weights: bool = True, **extra
) -> Path:
    """Fabricate what `pull` leaves behind: adapters/, listing.json, lock.json, weights."""
    bdir = store.bundle_dir(fm)
    (bdir / "adapters").mkdir(parents=True, exist_ok=True)
    (bdir / "adapters" / f"{fm}_adapter.py").write_text("# adapter\n" * 10, encoding="utf-8")
    store.save_listing(bdir, _listing(slug, fm, ref, **extra))
    if lock:
        (bdir / "lock.json").write_text(
            json.dumps(
                {
                    "slug": slug,
                    "tag": "latest",
                    "fmProject": fm,
                    "baseAdapter": fm,
                    "bundle": {"sha256": "b" * 64},
                    "checkpoint": {"sha256": "c" * 64},
                    "digest": "sha256:" + "d" * 64,
                }
            ),
            encoding="utf-8",
        )
    if weights:
        ck = store.checkpoint_cache_dir(ref)
        ck.mkdir(parents=True, exist_ok=True)
        (ck / "model.pt").write_bytes(b"w" * 2048)
    return bdir


def test_roots_follow_env(roots):
    b, c = roots
    assert store.store_root() == b and store.checkpoint_root() == c


def test_empty_store(roots):
    assert store.list_models() == []
    assert store.find_model("nothing") is None


def test_list_models_reads_lock_listing_and_checkpoint(roots):
    bdir = _bundle("alpha-base", "alpha", REF_A, tier="community")
    rows = store.list_models()
    assert len(rows) == 1
    r = rows[0]
    assert r["slug"] == "alpha-base" and r["tag"] == "latest" and r["fmProject"] == "alpha"
    assert r["baseAdapter"] == "alpha" and r["digest"] == "sha256:" + "d" * 64
    assert r["tier"] == "community" and r["title"] == "alpha-base title"
    assert r["bundle_dir"] == bdir and r["bundle_bytes"] > 0
    assert r["checkpoint_dir"] == store.checkpoint_cache_dir(REF_A)
    assert r["checkpoint_bytes"] == 2048
    assert (bdir / "listing.json").is_file() and (bdir / "listings" / "alpha-base.json").is_file()


def test_list_models_without_lock_or_weights(roots):
    _bundle("beta-base", "beta", REF_B, lock=False, weights=False)
    (r,) = store.list_models()
    assert r["digest"] is None and r["tag"] is None
    assert r["checkpoint_dir"] is None and r["checkpoint_bytes"] == 0
    assert r["tier"] is None


def test_lock_checkpoint_dir_is_honored(roots, tmp_path):
    bdir = _bundle("alpha-base", "alpha", REF_A, weights=False)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "model.pt").write_bytes(b"x")
    lock = json.loads((bdir / "lock.json").read_text())
    lock["checkpoint"]["dir"] = str(outside)
    (bdir / "lock.json").write_text(json.dumps(lock))
    (r,) = store.list_models()
    assert r["checkpoint_dir"] == outside
    # Outside the checkpoint root → never deleted, even when unshared.
    res = store.remove("alpha-base")
    assert res["removed_checkpoints"] == [] and res["kept_checkpoints"] == [outside]
    assert outside.is_dir() and res["removed_bundle"] == bdir and not bdir.exists()


def test_find_model_by_slug_tag_and_fm_project(roots):
    _bundle("alpha-base", "alpha", REF_A)
    assert store.find_model("alpha-base")["slug"] == "alpha-base"
    assert store.find_model("alpha-base:latest")["slug"] == "alpha-base"
    assert store.find_model("alpha-base:v9") is None
    assert store.find_model("alpha")["slug"] == "alpha-base"  # fmProject fallback


def test_remove_unknown_raises(roots):
    with pytest.raises(KeyError):
        store.remove("ghost")


def test_remove_deletes_bundle_and_owned_checkpoint(roots):
    bdir = _bundle("alpha-base", "alpha", REF_A)
    ck = store.checkpoint_cache_dir(REF_A)
    res = store.remove("alpha-base")
    assert res["removed_bundle"] == bdir and res["removed_checkpoints"] == [ck]
    assert not bdir.exists() and not ck.exists()
    assert store.list_models() == []


def test_remove_refuses_shared_checkpoint(roots):
    """Two models (different bundles) resolve to the same weights: removing one
    keeps the checkpoint; removing the last owner deletes it."""
    a = _bundle("alpha-base", "alpha", REF_A)
    b = _bundle("alpha-mirror", "alpha-mirror", REF_A)
    ck = store.checkpoint_cache_dir(REF_A)
    res = store.remove("alpha-base")
    assert res["removed_bundle"] == a and not a.exists()
    assert res["removed_checkpoints"] == [] and res["kept_checkpoints"] == [ck]
    assert ck.is_dir() and b.is_dir()
    res = store.remove("alpha-mirror")
    assert res["removed_checkpoints"] == [ck] and not ck.exists() and not b.exists()


def test_remove_keeps_bundle_shared_by_sibling_slug(roots):
    """Several listings share one fmProject (t0 alpha/beta/gamma): removing one
    slug forgets its listing + weights but keeps the bundle for the others."""
    bdir = _bundle("t0-alpha", "t0-alpha", REF_A, lock=False)
    _bundle("t0-beta", "t0-alpha", REF_B, lock=False)
    assert sorted(r["slug"] for r in store.list_models()) == ["t0-alpha", "t0-beta"]
    res = store.remove("t0-beta")
    assert res["removed_bundle"] is None and res["kept_bundle"] == bdir and bdir.is_dir()
    assert res["removed_checkpoints"] == [store.checkpoint_cache_dir(REF_B)]
    assert store.checkpoint_cache_dir(REF_A).is_dir()
    assert [r["slug"] for r in store.list_models()] == ["t0-alpha"]
    assert not (bdir / "listings" / "t0-beta.json").exists()
    # Removing by fmProject id drops the whole bundle and the remaining weights.
    res = store.remove("t0-alpha")
    assert res["removed_bundle"] == bdir and not bdir.exists()
    assert res["removed_checkpoints"] == [store.checkpoint_cache_dir(REF_A)]


def test_parse_ref():
    # A bare slug is slug:latest (Ollama semantics; the lock module owns the grammar).
    assert store.parse_ref("alpha-base") == ("alpha-base", "latest")
    assert store.parse_ref("alpha-base:v2") == ("alpha-base", "v2")


def test_checkpoint_cache_dir_naming(roots):
    _, c = roots
    assert store.checkpoint_cache_dir(REF_A).parent == c
    assert store.checkpoint_cache_dir(REF_A).name.startswith("<your-org>--<repo-name>--main--")
    assert store.checkpoint_cache_dir(json.dumps(REF_A)) == store.checkpoint_cache_dir(REF_A)
    assert store.checkpoint_cache_dir(
        dict(reversed(list(REF_A.items())))
    ) == store.checkpoint_cache_dir(REF_A)
    assert (
        store.checkpoint_cache_dir("hf://<your-org>/<repo-name>/w/model.pt").name
        == "<your-org>--<repo-name>--main--w--model.pt"
    )
    assert store.checkpoint_cache_dir(
        "https://github.com/<your-org>/<repo-name>/blob/main/m.pt"
    ).name.startswith("url--")
    assert store.checkpoint_cache_dir({"m": "https://example.invalid/m.pt"}).name.startswith(
        "manifest--"
    )
    assert store.checkpoint_cache_dir("") is None and store.checkpoint_cache_dir({}) is None


def test_checkpoint_cache_dir_matches_resolve_checkpoint(roots, tmp_path):
    """Parity with serving.hf.resolve_checkpoint on its local-path manifest
    branch (a copy, no download). Skipped where pfnstudio_core is not installed."""
    hf = pytest.importorskip("pfnstudio_core.serving.hf")
    src = tmp_path / "src.pt"
    src.write_bytes(b"w")
    manifest = {"model.pt": str(src)}
    resolved = hf.resolve_checkpoint(json.dumps(manifest))
    assert resolved == store.checkpoint_cache_dir(manifest)
    assert (resolved / "model.pt").is_file()


def test_human_bytes():
    assert store.human_bytes(0) == "-" and store.human_bytes(512) == "512 B"
    assert store.human_bytes(3 * 1024 * 1024) == "3.0 MB"
