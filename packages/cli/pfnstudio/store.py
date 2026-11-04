"""The local model store — what `pull / ls / rm / show` read and write.

Layout (both roots are the ones `run` already fills; same env overrides):

    <store_root>/<fmProject>/              adapter bundle (adapters/, *.yaml, requirements.txt)
        lock.json                          written by the lifecycle (pfnstudio.lock)
        listing.json                       marketplace listing of the slug pulled last
        listings/<slug>.json               one per slug ever pulled into this bundle
    <checkpoint_root>/<repo>--<rev>--<hash>/   weights, named by serving.hf.resolve_checkpoint

A bundle dir is keyed by ``fmProject``, and several listings can share one
(the t0 alpha/beta/gamma listings all run the ``t0-alpha`` adapter), so a
"model" here is one (bundle dir, slug) pair: the slug supplies the checkpoint,
the bundle supplies the adapter. Pure stdlib — no torch, no network — so
``pfnstudio ls`` is instant.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

LISTING_FILE = "listing.json"
LISTINGS_DIR = "listings"
LOCK_FILE = "lock.json"


# ── roots ────────────────────────────────────────────────────────────────────
def store_root() -> Path:
    """Where adapter bundles live (mirrors cli._fetch_basemodel_bundle)."""
    return Path(
        os.environ.get(
            "PFNSTUDIO_BASEMODELS_CACHE",
            str(Path.home() / ".cache" / "pfnstudio" / "basemodels"),
        )
    )


def checkpoint_root() -> Path:
    """Where checkpoints live (mirrors serving.hf.resolve_checkpoint)."""
    return Path(
        os.environ.get(
            "PFNSTUDIO_CHECKPOINT_CACHE",
            str(Path.home() / ".cache" / "pfnstudio" / "checkpoints"),
        )
    )


def safe_fm_project(fm_project: str) -> str:
    """The on-disk name of a bundle — the same sanitization the puller applies."""
    safe = "".join(ch for ch in str(fm_project) if ch.isalnum() or ch in "._-")
    if not safe or safe in (".", "..") or ".." in safe:
        raise ValueError(f"invalid fmProject id: {fm_project!r}")
    return safe


def bundle_dir(fm_project: str) -> Path:
    return store_root() / safe_fm_project(fm_project)


# ── small helpers ────────────────────────────────────────────────────────────
def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def dir_bytes(path: Path | None) -> int:
    if path is None or not path.is_dir():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def human_bytes(n: int | None) -> str:
    if not n:
        return "-"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def parse_ref(ref: str) -> tuple[str, str]:
    """``slug[:tag]`` → (slug, tag); a bare slug is ``slug:latest``, as in
    Ollama. One grammar for the whole lifecycle: pfnstudio.lock owns it."""
    from .lock import parse_model_ref

    slug, tag = parse_model_ref(ref)
    return str(slug), str(tag)


def read_lock(bdir: Path) -> dict | None:
    try:
        from .lock import read_lock as _read_lock

        lock = _read_lock(bdir)
        return lock if isinstance(lock, dict) else None
    except ImportError:
        return read_json(bdir / LOCK_FILE)


# ── listing index ────────────────────────────────────────────────────────────
def save_listing(bdir: Path, listing: dict) -> Path:
    """Index the marketplace listing beside the bundle so ls/show work offline.
    ``listing.json`` is the slug pulled last; ``listings/<slug>.json`` keeps
    every slug that shares this bundle."""
    bdir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(listing, indent=2, sort_keys=True) + "\n"
    (bdir / LISTING_FILE).write_text(text, encoding="utf-8")
    slug = listing.get("slug")
    if isinstance(slug, str) and slug:
        per = bdir / LISTINGS_DIR
        per.mkdir(exist_ok=True)
        (per / f"{safe_fm_project(slug)}.json").write_text(text, encoding="utf-8")
    return bdir / LISTING_FILE


def read_listings(bdir: Path) -> dict[str, dict]:
    """Every indexed listing of a bundle, keyed by slug."""
    out: dict[str, dict] = {}
    per = bdir / LISTINGS_DIR
    if per.is_dir():
        for p in sorted(per.glob("*.json")):
            d = read_json(p)
            if d and isinstance(d.get("slug"), str):
                out[d["slug"]] = d
    top = read_json(bdir / LISTING_FILE)
    if top and isinstance(top.get("slug"), str):
        out.setdefault(top["slug"], top)
    return out


# ── checkpoint location (mirrors serving.hf.resolve_checkpoint's cache key) ──
def _parse_hf_uri(uri: str) -> tuple[str, str, str] | None:
    s = uri.strip()
    if s.startswith("hf://"):
        parts = s[len("hf://") :].split("/", 2)
        if len(parts) == 3 and all(parts):
            return f"{parts[0]}/{parts[1]}", parts[2], "main"
        return None
    if s.startswith("https://huggingface.co/"):
        parts = s[len("https://huggingface.co/") :].split("/")
        if len(parts) >= 5 and parts[2] in {"blob", "tree", "resolve"} and parts[0] and parts[1]:
            path = "/".join(parts[4:])
            return (f"{parts[0]}/{parts[1]}", path, parts[3]) if path else None
    return None


def _github_blob_to_raw(url: str) -> str:
    import re

    m = re.match(r"^https://github\.com/([^/]+)/([^/]+)/blob/(.+)$", url)
    return (
        f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3).split('?', 1)[0]}"
        if m
        else url
    )


def checkpoint_cache_dir(ref: Any, root: Path | None = None) -> Path | None:
    """The directory ``resolve_checkpoint(ref)`` materializes into, computed
    WITHOUT fetching — same naming rule, so ls/rm can find weights offline.
    ``ref`` is the listing's checkpointRef (manifest dict, or a URI/path string)."""
    root = root or checkpoint_root()
    if isinstance(ref, str) and ref.startswith("{"):
        try:
            ref = json.loads(ref)
        except ValueError:
            return None
    if isinstance(ref, dict):
        if not ref or not all(isinstance(k, str) and isinstance(v, str) for k, v in ref.items()):
            return None
        digest = hashlib.sha256(json.dumps(sorted(ref.items())).encode("utf-8")).hexdigest()[:16]
        parsed = _parse_hf_uri(next(iter(ref.values())))
        if parsed:
            repo_id, _path, rev = parsed
            return root / f"{repo_id.replace('/', '--')}--{rev}--{digest}"
        return root / f"manifest--{digest}"
    if not isinstance(ref, str) or not ref:
        return None
    parsed = _parse_hf_uri(ref)
    if parsed:
        repo_id, path, rev = parsed
        return root / f"{repo_id.replace('/', '--')}--{rev}--{path.replace('/', '--')}"
    if ref.startswith(("http://", "https://")):
        raw = _github_blob_to_raw(ref)
        return root / f"url--{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"
    p = Path(ref).expanduser()
    return p if p.is_absolute() else p.resolve()


def locate_checkpoint(lock: dict | None, listing: dict | None) -> Path | None:
    """Existing checkpoint dir for a model, or None. Order: a dir the lock
    records → the dir its checkpointRef maps to → None."""
    ck = (lock or {}).get("checkpoint")
    if isinstance(ck, dict):
        for key in ("dir", "path"):
            v = ck.get(key)
            if isinstance(v, str) and v and Path(v).is_dir():
                return Path(v)
        ref = ck.get("ref")
        d = checkpoint_cache_dir(ref) if ref else None
        if d is not None and d.is_dir():
            return d
    ref = (listing or {}).get("checkpointRef")
    d = checkpoint_cache_dir(ref) if ref else None
    return d if d is not None and d.is_dir() else None


# ── the store ────────────────────────────────────────────────────────────────
def _is_bundle(d: Path) -> bool:
    return d.is_dir() and any(
        (d / name).exists()
        for name in ("adapters", LOCK_FILE, LISTING_FILE, LISTINGS_DIR, "pfnstudio.yaml")
    )


def list_models() -> list[dict]:
    """One row per (bundle, slug). Keys: slug, tag, fmProject, baseAdapter,
    digest, bundle_dir, bundle_bytes, checkpoint_dir, checkpoint_bytes, tier,
    title, listing (the indexed listing or None), lock (or None)."""
    root = store_root()
    if not root.is_dir():
        return []
    rows: list[dict] = []
    for bdir in sorted(p for p in root.iterdir() if _is_bundle(p)):
        lock = read_lock(bdir)
        listings = read_listings(bdir)
        bbytes = dir_bytes(bdir)
        slugs = list(listings)
        lock_slug = (lock or {}).get("slug")
        if isinstance(lock_slug, str) and lock_slug and lock_slug not in slugs:
            slugs.insert(0, lock_slug)
        for slug in slugs or [bdir.name]:
            listing = listings.get(slug)
            row_lock = lock if lock and (lock.get("slug") in (None, slug)) else None
            ckpt = locate_checkpoint(row_lock, listing)
            rows.append(
                {
                    "slug": slug,
                    "tag": (row_lock or {}).get("tag"),
                    "fmProject": bdir.name,
                    "baseAdapter": (row_lock or {}).get("baseAdapter")
                    or (listing or {}).get("baseAdapter"),
                    "digest": (row_lock or {}).get("digest"),
                    "bundle_dir": bdir,
                    "bundle_bytes": bbytes,
                    "checkpoint_dir": ckpt,
                    "checkpoint_bytes": dir_bytes(ckpt),
                    "tier": (listing or {}).get("tier"),
                    "title": (listing or {}).get("title"),
                    "listing": listing,
                    "lock": row_lock,
                }
            )
    return rows


def find_model(ref: str) -> dict | None:
    """Row for ``slug[:tag]`` or a bare fmProject; None when not in the store."""
    slug, tag = parse_ref(ref)
    rows = list_models()
    for r in rows:
        if r["slug"] == slug and r["tag"] in (None, tag):
            return r
    by_fm = [r for r in rows if r["fmProject"] == slug]
    return by_fm[0] if len(by_fm) == 1 else None


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def remove(ref: str) -> dict:
    """Remove ``slug[:tag]`` (or a whole fmProject bundle by its id).

    - A slug: drops its indexed listing; the bundle dir goes too unless another
      slug still shares it. Its checkpoint dir goes only when no OTHER model in
      the store resolves to the same dir (shared weights are never deleted).
    - An fmProject id (no slug matches): the bundle dir and every checkpoint
      owned by its slugs alone.
    Only paths under the two store roots are ever deleted. Raises KeyError when
    nothing matches. Returns {removed_bundle, kept_bundle, removed_checkpoints,
    kept_checkpoints (shared), slugs}.
    """
    slug, tag = parse_ref(ref)
    rows = list_models()
    targets = [r for r in rows if r["slug"] == slug and r["tag"] in (None, tag)]
    whole_bundle = False
    if not targets:
        targets = [r for r in rows if r["fmProject"] == slug]
        whole_bundle = bool(targets)
    if not targets:
        raise KeyError(ref)
    bdir: Path = targets[0]["bundle_dir"]
    target_slugs = {r["slug"] for r in targets}
    others = [r for r in rows if r["slug"] not in target_slugs or r["bundle_dir"] != bdir]
    shared_ckpts = {str(r["checkpoint_dir"]) for r in others if r["checkpoint_dir"]}

    removed_ckpts: list[Path] = []
    kept_ckpts: list[Path] = []
    for r in targets:
        ck = r["checkpoint_dir"]
        if ck is None or ck in removed_ckpts:
            continue
        if str(ck) in shared_ckpts or not _inside(ck, checkpoint_root()):
            kept_ckpts.append(ck)
        else:
            shutil.rmtree(ck)
            removed_ckpts.append(ck)

    siblings_left = any(r["bundle_dir"] == bdir for r in others)
    if not _inside(bdir, store_root()):
        raise PermissionError(f"refusing to delete outside the store: {bdir}")
    if whole_bundle or not siblings_left:
        shutil.rmtree(bdir)
        removed_bundle: Path | None = bdir
    else:
        removed_bundle = None
        for s in target_slugs:  # keep the bundle for its siblings; forget these slugs
            (bdir / LISTINGS_DIR / f"{safe_fm_project(s)}.json").unlink(missing_ok=True)
            top = read_json(bdir / LISTING_FILE)
            if top and top.get("slug") == s:
                (bdir / LISTING_FILE).unlink(missing_ok=True)
    return {
        "slugs": sorted(target_slugs),
        "removed_bundle": removed_bundle,
        "kept_bundle": None if removed_bundle else bdir,
        "removed_checkpoints": removed_ckpts,
        "kept_checkpoints": kept_ckpts,
    }
