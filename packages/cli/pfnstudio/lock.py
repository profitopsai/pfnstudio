"""The model lock — pins what `pfnstudio run <slug>[:tag]` actually ran.

A base model is three resolvable things: a marketplace listing (slug:tag), the
adapter bundle the marketplace serves, and the checkpoint the listing points at.
The lock records a content hash of each and folds them into ONE digest
(``sha256:<hex>``) so two machines that resolved the same inputs can prove it
by comparing a single line. ``lock.json`` sits next to the bundle; the server
stores the same object under the base run's ``hyperparams.lock`` and the
container image carries the digest as a label — no database change needed.

Pure stdlib, no network, no torch: the server and tests import this too.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

LOCK_FILENAME = "lock.json"
DEFAULT_TAG = "latest"

# A slug/tag is a DNS-label-ish token: lowercase alphanumerics with . _ - inside.
_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def parse_model_ref(ref: str) -> tuple[str, str]:
    """``"slug"`` or ``"slug:tag"`` → ``(slug, tag)``; the tag defaults to
    ``"latest"``. Rejects empty parts, a second colon and non-slug characters."""
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("model ref is empty — expected slug or slug:tag")
    ref = ref.strip()
    slug, sep, tag = ref.partition(":")
    if sep and ":" in tag:
        raise ValueError(f"invalid model ref {ref!r}: at most one ':'")
    tag = tag if sep else DEFAULT_TAG
    for label, value in (("slug", slug), ("tag", tag)):
        if not value:
            raise ValueError(f"invalid model ref {ref!r}: empty {label}")
        if not _TOKEN.match(value):
            raise ValueError(
                f"invalid model ref {ref!r}: {label} {value!r} must be lowercase "
                "letters, digits, '.', '_' or '-' and start with a letter or digit"
            )
    return slug, tag


def sha256_files(root: Path, rel_paths: list[str]) -> str:
    """One sha256 over the sorted ``(relative path, bytes)`` pairs under ``root``.
    Path separators are normalized to '/', so the hash is OS-independent and
    does not depend on the order ``rel_paths`` was given in."""
    h = hashlib.sha256()
    for rel in sorted(Path(p).as_posix() for p in rel_paths):
        data = (root / rel).read_bytes()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def _bundle_files(bundle_dir: Path) -> list[str]:
    """Everything the bundle endpoint serves: every regular file under the
    bundle dir except the lock itself and ``__pycache__`` byte-code."""
    out: list[str] = []
    for p in bundle_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(bundle_dir)
        if rel.as_posix() == LOCK_FILENAME or "__pycache__" in rel.parts:
            continue
        out.append(rel.as_posix())
    return sorted(out)


def _checkpoint_files(checkpoint_dir: Path) -> list[str]:
    """The staged weight files: every regular file under the resolved checkpoint
    dir. Dot-prefixed entries are the hub's own bookkeeping, not weights."""
    return sorted(
        p.relative_to(checkpoint_dir).as_posix()
        for p in checkpoint_dir.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(checkpoint_dir).parts)
    )


def _revision_from_cache_dir(checkpoint_dir: Path) -> str | None:
    """``resolve_checkpoint`` names its cache dirs ``<owner>--<repo>--<rev>--<hash>``
    (``manifest--<hash>`` / ``url--<hash>`` when the source is not a hub repo).
    Returns the ``<rev>`` segment when the name has that shape, else None."""
    parts = checkpoint_dir.name.split("--")
    if len(parts) < 4 or parts[0] in ("manifest", "url"):
        return None
    return parts[-2] or None


def build_lock(
    *,
    slug: str,
    tag: str,
    fm_project: str,
    base_adapter: str | None,
    bundle_dir: Path,
    checkpoint_dir: Path,
    checkpoint_ref: object,
) -> dict:
    """Hash the resolved bundle + checkpoint and assemble the lock document
    (``digest`` included). ``checkpoint_ref`` is recorded exactly as the listing
    gave it (a string URI or a filename→URI manifest)."""
    bundle_dir = Path(bundle_dir)
    checkpoint_dir = Path(checkpoint_dir)
    bundle_files = _bundle_files(bundle_dir)
    ckpt_files = _checkpoint_files(checkpoint_dir)
    req = bundle_dir / "requirements.txt"
    lock: dict = {
        "schema": 1,
        "slug": slug,
        "tag": tag,
        "fmProject": fm_project,
        "baseAdapter": base_adapter,
        "bundle": {"sha256": sha256_files(bundle_dir, bundle_files), "files": len(bundle_files)},
        "checkpoint": {
            "ref": checkpoint_ref,
            "sha256": sha256_files(checkpoint_dir, ckpt_files),
            "revision": _revision_from_cache_dir(checkpoint_dir),
        },
        "requirements_sha256": hashlib.sha256(req.read_bytes()).hexdigest()
        if req.is_file()
        else None,
    }
    lock["digest"] = digest_of(lock)
    lock["created_at"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    return lock


def digest_of(lock: dict) -> str:
    """``sha256:<hex>`` over the canonical JSON (sorted keys, no whitespace) of
    the identity fields only — never ``created_at``, so re-running the same
    inputs yields the same digest."""
    identity = {
        "slug": lock.get("slug"),
        "tag": lock.get("tag"),
        "fmProject": lock.get("fmProject"),
        "baseAdapter": lock.get("baseAdapter"),
        "bundle": {"sha256": (lock.get("bundle") or {}).get("sha256")},
        "checkpoint": {"sha256": (lock.get("checkpoint") or {}).get("sha256")},
        "requirements_sha256": lock.get("requirements_sha256"),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_lock(bundle_dir: Path, lock: dict) -> Path:
    """Write ``<bundle_dir>/lock.json`` (pretty, key-sorted) and return its path."""
    path = Path(bundle_dir) / LOCK_FILENAME
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_lock(bundle_dir: Path) -> dict | None:
    """The lock next to a bundle, or None when there is none (or it is not a
    JSON object)."""
    path = Path(bundle_dir) / LOCK_FILENAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None
