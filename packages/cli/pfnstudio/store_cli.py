"""``pfnstudio pull / ls / ps / rm / show`` — thin clients of the local model
store (pfnstudio.store) and the daemon (pfnstudio.client), à la ollama.

Registration (cli.py, right after ``app = typer.Typer(...)``)::

    from .store_cli import store_commands
    store_commands(app)

The verbs land on the ROOT app (``pfnstudio pull t0-alpha``), not under a
sub-app. ``pull``, ``ls`` and ``rm`` work without the daemon; ``ps`` and the
loaded? column need it; ``show`` asks the daemon for capabilities when it runs
and otherwise takes the same in-process path ``run`` uses. ``rm`` is the only
destructive verb and confirms before deleting weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import store

console = Console()
_err = Console(stderr=True)

DEFAULT_API = "https://cloud.pfnstudio.com/api"


# ── daemon (pfnstudio.client) — optional at every call site ─────────────────
def _client() -> Any | None:
    """The daemon client module, or None when it is absent or not running."""
    try:
        from . import client
    except ImportError:
        return None
    try:
        return client if client.is_running() else None
    except Exception:
        return None


def _row_slug(row: dict) -> str | None:
    for k in ("slug", "model", "name"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v.split(":", 1)[0]
    return None


def _loaded() -> set[str] | None:
    """Slugs loaded in the daemon; None when no daemon answers."""
    c = _client()
    if c is None:
        return None
    try:
        return {s for s in (_row_slug(r) for r in c.ps() if isinstance(r, dict)) if s}
    except Exception:
        return None


# ── marketplace ──────────────────────────────────────────────────────────────
def _fetch_listing(api_url: str, slug: str) -> dict:
    import requests

    base = api_url.rstrip("/")
    try:
        resp = requests.get(f"{base}/marketplace/listings", timeout=20)
        resp.raise_for_status()
        listings = resp.json()
    except Exception as e:
        _err.print(f"[red]Couldn't reach the marketplace at {base}/marketplace/listings: {e}[/red]")
        raise typer.Exit(code=1) from e
    base_models = [
        x
        for x in (listings if isinstance(listings, list) else [])
        if isinstance(x, dict) and x.get("type") == "basemodel"
    ]
    listing = next((x for x in base_models if x.get("slug") == slug), None)
    if listing is None:
        _err.print(f"[red]No base model with slug '{slug}'.[/red]")
        if base_models:
            _err.print(
                "[dim]Available:[/dim] "
                + ", ".join(sorted(str(x.get("slug")) for x in base_models))
            )
        raise typer.Exit(code=1)
    return listing


def _short(digest: Any) -> str:
    s = str(digest or "")
    s = s.split(":", 1)[1] if s.startswith("sha256:") else s
    return s[:12] if s else "-"


def _capabilities(row: dict) -> dict:
    """The adapter's capabilities() — from the daemon when it runs, else the
    in-process path `run` takes (run_inference with {"__capabilities__": True}).
    Returns {"capabilities": {...}} or {"error": "..."}; never hand-written."""
    c = _client()
    if c is not None and hasattr(c, "capabilities"):
        try:
            out = c.capabilities(row["slug"])
            caps = (out.get("result") or out).get("capabilities") if isinstance(out, dict) else None
            if isinstance(caps, dict):
                return {"capabilities": caps, "source": "daemon"}
        except Exception as e:
            _err.print(
                f"[dim]daemon capabilities unavailable ({e}); loading the adapter in-process[/dim]"
            )
    try:
        from pfnstudio_core.training.predict import run_inference
    except ImportError as e:
        return {"error": f"predict module unavailable: {e}"}
    import tempfile

    listing = row.get("listing") or {}
    adapter = row.get("baseAdapter") or listing.get("baseAdapter")
    if not adapter:
        return {"error": "no baseAdapter recorded for this model (pull it again)"}
    ckpt = row.get("checkpoint_dir") or store.checkpoint_cache_dir(
        listing.get("checkpointRef") or ""
    )
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "runs" / f"{row['slug']}.yaml"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            f"description: show\nhyperparams:\n  adapter: {adapter}\n  baseAdapter: {adapter}\n",
            encoding="utf-8",
        )
        try:
            out = run_inference(
                manifest_path=manifest,
                project_root=row["bundle_dir"],
                checkpoint_dir=Path(ckpt) if ckpt else row["bundle_dir"],
                payload={"__capabilities__": True},
            )
        except ModuleNotFoundError as e:
            return {
                "error": f"this model needs a package that isn't installed: {e.name or e} "
                f"(pip install -r {row['bundle_dir'] / 'requirements.txt'})"
            }
        except Exception as e:
            return {"error": f"capabilities failed: {type(e).__name__}: {e}"}
    result = out.get("result") or {}
    if result.get("error"):
        return {"error": str(result["error"])}
    return {"capabilities": result.get("capabilities") or {}, "source": "in-process"}


# ── commands ─────────────────────────────────────────────────────────────────
def store_commands(app: typer.Typer) -> None:
    """Register pull / ls / ps / rm / show on the root typer app."""

    @app.command("pull")
    def pull(
        ref: str = typer.Argument(..., help="Base-model slug, optionally slug:tag."),
        api_url: str = typer.Option(
            DEFAULT_API,
            "--api-url",
            envvar="PFNSTUDIO_API_URL",
            help="PFN Studio API base (public marketplace listings are unauthenticated).",
        ),
    ) -> None:
        """Prefetch a base model — adapter bundle, checkpoint and listing — into
        the local store without running it."""
        slug, tag = store.parse_ref(ref)
        listing = _fetch_listing(api_url, slug)
        served_tag = listing.get("tag")
        if tag and served_tag and str(served_tag) != tag:
            _err.print(f"[red]The marketplace serves {slug}:{served_tag}, not :{tag}.[/red]")
            raise typer.Exit(code=1)
        fm_project, checkpoint_ref = listing.get("fmProject"), listing.get("checkpointRef")
        if not fm_project or not checkpoint_ref:
            _err.print("[red]This listing has no fmProject/checkpointRef — nothing to pull.[/red]")
            raise typer.Exit(code=1)
        console.print(f"[bold]{listing.get('title') or slug}[/bold]")

        from .cli import _fetch_basemodel_bundle  # lazy: cli imports this module

        try:
            bdir = _fetch_basemodel_bundle(api_url.rstrip("/"), slug, str(fm_project))
        except Exception as e:
            _err.print(f"[red]Couldn't fetch the adapter bundle: {e}[/red]")
            raise typer.Exit(code=1) from e
        console.print(f"[green]✓[/green] adapter bundle  {bdir}")
        store.save_listing(bdir, listing)

        try:
            from pfnstudio_core.serving.hf import resolve_checkpoint
        except ImportError as e:
            _err.print(f"[red]checkpoint resolver unavailable: {e}[/red]")
            raise typer.Exit(code=1) from e
        ref_arg = (
            json.dumps(checkpoint_ref) if isinstance(checkpoint_ref, dict) else str(checkpoint_ref)
        )
        console.print("[dim]pulling checkpoint…[/dim]")
        try:
            ckpt = resolve_checkpoint(ref_arg)
        except Exception as e:
            _err.print(f"[red]Couldn't resolve checkpoint: {type(e).__name__}: {e}[/red]")
            raise typer.Exit(code=1) from e
        console.print(
            f"[green]✓[/green] checkpoint      {ckpt}  ({store.human_bytes(store.dir_bytes(ckpt))})"
        )
        # The lifecycle (pfnstudio.lock) owns lock.json; `ls` shows its digest once it is written.
        console.print(
            f"[green]✓[/green] {slug}{f':{tag}' if tag else ''} is in the store — run it with:  pfnstudio run {slug}"
        )

    @app.command("ls")
    def ls(as_json: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
        """List base models in the local store."""
        rows = store.list_models()
        loaded = _loaded()
        if as_json:
            out = [
                {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in r.items()
                    if k not in ("listing", "lock")
                }
                | {"loaded": (r["slug"] in loaded) if loaded is not None else None}
                for r in rows
            ]
            sys.stdout.write(json.dumps(out, indent=2) + "\n")
            return
        if not rows:
            console.print(
                f"[dim]No models in {store.store_root()}. Pull one with:[/dim]  pfnstudio pull <slug>"
            )
            return
        t = Table(box=None, header_style="bold")
        for col in ("NAME", "TAG", "DIGEST", "SIZE", "TIER", "LOADED"):
            t.add_column(col)
        for r in rows:
            t.add_row(
                r["slug"],
                str(r["tag"] or "-"),
                _short(r["digest"]),
                store.human_bytes(r["bundle_bytes"] + r["checkpoint_bytes"]),
                str(r["tier"] or "-"),
                "-" if loaded is None else ("yes" if r["slug"] in loaded else "no"),
            )
        console.print(t)
        if loaded is None:
            console.print("[dim]daemon not running — LOADED unknown[/dim]")

    @app.command("ps")
    def ps(as_json: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
        """Models currently loaded in the daemon."""
        c = _client()
        if c is None:
            console.print("daemon not running")
            raise typer.Exit(code=1)
        try:
            rows = [r for r in c.ps() if isinstance(r, dict)]
        except Exception as e:
            _err.print(f"[red]daemon ps failed: {e}[/red]")
            raise typer.Exit(code=1) from e
        if as_json:
            sys.stdout.write(json.dumps(rows, indent=2, default=str) + "\n")
            return
        if not rows:
            console.print("[dim]nothing loaded[/dim]")
            return
        preferred = ["slug", "tag", "digest", "device", "loaded_at", "last_used", "requests"]
        keys = [k for k in preferred if any(k in r for r in rows)]
        keys += sorted({k for r in rows for k in r} - set(keys))
        t = Table(box=None, header_style="bold")
        for k in keys:
            t.add_column(k.upper())
        for r in rows:
            t.add_row(*(_short(r.get(k)) if k == "digest" else str(r.get(k, "-")) for k in keys))
        console.print(t)

    @app.command("rm")
    def rm(
        ref: str = typer.Argument(
            ..., help="slug[:tag] (or an fmProject id to drop a whole bundle)."
        ),
        yes: bool = typer.Option(False, "--yes", "-y", help="Delete without asking."),
    ) -> None:
        """Remove a model's adapter bundle and, when no other model shares them,
        its weights. Refuses while the model is loaded in the daemon."""
        row = store.find_model(ref)
        if row is None:
            _err.print(f"[red]'{ref}' is not in the store.[/red] [dim]See: pfnstudio ls[/dim]")
            raise typer.Exit(code=1)
        loaded = _loaded()
        if loaded and row["slug"] in loaded:
            _err.print(f"[red]{row['slug']} is loaded in the daemon — unload it first.[/red]")
            raise typer.Exit(code=1)
        rows = store.list_models()
        others = [r for r in rows if r["slug"] != row["slug"]]
        shared_ckpt = row["checkpoint_dir"] is not None and any(
            r["checkpoint_dir"] == row["checkpoint_dir"] for r in others
        )
        shared_bundle = any(r["bundle_dir"] == row["bundle_dir"] for r in others)
        tag_sfx = f":{row['tag']}" if row["tag"] else ""
        console.print(f"[bold]{row['slug']}[/bold]{tag_sfx}")
        console.print(
            f"  bundle      {row['bundle_dir']}  ({store.human_bytes(row['bundle_bytes'])})"
            + ("  [dim]shared — kept[/dim]" if shared_bundle else "")
        )
        if row["checkpoint_dir"] is None:
            console.print("  checkpoint  [dim]not in the cache[/dim]")
        else:
            console.print(
                f"  checkpoint  {row['checkpoint_dir']}  ({store.human_bytes(row['checkpoint_bytes'])})"
                + ("  [dim]shared — kept[/dim]" if shared_ckpt else "")
            )
        if not yes and not typer.confirm("Delete?", default=False):
            console.print("[dim]aborted[/dim]")
            raise typer.Exit(code=1)
        try:
            res = store.remove(ref)
        except KeyError:
            _err.print(f"[red]'{ref}' is not in the store.[/red]")
            raise typer.Exit(code=1) from None
        for p in res["removed_checkpoints"]:
            console.print(f"[green]✓[/green] removed checkpoint {p}")
        if res["removed_bundle"]:
            console.print(f"[green]✓[/green] removed bundle {res['removed_bundle']}")
        for p in res["kept_checkpoints"]:
            console.print(f"[dim]kept shared checkpoint {p}[/dim]")

    @app.command("show")
    def show(
        slug: str = typer.Argument(..., help="Base-model slug."),
        as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
    ) -> None:
        """Describe a model in the store: title, license, params, tier, digest,
        upstream package, and its tasks with input/output schemas (from the
        adapter's own capabilities())."""
        row = store.find_model(slug)
        if row is None:
            _err.print(
                f"[red]'{slug}' is not in the store.[/red] [dim]Pull it first: pfnstudio pull {slug}[/dim]"
            )
            raise typer.Exit(code=1)
        listing = row.get("listing") or {}
        req = row["bundle_dir"] / "requirements.txt"
        upstream = (
            [
                ln.strip()
                for ln in req.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
            if req.is_file()
            else []
        )
        caps_out = _capabilities(row)
        info = {
            "slug": row["slug"],
            "tag": row["tag"],
            "title": listing.get("title"),
            "tagline": listing.get("tagline"),
            "license": listing.get("license"),
            "params": listing.get("paramsLabel"),
            "tier": row["tier"],
            "digest": row["digest"],
            "fmProject": row["fmProject"],
            "baseAdapter": row["baseAdapter"],
            "repoUrl": listing.get("repoUrl"),
            "upstream": upstream,
            "bundle_dir": str(row["bundle_dir"]),
            "checkpoint_dir": str(row["checkpoint_dir"]) if row["checkpoint_dir"] else None,
            **caps_out,
        }
        if as_json:
            sys.stdout.write(json.dumps(info, indent=2) + "\n")
            return
        console.print(f"[bold]{info['title'] or row['slug']}[/bold]")
        if info["tagline"]:
            console.print(f"[dim]{info['tagline']}[/dim]")
        for label, key in (
            ("slug", "slug"),
            ("tag", "tag"),
            ("license", "license"),
            ("params", "params"),
            ("tier", "tier"),
            ("digest", "digest"),
            ("adapter", "baseAdapter"),
            ("repo", "repoUrl"),
            ("bundle", "bundle_dir"),
            ("checkpoint", "checkpoint_dir"),
        ):
            console.print(f"  {label:<11}{info[key] if info[key] is not None else '-'}")
        console.print(f"  {'upstream':<11}{', '.join(upstream) or '-'}")
        if caps_out.get("error"):
            console.print(f"\n[yellow]tasks unavailable:[/yellow] {caps_out['error']}")
            return
        caps = caps_out.get("capabilities") or {}
        console.print(
            f"\n[bold]Tasks[/bold] ({len(caps)})  [dim]from capabilities(), {caps_out.get('source')}[/dim]"
        )
        for key, spec in caps.items():
            spec = spec if isinstance(spec, dict) else {}
            console.print(f"\n  [cyan]{key}[/cyan] — {spec.get('label') or key}")
            if spec.get("description"):
                console.print(f"  [dim]{spec['description']}[/dim]")
            for which in ("input_schema", "output_schema"):
                if which in spec:
                    console.print(f"  {which}:")
                    console.print(
                        "    " + json.dumps(spec[which], indent=2).replace("\n", "\n    ")
                    )
