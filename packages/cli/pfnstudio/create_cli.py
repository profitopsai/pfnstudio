"""`pfnstudio create <name> -f Pfnfile` — compose a named model from a base.

Flow (mirrors the MCP chain list_base_models → install_base_model →
continue_pretraining, over the same HTTP routes):

  1. parse the Pfnfile                              (pfnfile.py, no network)
  2. GET  {api}/marketplace/listings                 resolve FROM (type=basemodel)
  3. POST {api}/organizations/{org}/base-models/install   new project + base run
  4. PRIOR/MODEL/EVAL given as paths → POST {api}/projects/{pid}/{priors|models|evals}
     given as slugs → verified against GET {api}/projects/{pid}/{priors|models|evals}
  5. training spec present → POST {api}/projects/{pid}/runs with
     initWeightsFromRunId=<base run>; --execute adds POST .../runs/{id}/execute
  6. write the local alias ~/.cache/pfnstudio/models/<name>.json so a later
     `pfnstudio run <name>` can address the result.

Registration (cli.py is owned elsewhere):

    from .create_cli import create_commands
    create_commands(app)          # adds the top-level `create` command

Auth/base-URL pattern copied from container.py: PFNSTUDIO_TOKEN,
"Authorization: Token <t>", PFNSTUDIO_CLOUD_URL / PFNSTUDIO_API_URL.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from .pfnfile import ParseError, Pfnfile, parse_file, requires_satisfied, slugify, to_request

console = Console()
err_console = Console(stderr=True)

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


# ── HTTP (same shape as container.py) ────────────────────────────────────────
def _api_base(cloud_url: str | None) -> str:
    base = (
        cloud_url
        or os.environ.get("PFNSTUDIO_CLOUD_URL")
        or os.environ.get("PFNSTUDIO_API_URL")
        or "http://localhost:3000/api"
    ).rstrip("/")
    return base if base.endswith("/api") else f"{base}/api"


def _token(explicit: str | None) -> str:
    tok = explicit or os.environ.get("PFNSTUDIO_TOKEN") or os.environ.get("PFNSTUDIO_API_TOKEN")
    if not tok:
        err_console.print(
            "[red]No API token.[/red] Set PFNSTUDIO_TOKEN or pass --token "
            "(create one in Studio → Avatar → API tokens)."
        )
        raise typer.Exit(2)
    return tok


def _request(method: str, url: str, *, token: str | None, **kwargs: Any) -> Any:
    import requests

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    try:
        resp = requests.request(
            method, url, headers=headers, timeout=kwargs.pop("timeout", 60), **kwargs
        )
    except requests.RequestException as e:
        err_console.print(f"[red]Request failed:[/red] {method} {url}: {e}")
        raise typer.Exit(2) from e
    if resp.status_code == 401:
        err_console.print("[red]Unauthorized.[/red] Check your API token.")
        raise typer.Exit(2)
    if not resp.ok:
        detail = resp.text[:400]
        try:
            detail = json.dumps(resp.json(), indent=2)
        except Exception:
            pass
        err_console.print(f"[red]{resp.status_code}[/red] {method} {url}\n{detail}")
        raise typer.Exit(2)
    return resp.json() if resp.content else {}


# ── caches ───────────────────────────────────────────────────────────────────
def models_cache_dir() -> Path:
    return Path(
        os.environ.get(
            "PFNSTUDIO_MODELS_CACHE", str(Path.home() / ".cache" / "pfnstudio" / "models")
        )
    )


def basemodels_cache_dir() -> Path:
    return Path(
        os.environ.get(
            "PFNSTUDIO_BASEMODELS_CACHE", str(Path.home() / ".cache" / "pfnstudio" / "basemodels")
        )
    )


def _base_lock(fm_project: str | None) -> dict | None:
    """Sibling module lock.py: read_lock(bundle_dir) -> {slug, tag, fmProject, digest} | None.
    Degrades to None when the module or the cached bundle is absent."""
    if not fm_project:
        return None
    try:
        from .lock import read_lock  # type: ignore[import-not-found]
    except Exception:
        return None
    safe = "".join(ch for ch in fm_project if ch.isalnum() or ch in "._-")
    try:
        lock = read_lock(basemodels_cache_dir() / safe)
    except Exception:
        return None
    return lock if isinstance(lock, dict) else None


def alias_path(name: str) -> Path:
    return models_cache_dir() / f"{name}.json"


def write_alias(record: dict[str, Any]) -> Path:
    p = alias_path(record["name"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return p


# ── resolution helpers ───────────────────────────────────────────────────────
def _resolve_base(base: str, pf: Pfnfile) -> dict[str, Any]:
    rows = _request("GET", f"{base}/marketplace/listings", token=None)
    rows = rows if isinstance(rows, list) else []
    bases = [r for r in rows if isinstance(r, dict) and r.get("type") == "basemodel"]
    hit = next((r for r in bases if r.get("slug") == pf.from_slug), None)
    if not hit:
        avail = ", ".join(sorted(str(r.get("slug")) for r in bases)) or "(none)"
        err_console.print(
            f"[red]FROM {pf.from_ref}:[/red] no base model with slug {pf.from_slug!r}. Available: {avail}"
        )
        raise typer.Exit(1)
    if not hit.get("fmProject"):
        err_console.print(
            f"[red]FROM {pf.from_ref}:[/red] base model has no FM-project yet and cannot be installed."
        )
        raise typer.Exit(1)
    listing_tag = hit.get("tag") or hit.get("version")
    if pf.from_tag and pf.from_tag != "latest" and listing_tag and str(listing_tag) != pf.from_tag:
        err_console.print(
            f"[red]FROM {pf.from_ref}:[/red] the marketplace serves tag {listing_tag!r}; "
            f"pin that tag or drop the tag to take the latest."
        )
        raise typer.Exit(1)
    return hit


def _install_payload(listing: dict[str, Any], name: str, pf: Pfnfile) -> dict[str, Any]:
    body: dict[str, Any] = {
        "fmProject": listing["fmProject"],
        "slug": slugify(name),
        "title": name,
        "description": f"{name} — FROM {pf.from_ref}",
        "checkpointRef": listing.get("checkpointRef"),
    }
    for k in ("baseAdapter", "keymap", "modelRef"):
        if listing.get(k) is not None:
            body[k] = listing[k]
    return body


def _resolve_org(base: str, tok: str, explicit: str | None) -> str:
    org = explicit or os.environ.get("PFNSTUDIO_ORG")
    if org:
        return org
    orgs = _request("GET", f"{base}/organizations", token=tok)
    orgs = orgs if isinstance(orgs, list) else []
    if not orgs:
        err_console.print(
            "[red]You are not a member of any organization.[/red] Create one in Studio first."
        )
        raise typer.Exit(1)
    if len(orgs) > 1:
        names = ", ".join(f"{o.get('slug')} ({o.get('id')})" for o in orgs)
        console.print(
            f"[dim]several organizations — using the first ({orgs[0].get('slug')}); pass --org to pick: {names}[/dim]"
        )
    return str(orgs[0]["id"])


def _looks_like_path(value: str, cwd: Path) -> Path | None:
    """A PRIOR/MODEL/EVAL value is a path when it points at something on disk
    (relative to the Pfnfile) or is spelled like one; otherwise it is a slug."""
    p = (cwd / value).expanduser()
    if p.exists():
        return p.resolve()
    if "/" in value or value.endswith((".yaml", ".yml")) or value.startswith((".", "~")):
        err_console.print(
            f"[red]{value}:[/red] looks like a path but does not exist (resolved {p})."
        )
        raise typer.Exit(1)
    return None


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or "id" not in data:
        err_console.print(f"[red]{path}:[/red] expected a YAML mapping with an `id`.")
        raise typer.Exit(1)
    return data


def _artifact_body(kind: str, path: Path) -> dict[str, Any]:
    """Same YAML → DTO bridge as `pfnstudio push` (cli.py), one artifact at a time."""
    if kind == "priors":
        y = path / "prior.yaml" if path.is_dir() else path
        spec, d = _yaml(y), y.parent
        code, md, req = (
            d / spec.get("implementation", "prior.py"),
            d / "prior.md",
            d / "requirements.txt",
        )
        return {
            "name": spec.get("name") or spec["id"],
            "slug": spec["id"],
            "version": spec["version"],
            "kind": spec["kind"],
            "description": spec.get("description"),
            "rationale": md.read_text() if md.exists() else None,
            "parameters": spec.get("parameters") or {},
            "outputs": spec.get("outputs") or {"variables": []},
            "citations": spec.get("citations") or [],
            "code": code.read_text() if code.exists() else None,
            "requirements": req.read_text() if req.exists() else None,
            "pythonVersion": spec.get("python_version") or spec.get("pythonVersion"),
        }
    spec = _yaml(path / "eval.yaml" if path.is_dir() else path)
    if kind == "models":
        return {
            "name": spec.get("name") or spec["id"],
            "slug": spec["id"],
            "version": spec["version"],
            "description": spec.get("description"),
            "inputShape": spec.get("input_shape") or spec.get("inputShape"),
            "blocks": spec.get("blocks") or [],
            "outputHeads": spec.get("output_heads") or spec.get("outputHeads") or [],
            "citations": spec.get("citations") or [],
            **({"capabilities": spec["capabilities"]} if spec.get("capabilities") else {}),
        }
    return {
        "name": spec.get("name") or spec["id"],
        "slug": spec["id"],
        "version": spec["version"],
        "task": spec["task"],
        "description": spec.get("description"),
        "dataset": spec.get("dataset") or {},
        "metrics": spec.get("metrics") or [],
        "baselines": spec.get("baselines") or [],
        "citations": spec.get("citations") or [],
    }


def _resolve_artifact(
    base: str, tok: str, pid: str, kind: str, value: str | None, cwd: Path, dry: bool
) -> str | None:
    """Return the slug to reference for one PRIOR/MODEL/EVAL value.
    None value → project default (first artifact), like MCP continue_pretraining."""
    rows = [] if dry else _request("GET", f"{base}/projects/{pid}/{kind}", token=tok)
    rows = rows if isinstance(rows, list) else []
    if value is None:
        if dry:
            return None
        if not rows:
            err_console.print(
                f"[red]project has no {kind}[/red] to compose the run; the base install should have added some."
            )
            raise typer.Exit(1)
        return str(rows[0]["slug"])
    path = _looks_like_path(value, cwd)
    if path is None:
        if not dry and not any(r.get("slug") == value for r in rows):
            avail = ", ".join(sorted(str(r.get("slug")) for r in rows)) or "(none)"
            err_console.print(
                f"[red]{kind[:-1]} {value!r} not found in the project.[/red] Available: {avail}"
            )
            raise typer.Exit(1)
        return value
    body = _artifact_body(kind, path)
    if dry:
        console.print(f"[dim]would POST /projects/<pid>/{kind} {body['slug']} from {path}[/dim]")
        return str(body["slug"])
    if any(r.get("slug") == body["slug"] for r in rows):
        console.print(
            f"[yellow]⤳[/yellow] {kind[:-1]} {body['slug']} already in the project — using it"
        )
    else:
        _request("POST", f"{base}/projects/{pid}/{kind}", token=tok, json=body)
        console.print(f"[green]✓[/green] {kind[:-1]} {body['slug']} ← {path}")
    return str(body["slug"])


# ── the command ──────────────────────────────────────────────────────────────
def create(
    name: str | None = typer.Argument(
        None, help="Model name (lowercase, digits, . _ -). Falls back to the Pfnfile's NAME."
    ),
    file: Path = typer.Option(Path("Pfnfile"), "--file", "-f", help="Path to the Pfnfile."),
    org: str | None = typer.Option(
        None, "--org", envvar="PFNSTUDIO_ORG", help="Organization id (default: your first org)."
    ),
    execute: bool = typer.Option(
        False, "--execute", help="Dispatch the training run right after creating it."
    ),
    runner_id: str | None = typer.Option(
        None, "--runner", help="Self-hosted runner id (compute target 'runner')."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved requests; call only the public marketplace listing.",
    ),
    token: str | None = typer.Option(None, "--token", envvar="PFNSTUDIO_TOKEN"),
    cloud_url: str | None = typer.Option(
        None, "--cloud-url", help="API base, e.g. http://localhost:3000/api"
    ),
) -> None:
    """Compose a new model from a base model, as described by a Pfnfile.

    Example:  pfnstudio create my-model -f Pfnfile
    """
    try:
        pf = parse_file(file)
    except ParseError as e:
        err_console.print(f"[red]Pfnfile error:[/red] {e}")
        raise typer.Exit(1) from e

    name = name or pf.name
    if not name:
        err_console.print(
            "[red]No model name.[/red] Pass one (`pfnstudio create <name>`) or add `NAME <name>` to the Pfnfile."
        )
        raise typer.Exit(1)
    if not _NAME_RE.match(name):
        err_console.print(
            f"[red]Invalid name {name!r}.[/red] Use lowercase letters, digits, . _ - (no slashes)."
        )
        raise typer.Exit(1)

    from . import __version__

    ok = requires_satisfied(pf, __version__)
    if ok is False:
        err_console.print(
            f"[red]REQUIRES {pf.requires}:[/red] installed pfnstudio is {__version__}. Upgrade: pip install -U pfnstudio"
        )
        raise typer.Exit(1)

    base = _api_base(cloud_url)
    listing = _resolve_base(base, pf)
    lock = _base_lock(listing.get("fmProject"))
    tag = pf.from_tag or (lock or {}).get("tag") or "latest"
    digest = (lock or {}).get("digest")
    console.print(f"FROM {pf.from_slug}:{tag} digest {digest or '(not cached locally)'}")

    install_body = _install_payload(listing, name, pf)
    cwd = file.resolve().parent
    tok = None if dry_run else _token(token)

    if dry_run:
        org_id = org or os.environ.get("PFNSTUDIO_ORG") or "<org>"
        console.print(f"[bold]POST[/bold] {base}/organizations/{org_id}/base-models/install")
        console.print_json(data=install_body)
        if pf.has_training_spec:
            prior = _resolve_artifact(base, "", "<pid>", "priors", pf.prior, cwd, True)
            model = _resolve_artifact(base, "", "<pid>", "models", pf.model, cwd, True)
            evals = [_resolve_artifact(base, "", "<pid>", "evals", e, cwd, True) for e in pf.evals]
            run_body = to_request(
                pf,
                name,
                base_run_id="<base-run-id>",
                prior_slug=prior or "<project default>",
                model_slug=model or "<project default>",
                eval_slugs=[e for e in evals if e] or ["<project default>"],
                digest=digest,
            )
            if runner_id:
                run_body["compute"] = {"target": "runner", "runner_id": runner_id}
            console.print(f"[bold]POST[/bold] {base}/projects/<pid>/runs")
            console.print_json(data=run_body)
        console.print(f"[dim]alias → {alias_path(name)}[/dim]")
        return

    if alias_path(name).exists():
        err_console.print(
            f"[red]{name!r} already exists[/red] ({alias_path(name)}). Pick another name or remove the alias."
        )
        raise typer.Exit(1)

    org_id = _resolve_org(base, tok, org)
    installed = _request(
        "POST", f"{base}/organizations/{org_id}/base-models/install", token=tok, json=install_body
    )
    pid, base_run = str(installed["projectId"]), str(installed["runId"])
    console.print(
        f"[green]✓[/green] project {installed.get('projectSlug')} ({pid}) base run {base_run}"
    )

    run_id = base_run
    if pf.has_training_spec:
        prior = _resolve_artifact(base, tok, pid, "priors", pf.prior, cwd, False)
        model = _resolve_artifact(base, tok, pid, "models", pf.model, cwd, False)
        evals = [_resolve_artifact(base, tok, pid, "evals", e, cwd, False) for e in pf.evals] or [
            _resolve_artifact(base, tok, pid, "evals", None, cwd, False)
        ]
        run_body = to_request(
            pf,
            name,
            base_run_id=base_run,
            prior_slug=prior,
            model_slug=model,
            eval_slugs=[e for e in evals if e],
            digest=digest,
        )
        if runner_id:
            run_body["compute"] = {"target": "runner", "runner_id": runner_id}
        run = _request("POST", f"{base}/projects/{pid}/runs", token=tok, json=run_body)
        run_id = str(run["id"])
        console.print(f"[green]✓[/green] training run {run_id} (warm-start from {base_run})")
        if execute:
            _request("POST", f"{base}/projects/{pid}/runs/{run_id}/execute", token=tok, json={})
            console.print(f"[green]▶[/green] dispatched {run_id}")

    record = {
        "name": name,
        "from": {
            "slug": pf.from_slug,
            "tag": tag,
            "digest": digest,
            "fmProject": listing.get("fmProject"),
        },
        "projectId": pid,
        "projectSlug": installed.get("projectSlug"),
        "runId": run_id,
        "baseRunId": base_run,
        "license": pf.license,
        "requires": pf.requires,
        "api": base,
        "pfnfile": str(file.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    p = write_alias(record)
    console.print(f"[bold]{name}[/bold] → run {run_id}  (alias {p})")
    console.print(f"FROM {pf.from_slug}:{tag} digest {digest or 'null'}")


def create_commands(app: typer.Typer) -> None:
    """Register `create` on the root Typer app (call from cli.py)."""
    app.command("create")(create)
