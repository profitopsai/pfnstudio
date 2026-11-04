"""PFN Studio CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .compute import ADAPTERS, get_adapter
from .create_cli import alias_path, create_commands
from .lint import lint_project
from .lock import build_lock, parse_model_ref, write_lock
from .runner import runner_app
from .scaffold import scaffold_project
from .store_cli import store_commands
from .tracking import select_trackers
from .validate import validate_project

app = typer.Typer(
    name="pfnstudio",
    help="Build, validate, and run prior-fitted foundation model projects.",
    no_args_is_help=True,
)

studio_app = typer.Typer(
    name="studio", help="Static-site Studio: render the project as a browsable site."
)
app.add_typer(studio_app, name="studio")

author_app = typer.Typer(
    name="author", help="Local prior-authoring helpers (no API access required)."
)
app.add_typer(author_app, name="author")

# Self-hosted runner: this machine pulls jobs from PFN Studio cloud and
# trains them locally. Mirrors GitHub Actions' self-hosted runner.
app.add_typer(runner_app, name="runner")

# Base-model lifecycle (Ollama-shaped): pull / ls / ps / rm / show live in
# store_cli, `create -f Pfnfile` in create_cli; `run` and `serve` are below.
store_commands(app)
create_commands(app)

console = Console()


@app.command()
def version() -> None:
    """Print the CLI version."""
    console.print(f"pfnstudio {__version__}")


@app.command()
def init(
    name: str = typer.Argument(..., help="Project name (kebab-case)."),
    path: Path = typer.Option(Path.cwd(), help="Parent directory."),
    description: str = typer.Option("A prior-fitted foundation model project."),
    org: str = typer.Option("your-org", help="GitHub org/user for repo URL substitutions."),
) -> None:
    """Scaffold a new FM project from the template."""
    target = path / name
    if target.exists():
        console.print(f"[red]Refusing to overwrite existing path:[/red] {target}")
        raise typer.Exit(code=1)
    scaffold_project(target=target, project_name=name, description=description, org=org)
    console.print(f"[green]Created[/green] {target}")


@app.command()
def validate(project: Path = typer.Argument(Path.cwd())) -> None:
    """Validate every artifact against its JSON Schema."""
    errors = validate_project(project)
    if errors:
        for err in errors:
            console.print(f"[red]✗[/red] {err}")
        raise typer.Exit(code=1)
    console.print("[green]✓ All artifacts valid.[/green]")


@app.command()
def lint(project: Path = typer.Argument(Path.cwd())) -> None:
    """Cross-reference lint: do runs reference priors/models/evals that exist?"""
    errors = lint_project(project)
    if errors:
        for err in errors:
            console.print(f"[red]✗[/red] {err}")
        raise typer.Exit(code=1)
    console.print("[green]✓ All cross-references valid.[/green]")


@app.command()
def list_artifacts(project: Path = typer.Argument(Path.cwd())) -> None:
    """List every artifact in a project."""
    table = Table(title=str(project))
    table.add_column("kind")
    table.add_column("id")
    table.add_column("version")
    table.add_column("path")
    for prior_yaml in (project / "priors").rglob("prior.yaml"):
        import yaml as _y

        spec = _y.safe_load(prior_yaml.read_text()) or {}
        table.add_row(
            "prior",
            str(spec.get("id", "?")),
            str(spec.get("version", "?")),
            str(prior_yaml.relative_to(project)),
        )
    for model_yaml in (project / "models").glob("*.yaml"):
        import yaml as _y

        spec = _y.safe_load(model_yaml.read_text()) or {}
        table.add_row(
            "model",
            str(spec.get("id", "?")),
            str(spec.get("version", "?")),
            str(model_yaml.relative_to(project)),
        )
    for eval_yaml in (project / "evals").glob("*.yaml"):
        import yaml as _y

        spec = _y.safe_load(eval_yaml.read_text()) or {}
        table.add_row(
            "eval",
            str(spec.get("id", "?")),
            str(spec.get("version", "?")),
            str(eval_yaml.relative_to(project)),
        )
    for run_yaml in (project / "runs").glob("*.yaml"):
        import yaml as _y

        spec = _y.safe_load(run_yaml.read_text()) or {}
        table.add_row("run", str(spec.get("id", "?")), "—", str(run_yaml.relative_to(project)))
    console.print(table)


@app.command()
def sample(
    project: Path = typer.Argument(..., help="Project root containing priors/ and prior code."),
    prior: str = typer.Argument(..., help="Prior slug (matches priors/<slug>/ directory)."),
    count: int = typer.Option(9, help="How many tasks to sample."),
    seed: int = typer.Option(42, help="Base seed; per-sample seeds = base + i."),
    overrides_json: str = typer.Option("{}", help="JSON dict of parameter overrides."),
) -> None:
    """Sample N tasks from a prior and emit JSON to stdout.

    Used by the API's Visual Prior Inspector. Output is a single JSON object:
        {
          "prior_id": ...,
          "version": ...,
          "kind": ...,
          "outputs": [...],
          "samples": [ { "seed": 42, "X": [...], "y": [...], ... }, ... ]
        }
    Numpy arrays are converted to nested lists. Non-serialisable values become null.
    """
    import sys

    project = project.resolve()
    sys.path.insert(0, str(project))

    from pfnstudio_core.loaders import load_prior
    from pfnstudio_core.registry import discover_in_project, get_prior, list_priors

    # rich's Console.print() doesn't accept err=; use a separate stderr Console
    # for diagnostics so they don't pollute the JSON stdout the API parses.
    err_console = Console(stderr=True)

    prior_dir = project / "priors" / prior
    prior_yaml = prior_dir / "prior.yaml"
    if not prior_yaml.exists():
        err_console.print(f"[red]not found:[/red] {prior_yaml}")
        raise typer.Exit(code=1)

    spec = load_prior(prior_yaml)
    discover_in_project(project)
    try:
        prior_cls = get_prior(spec.id)
    except KeyError as e:
        # Fork tolerance: a forked prior gets a new slug but inherits its
        # parent's @register_prior("...") decorator unchanged. When the API
        # exports a single-prior project to a temp dir for sampling, the
        # registry then has exactly one entry — whatever the decorator names
        # it. Use that one regardless of the slug the YAML thinks it has.
        registered = list_priors()
        if len(registered) == 1:
            prior_cls = get_prior(registered[0])
        else:
            err_console.print(f"[red]registry miss:[/red] {e}")
            err_console.print(f"  registered priors: {registered}")
            raise typer.Exit(code=1) from e

    prior_obj = prior_cls()
    prior_obj.spec = spec

    overrides = json.loads(overrides_json)

    import math as _math

    def _to_jsonable(v):
        # JSON.parse() on the API side (TypeScript) is standards-compliant
        # and rejects the non-finite floats Python's json.dumps emits by
        # default ('NaN', 'Infinity', '-Infinity'). Priors that emit NaN
        # — e.g. Do-PFN's prior, which appends a y-feature column with
        # NaN at query rows so the preprocessor knows what to mean-impute
        # — would otherwise produce stdout the API can't parse. Convert
        # non-finite floats to None at every level so the JSON we emit is
        # standards-compliant; the API + UI render None as "missing".
        try:
            import numpy as np

            if isinstance(v, np.ndarray):
                if v.dtype.kind == "f":
                    # Replace non-finite cells with None before tolist().
                    # astype(object) so the array can hold None alongside floats.
                    finite = np.isfinite(v)
                    if not finite.all():
                        cleaned = np.where(finite, v, None).astype(object)
                        return cleaned.tolist()
                return v.tolist()
            if isinstance(v, np.floating):
                x = float(v.item())
                return x if _math.isfinite(x) else None
            if isinstance(v, np.integer):
                return v.item()
        except ImportError:
            pass
        try:
            import torch

            if isinstance(v, torch.Tensor):
                return _to_jsonable(v.detach().cpu().numpy())
        except ImportError:
            pass
        if isinstance(v, dict):
            return {k: _to_jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_to_jsonable(x) for x in v]
        if isinstance(v, float):
            return v if _math.isfinite(v) else None
        if isinstance(v, (int, str, bool)) or v is None:
            return v
        return None  # silently drop non-serialisable

    samples = []
    for i in range(count):
        s = prior_obj.sample(seed=seed + i, **overrides)
        samples.append({"seed": seed + i, **{k: _to_jsonable(v) for k, v in s.items()}})

    payload = {
        "prior_id": spec.id,
        "name": spec.name,
        "version": spec.version,
        "kind": spec.kind,
        "outputs": [o.model_dump() for o in spec.outputs.variables],
        "parameters": {k: p.model_dump() for k, p in spec.parameters.items()},
        "overrides": overrides,
        "samples": samples,
    }
    # allow_nan=False — if any non-finite float slips past _to_jsonable
    # we want a hard error here, NOT silent emission of invalid JSON
    # the API can't parse. The scrub above should make this unreachable;
    # the strict flag is the safety net.
    print(json.dumps(payload, allow_nan=False))


@app.command("check-model")
def check_model(
    project: Path = typer.Argument(..., help="Project root containing models/<slug>.yaml."),
    model: str = typer.Argument(..., help="Model slug (matches models/<slug>.yaml)."),
) -> None:
    """Assemble a model's blocks and emit JSON to stdout.

    Instantiates every block via the registry exactly as the trainer does
    (load_model normalizes, then Model(spec) builds each block). Catches
    unknown block types / bad config / dimension errors. Used by the API's
    Plip test_model tool. Emits a single JSON object (exit 0 even when the
    build fails, so the caller can read the reason):
        {"ok": true, "blocks": [{"name","type"}], "heads": ["name:task"]}
        {"ok": false, "error": "KeyError: ..."}
    """
    import sys

    project = project.resolve()
    sys.path.insert(0, str(project))

    from pfnstudio_core import Model
    from pfnstudio_core.loaders import load_model
    from pfnstudio_core.registry import discover_in_project

    model_yaml = project / "models" / f"{model}.yaml"
    if not model_yaml.exists():
        print(json.dumps({"ok": False, "error": f"models/{model}.yaml not found"}))
        raise typer.Exit(code=1)

    # Register any project-local blocks (built-ins register on core import).
    try:
        discover_in_project(project)
    except Exception:
        pass

    try:
        spec = load_model(model_yaml)
        built = Model(spec)
        blocks = [{"name": label, "type": mod.__class__.__name__} for label, mod in built.modules]
        heads = [f"{h.name}:{h.task}" for h in spec.output_heads]
        print(json.dumps({"ok": True, "blocks": blocks, "heads": heads}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))


@app.command()
def push(
    project_dir: Path = typer.Argument(
        ...,
        help="Local project directory in canonical PFN Studio layout (priors/, models/, evals/, runs/).",
    ),
    project_id: str = typer.Option(
        ..., "--project-id", help="Target project ID or slug in the PFN Studio API."
    ),
    api_url: str = typer.Option(
        "http://localhost:3000/api", "--api-url", envvar="PFNSTUDIO_API_URL"
    ),
    token: str = typer.Option(
        ...,
        "--token",
        envvar="PFNSTUDIO_TOKEN",
        help="API token (issue one at /api-tokens in the web UI).",
    ),
    only: str = typer.Option(
        "priors,models", help="Comma-separated kinds to push: priors,models,evals,runs."
    ),
    skip_existing: bool = typer.Option(
        True, help="Skip artifacts whose slug already exists in the target project."
    ),
) -> None:
    """Bulk-upload artifacts from a local project directory to PFN Studio.

    Example:
        pfnstudio push my-domain-priors --project-id my-project --token ps_xxx
    """
    import requests
    import yaml as _y

    project_dir = project_dir.resolve()
    if not project_dir.exists():
        console.print(f"[red]not found:[/red] {project_dir}")
        raise typer.Exit(code=1)

    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    base = api_url.rstrip("/")
    kinds = [k.strip() for k in only.split(",") if k.strip()]

    # Resolve the project (slug or id)
    res = requests.get(f"{base}/projects/{project_id}", headers=headers, timeout=10)
    if res.status_code == 401:
        _die_token()
    if res.status_code != 200:
        console.print(f"[red]project lookup failed:[/red] {res.status_code} {res.text[:300]}")
        raise typer.Exit(code=1)
    resolved_id = res.json()["id"]
    console.print(f"[blue]→[/blue] target project: {res.json()['name']} ({resolved_id})")

    pushed, skipped, failed = 0, 0, 0

    if "priors" in kinds:
        for prior_yaml in sorted((project_dir / "priors").rglob("prior.yaml")):
            spec = _y.safe_load(prior_yaml.read_text()) or {}
            prior_dir = prior_yaml.parent
            code_path = prior_dir / spec.get("implementation", "prior.py")
            rationale_path = prior_dir / "prior.md"
            requirements_path = prior_dir / "requirements.txt"
            wheel_dir = prior_dir / "wheels"

            body = {
                "name": spec.get("name") or spec["id"],
                "slug": spec["id"],
                "version": spec["version"],
                "kind": spec["kind"],
                "description": spec.get("description"),
                "rationale": rationale_path.read_text() if rationale_path.exists() else None,
                "parameters": spec.get("parameters") or {},
                "outputs": spec.get("outputs") or {"variables": []},
                "citations": spec.get("citations") or [],
                "code": code_path.read_text() if code_path.exists() else None,
                "requirements": requirements_path.read_text()
                if requirements_path.exists()
                else None,
                "pythonVersion": spec.get("python_version") or spec.get("pythonVersion"),
            }
            r = requests.post(
                f"{base}/projects/{resolved_id}/priors",
                headers=headers,
                json=body,
                timeout=30,
            )

            # Did we end up with a server-side prior to attach wheels to?
            target_prior_id: str | None = None
            if r.status_code in (200, 201):
                created = r.json()
                target_prior_id = created["id"]
                console.print(f"[green]✓[/green] prior {spec['id']}")
                pushed += 1
            elif r.status_code == 409 or "unique" in (r.text or "").lower():
                if skip_existing:
                    # Look up the existing prior so we can still attach/refresh wheels.
                    rg = requests.get(
                        f"{base}/projects/{resolved_id}/priors/{spec['id']}",
                        headers=headers,
                        timeout=10,
                    )
                    if rg.status_code == 200:
                        target_prior_id = rg.json()["id"]
                        console.print(
                            f"[yellow]⤳[/yellow] prior {spec['id']} already exists — skipping body, will refresh wheels"
                        )
                        skipped += 1
                    else:
                        console.print(
                            f"[yellow]⤳[/yellow] prior {spec['id']} already exists — could not look up id ({rg.status_code}); skipping wheels too"
                        )
                        skipped += 1
                else:
                    console.print(
                        f"[red]✗[/red] prior {spec['id']} conflict; pass --no-skip-existing to fail loudly"
                    )
                    failed += 1
            else:
                console.print(f"[red]✗[/red] prior {spec['id']}: {r.status_code} {r.text[:200]}")
                failed += 1

            # Upload any wheels in the local wheels/ dir to whichever prior we resolved.
            if target_prior_id:
                wheels = (
                    sorted(
                        [
                            p
                            for p in wheel_dir.glob("*")
                            if p.suffix in (".whl",) or p.name.endswith(".tar.gz")
                        ]
                    )
                    if wheel_dir.is_dir()
                    else []
                )
                for wheel_path in wheels:
                    with wheel_path.open("rb") as fh:
                        rw = requests.post(
                            f"{base}/projects/{resolved_id}/priors/{target_prior_id}/wheels",
                            headers={"Authorization": f"Token {token}"},
                            files={"file": (wheel_path.name, fh, "application/octet-stream")},
                            timeout=120,
                        )
                    if rw.status_code in (200, 201):
                        console.print(f"  [green]+[/green] wheel {wheel_path.name}")
                    else:
                        console.print(
                            f"  [red]✗[/red] wheel {wheel_path.name}: {rw.status_code} {rw.text[:160]}"
                        )

    if "models" in kinds:
        for model_yaml in sorted((project_dir / "models").glob("*.yaml")):
            spec = _y.safe_load(model_yaml.read_text()) or {}

            # Translate the canonical YAML keys (input_shape, output_heads,
            # snake_case) into the API's camelCase DTO. The CLI is the
            # YAML-side bridge — we don't want callers writing the API's
            # internal naming directly.
            body = {
                "name": spec.get("name") or spec["id"],
                "slug": spec["id"],
                "version": spec["version"],
                "description": spec.get("description"),
                "inputShape": spec.get("input_shape") or spec.get("inputShape"),
                "blocks": spec.get("blocks") or [],
                "outputHeads": spec.get("output_heads") or spec.get("outputHeads") or [],
                "citations": spec.get("citations") or [],
            }
            r = requests.post(
                f"{base}/projects/{resolved_id}/models",
                headers=headers,
                json=body,
                timeout=30,
            )
            if r.status_code in (200, 201):
                console.print(f"[green]✓[/green] model {spec['id']}")
                pushed += 1
            elif r.status_code == 409 or "unique" in (r.text or "").lower():
                if skip_existing:
                    console.print(f"[yellow]⤳[/yellow] model {spec['id']} already exists — skipped")
                    skipped += 1
                else:
                    console.print(
                        f"[red]✗[/red] model {spec['id']} conflict; pass --no-skip-existing to fail loudly"
                    )
                    failed += 1
            else:
                console.print(f"[red]✗[/red] model {spec['id']}: {r.status_code} {r.text[:200]}")
                failed += 1

    if "evals" in kinds:
        # evals/ may be flat (one YAML per file) or nested (one dir per
        # eval with eval.yaml inside) — accept both shapes.
        eval_yamls = sorted((project_dir / "evals").glob("*.yaml"))
        eval_yamls += sorted((project_dir / "evals").rglob("eval.yaml"))
        for eval_yaml in eval_yamls:
            spec = _y.safe_load(eval_yaml.read_text()) or {}
            # API DTO is camelCase / flat:
            #   task, dataset (object), metrics (array), baselines (array)
            body = {
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
            r = requests.post(
                f"{base}/projects/{resolved_id}/evals",
                headers=headers,
                json=body,
                timeout=30,
            )
            if r.status_code in (200, 201):
                console.print(f"[green]✓[/green] eval {spec['id']}")
                pushed += 1
            elif r.status_code == 409 or "unique" in (r.text or "").lower():
                if skip_existing:
                    console.print(f"[yellow]⤳[/yellow] eval {spec['id']} already exists — skipped")
                    skipped += 1
                else:
                    console.print(
                        f"[red]✗[/red] eval {spec['id']} conflict; pass --no-skip-existing to fail loudly"
                    )
                    failed += 1
            else:
                console.print(f"[red]✗[/red] eval {spec['id']}: {r.status_code} {r.text[:200]}")
                failed += 1

    console.print()
    console.print(f"[bold]done:[/bold] {pushed} pushed, {skipped} skipped, {failed} failed.")
    if failed > 0:
        raise typer.Exit(code=1)


def _die_token() -> None:
    console.print(
        "[red]401 unauthorized.[/red] Issue an API token at the web UI's /api-tokens page, then export PFNSTUDIO_TOKEN=ps_xxx"
    )
    raise typer.Exit(code=1)


# ──────────────────────────────────────────────────────────────────────────
# `pfnstudio author wrap` — generate a PFN Studio prior from any
# Python class with a `sample_batch(batch_size, seed, **kw)` method (or any
# entry-point you can describe).
#
# Runs locally in the user's own venv — that venv must have the source class
# importable (e.g. via `pip install -e .` or `pip install some_pkg`).
# ──────────────────────────────────────────────────────────────────────────
@author_app.command("wrap")
def author_wrap(
    target: str = typer.Option(
        ...,
        "--class",
        help="Importable class, e.g. tcpfn.discovery.discovery_prior:CausalDiscoveryPrior",
    ),
    slug: str = typer.Option(..., help="Slug for the new PFN Studio prior."),
    out: Path = typer.Option(
        Path.cwd(), help="Output project root (priors/<slug>/ will be created here)."
    ),
    requires: list[str] = typer.Option(
        [],
        "--requires",
        "-r",
        help="pip-installable lines for requirements.txt. Repeat the flag (-r tcpfn -r torch>=2.2).",
    ),
    sample_method: str = typer.Option(
        "sample_batch",
        help="Method name on the class that produces samples (default: sample_batch).",
    ),
    name: str = typer.Option(None, help="Display name (default: derived from class name)."),
    kind: str = typer.Option(
        "custom", help="Prior kind: scm, tabular, temporal, temporal_causal, graph, custom."
    ),
) -> None:
    """Generate a PFN Studio-format prior wrapping any Python class.

    Introspects the class constructor to populate parameters, emits a
    PFN Studio-canonical directory:

        priors/<slug>/
          ├── prior.py           — adapter calling the wrapped class
          ├── prior.yaml         — spec with parameters from __init__
          ├── prior.md           — placeholder rationale
          └── requirements.txt   — pip-installable deps

    After running this for each prior you want to migrate, push them with:
        pfnstudio push <out-dir> --project-id <slug>
    """
    import importlib
    import inspect

    err_console = Console(stderr=True)

    if ":" not in target:
        err_console.print("[red]--class must look like 'package.module:ClassName'[/red]")
        raise typer.Exit(code=1)

    module_path, class_name = target.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        err_console.print(f"[red]Could not import {module_path}:[/red] {e}")
        err_console.print(
            "[yellow]Hint:[/yellow] are you in the venv that has this package installed?"
        )
        raise typer.Exit(code=1) from e

    cls = getattr(module, class_name, None)
    if cls is None:
        err_console.print(f"[red]No class named {class_name} in {module_path}[/red]")
        raise typer.Exit(code=1)

    # Introspect __init__ params (skip self, and skip `seed` since the wrapper
    # already exposes seed as its own required argument — emitting it again
    # would cause a duplicate-kwarg SyntaxError).
    SKIP = {"self", "seed"}
    init_accepts_seed = False
    try:
        init_sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        err_console.print(
            f"[yellow]Warning:[/yellow] could not introspect {class_name}.__init__; emitting empty parameters."
        )
        init_params = []
    else:
        init_accepts_seed = "seed" in init_sig.parameters
        init_params = [
            (p.name, p)
            for p in init_sig.parameters.values()
            if p.name not in SKIP
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]

    sample_method_name = sample_method
    has_sample_method = hasattr(cls, sample_method_name)
    if not has_sample_method:
        err_console.print(
            f"[yellow]Warning:[/yellow] {class_name} has no .{sample_method_name}(); the wrapper will be a stub you must edit."
        )

    # ── Generate prior.py ──
    prior_dir = out / "priors" / slug
    prior_dir.mkdir(parents=True, exist_ok=True)

    # Class name for the wrapper, e.g. WrappedCausalDiscoveryPrior
    wrapper_class = "Wrapped" + class_name + ("" if class_name.endswith("Prior") else "Prior")

    init_arg_lines = []
    init_kwargs_pass = []
    for pname, pinfo in init_params:
        default = pinfo.default
        if default is inspect.Parameter.empty:
            init_arg_lines.append(f"        {pname},")
            init_kwargs_pass.append(f"{pname}={pname}")
        else:
            # Python repr of the default. For non-trivial defaults, fall back to a placeholder.
            try:
                rep = repr(default)
                # Avoid emitting class instances or unrepresentable defaults verbatim
                if rep.startswith("<") or "object at 0x" in rep:
                    rep = "None  # FIXME: introspected default could not be serialised"
            except Exception:
                rep = "None  # FIXME: introspected default could not be serialised"
            init_arg_lines.append(f"        {pname}={rep},")
            init_kwargs_pass.append(f"{pname}={pname}")
    if init_accepts_seed:
        # Forward the wrapper's seed into the wrapped class's __init__ so
        # sampling is deterministic per seed even when sample_batch() itself
        # doesn't accept a seed argument.
        init_kwargs_pass.insert(0, "seed=seed")
    init_args_block = "\n".join(init_arg_lines)
    init_kwargs_block = ", ".join(init_kwargs_pass)

    # Build sample_batch call kwargs based on what the method actually accepts.
    sample_batch_kwargs: list[str] = []
    sample_batch_fixme = ""
    if has_sample_method:
        try:
            sb_sig = inspect.signature(getattr(cls, sample_method_name))
            sb_params = sb_sig.parameters
            if "batch_size" in sb_params:
                sample_batch_kwargs.append("batch_size=1")
            if "seed" in sb_params:
                sample_batch_kwargs.append("seed=seed")
            if "n_context_units" in sb_params:
                sample_batch_kwargs.append("n_context_units=10")
                sample_batch_fixme = "        # FIXME: n_context_units / n_query_units default to 10/5; tune for your task.\n"
            if "n_query_units" in sb_params:
                sample_batch_kwargs.append("n_query_units=5")
        except (TypeError, ValueError):
            sample_batch_kwargs = ["batch_size=1"]

    sb_call = ", ".join(sample_batch_kwargs)
    sample_call_block = (
        f"        instance = self._cls({init_kwargs_block})\n"
        f"{sample_batch_fixme}"
        f"        batch = instance.{sample_method_name}({sb_call})\n"
        f"        if isinstance(batch, dict):\n"
        f"            return {{k: _drop_batch_dim(v) for k, v in batch.items()}}\n"
        f"        return {{'sample': batch}}"
        if has_sample_method
        else "        raise NotImplementedError('sample method not found on the wrapped class — edit this file')"
    )

    prior_py = f'''"""Auto-generated wrapper for {module_path}:{class_name}.

Generated by `pfnstudio author wrap`. Edit freely — this is just a starting
point. The PFN Studio sampler will install requirements.txt before importing
this file, so the wrapped class needs to be available via that.
"""

from __future__ import annotations
from typing import Any

from pfnstudio_core import Prior, register_prior

# Import the original class. This lookup must succeed in the cached install
# environment that PFN Studio creates from requirements.txt.
from {module_path} import {class_name} as _Wrapped


def _drop_batch_dim(v: Any) -> Any:
    """If v looks like a tensor/array with a leading batch dim of size 1, drop it."""
    shape = getattr(v, "shape", None)
    if shape is not None and len(shape) > 0 and shape[0] == 1:
        return v[0]
    return v


@register_prior("{slug}")
class {wrapper_class}(Prior):
    """Adapter exposing {class_name} through PFN Studio's Prior API."""

    _cls = _Wrapped

    def sample(
        self,
        *,
        seed: int,
{init_args_block}
        **_: Any,
    ) -> dict[str, Any]:
{sample_call_block}
'''

    (prior_dir / "prior.py").write_text(prior_py)

    # ── Generate prior.yaml ──
    yaml_params: dict = {}
    for pname, pinfo in init_params:
        default = pinfo.default
        ann = pinfo.annotation
        ptype = "float"  # safe default
        # Order matters: bool is a subclass of int in Python, so check bool first.
        if ann is bool:
            ptype = "bool"
        elif ann is int:
            ptype = "int"
        elif ann is float:
            ptype = "float"
        elif ann is str:
            ptype = "str"
        entry: dict = {"type": ptype, "description": f"From {class_name}.__init__"}
        if default is not inspect.Parameter.empty and isinstance(default, (int, float)):
            entry["default"] = default
        yaml_params[pname] = entry

    display_name = name or " ".join(
        word for word in [class_name.replace("Prior", ""), "Prior"] if word
    )

    yaml_body = f"""id: {slug}
name: {display_name}
version: 0.1.0
kind: {kind}
description: |
  Wrapped from {module_path}:{class_name}.
  Generated by `pfnstudio author wrap`. Edit this file to fill in
  output variables and tighten parameter ranges.

parameters:
"""
    for pname, entry in yaml_params.items():
        yaml_body += f"  {pname}:\n"
        for k, v in entry.items():
            if isinstance(v, str):
                yaml_body += f'    {k}: "{v}"\n'
            else:
                yaml_body += f"    {k}: {v}\n"

    yaml_body += """
outputs:
  variables:
    # FIXME: fill in based on what the wrapped sample method returns.
    - name: data
      type: tensor
      description: Sample output from the wrapped class

implementation: prior.py
"""
    (prior_dir / "prior.yaml").write_text(yaml_body)

    # ── Generate prior.md ──
    md = f"""# {display_name}

Wrapper around `{module_path}:{class_name}`.

## TODO before publishing

- [ ] Verify the `sample()` method correctly unwraps the batch returned by the original class
- [ ] Fill in `outputs.variables` in prior.yaml with actual shapes and descriptions
- [ ] Tighten parameter ranges in prior.yaml (e.g., add `range: [lo, hi]` for sliders)
- [ ] Test locally with `pfnstudio sample <project-dir> {slug} --count 3`
- [ ] Push with `pfnstudio push <project-dir> --project-id <project>`

## Why this prior exists

(Edit this section to describe the pedagogical or research purpose.)
"""
    (prior_dir / "prior.md").write_text(md)

    # ── requirements.txt ──
    if requires:
        (prior_dir / "requirements.txt").write_text("\n".join(requires) + "\n")

    console.print(f"[green]✓[/green] Wrote {prior_dir}/")
    console.print(
        f"  Files: prior.py, prior.yaml, prior.md{', requirements.txt' if requires else ''}"
    )
    console.print()
    console.print("[bold]Next:[/bold]")
    console.print(f"  1. Edit [cyan]{prior_dir}/prior.yaml[/cyan] — fill in outputs[].variables")
    console.print(f"  2. Edit [cyan]{prior_dir}/prior.py[/cyan] — verify the sample() unwrap logic")
    console.print(f"  3. Test: [cyan]pfnstudio sample {out} {slug} --count 3[/cyan]")
    console.print(f"  4. Push: [cyan]pfnstudio push {out} --project-id <slug>[/cyan]")


@app.command()
def run(
    what: str = typer.Argument(
        ...,
        metavar="RUN_YAML|MODEL[:TAG]|NAME",
        help="A run YAML to execute (runs/<slug>.yaml), a marketplace base-model "
        "ref — 'cdfm-base', 't0-alpha-base:latest' — or the name of a "
        "model made with `pfnstudio create`.",
    ),
    project: Path = typer.Option(
        None, help="Project root (default: parent of runs/). Run-YAML form only."
    ),
    target: str = typer.Option(
        None,
        help=f"Override compute target. Available: {sorted(ADAPTERS)}. Run-YAML form only.",
    ),
    api_url: str = typer.Option(
        "https://cloud.pfnstudio.com/api",
        "--api-url",
        envvar="PFNSTUDIO_API_URL",
        help="PFN Studio API base (public marketplace listings are unauthenticated). "
        "Defaults to the hosted marketplace; point it at a local studio with "
        "--api-url or PFNSTUDIO_API_URL. Model form only.",
    ),
    basemodels: Path = typer.Option(
        None,
        "--basemodels",
        envvar="PFNSTUDIO_BASEMODELS_DIR",
        help="Directory containing <fmProject>/adapters/. Auto-detected from a "
        "repo checkout when omitted. Model form only.",
    ),
    task: str = typer.Option(
        None,
        help="Which capability task to run (default: the first the model declares). "
        "Model form only.",
    ),
    input_json: Path = typer.Option(
        None,
        "--input",
        help="JSON payload to run instead of the built-in self-diagnostic "
        "(same shape the web Try-it posts). Omit to run the model's own "
        "held-out sample. Model form only.",
    ),
    install_deps: bool = typer.Option(
        False,
        "--install-deps",
        help="pip-install what the model needs before running: PyTorch when it "
        "is missing, then the adapter's requirements.txt (its upstream model "
        "package). Off by default: nothing is installed silently. Model form only.",
    ),
) -> None:
    """Run a training manifest, or a base-model PFN from the marketplace.

    `pfnstudio run runs/<slug>.yaml` executes that run via the configured (or
    --target) compute adapter — the path the studio API and the self-hosted
    runner spawn.

    `pfnstudio run <slug>[:tag]` is the fastest way to see a PFN work, à la
    `ollama run`: resolves the listing by slug (tag defaults to "latest") from
    the public marketplace, pulls its adapter bundle from the marketplace and
    its pretrained checkpoint from HuggingFace (both cached), writes a
    lock.json next to the bundle pinning exactly what was resolved (one
    `digest sha256:…` line — the same inputs always give the same digest), then
    loads the model into the local daemon (`pfnstudio serve`, started on demand
    like `ollama serve`) and runs a sample through it, printing the tasks it
    supports and a live result. The model stays loaded for the next call. No
    repo checkout needed; a checkout or --basemodels still takes precedence.
    With no --input, it runs the adapter's built-in self-diagnostic on held-out
    data, which proves the real weights load AND predict.

    `pfnstudio run <name>` for a model made with `pfnstudio create` sends the
    request to that model's run in the studio (it lives there, not locally).
    """
    # A path (an existing file, a .yaml/.yml name, or anything with a separator)
    # is a run manifest; a name `create` wrote an alias for is a studio model;
    # every other token is a marketplace model ref.
    is_manifest = (
        Path(what).is_file() or what.endswith((".yaml", ".yml")) or "/" in what or "\\" in what
    )
    if is_manifest:
        _run_manifest(Path(what), project, target)
    elif alias_path(what).is_file():
        _run_alias(what, task, input_json)
    else:
        _run_model(what, api_url, basemodels, task, input_json, install_deps)


def _run_manifest(manifest: Path, project: Path | None, target: str | None) -> None:
    """Execute a run via the configured (or overridden) compute adapter."""
    manifest = manifest.resolve()
    if project is None:
        if manifest.parent.name == "runs":
            project = manifest.parent.parent
        else:
            project = Path.cwd()
    project = project.resolve()

    import yaml as _y

    spec = _y.safe_load(manifest.read_text())
    chosen_target = target or (spec.get("compute") or {}).get("target", "local")
    console.print(f"[blue]→[/blue] dispatching {spec.get('id')} via [bold]{chosen_target}[/bold]")

    adapter = get_adapter(chosen_target)

    from pfnstudio_core.run import RunSpec

    run_spec = RunSpec.model_validate(spec)
    trackers = select_trackers(run_spec.tracking)
    for t in trackers:
        t.start(run_id=run_spec.id, config=spec, project_root=project)

    try:
        results = adapter.submit(run_yaml=manifest, project_root=project)
    finally:
        for t in trackers:
            t.finish(results if "results" in dir() else {})

    console.print_json(json.dumps(results, default=str))
    if results.get("status") == "error":
        raise typer.Exit(code=1)


@app.command()
def predict(
    manifest: Path = typer.Argument(..., help="Path to a run YAML."),
    checkpoint: str = typer.Option(
        ...,
        help=(
            "Checkpoint reference. One of: (a) absolute local path to a "
            "directory containing model.pt + topology.json, (b) a single "
            "HF URI like hf://OWNER/REPO/PATH, or (c) a JSON-encoded "
            "manifest object mapping local filenames to URIs. The studio "
            "API passes (c) when the catalog template ships a multi-file "
            "checkpoint published on HuggingFace; see "
            "serving/hf.py:resolve_checkpoint for the full grammar."
        ),
    ),
    input_json: Path = typer.Option(
        None,
        "--input",
        help="Path to a JSON file with the inference payload. Reads stdin if omitted.",
    ),
    project: Path = typer.Option(None, help="Project root (default: parent of runs/)."),
) -> None:
    """Load a trained checkpoint and run inference.

    The input JSON shape depends on the prior. For a regression prior
    (linear-regression-style PFN) it should be::

        {
          "context": {"x": [[1.0],[2.0]], "y": [2.0, 4.1]},
          "query":   {"x": [[3.0],[4.0]]}
        }

    Output JSON shape::

        {"predictions": [<y for each query x>]}

    Output is emitted on stdout so the API can capture it. Errors are
    surfaced as ``{"error": "..."}`` with a non-zero exit code.
    """
    import json as _json
    import sys as _sys

    manifest = manifest.resolve()
    if project is None:
        project = manifest.parent.parent if manifest.parent.name == "runs" else Path.cwd()
    project = project.resolve()

    payload_text = input_json.read_text() if input_json is not None else _sys.stdin.read()
    try:
        payload = _json.loads(payload_text)
    except _json.JSONDecodeError as e:
        console.print_json(_json.dumps({"error": f"invalid input JSON: {e}"}))
        raise typer.Exit(code=1) from e

    # Lazy-import the inference module so the CLI's other commands
    # (init / validate / lint) don't pay torch's import cost.
    try:
        from pfnstudio_core.training.predict import run_inference
    except ImportError as e:
        console.print_json(_json.dumps({"error": f"predict module unavailable: {e}"}))
        raise typer.Exit(code=1) from e

    # Optional `tag` field for promptable priors. Extracted from the
    # top-level payload and passed as a separate argument so the rest
    # of the payload (context/query/X) keeps its existing shape.
    # Brains trained without promptable_training ignore the tag (no
    # tag_embedder → no tag pathway).
    tag = payload.pop("tag", None) if isinstance(payload, dict) else None
    # Optional `detect` flag — when True and the checkpoint ships a
    # companion detector, the response includes a `detected_tag`
    # field with the detector's per-axis proposal.
    detect_flag = bool(payload.pop("detect", False)) if isinstance(payload, dict) else False

    # Resolve --checkpoint into a local directory. For plain filesystem
    # paths this is a no-op; for HF URIs / manifests it pulls (and
    # caches) the files into ~/.cache/pfnstudio/checkpoints/... before
    # the predict module can load them. The resolver is the single
    # entrypoint that knows about all the catalog-supported forms.
    try:
        from pfnstudio_core.serving.hf import resolve_checkpoint
    except ImportError as e:
        console.print_json(_json.dumps({"error": f"checkpoint resolver unavailable: {e}"}))
        raise typer.Exit(code=1) from e

    # A capabilities request answers a static manifest BEFORE any weights are
    # loaded, so it needs no checkpoint. Skip the resolve — otherwise a private
    # or unpublished HF checkpoint the caller can't pull (e.g. a self-hosted
    # runner without the org's HF token) would 404 here and the Try-it /
    # Inference surfaces would fall back to the legacy form even though the
    # adapter can describe its tasks perfectly well. Pass the raw path through
    # unresolved; the capabilities branch never reads it.
    is_capabilities = isinstance(payload, dict) and payload.get("__capabilities__")
    if is_capabilities:
        resolved_checkpoint_dir = Path(str(checkpoint))
    else:
        try:
            resolved_checkpoint_dir = resolve_checkpoint(checkpoint)
        except Exception as e:
            console.print_json(
                _json.dumps(
                    {
                        "error": (
                            f"could not resolve checkpoint: {type(e).__name__}: {e}. "
                            "If the catalog template references an unpublished "
                            "checkpoint (e.g. hf://<org>/<repo>/<file-not-yet-uploaded>), "
                            "the Solution authors need to finish training and "
                            "upload to HuggingFace before this Solution can run."
                        ),
                    }
                )
            )
            raise typer.Exit(code=1) from e

    try:
        out = run_inference(
            manifest_path=manifest,
            project_root=project,
            checkpoint_dir=resolved_checkpoint_dir,
            payload=payload,
            tag=tag,
            detect=detect_flag,
        )
    except Exception as e:
        console.print_json(
            _json.dumps(
                {
                    "error": f"{type(e).__name__}: {e}",
                }
            )
        )
        raise typer.Exit(code=1) from e

    # Emit ONLY the JSON output on stdout so the API can parse it
    # cleanly. Anything else (progress, logs) should go on stderr.
    _sys.stdout.write(_json.dumps(out) + "\n")
    _sys.stdout.flush()


def _fetch_basemodel_bundle(api_base: str, slug: str, fm_project: str) -> Path:
    """Pull basemodels/<fmProject>/ (adapter, yaml, requirements) from the public
    marketplace and lay it out under the local basemodels cache — the same move
    `run` already makes for the checkpoint, so no checkout is needed. Returns the
    fm-project dir. Every entry's path is validated before it is written."""
    import os

    import requests

    resp = requests.get(f"{api_base}/marketplace/basemodels/{slug}/bundle", timeout=30)
    resp.raise_for_status()
    bundle = resp.json()
    files = bundle.get("files") if isinstance(bundle, dict) else None
    if not isinstance(files, list) or not files:
        raise RuntimeError("the marketplace returned an empty bundle")
    cache_root = Path(
        os.environ.get(
            "PFNSTUDIO_BASEMODELS_CACHE",
            str(Path.home() / ".cache" / "pfnstudio" / "basemodels"),
        )
    )
    safe = "".join(ch for ch in fm_project if ch.isalnum() or ch in "._-")
    if not safe or safe in (".", "..") or ".." in safe:
        raise RuntimeError(f"invalid fmProject id: {fm_project!r}")
    dest = (cache_root / safe).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for entry in files:
        rel = str((entry or {}).get("path") or "")
        content = (entry or {}).get("content")
        p = Path(rel)
        if not rel or p.is_absolute() or ".." in p.parts or not isinstance(content, str):
            raise RuntimeError(f"refusing unsafe bundle entry: {rel!r}")
        target = (dest / p).resolve()
        if target != dest and dest not in target.parents:
            raise RuntimeError(f"refusing bundle entry outside the cache: {rel!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return dest


def _find_basemodels_dir(fm_project: str, override: Path | None) -> Path | None:
    """Locate the on-disk fm-project (basemodels/<fmProject>/) that carries the
    adapter needed to run a base model. Order: explicit --basemodels override →
    walk up from cwd looking for basemodels/<fmProject>. Returns the fm-project
    dir or None when it can't be found (a pip-installed CLI outside a checkout)."""
    if override is not None:
        cand = override / fm_project
        return cand if (cand / "adapters").is_dir() or cand.is_dir() else None
    here = Path.cwd().resolve()
    for base in [here, *here.parents]:
        cand = base / "basemodels" / fm_project
        if cand.is_dir():
            return cand
    return None


# Same spec as the `torch` extra of pfnstudio-core (`pip install "pfnstudio[torch]"`).
_TORCH_SPEC = "torch>=2.2"


def _ensure_torch(*, install: bool, ref: str) -> None:
    """Every base model runs on PyTorch, which plain `pip install pfnstudio`
    leaves out (it is a large download the authoring commands never need).

    With ``install`` (``--install-deps``) a missing torch is pip-installed into
    this Python. Without it, stop with the one-line fix instead of failing later
    inside the daemon on a bare ImportError. Torch already present: no-op.
    """
    import importlib
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("torch") is not None:
        return
    if not install:
        console.print(
            "[red]PyTorch isn't installed, and every base model needs it.[/red]\n"
            f"  [bold]pfnstudio run {ref} --install-deps[/bold]   [dim]installs it, then the model's own packages[/dim]\n"
            '  [dim]or[/dim] pip install "pfnstudio[torch]"'
        )
        raise typer.Exit(code=1)
    console.print(
        f"[dim]Installing PyTorch ({_TORCH_SPEC}) — needed by every base model; "
        "a one-time download of 1–3 GB…[/dim]"
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", _TORCH_SPEC])
    importlib.invalidate_caches()


def _run_model(
    ref: str,
    api_url: str,
    basemodels: Path | None,
    task: str | None,
    input_json: Path | None,
    install_deps: bool,
) -> None:
    """The `pfnstudio run <slug>[:tag]` path: resolve listing → bundle →
    checkpoint, pin them in lock.json, then run the adapter's sample through
    the local daemon."""
    import json as _json
    import os as _os
    import sys as _sys

    import requests

    def err(msg: str) -> None:
        console.print(f"[red]{msg}[/red]")

    try:
        slug, tag = parse_model_ref(ref)
    except ValueError as e:
        err(str(e))
        raise typer.Exit(code=1) from e

    # 1) Resolve the listing from the public marketplace catalog.
    base = api_url.rstrip("/")
    try:
        resp = requests.get(f"{base}/marketplace/listings", timeout=20)
        resp.raise_for_status()
        listings = resp.json()
    except Exception as e:
        err(f"Couldn't reach the marketplace at {base}/marketplace/listings: {e}")
        raise typer.Exit(code=1) from e

    base_models = [
        x
        for x in (listings if isinstance(listings, list) else [])
        if isinstance(x, dict) and x.get("type") == "basemodel"
    ]
    listing = next((x for x in base_models if x.get("slug") == slug), None)
    if listing is None:
        err(f"No base model with slug '{slug}'.")
        if base_models:
            avail = ", ".join(sorted(str(x.get("slug")) for x in base_models))
            console.print(f"[dim]Available base models:[/dim] {avail}")
        raise typer.Exit(code=1)

    title = str(listing.get("title") or slug)
    console.print(f"\n[bold]{title}[/bold]")
    if listing.get("tagline"):
        console.print(f"[dim]{listing['tagline']}[/dim]")
    if listing.get("paramsLabel"):
        console.print(f"[dim]{listing['paramsLabel']}[/dim]")

    fm_project = listing.get("fmProject")
    adapter_name = listing.get("baseAdapter")
    # Version tags: a listing may define them as a {tag: checkpointRef | null}
    # map under "tags" (null = the listing's own checkpointRef). A plain list
    # under "tags" is the catalog's topic tags, not versions — only "latest".
    version_tags = listing.get("tags") if isinstance(listing.get("tags"), dict) else {}
    available_tags = sorted({"latest", *version_tags})
    if tag not in available_tags:
        err(f"'{slug}' has no tag '{tag}'.")
        console.print(f"[dim]Available tags:[/dim] {', '.join(available_tags)}")
        raise typer.Exit(code=1)
    checkpoint_ref = version_tags.get(tag) or listing.get("checkpointRef")
    if not checkpoint_ref:
        err("This base model has no checkpoint reference — nothing to pull.")
        raise typer.Exit(code=1)

    # 2) Locate the adapter (needed to describe + run the model): a checkout or
    #    --basemodels first; otherwise pull the bundle from the marketplace.
    fm_dir = _find_basemodels_dir(str(fm_project), basemodels) if fm_project else None
    if fm_dir is None and fm_project:
        try:
            fm_dir = _fetch_basemodel_bundle(base, slug, str(fm_project))
            console.print(f"[green]✓[/green] Adapter bundle cached at [bold]{fm_dir}[/bold]")
        except Exception as e:
            console.print(
                f"[yellow]Couldn't fetch the adapter bundle from the marketplace:[/yellow] {e}"
            )
            fm_dir = None

    # The checkpoint resolver + inference module live in core; import lazily so
    # `pfnstudio --help` doesn't pay torch's import cost.
    try:
        from pfnstudio_core.serving.hf import resolve_checkpoint
    except ImportError as e:
        err(f"checkpoint resolver unavailable: {e}")
        raise typer.Exit(code=1) from e

    ref_arg = (
        _json.dumps(checkpoint_ref) if isinstance(checkpoint_ref, dict) else str(checkpoint_ref)
    )

    if fm_dir is None:
        # No adapter on disk — we can still pull the weights so they're cached,
        # but running needs the adapter. Tell the user exactly how to get it.
        console.print(
            "\n[yellow]Adapter not found locally[/yellow] — pulling the checkpoint anyway."
        )
        try:
            ckpt = resolve_checkpoint(ref_arg)
        except Exception as e:
            err(f"Couldn't resolve checkpoint: {type(e).__name__}: {e}")
            raise typer.Exit(code=1) from e
        console.print(f"[green]✓[/green] Checkpoint cached at [bold]{ckpt}[/bold]")
        console.print(
            "\nTo RUN it, point --basemodels at a checkout that has "
            f"[bold]basemodels/{fm_project}/[/bold] (the adapter), then re-run:\n"
            f"  [dim]pfnstudio run {slug}:{tag} --basemodels /path/to/basemodels[/dim]"
        )
        return

    # 2b) PyTorch, then the upstream model package — only on request. Torch is
    #     checked before the (large) checkpoint download so a missing install
    #     fails fast with the fix, not after minutes of pulling weights.
    _ensure_torch(install=install_deps, ref=f"{slug}:{tag}")
    req_file = fm_dir / "requirements.txt"
    if install_deps and req_file.is_file() and req_file.read_text(encoding="utf-8").strip():
        import subprocess as _subprocess

        console.print(f"[dim]Installing adapter dependencies from {req_file}…[/dim]")
        _subprocess.check_call([_sys.executable, "-m", "pip", "install", "-r", str(req_file)])

    # 3) Resolve the checkpoint into a local dir (pulls + caches from HF).
    console.print("\n[dim]Pulling checkpoint…[/dim]", style=None)
    try:
        ckpt_dir = resolve_checkpoint(ref_arg)
    except Exception as e:
        err(f"Couldn't resolve checkpoint: {type(e).__name__}: {e}")
        raise typer.Exit(code=1) from e

    # 3b) Pin what was resolved: hash the bundle + weights into lock.json next
    #     to the bundle and print the one-line digest (same inputs → same digest).
    lock = build_lock(
        slug=slug,
        tag=tag,
        fm_project=str(fm_project),
        base_adapter=adapter_name,
        bundle_dir=fm_dir,
        checkpoint_dir=ckpt_dir,
        checkpoint_ref=checkpoint_ref,
    )
    lock_path = write_lock(fm_dir, lock)
    console.print(f"[green]✓[/green] Lock written to [bold]{lock_path}[/bold]")
    _sys.stdout.write(f"digest {lock['digest']}\n")
    _sys.stdout.flush()

    # 4) Load it into the local daemon (started on demand, like `ollama serve`)
    #    and talk to it over HTTP — the CLI is a client, never a model host.
    #    The daemon resolves the same caches this command just filled, so the
    #    load is a weights load, not a second download. A fresh daemon inherits
    #    this command's --api-url / --basemodels through the environment; one
    #    that is already running keeps the settings it started with.
    from . import client

    _os.environ["PFNSTUDIO_API_URL"] = base
    if basemodels is not None:
        _os.environ["PFNSTUDIO_BASEMODELS_DIR"] = str(basemodels)

    def daemon_err(prefix: str, e: Exception) -> None:
        detail = e.detail if isinstance(e, client.DaemonError) else None
        if isinstance(detail, dict):
            err(f"{prefix}: {detail.get('detail') or detail.get('error') or detail}")
            if detail.get("error") == "missing_dependency":
                console.print(
                    "[dim]Install the adapter's upstream package into the daemon's Python:[/dim]\n"
                    f"  pip install -r {fm_dir / 'requirements.txt'}\n"
                    "[dim]or re-run with[/dim] --install-deps[dim]; a daemon that was already "
                    "running needs a restart to see new packages.[/dim]"
                )
            elif detail.get("hint"):
                console.print(f"[dim]{detail['hint']}[/dim]")
        else:
            err(f"{prefix}: {detail if detail is not None else e}")

    try:
        client.ensure_running()
    except RuntimeError as e:
        err(f"local daemon unavailable: {e}")
        raise typer.Exit(code=1) from e

    console.print(f"\n[dim]Loading into the daemon at {client.base_url()}…[/dim]")
    try:
        info = client.load(f"{slug}:{tag}")
    except (client.DaemonError, Exception) as e:
        daemon_err("load failed", e)
        raise typer.Exit(code=1) from e
    state = "already loaded" if info.get("already_loaded") else "loaded"
    console.print(f"[green]✓[/green] {slug} {state} at {client.base_url()}/models/{slug}/")

    # 4a) Capabilities — what tasks does this model support?
    try:
        caps_body = client.capabilities(slug)
    except Exception as e:
        daemon_err("capabilities failed", e)
        raise typer.Exit(code=1) from e
    caps_result = (
        caps_body.get("result") if isinstance(caps_body.get("result"), dict) else caps_body
    )
    caps = caps_result.get("capabilities") if isinstance(caps_result, dict) else None
    caps = caps if isinstance(caps, dict) else {}
    if caps:
        console.print(f"\n[bold]Tasks[/bold] ({len(caps)}):")
        for key, spec in caps.items():
            label = (spec or {}).get("label") or key
            console.print(f"  • [cyan]{key}[/cyan] — {label}")
    chosen = task or (next(iter(caps)) if caps else None)

    # 4b) Run it. --input if given, else the adapter's self-diagnostic
    #     (empty inputs → the adapter samples a held-out task and reports
    #     predicted-vs-oracle, proving the real weights work).
    if input_json is not None:
        payload = _json.loads(input_json.read_text())
    elif chosen:
        payload = {"task": chosen}
    else:
        payload = {}

    console.print(f"\n[dim]Running{f' {chosen}' if chosen else ''}…[/dim]")
    try:
        out = client.predict(slug, payload)
    except Exception as e:
        daemon_err("inference failed", e)
        raise typer.Exit(code=1) from e

    _print_model_result(out)


def _print_model_result(out: dict) -> None:
    """Shared tail of the model and alias forms of `run`: the result JSON on
    stdout, or the model's own error with the --input hint."""
    import json as _json
    import sys as _sys

    result = out.get("result", out) if isinstance(out, dict) else out
    if isinstance(result, dict) and result.get("error"):
        console.print(f"[red]model returned an error: {result['error']}[/red]")
        console.print(
            "[dim]Some adapters need real input data — pass one with "
            "--input <payload.json> (same shape the web Try-it posts).[/dim]"
        )
        raise typer.Exit(code=1)

    console.print("\n[green]✓ Result[/green]")
    _sys.stdout.write(_json.dumps(result, indent=2) + "\n")
    _sys.stdout.flush()


def _run_alias(name: str, task: str | None, input_json: Path | None) -> None:
    """The `pfnstudio run <name>` path for a model made with `pfnstudio create`.

    `create` writes ~/.cache/pfnstudio/models/<name>.json pointing at the run it
    composed in the studio; the model lives there (weights on the studio's
    runner), so this sends the request to that run's predict endpoint with the
    same token `create` used. Studio-side errors are reported as returned."""
    import json as _json

    from .create_cli import _request, _token

    record = _json.loads(alias_path(name).read_text(encoding="utf-8"))
    api = str(record.get("api") or "").rstrip("/")
    pid, run_id = record.get("projectId"), record.get("runId")
    if not (api and pid and run_id):
        console.print(
            f"[red]alias {alias_path(name)} is missing api/projectId/runId — re-create the model.[/red]"
        )
        raise typer.Exit(code=1)
    src = record.get("from") or {}
    console.print(
        f"\n[bold]{name}[/bold] → run {run_id} in project {record.get('projectSlug') or pid}"
    )
    if src.get("slug"):
        console.print(
            f"[dim]FROM {src['slug']}:{src.get('tag') or 'latest'} digest {src.get('digest') or 'null'}[/dim]"
        )

    tok = _token(None)
    runs = f"{api}/projects/{pid}/runs/{run_id}"
    caps_body = _request("GET", f"{runs}/capabilities", token=tok)
    caps_result = (
        caps_body.get("result") if isinstance(caps_body.get("result"), dict) else caps_body
    )
    caps = caps_result.get("capabilities") if isinstance(caps_result, dict) else None
    caps = caps if isinstance(caps, dict) else {}
    if caps:
        console.print(f"\n[bold]Tasks[/bold] ({len(caps)}):")
        for key, spec in caps.items():
            console.print(f"  • [cyan]{key}[/cyan] — {(spec or {}).get('label') or key}")
    chosen = task or (next(iter(caps)) if caps else None)

    if input_json is not None:
        payload = _json.loads(input_json.read_text())
    elif chosen:
        payload = {"task": chosen}
    else:
        payload = {}

    console.print(f"\n[dim]Running{f' {chosen}' if chosen else ''} in the studio…[/dim]")
    # Inference blocks until the studio's runner answers; no client-side deadline.
    out = _request("POST", f"{runs}/predict", token=tok, json=payload, timeout=None)
    _print_model_result(out)


@app.command()
def serve(
    manifest: Path = typer.Argument(
        None,
        help="Path to a run YAML — serve that one trained model (container / "
        "Deployments form). Omit it to run the local model daemon instead.",
    ),
    checkpoint: Path = typer.Option(
        None,
        help="Path to the checkpoint directory (contains model.pt + topology.json). "
        "Required with a manifest.",
    ),
    project: Path = typer.Option(
        None, help="Project root (default: parent of runs/). Manifest form only."
    ),
    host: str = typer.Option(
        "127.0.0.1",
        help="Bind address. Defaults to loopback — only this machine's clients.",
    ),
    port: int = typer.Option(
        0,
        help="TCP port to bind. Manifest form: 0 lets the OS pick a free port, "
        'emitted in the {"event":"ready"} stdout line. Daemon form: 0 means '
        "the daemon's default 11440 (what `pfnstudio run` connects to).",
    ),
    keep_alive: float = typer.Option(
        None,
        "--keep-alive",
        help="Daemon form only: seconds a loaded model may sit idle before it is "
        "unloaded. Default: never — models stay until `pfnstudio rm` or exit.",
    ),
) -> None:
    """Serve one trained model, or run the local model daemon.

    With a manifest: load that checkpoint once and serve predictions over
    HTTP. Used by the Deployments path and the model container: the studio
    API spawns one of these per Deployment, watches stdout for the structured
    event stream ({"event":"starting"|"ready"|"error"|"shutdown",...}), and
    forwards user predict() calls to ``POST /predict`` on the reported port.
    Blocks until SIGTERM/SIGINT; a loader failure exits with code 1 after
    emitting {"event":"error","stage":"load",...}.

    Without one: run the daemon in the foreground, à la `ollama serve` — one
    process hosting many marketplace base models, each under
    /models/<slug>/ with the container's exact API (/predict, /capabilities,
    /mcp, /docs), plus /ps, /load and /unload for the CLI. `pfnstudio run`
    starts this on demand when it is not already running.
    """
    import sys as _sys

    if manifest is None:
        from .daemon import DEFAULT_PORT, main  # lazy: fastapi/uvicorn only when serving

        main(host, port or DEFAULT_PORT, keep_alive=keep_alive)
        return
    if checkpoint is None:
        _sys.stderr.write("--checkpoint is required when serving a manifest\n")
        raise typer.Exit(code=2)

    manifest = manifest.resolve()
    if project is None:
        project = manifest.parent.parent if manifest.parent.name == "runs" else Path.cwd()
    project = project.resolve()

    # Lazy import — keeps torch / serving deps out of the lightweight
    # CLI commands' import path (init / validate / lint).
    try:
        from pfnstudio_core.serving import serve as _serve
    except ImportError as e:
        _sys.stderr.write(f"serving module unavailable: {e}\n")
        raise typer.Exit(code=1) from e

    _serve(
        manifest_path=manifest,
        project_root=project,
        checkpoint_dir=checkpoint.resolve(),
        host=host,
        port=port,
    )


@app.command("hf-push")
def hf_push(
    folder: Path = typer.Argument(..., help="Local folder to publish."),
    repo_id: str = typer.Option(..., "--repo-id", help='"owner/repo-name" form.'),
    revision: str = typer.Option("main", help="Branch to commit on."),
    private: bool = typer.Option(True, "--private/--public", help="Private repo?"),
    commit_message: str = typer.Option(
        None, "--commit-message", help="Override the default commit message."
    ),
    token_env: str = typer.Option(
        "HF_TOKEN",
        help="Env var holding the HF write token. Passing the token via env "
        "(instead of a flag) keeps it out of `ps` listings.",
    ),
) -> None:
    """Publish a folder to a HuggingFace model repo.

    Used by the Deployments promote flow: NestJS exports the trained
    run's project tree + checkpoint to a local dir, then invokes this
    command to mirror it into the user's HF account. Output JSON on
    stdout: ``{"commit_sha": "...", "url": "..."}``.
    """
    import json as _json
    import os as _os
    import sys as _sys

    token = _os.environ.get(token_env, "").strip()
    if not token:
        _sys.stderr.write(f"missing token in env var {token_env}\n")
        raise typer.Exit(code=1)

    try:
        from pfnstudio_core.serving.hf import push_folder
    except ImportError as e:
        _sys.stderr.write(f"huggingface_hub not installed: {e}. Run: pip install pfnstudio[hf]\n")
        raise typer.Exit(code=1) from e

    try:
        out = push_folder(
            local_dir=folder.resolve(),
            repo_id=repo_id,
            token=token,
            revision=revision,
            private=private,
            commit_message=commit_message,
        )
    except Exception as e:
        _sys.stdout.write(_json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
        _sys.stdout.flush()
        raise typer.Exit(code=1) from e

    _sys.stdout.write(_json.dumps(out) + "\n")
    _sys.stdout.flush()


@app.command("hf-pull")
def hf_pull(
    repo_id: str = typer.Argument(..., help='"owner/repo-name" form.'),
    dest: Path = typer.Option(..., help="Local directory to materialize into."),
    revision: str = typer.Option("main", help="Branch / commit to pull."),
    token_env: str = typer.Option(
        "HF_TOKEN",
        help="Env var holding the HF read-or-write token. Private repos need a real token.",
    ),
) -> None:
    """Materialize a HuggingFace model repo into a local directory.

    Used by the deployment worker boot path: when artifactRef.kind == 'hf',
    NestJS calls this command to populate the worker's workspace before
    spawning `pfnstudio serve`.
    """
    import json as _json
    import os as _os
    import sys as _sys

    token = _os.environ.get(token_env, "").strip()
    if not token:
        _sys.stderr.write(f"missing token in env var {token_env}\n")
        raise typer.Exit(code=1)

    try:
        from pfnstudio_core.serving.hf import pull_snapshot
    except ImportError as e:
        _sys.stderr.write(f"huggingface_hub not installed: {e}. Run: pip install pfnstudio[hf]\n")
        raise typer.Exit(code=1) from e

    try:
        out_dir = pull_snapshot(
            repo_id=repo_id,
            dest_dir=dest.resolve(),
            token=token,
            revision=revision,
        )
    except Exception as e:
        _sys.stdout.write(_json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
        _sys.stdout.flush()
        raise typer.Exit(code=1) from e

    _sys.stdout.write(_json.dumps({"dest": str(out_dir)}) + "\n")
    _sys.stdout.flush()


@app.command("resolve-checkpoint")
def resolve_checkpoint_cmd(
    ref: str = typer.Option(
        ...,
        help="checkpoint_ref: a local path, a single hf://OWNER/REPO/PATH URI, "
        'or a JSON manifest map {"model.pt": "hf://…"}.',
    ),
    dest: Path = typer.Option(..., help="Local directory to materialize weights into."),
    all_files: bool = typer.Option(
        False,
        "--all",
        help=(
            "Copy EVERY file in the resolved checkpoint instead of just "
            "model.pt + topology.json. Required when the consumer is a base "
            "model's own adapter (one that reads, say, model.safetensors "
            "rather than model.pt) — see the command help."
        ),
    ),
    token_env: str = typer.Option(
        "HF_TOKEN",
        help="Env var holding the HF read token (private base-model repos need it).",
    ),
) -> None:
    """Resolve a base-model `checkpoint_ref` and stage it into `--dest`.

    Two modes, because the two consumers want different things:

      * default — stage `model.pt` (+ optional `topology.json`), flattening a
        single-URI ref onto the trainer's expected name. NestJS's runner calls
        this when a warm-start parent's checkpointPath is a remote ref (an
        installed base model) rather than a local run directory, so the
        training loop's `init_checkpoint/model.pt` loader works unchanged.

      * ``--all`` — copy the resolved checkpoint verbatim. An adapter loads a
        base model through its OWN architecture and reads whatever file names
        that model publishes (e.g. `model.safetensors` rather than `model.pt`);
        reducing the directory to `model.pt` silently strips what it needs.
        The Deployments promote flow uses this so a served base model sees the
        same bytes `pfnstudio predict` resolves for it.
    """
    import json as _json
    import os as _os
    import shutil as _shutil
    import sys as _sys

    token = _os.environ.get(token_env, "").strip() or None
    try:
        from pfnstudio_core.serving.hf import resolve_checkpoint

        resolved_dir = resolve_checkpoint(ref, token=token)
        dest.mkdir(parents=True, exist_ok=True)
        staged: list[str] = []

        if all_files:
            # Verbatim copy. `dirs_exist_ok` so a re-resolve over a populated
            # dest refreshes it rather than raising. A resolved checkpoint is a
            # flat weights directory (no venv, no dataset), so there is nothing
            # to prune here.
            _shutil.copytree(resolved_dir, dest, dirs_exist_ok=True)
            staged = sorted(p.name for p in dest.iterdir() if p.is_file())
            if not staged:
                raise FileNotFoundError(f"resolved checkpoint {resolved_dir} is empty")
            _sys.stdout.write(_json.dumps({"dest": str(dest.resolve()), "files": staged}) + "\n")
            _sys.stdout.flush()
            return

        # model.pt is required; topology.json is optional shape metadata.
        src_model = resolved_dir / "model.pt"
        if not src_model.exists():
            # Single-URI refs land under the URI basename; take the first
            # weights file in the resolved dir. Covers bare state_dicts (.pt)
            # and Lightning/submitit checkpoints (.ckpt / .cpkt) — the trainer
            # unwraps the container before loading.
            cands = (
                sorted(resolved_dir.glob("*.pt"))
                + sorted(resolved_dir.glob("*.ckpt"))
                + sorted(resolved_dir.glob("*.cpkt"))
            )
            if not cands:
                raise FileNotFoundError(f"no weights (.pt/.ckpt/.cpkt) found under {resolved_dir}")
            src_model = cands[0]
        _shutil.copyfile(src_model, dest / "model.pt")
        staged.append("model.pt")
        src_topo = resolved_dir / "topology.json"
        if src_topo.exists():
            _shutil.copyfile(src_topo, dest / "topology.json")
            staged.append("topology.json")
    except Exception as e:
        _sys.stdout.write(_json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
        _sys.stdout.flush()
        raise typer.Exit(code=1) from e

    _sys.stdout.write(_json.dumps({"dest": str(dest.resolve()), "files": staged}) + "\n")
    _sys.stdout.flush()


@studio_app.command("build")
def studio_build(
    project: Path = typer.Argument(Path.cwd()),
    out: Path = typer.Option(Path("_studio"), help="Output directory."),
) -> None:
    """Render a static site for the project."""
    try:
        from pfnstudio_studio.build import build_site
    except ImportError as e:
        console.print(f"[red]pfnstudio-studio not installed:[/red] {e}")
        console.print("Install: pip install pfnstudio-studio")
        raise typer.Exit(code=1) from e
    out_dir = build_site(project_root=project, out_dir=out)
    console.print(f"[green]Built site at[/green] {out_dir}")


@studio_app.command("serve")
def studio_serve(
    project: Path = typer.Argument(Path.cwd()),
    port: int = typer.Option(8000),
) -> None:
    """Build the static site and serve it locally."""
    try:
        from pfnstudio_studio.build import build_site
        from pfnstudio_studio.serve import serve_dir
    except ImportError as e:
        console.print(f"[red]pfnstudio-studio not installed:[/red] {e}")
        raise typer.Exit(code=1) from e
    out_dir = build_site(project_root=project, out_dir=Path("_studio"))
    serve_dir(out_dir, port=port)


if __name__ == "__main__":
    app()
