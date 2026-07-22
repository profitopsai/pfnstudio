# Changelog

Versions are per-package; tags are `core-v<version>` and `cli-v<version>`.

## pfnstudio-core 0.9.5 (core)

### Fixed
- **`pfnstudio serve` now works for adapter base models.** The serve worker
  only built the studio block-model `ModelLoader`, so a `{task, inputs}`
  request to a served TCPFN/adapter model 400'd with `invalid_payload`. It now
  detects a run's adapter and serves through the adapter's own
  `predict()` / `capabilities()` (loaded once, warm), matching the CLI/predict
  path. Fixes runner-served (and local-worker) endpoints for adapter models.

## pfnstudio-core 0.9.4 (core)

### Added
- **Model adapter subsystem** — the plug-in seam for continued-pretraining
  external base models (TCPFN, Do-PFN, …) through their OWN architecture + loss
  instead of the studio's block-composed model. Adds `@register_adapter` /
  `get_adapter` / `resolve_adapter`, the generic `run_adapter_training` loop,
  and the `Adapter` protocol. Inference routes through the adapter's
  `predict()` / `capabilities()` when a run carries `hyperparams.adapter` (or a
  base run's `baseAdapter`). `discover_in_project` now also imports
  `adapters/<name>.py`, so a per-project adapter self-registers at run time.

## pfnstudio 0.8.17 (CLI)

### Added
- **Adapter training dispatch** — the `local` compute path now routes a run
  with `hyperparams.adapter` to `run_adapter_training` (needs
  `pfnstudio-core>=0.9.4`). Previously such a run fell through to block-model
  training and skipped with `step_fn returned None` (0 steps, no checkpoint),
  because core had no adapter concept.

## pfnstudio 0.8.16 (CLI)

### Changed
- **Renamed `templates/` → `starters/`.** The directory `pfnstudio init` copies
  is now `starters/fm-project`, clearly distinct from `studies/` (complete,
  reproducible example projects). `init` behaviour is unchanged. Internal
  helpers `templates_root()` / `fm_project_template()` are now
  `starters_root()` / `fm_project_starter()`.

### Fixed
- **`__version__` was a hardcoded literal stuck at 0.8.12.** It was never
  bumped with `pyproject.toml`, so `pfnstudio runner status` and the
  capabilities report showed a stale version for several releases (a runner
  on 0.8.15 reported 0.8.12). `__version__` is now read from the installed
  distribution metadata, so it can never drift from what pip installed.

### Added
- **Runner reports and prints its full version set.** `pfnstudio runner status`
  now shows the CLI version, the **pfnstudio-core** (training engine) version,
  and Python — not just the CLI. The core version drifts independently of the
  CLI, so a runner can look current while its training engine is stale; that
  mismatch was previously invisible. The same versions are added to the
  capabilities payload the runner posts to the cloud, so Studio can display
  cli/core/torch/python per runner (and flag a missing core). CUDA toolkit
  version is now shown in the capabilities line too.

## pfnstudio 0.8.15 (CLI)

### Fixed
- **A NaN/Inf metric no longer discards the whole run report.** When a trainer
  or eval emitted a non-finite float (e.g. `loss=nan`, a NaN eval metric), the
  runner's result POST failed with `Out of range float values are not JSON
  compliant: nan` (`requests` raises `InvalidJSONError`) and the entire run —
  events *and* final metrics — was lost even though training completed and the
  checkpoint was saved. Non-finite floats are now scrubbed to `null` (their
  correct JSON representation) at event ingestion, on the final result post, and
  on predict responses. The NaN stays visible in Studio as "no value" instead of
  silently dropping everything.

## pfnstudio-core 0.9.3

### Fixed
- **Training now uses the GPU.** The trainer resolved no device — model and
  batch tensors stayed on CPU even on a CUDA box, so a heavy model (12-layer
  axial transformer) exhausted host RAM instead of training in VRAM. It now
  resolves `PFNSTUDIO_DEVICE` → CUDA if available → CPU, moves the model there
  before the optimizer captures params, and moves each batch tensor (X, tag,
  A, labels, y) onto it. The `setup()` hook gets the resolved device too.

## pfnstudio-core 0.9.2

### Fixed
- **Head detection no longer misclassifies pooling stages.** `_is_head_module`
  used a loose `name.endswith("head")` heuristic, which treated
  `RowPoolForHead` (a stage that feeds INTO a head) as a parallel output head.
  A model with a pool before its head (e.g. Do-PFN's `row_pool_for_head` →
  `bar_distribution_head`) then assembled wrong — the head ran on the pre-pool
  encoder output — surfacing at eval time as `size of tensor a (192) must match
  tensor b (100)`. Now a strict class-name allowlist + the `is_head` duck-type.

## pfnstudio-core 0.9.1

### Fixed
- **Trainer support for the axial-attention blocks + distributional heads.**
  0.9.0 shipped the blocks (`grid_preprocessor`, `axial_attention_block`, …) and
  a bar-distribution head pattern, but the training/predict loops didn't feed
  them the plumbing they need, so a study using them failed with
  "grid_preprocessor requires single_eval_pos > 0". The default step and the
  predict forward now:
  - thread **`single_eval_pos`** (the train/test boundary from the prior's
    `n_ctx`) to blocks declaring `needs_single_eval_pos`, and unwrap
    `(out, kv)` tuples;
  - run the generic **`setup(*, prior, hp, device)`** hook before training (so a
    bar-distribution head fits its bucket borders from the prior);
  - honor a head's **`loss(query_output, target)`** hook (bucketized NLL)
    before falling back to MSE.

## pfnstudio 0.8.14

### Fixed
- **A failing job no longer stops the runner.** The main poll loop didn't wrap
  `_run_job`, so an exception escaping the executor killed the whole daemon.
  It's now isolated: the job is logged, best-effort marked failed on the cloud,
  and the runner keeps polling.

### Added
- **Runner per-job core sync (opt-in)** — a self-hosted runner can refresh
  `pfnstudio-core` before each job so it picks up newer core blocks (e.g. the
  axial-attention library) instead of failing with "No block registered".
  Enable when launching the runner:
  - `PFNSTUDIO_RUNNER_SYNC_CORE=1` — `pip install -U --no-deps pfnstudio-core`
  - `PFNSTUDIO_RUNNER_CORE_SPEC=<spec>` — upgrade an exact spec (pinned version
    or a git URL) instead.
  `--no-deps` keeps it fast; a failed sync is non-fatal (the job runs on the
  installed core).

## pfnstudio-core 0.9.0

Headline: the **axial-attention block library** and the **scorer registry** are
now public, so paper-faithful studies (Do-PFN, cascade discovery) run and
reproduce entirely from open packages.

### Added
- **Axial-attention block library** (`blocks/grid.py` + `_grid_preprocessing.py`):
  `grid_preprocessor`, `tabular_cell_embedder`, `along_row_attention`,
  `along_column_attention`, `axial_attention_block`, `row_pool_for_head`, and
  the `TabularPreprocessor` they wrap. Registered on import.
- **Scorer registry** — `@register_scorer(slug)` / `get_scorer(slug)`. Studies
  ship their own eval scorers (`evals/<slug>.py`) discovered at run time;
  `pfnstudio_core.scorers.BUILTIN_SCORERS` remains for the generic scorers.
- **Generic distributional-head hooks** — a head block can declare
  `is_head = True` (so the trainer fans it out without a hardcoded allowlist)
  and expose `to_prediction(output)` (so the predict path reduces per-bucket
  logits to a point estimate). Composes with the existing `setup()` / `loss()`
  training hooks — enough to build a bar-distribution head as project code.
- **Project-block discovery** — `discover_in_project` imports `blocks/*.py`;
  `get_block` tolerates `-`/`_` slug variants.

### Fixed
- **Multi-submodule checkpoints** — a block holding more than one `nn.Module`
  (e.g. a gated residual with both an `mlp` and a `gate`) no longer collides in
  the checkpoint. Submodules are namespaced by attribute; single-module blocks
  keep the legacy flat keying, so existing checkpoints still load.

### Notes
- `networkx` is **not** a core dependency. Studies that need it (the Do-PFN
  random-DAG prior) carry it in `priors/<slug>/requirements.txt`.

## pfnstudio 0.8.13

### Changed
- Eval scoring resolves a project's `@register_scorer` (loaded by
  `discover_in_project`) **before** falling back to core's builtins, so a
  study's own scorer wins.
- Requires `pfnstudio-core>=0.9.0`.
