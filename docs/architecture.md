# Architecture

PFN Studio is split into three Python packages plus a JSON Schema set and a project template.

```
priorstudio/                            # repo root
├── schemas/                             # JSON Schema for every artifact type (single source of truth)
├── templates/fm-project/                # cookiecutter scaffold
└── packages/
    ├── core/   pfnstudio-core        # Prior, Model, Eval, Run, registry, blocks, training loop
    ├── cli/    priorstudio             # init, validate, lint, run, list, studio (commands)
    │           ├── compute/             # local, vast, modal, runpod, hf_spaces
    │           └── tracking/            # local (always), wandb, mlflow
    └── studio/ pfnstudio-studio      # static-site renderer for an FM project
```

## Why three packages

- **core** is the only thing a project's own Python code imports. It has minimal deps (pydantic, pyyaml, numpy). PyTorch is optional (`pip install pfnstudio-core[torch]`). Projects that bring their own framework can use core's spec/loader/registry surface without pulling torch.
- **cli** is the developer tool — assumes core, adds Typer/Rich for UX and jsonschema for validation. Compute and tracking adapters live here so they can shell out / make HTTP calls without contaminating core.
- **studio** is optional. The framework is fully usable from the terminal; Studio is a convenience for browsing.

## Data flow

```
prior.yaml                   model.yaml           eval.yaml           run.yaml
     │                            │                   │                  │
     ▼                            ▼                   ▼                  ▼
PriorSpec                    ModelSpec            EvalSpec           RunSpec      (loaders.py)
     │                            │                   │                  │
     │  (registry decorator       │                   │                  │
     │   binds yaml id to         │                   │                  │
     │   Python class)            │                   │                  │
     ▼                            ▼                   ▼                  │
Prior subclass               Model(spec)          Eval subclass          │
     │                       composes blocks                             │
     │                       from registry                               │
     │                            │                                      │
     └──────────► train_pfn(model, prior, run) ◄──────────────────────────┘
                       │
                       ▼
                  results dict ──► trackers (local, wandb, mlflow)
                       │
                       ▼
            runs/<run_id>/results.json
```

## Block registry

Architecture is config-driven. A model YAML names blocks by string; the registry resolves each to a Python class.

```python
from pfnstudio_core import register_block


@register_block("my_attention")
class MyAttention:
    def __init__(self, d_model: int, n_heads: int): ...
    def __call__(self, x): ...
```

```yaml
# in models/foo.yaml
blocks:
  - type: my_attention
    config: { d_model: 256, n_heads: 8 }
```

Blocks are decoupled from any framework — the contract is `__init__(**config)` + `__call__(x)`. Built-in blocks happen to use PyTorch and import torch lazily; for JAX/MLX, register your own.

## Discovery

When the CLI runs, it imports every `.py` file under `priors/`, `models/`, and `evals/` so the decorators fire. This means a project's Python code doesn't need an `__init__.py` or any explicit registration plumbing — *being there* is enough.

## Validation layers

Three layers, all run on every PR via CI:

1. **JSON Schema** (`pfnstudio validate`) — structural shape of every YAML.
2. **Cross-reference lint** (`pfnstudio lint`) — does run X reference a real prior, model, eval? Do citations resolve in `references.bib`? Does the prior's directory name match its `id`?
3. **Pydantic** (in core's `loaders.py`) — strong typing for any code that reads artifacts.
