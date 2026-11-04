# pfnstudio-core

The Python contract for PFN Studio FM projects.

```python
from pfnstudio_core import Prior, Model, Run, register_block, register_prior, register_scorer
from pfnstudio_core.scorers.base import DatasetScorer, ScorerResult


@register_prior("my_prior")
class MyPrior(Prior):
    def sample(self, seed: int): ...


@register_block("my_attention")
class MyAttention:
    def __init__(self, d_model: int, n_heads: int): ...


# Paper-specific scorer — ships in the template beside evals/<slug>.yaml.
@register_scorer("my_eval")
class MyScorer(DatasetScorer):
    def score(self, *, model, eval_spec, loader, run_spec) -> ScorerResult: ...
```

The CLI discovers anything registered via these decorators and validates `models/*.yaml` references against the registry.

## Layout

- `prior.py` — `Prior` ABC and built-in prior loader
- `model.py` — `Model` config + block-composition
- `eval.py` — `EvalSpec` — the declarative benchmark spec (dataset + metrics + baselines)
- `scorers/` — `DatasetScorer` — the executable scoring pipeline; core ships only *generic* scorers, paper-specific ones live in templates
- `run.py` — `Run` manifest + executor protocol
- `registry.py` — `@register_prior`, `@register_block`, `@register_scorer` and discovery
- `loaders.py` — load YAML artifacts into typed objects
- `blocks/` — built-in architecture blocks (transformer encoder, causal attention, heads)
- `training/` — minimal in-process training loop for the `local` compute adapter
