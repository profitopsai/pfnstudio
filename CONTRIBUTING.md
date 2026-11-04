# Contributing to PFN Studio

We want this repo to be the canonical marketplace + reference toolkit
for prior-fitted networks. That only works if contributing is easy
and low-friction. This file has the basics — open an issue if
anything's unclear.

---

## Development setup

```bash
git clone https://github.com/profitopsai/pfnstudio
cd priorstudio

# Python 3.10+. Recommend a venv:
python -m venv .venv
source .venv/bin/activate

# Editable installs of the three packages
pip install -e packages/core[torch]
pip install -e packages/cli
pip install -e packages/studio

# Run the tests
pytest tests/
```

That's the full bootstrap. If a step fails on a fresh machine, file
a bug — that's a release blocker, not an "edge case."

## Quick checks before pushing

```bash
ruff check .            # lint
ruff format --check .   # format
mypy packages/          # types
pytest tests/           # tests
pfnstudio validate priors/linear-regression/   # any prior you touched
```

CI runs the same set on every PR.

---

## Adding a new prior

A prior is a synthetic data generator. To add one:

1. Create `priors/<your-slug>/` with three files:

   ```
   priors/<your-slug>/
     prior.py       # the registered Python class
     prior.yaml     # parameter spec, outputs, citations
     README.md      # title, tagline, summary, how-to-use
   ```

2. **prior.py** registers the sampler with `@register_prior("<id>")`.
   The smallest valid shape:

   ```python
   from __future__ import annotations
   from typing import Any
   import numpy as np
   from pfnstudio_core import Prior, register_prior


   @register_prior("your_prior_id")
   class YourPrior(Prior):
       def sample(self, *, seed: int, num_points: int = 100, **_) -> dict[str, Any]:
           rng = np.random.default_rng(seed)
           # ... your sampling logic ...
           return {"X": X, "y": y}
   ```

3. **prior.yaml** declares the parameter spec + outputs + citations.
   See any `priors/*/prior.yaml` for a reference shape. JSON Schema in
   [`schemas/prior.schema.json`](schemas/prior.schema.json) — `pfnstudio validate`
   checks against it.

4. **README.md** is for humans browsing the marketplace. Title,
   tagline, one-paragraph summary, the `pfnstudio sample` invocation,
   citations.

5. Run the smoke test:

   ```bash
   pfnstudio validate priors/<your-slug>/
   pfnstudio sample priors/<your-slug>/prior.yaml --seeds 3 --num-points 5
   ```

   Both must pass before you open the PR.

6. Open the PR with a short `What this prior models` paragraph + a
   pointer to whatever paper or use-case motivated it. We'll merge once
   CI is green and the schema checks pass.

The 13 existing priors in `priors/` are reference implementations —
copy the shape that's closest to what you're trying to build.

---

## Adding a dataset scorer

A scorer runs after training and computes real-data metrics. To add one:

1. Implement a subclass of `DatasetScorer` in
   `packages/core/pfnstudio_core/scorers/`. Override `score(self, *,
   model, eval_spec, loader, run_spec) -> ScorerResult`.

2. Register it in `BUILTIN_SCORERS` (`scorers/__init__.py`) keyed by
   the eval slug it scores.

3. Add a smoke test under `tests/core/scorers/`.

4. Update the registry catalog in
   [`packages/core/pfnstudio_core/datasets/`](packages/core/pfnstudio_core/datasets/)
   if your scorer needs a new dataset.

The M4-monthly forecast scorer is the worked example
([`m4_monthly_forecast.py`](packages/core/pfnstudio_core/scorers/m4_monthly_forecast.py)).
Copy its shape.

---

## Pull request conventions

- **One concern per PR.** Adding a prior + fixing a scorer = two PRs.
- **Commit messages**: imperative mood, first line ≤72 chars, body
  explains *why*. Group related commits into one PR.
- **Public-API changes** (anything in `pfnstudio_core/`,
  `priorstudio/`, the schemas, or `priors/<slug>/prior.yaml`'s required
  fields) need a note in `CHANGELOG.md` and a heads-up in the PR
  description.

## Reviews

We review with a few rules:
1. **It runs on CPU.** Every prior, every example, every smoke test.
   GPU is for paper-scale work; the public repo stays approachable.
2. **It's honest.** If a scorer can't reproduce a paper's numbers,
   the README says so.
3. **No vendor lock-in.** Nothing in this repo should depend on the
   hosted studio at pfnstudio.com. If a change adds such a
   dependency, it belongs in the cloud repo, not here.

If a PR sits without review for a week, ping
[hello@profitops.ai](mailto:hello@profitops.ai) — we're a small team.

---

## Code of conduct

We follow the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md). TL;DR:
be kind, assume good faith, focus on the work.

## License

By contributing you agree your contribution is licensed under the
[Apache License 2.0](LICENSE), matching the project.
