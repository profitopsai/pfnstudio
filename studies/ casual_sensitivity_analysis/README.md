# CSA-PFN MSM — Studio replication of the paper's PRIMARY model (v8)

Replicates the **marginal sensitivity model (MSM)** foundation model from
Javurek et al., *Amortizing Causal Sensitivity Analysis via PFNs*
(arXiv:2605.10590), with the author's permission. This is the paper's main
model: *"We trained a FM for the marginal sensitivity model (MSM), using
10,000 synthetic DGPs… over 1B training pairs per bound."*

**Status: sanity PASSED (2026-07-07).** Flat model on RTX 4090: d_in=15,
loss 355 → 0.86 over 50 steps, no NaN. Pipeline verified end-to-end.
Next: `runs/msm-main-v1.yaml`.

## What the trained model does

One forward pass, no retraining:
INPUT  = context dataset rows (x1..x10, a, y) + query rows (x, a, Γ, bound_flag)
OUTPUT = θ̂ per query row — the estimated upper (bound=0) or lower (bound=1)
bound on E[Y(a)|x] under the MSM at confounding level Γ ∈ [1, 5].
Sweep Γ at both bounds to trace the sensitivity interval; CATE bounds are
combinations across arms (upper CATE = θ₊(x,1,Γ) − θ₋(x,0,Γ)).
The prior's packing/normalization code is part of the model's API — any
real-data (CSV) usage must standardize X per feature, normalize y, and pack
the 15-column tokens exactly as `prior.py` does.

## Models

- `causal-sensitivity-flat-pfn` — **DEFAULT.** tabular_embedder →
  transformer_encoder (d=128, 4 heads, FF 512, 10 layers) → scalar_head.
  Matches Table 1 dims; deviation: flat attention instead of per-feature.
- `causal-sensitivity-axial-pfn.EXPERIMENTAL` — paper-faithful axial stack;
  currently produces NaN gradients on the first backward in Studio
  (reproduced across token styles and grouping configs on data that trains
  cleanly on the flat model). Kept for the Studio bug report; do not use.

Other documented deviations from the paper (both models): single scalar
head + bound_flag row duplication instead of two 5-component GMM heads with
NLL (point estimates, not PPDs); no monotonicity penalty (apply cumulative
max/min over Γ post-hoc).

## Structure

```
priors/causal_sensitivity_msm_analytical/   closed-form MSM labels (CPU-cheap)
models/causal-sensitivity-flat-pfn.yaml     DEFAULT model (proven)
models/causal-sensitivity-axial-pfn.EXPERIMENTAL.yaml   known Studio NaN bug
evals/causal-sensitivity-msm-bounds.yaml    RMSE/MAE/R2 + monotonicity gate
runs/msm-sanity-v1.yaml                     sanity run (PASSED; regression test)
runs/msm-main-v1.yaml                       main run — see its checklist
```

## v8: stderr diagnostics + paper-scale run

All [prior] diagnostic prints now go to stderr — stdout stays clean JSON,
which fixes the Try-it widget ("could not parse sample output") and any
serving path that parses the prior's stdout. New run
`msm-main-v2-paperscale.yaml`: 100k steps = ~1.02B pair-visits per bound,
matching the paper's training budget with 320x the DGP diversity (every
DGP fresh instead of 150x reuse). ~9-11h on one 4090.

## v7: parallel prefetch sampling (fork workers)

v6's ProcessPoolExecutor failed under Studio because the prior module has a
dynamic name workers can't re-import (PicklingError). v7 uses raw fork
Processes — children inherit the module in memory, only seeds cross the
queue. Verified under a simulated dynamic module name: pool healthy, ~4x on
4 cores, results bitwise-identical to serial. Auto-fallback to serial on any
failure.

## v6 (superseded): parallel prefetch sampling

The prior now prefetches upcoming seeds across a process pool (the trainer
requests consecutive seeds, so seed s..s+63 are computed on all CPU cores
while the GPU trains). Outputs are BITWISE IDENTICAL to serial mode (same
seeded _sample_impl per item — verified); pure throughput. Auto-fallback to
serial on any pool failure. Tune with env PRIOR_WORKERS (default min(16,
cpus)) and PRIOR_PREFETCH (64). Watch for the "[prior] parallel sampling: N
workers" line at run start. Expected: step time ~106s -> ~15-25s at main
scale on a 32-core box; also lifts GPU utilization since the GPU stops
waiting on data.

## v5: vectorized DGP (62x faster sampling)

The covariate DAG is now evaluated for all N samples at once (numpy) instead
of the authors' per-sample Python loop. Verified statistically identical
(means/stds/correlations within sampling error vs the loop implementation);
only the internal RNG call order differs. Effect: ~50 ms per N=1024 dataset
vs ~3 s — main-run step time drops from ~2 min to ~15-20 s, so 2500 steps
finish in ~12 hours instead of 3-4 days. Documented here because it is an
implementation optimization, not a distributional deviation.

## Prior — Table 2 fidelity ("Analytical MSM" column)

| Item | Paper | This prior |
|---|---|---|
| Sensitivity model | MSM (closed form) | closed form ✅ (verified vs brute-force LP, 800 cases, exact; Γ=1 recovers E[Y]) |
| Grid range / spacing | [1.0, 5.0], log-uniform | same ✅ |
| Per-DGP grid randomization | yes | redrawn per query group ✅ |
| MC samples | 128 (single bank, reused) | same ✅ |
| Grid size | 50 | n_gamma per group (4–5) ⚠️ fresh random Γ per group, equivalent coverage at fixed seq length |
| DGP | Setting_standard | vendored verbatim ✅ |

Includes a degenerate-DGP guard (re-rolls non-finite or |θ|>50 items) and a
`[prior] N=... D=...` log line — always check it shows the intended scale.

## Hard-won Studio facts (read before editing)

- Token contract is `zero_packed` (15 cols, is_context flag, no NaN).
  `nan_grid` NaN-masking broke gradients in the grid path.
- Studio passes the prior Parameters form into `sample()` kwargs; every
  param accepted by `sample()` must also exist on `_sample_impl()`.
- Code defaults win when nothing else applies — keep defaults at the scale
  you actually want to run, and verify via the `[prior]` log line.
- Reuse a box once torch is installed; prefer PyTorch-image instances,
  ≥30GB disk, RTX 4090/A100/H100 class (Blackwell cards break torch wheels).
- Stale GPU processes from crashed runs hold VRAM — `nvidia-smi` + `kill`
  before relaunching on a reused box.

## Roadmap

1. `msm-main-v1` (see checklist in the yaml) → gates: loss ≪ 0.9, RMSE beats
   predict-0 and context-mean, R² > 0, bounds monotone in Γ.
2. Ground-truth validation on held-out DGPs vs closed-form labels
   (exact — stronger than anything possible for the KL run).
3. HF release: checkpoint + topology + packing code + inference example +
   model card (paper, permission, deviations, eval numbers).
4. KL project: swap to the flat model, cost pilot, offline prior pool.
5. Report the axial NaN bug to Studio with this repro.

## Reference

Javurek, Frauen, Brockschmidt, Schweisthal, Feuerriegel (2026).
*Amortizing Causal Sensitivity Analysis via Prior-Data Fitted Networks.*
arXiv:2605.10590. Replicated with the corresponding author's permission.
