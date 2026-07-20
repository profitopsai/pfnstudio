# Study: Do-PFN in-context interventional outcome prediction

This study reproduces the Do-PFN pipeline from Robertson et al. (2025) as a
PFN Studio project. A newly imported project contains the prior, the complete
default architecture, the distributional output head, the evaluation scorer,
and a runnable training configuration.

## Default project contents

- `do_pfn_scm`: synthetic random-DAG structural-causal-model prior.
- `separate_xy_input_encoder`: reusable separate X/Y context-query encoder.
- `context_query_axial_attention`: reusable context-safe axial-attention layer.
- `bar_distribution_head`: full-support 100-bucket outcome-density decoder.
- `cid_recovery`: random-DAG evaluation plus six structured causal studies.
- `v0_1`: single-GPU reproduction-scale training run.

The Python implementations are project files. Users can inspect, edit, and
reuse them from the Files and Models tabs without changing PFN Studio core.

## Default model architecture

The imported `do_pfn` model contains 15 blocks:

1. One `separate_xy_input_encoder` (`d_model=192`, `features_per_group=85`).
2. Twelve `context_query_axial_attention` layers (`6` heads, FF multiplier `4`).
3. One `row_pool_for_head` that selects token `1`, the separately encoded Y token.
4. One `bar_distribution_head` (`192 -> 768 -> 100` logits).

The custom input encoder replaces both `grid_preprocessor` and
`tabular_cell_embedder`. It performs context-only X normalization, clamps
standardized values to `[-100, 100]`, groups/pads X as expected by the released
configuration, and appends the masked outcome as its own token.

Each custom axial layer performs:

1. Full multi-head attention across feature tokens within each row.
2. Attention across rows, with query rows restricted to context keys/values.
3. A GELU feed-forward residual update.

All three residual stages use parameter-free post-LayerNorm. The attention
output projections and final feed-forward projection use the released
zero-initialization convention.

The output head fits equal-mass bucket borders from the training prior and
optimizes full-support bar-distribution negative log density, including
half-normal edge tails. Its distribution mean is used for point prediction.

## Prior

Every task samples a fresh DAG and structural equations, then produces:

- observational context rows `[T, X, Y]`;
- intervention query rows `[do(T), X, NaN]`;
- normalized interventional outcomes for training;
- raw and normalized oracle quantities for evaluation;
- validity and diagnostic metadata.

The prior includes bounded nonlinearities, non-finite-task handling, variable
context lengths, variable feature counts, variable latent-node counts, and
structured logs for diagnosing unstable samples.

## Evaluation

`cid_recovery.py` reports CID, CATE, ATE error, naive-ridge comparisons, and
90-percent predictive-interval coverage on fresh random-DAG tasks. It also
evaluates 100 datasets for each of six fixed structures:

- observed confounder;
- observed mediator;
- confounder plus mediator;
- unobserved confounder;
- back-door criterion;
- front-door criterion.

## Reproduce

```bash
pip install "pfnstudio-core[torch]" pfnstudio
pip install -r studies/do-pfn/priors/do_pfn_scm/requirements.txt

pfnstudio validate studies/do-pfn/priors/do_pfn_scm/
pfnstudio run studies/do-pfn/runs/v0_1.yaml
pfnstudio eval studies/do-pfn/evals/cid_recovery.yaml
```

Or run `./studies/do-pfn/reproduce.sh`.

The checked-in run is a reproduction-scale starter. Matching the paper's full
training budget requires substantially more synthetic tasks and GPU time.

## Layout

```text
studies/do-pfn/
|-- blocks/
|   |-- separate_xy_input_encoder.py
|   |-- separate_xy_input_encoder.yaml
|   |-- context_query_axial_attention.py
|   |-- context_query_axial_attention.yaml
|   |-- bar_distribution_head.py
|   `-- bar_distribution_head.yaml
|-- priors/do_pfn_scm/
|   |-- prior.py
|   |-- prior.yaml
|   `-- requirements.txt
|-- models/do_pfn.yaml
|-- evals/
|   |-- cid_recovery.py
|   `-- cid_recovery.yaml
`-- runs/v0_1.yaml
```

## Citation

```bibtex
@misc{robertson2025dopfn,
  title = {Do-PFN: In-Context Learning for Causal Effect Estimation},
  author = {Robertson, Jake and Reuter, Arik and Guo, Siyuan and Hollmann, Noah and Hutter, Frank and Scholkopf, Bernhard},
  year = {2025},
  eprint = {2506.06039},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2506.06039}
}
```

This PFN Studio study is Apache-2.0. The upstream Do-PFN repository should be
consulted separately for its current licensing terms.
