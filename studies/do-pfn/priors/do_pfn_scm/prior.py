from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from pfnstudio_core import Prior, register_prior


def _identity(x):
    return x


def activation_sampling(nonlins: str):
    if nonlins == "paper_text":
        return np.random.choice([torch.square, torch.relu, torch.tanh])
    if nonlins in ("mixed", "post"):
        return np.random.choice([torch.square, torch.relu, torch.tanh, _identity])
    if nonlins == "tanh":
        return torch.tanh
    if nonlins == "relu":
        return torch.relu
    if nonlins == "id":
        return _identity
    return np.random.choice([torch.square, torch.relu, torch.tanh])


def make_exo_dist_samples(shape: tuple[int, ...], exo_std: float):
    def sample():
        return torch.normal(0, exo_std, shape)

    return sample


def make_additive_noise_gaussian(shape: tuple[int, ...], std: float):
    def sample():
        return torch.normal(0, std, shape)

    return sample


class MakeStructuralEquations(nn.Module):
    def __init__(
        self,
        parents: list[str],
        samples_shape: tuple[int, ...],
        noise_std: float,
        noise_dist: str = "gaussian",
        nonlins: str = "paper_text",
        max_hidden_layers: int = 0,
    ):
        super().__init__()
        self.parents = parents
        self.nonlins = nonlins
        self.layers = nn.Linear(len(parents), 1, bias=False) if parents else None
        self.activation = activation_sampling(nonlins)

        if noise_dist != "gaussian":
            raise ValueError("Do-PFN v1 training prior uses gaussian noise.")
        self.additive_noise = make_additive_noise_gaussian(samples_shape, noise_std)()

    def forward(self, **kwargs):
        if not self.parents:
            return self.additive_noise

        parent_values = [kwargs[parent] for parent in self.parents]
        parent_tensor = torch.stack(parent_values, dim=-1)

        with torch.no_grad():
            out = self.layers(parent_tensor).squeeze(-1)
            if self.nonlins == "post":
                return self.activation(out + self.additive_noise)
            return self.activation(out) + self.additive_noise


class StructuralCausalModel:
    def __init__(self):
        self.endogenous_vars = {}
        self.exogenous_vars = {}
        self.functions = {}
        self.exogenous_distributions = {}
        self.saved_functions = {}
        self.binary_strategy = "mean"

    def add_endogenous_var(self, name: str, function: Callable, param_varnames: dict):
        name = name.upper()
        self.endogenous_vars[name] = None
        self.functions[name] = (function, param_varnames)

    def add_exogenous_var(
        self, name: str, distribution: Callable, distribution_kwargs: dict
    ):
        name = name.upper()
        self.exogenous_vars[name] = None
        self.exogenous_distributions[name] = (distribution, distribution_kwargs)

    def create_graph(self):
        graph = nx.DiGraph()
        [graph.add_node(v.upper(), type="endo") for v in self.endogenous_vars]
        [graph.add_node(v.upper(), type="exo") for v in self.exogenous_vars]
        for var in self.functions:
            for parent in self.functions[var][1].values():
                graph.add_edge(parent.upper(), var.upper())
        return graph

    def set_binarization_params(self, treatment):
        threshs, t1s, t2s = [], [], []
        for b in range(treatment.shape[0]):
            vals = torch.nan_to_num(treatment[b])
            not_min_max = (vals > vals.min()) & (vals < vals.max())
            if not bool(not_min_max.any()):
                threshs.append(vals[0])
                t1s.append(vals[0])
                t2s.append(vals[0])
                continue

            thresh = vals[not_min_max].mean()
            low = vals[vals < thresh]
            high = vals[vals >= thresh]
            t1 = low.mean() if len(low) else vals.min()
            t2 = high.mean() if len(high) else vals.max()
            threshs.append(thresh)
            t1s.append(t1)
            t2s.append(t2)

        self.t_threshs = torch.stack([x.reshape(()) for x in threshs])
        self.t1s = torch.stack([x.reshape(()) for x in t1s])
        self.t2s = torch.stack([x.reshape(()) for x in t2s])

    def get_binarized_treatment(self, treatment):
        for b in range(treatment.shape[0]):
            lt = treatment[b] < self.t_threshs[b]
            treatment[b][lt] = self.t1s[b]
            treatment[b][~lt] = self.t2s[b]
        return treatment

    def get_zero_one_treatment(self, treatment):
        for b in range(treatment.shape[1]):
            treatment[:, b] = (treatment[:, b] < treatment[:, b].mean()).float()
        return treatment

    def get_next_sample(self, exogenous_vars=None, binarize=False, graph=None):
        if exogenous_vars is None:
            for key, dist in self.exogenous_distributions.items():
                self.exogenous_vars[key] = dist[0](**dist[1])
        else:
            self.exogenous_vars = exogenous_vars

        if binarize and self.t_key in self.exogenous_vars and exogenous_vars is None:
            self.set_binarization_params(self.exogenous_vars[self.t_key])
            self.exogenous_vars[self.t_key] = self.get_binarized_treatment(
                self.exogenous_vars[self.t_key]
            )

        structure = graph if graph is not None else self.create_graph()
        for node in nx.topological_sort(structure):
            if node in self.exogenous_vars:
                continue
            lookup = {**self.exogenous_vars, **self.endogenous_vars}
            param_map = self.functions[node][1]
            params = {p: lookup[param_map[p]] for p in param_map}
            self.endogenous_vars[node] = self.functions[node][0](**params)

            if binarize and self.t_key == node:
                self.set_binarization_params(self.endogenous_vars[node])
                self.endogenous_vars[node] = self.get_binarized_treatment(
                    self.endogenous_vars[node]
                )

        return dict(self.endogenous_vars), dict(self.exogenous_vars)

    def do_interventions(self, interventions):
        self.saved_functions = {}
        for target, intervention in interventions:
            self.saved_functions[target] = self.functions[target]
            self.functions[target] = intervention

    def undo_interventions(self):
        for key, value in self.saved_functions.items():
            self.functions[key] = value
        self.saved_functions.clear()


class SCMGenerator:
    def __init__(
        self,
        all_functions: dict[str, Callable],
        seed: int,
        samples_shape: tuple[int, ...],
        noise_std: float,
        noise_dist: str,
        nonlins: str,
        max_hidden_layers: int = 0,
    ):
        self.all_functions = all_functions
        self.seed = seed
        self.samples_shape = samples_shape
        self.noise_std = noise_std
        self.noise_dist = noise_dist
        self.nonlins = nonlins
        self.max_hidden_layers = max_hidden_layers

    def create_graph_from_nodes(self, num_nodes: int, p: float):
        graph = nx.DiGraph()
        nodes = list(range(num_nodes))
        graph.add_nodes_from(nodes)
        perm = np.random.permutation(nodes)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < p:
                    graph.add_edge(perm[i], perm[j])
        return graph

    def create_scm_from_graph(
        self, graph, possible_functions, exo_distribution, exo_distribution_kwargs
    ):
        scm = StructuralCausalModel()

        mapping = {}
        for n in graph.nodes:
            parents = list(graph.predecessors(n))
            mapping[n] = "X" + str(n) if parents else "U" + str(n)
        graph = nx.relabel_nodes(graph, mapping, copy=True)

        random.seed(self.seed)
        for n in graph.nodes:
            parents = list(graph.predecessors(n))
            if parents:
                fn_name = random.choice(possible_functions)
                scm.add_endogenous_var(
                    n,
                    self.all_functions[fn_name](
                        parents=parents,
                        samples_shape=self.samples_shape,
                        noise_std=self.noise_std,
                        noise_dist=self.noise_dist,
                        nonlins=self.nonlins,
                        max_hidden_layers=self.max_hidden_layers,
                    ),
                    {p: p for p in parents},
                )
            else:
                scm.add_exogenous_var(n, exo_distribution, exo_distribution_kwargs)

        return scm


def _idx(name: str) -> int:
    return int(name[1:])


def _choice(items):
    return items[int(torch.randint(0, len(items), (1,)).item())]


def _adjacency_from_graph(graph, k: int):
    adj = np.zeros((k, k), dtype=np.int8)
    for src, dst in graph.edges:
        adj[_idx(src), _idx(dst)] = 1
    return adj


# ── Monte-Carlo oracle CATE ────────────────────────────────────────────────
# The training SCM is torch-based with the exogenous noise baked into each
# equation, so it can't be re-integrated in place. But the oracle
# E[Y|do(T=high),X] − E[Y|do(T=low),X] only needs the sampled *structure*
# (graph + linear weights + noise scales), not the specific noise draws — it
# is an expectation. So we extract the SCM into plain numpy once and Monte-
# Carlo integrate over fresh noise, pinning the observed covariates per row
# and forcing the treatment node to its low/high binarized levels.
#
# Known limitation (documented, not a bug): covariates are pinned at their
# observed values, i.e. the conditioning set is treated with do(X=x). When a
# covariate is a *descendant* of the treatment (a mediator), this blocks the
# indirect path — exactly the estimand the model is also asked for (it
# conditions on the same covariate columns), so oracle and model stay
# consistent, but it is a controlled-direct-effect flavour of CATE, not the
# total effect. The paper's structured case studies separate these; v0.1 does
# not. See README "What this study does *not* show (yet)".
_ORACLE_CLAMP: float = 1.0e4
_STABLE_SAMPLE_ABS_MAX: float = 1.0e4


_PRIOR_WARN_ABS_MAX: float = 1.0e4
_PRIOR_ACCEPT_LOG_EVERY: int = 5000


def _np_stats(name: str, arr):
    a = np.asarray(arr)
    finite = np.isfinite(a)
    out = {
        "name": name,
        "shape": tuple(a.shape),
        "finite": int(finite.sum()),
        "nan": int(np.isnan(a).sum()),
        "inf": int(np.isinf(a).sum()),
    }
    if finite.any():
        vals = a[finite]
        out["min"] = float(vals.min())
        out["max"] = float(vals.max())
        out["max_abs"] = float(np.abs(vals).max())
    return out


def _should_log_prior_accept(seed: int, stats: list[dict]) -> bool:
    if int(seed) % _PRIOR_ACCEPT_LOG_EVERY == 0:
        return True

    for s in stats:
        # X is allowed to contain NaNs because query y is masked in X.
        if s.get("name") != "X" and s.get("nan", 0):
            return True

        # Inf is never expected.
        if s.get("inf", 0):
            return True

        # Very large finite values are suspicious.
        if s.get("max_abs", 0.0) > _PRIOR_WARN_ABS_MAX:
            return True

    return False


def _assert_stable_torch_sample(name: str, tensor: torch.Tensor) -> None:
    if tensor.numel() == 0:
        raise FloatingPointError(f"{name} is empty")

    if not torch.isfinite(tensor).all():
        bad = int((~torch.isfinite(tensor)).sum().item())
        raise FloatingPointError(f"{name} has {bad} NaN/Inf values")

    max_abs = float(tensor.detach().abs().max().cpu().item())
    if max_abs > _STABLE_SAMPLE_ABS_MAX:
        raise FloatingPointError(
            f"{name} is unstable: max_abs={max_abs:.4g} > {_STABLE_SAMPLE_ABS_MAX:.4g}"
        )


def _apply_nonlinearity(z, gamma_idx: int):
    if gamma_idx == 0:
        return z * z
    if gamma_idx == 1:
        return np.tanh(z)
    if gamma_idx == 2:
        return np.maximum(0.0, z)
    return z  # identity / unknown


def _act_to_gamma_idx(activation) -> int:
    if activation is torch.square:
        return 0
    if activation is torch.tanh:
        return 1
    if activation is torch.relu:
        return 2
    return 3  # identity


def _extract_equations(scm, graph, k: int):
    """Pull the sampled SCM into a numpy structure indexed by node id 0..k-1.

    Endogenous node → {parents:[int], weights:(n_par,) float64, gamma_idx:int}
    Exogenous root  → {parents:[], weights:None, gamma_idx:3}
    """
    eqs: list[dict | None] = [None] * k
    for name in graph.nodes:
        idx = _idx(name)
        fn = scm.functions.get(name)
        if fn is None:  # exogenous root — value is pure exo noise
            eqs[idx] = {"parents": [], "weights": None, "gamma_idx": 3}
            continue
        module = fn[0]
        parent_idxs = [_idx(p) for p in module.parents]
        weights = (
            module.layers.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
            if module.layers is not None
            else None
        )
        eqs[idx] = {
            "parents": parent_idxs,
            "weights": weights,
            "gamma_idx": _act_to_gamma_idx(module.activation),
        }
    if any(e is None for e in eqs):
        raise RuntimeError("failed to extract every SCM node into numpy")
    return eqs


def _build_oracle_noise(
    n: int, k: int, equations, sigma_exo: float, sigma_eps: float, rng
):
    """Root (exogenous) nodes ~ N(0, sigma_exo); non-roots ~ N(0, sigma_eps) —
    matching the torch prior's exogenous vs additive-noise split."""
    eps = np.empty((n, k), dtype=np.float64)
    for node in range(k):
        scale = sigma_exo if not equations[node]["parents"] else sigma_eps
        eps[:, node] = rng.normal(0.0, scale, size=n)
    return eps


def _oracle_forward(equations, topo_order, eps, overrides):
    n, k = eps.shape
    endo = np.zeros((n, k), dtype=np.float64)
    for node in topo_order:
        node = int(node)
        if node in overrides:
            endo[:, node] = overrides[node]
            continue
        eq = equations[node]
        parents = eq["parents"]
        if not parents:
            endo[:, node] = eps[:, node]
        else:
            linear = endo[:, parents] @ eq["weights"]
            endo[:, node] = _apply_nonlinearity(linear, eq["gamma_idx"]) + eps[:, node]
        np.clip(endo[:, node], -_ORACLE_CLAMP, _ORACLE_CLAMP, out=endo[:, node])
    return endo


def monte_carlo_oracle_cate(
    *,
    equations,
    topo_order,
    k: int,
    cov_indices,
    observed_values,
    t_idx: int,
    y_idx: int,
    t_level_for_one: float,
    t_level_for_zero: float,
    sigma_exo: float,
    sigma_eps: float,
    n_mc: int,
    rng,
):
    """Per-row CATE by Monte-Carlo integration over the exogenous noise, with
    observed covariates pinned per row.

    Sign convention MUST match how the scorer queries the model. The model's
    treatment column is 0/1, and `get_zero_one_treatment` maps the LOW binarized
    node level (t1) → 1 and the HIGH level (t2) → 0. The scorer computes
    `pred(col=1) − pred(col=0)`, so the oracle is
        E[Y | do(T = t_level_for_one), X] − E[Y | do(T = t_level_for_zero), X]
    with t_level_for_one = t1 (low) and t_level_for_zero = t2 (high). Getting
    this backwards silently negates every CATE metric — see the sign check in
    the study's verification notes.
    """
    n = observed_values.shape[0]
    base = {
        int(node): observed_values[:, i].astype(np.float64)
        for i, node in enumerate(cov_indices)
    }
    one_lvl = np.full(n, float(t_level_for_one), dtype=np.float64)
    zero_lvl = np.full(n, float(t_level_for_zero), dtype=np.float64)
    acc_one = np.zeros(n, dtype=np.float64)
    acc_zero = np.zeros(n, dtype=np.float64)
    for _ in range(int(n_mc)):
        eps = _build_oracle_noise(n, k, equations, sigma_exo, sigma_eps, rng)
        acc_one += _oracle_forward(
            equations, topo_order, eps, {**base, int(t_idx): one_lvl}
        )[:, y_idx]
        acc_zero += _oracle_forward(
            equations, topo_order, eps, {**base, int(t_idx): zero_lvl}
        )[:, y_idx]
    return ((acc_one - acc_zero) / int(n_mc)).astype(np.float32)


class DoPfnBatch:
    def __init__(self, x, y, target_y, x_int, **extra):
        self.x = x
        self.y = y
        self.target_y = target_y
        self.x_int = x_int
        for key, value in extra.items():
            setattr(self, key, value)


def sample_do_pfn_torch_batch(
    *,
    seed: int,
    batch_size: int,
    seq_len: int,
    num_features: int,
    num_unobserved: int = 1,
    noise_dist: str = "gaussian",
    exo_dist: str = "gaussian",
    nonlins: str = "paper_text",
    max_hidden_layers: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if exo_dist != "gaussian":
        raise ValueError("Do-PFN v1 training prior uses gaussian exogenous noise.")

    k = int(num_features + num_unobserved + 2)
    samples_shape = (batch_size, seq_len)
    p_edge = float(np.random.uniform(1.0 / (num_features + 1), 1.0))
    sigma_exo = float(np.random.uniform(1.0, 3.0))
    sigma_eps = float(0.3 * np.random.beta(1.0, 5.0))

    gen = SCMGenerator(
        all_functions={"nonlinear": MakeStructuralEquations},
        seed=seed,
        samples_shape=samples_shape,
        noise_std=sigma_eps,
        noise_dist=noise_dist,
        nonlins=nonlins,
        max_hidden_layers=max_hidden_layers,
    )

    raw_graph = gen.create_graph_from_nodes(k, p_edge)
    scm = gen.create_scm_from_graph(
        raw_graph,
        possible_functions=["nonlinear"],
        exo_distribution=make_exo_dist_samples(samples_shape, sigma_exo),
        exo_distribution_kwargs={},
    )
    graph = scm.create_graph()
    nodes = list(graph.nodes)

    t_candidates = [n for n in nodes if graph.out_degree(n) > 0]
    if not t_candidates:
        raise RuntimeError("sampled graph has no treatment candidate")
    scm.t_key = _choice(t_candidates)

    descendants = list(nx.descendants(graph, scm.t_key))
    if not descendants:
        raise RuntimeError("sampled treatment has no descendants")
    scm.y_key = _choice(descendants)

    endo_obs, exo_obs = scm.get_next_sample(binarize=True, graph=graph)
    sample_obs = endo_obs | exo_obs

    coin = torch.randint(0, 2, (batch_size, seq_len))
    t1s = scm.t1s.unsqueeze(1).expand(-1, seq_len)
    t2s = scm.t2s.unsqueeze(1).expand(-1, seq_len)
    t_int = torch.where(coin == 0, t1s, t2s)

    if scm.t_key in scm.endogenous_vars:
        scm.do_interventions([(scm.t_key, (lambda: t_int, {}))])
    else:
        exo_obs[scm.t_key] = t_int

    endo_int, exo_int = scm.get_next_sample(exogenous_vars=exo_obs, graph=graph)
    sample_int = endo_int | exo_int
    scm.undo_interventions()

    x_candidates = list(set(graph.nodes) - {scm.t_key, scm.y_key})
    x_keys = [
        scm.t_key,
        *np.random.choice(x_candidates, size=num_features, replace=False),
    ]

    x_obs = torch.stack([sample_obs[key] for key in x_keys]).permute(-1, 1, 0)
    x_int = torch.stack([sample_int[key] for key in x_keys]).permute(-1, 1, 0)
    x_obs[:, :, 0] = scm.get_zero_one_treatment(x_obs[:, :, 0])
    x_int[:, :, 0] = scm.get_zero_one_treatment(x_int[:, :, 0])

    y_obs = sample_obs[scm.y_key].T.unsqueeze(-1)
    y_int = sample_int[scm.y_key].T.unsqueeze(-1)

    _assert_stable_torch_sample("x_obs", x_obs)
    _assert_stable_torch_sample("x_int", x_int)
    _assert_stable_torch_sample("y_obs", y_obs)
    _assert_stable_torch_sample("y_int", y_int)

    adj = _adjacency_from_graph(graph, k)
    cov_indices = [_idx(key) for key in x_keys[1:]]
    t_idx = _idx(scm.t_key)
    y_idx = _idx(scm.y_key)

    # Structure extracted for the numpy Monte-Carlo oracle CATE. The two
    # binarized treatment levels (t_low/t_high) are what the model's 0/1
    # treatment column maps to in the DGP, so the oracle intervenes with these
    # rather than literal 0/1.
    topo_order = np.array([_idx(n) for n in nx.topological_sort(graph)], dtype=np.int64)
    equations = _extract_equations(scm, graph, k)
    t_low = float(scm.t1s.reshape(-1)[0].item())
    t_high = float(scm.t2s.reshape(-1)[0].item())

    return DoPfnBatch(
        x=x_obs.detach(),
        y=y_obs.detach(),
        target_y=y_int.detach(),
        x_int=x_int.detach(),
        adjacency=adj,
        equations=equations,
        topo_order=topo_order,
        t_low=t_low,
        t_high=t_high,
        task_meta={
            "K": k,
            "num_features": int(num_features),
            "num_unobserved": int(num_unobserved),
            "sigma_exo": sigma_exo,
            "sigma_eps": sigma_eps,
            "p_edge": p_edge,
            "edge_density": float(adj.sum() / (k * k)),
            "t_idx": t_idx,
            "y_idx": y_idx,
            "x_indices": cov_indices,
            "t_key": scm.t_key,
            "y_key": scm.y_key,
            "x_keys": x_keys,
        },
    )


def sample_do_pfn_studio_task(
    *,
    seed: int,
    num_samples: int = 2200,
    ctx_frac: float = 0.75,
    num_features: int = 6,
    num_unobserved: int = 1,
    K_max: int = 10,
    max_retries: int = 50,
    oracle_mc: int = 0,
    **params: Any,
):
    k = int(num_features + num_unobserved + 2)
    if k > K_max:
        raise ValueError(f"K={k} exceeds K_max={K_max}")

    last_exc = None
    for attempt in range(max_retries):
        try:
            batch = sample_do_pfn_torch_batch(
                seed=seed + attempt * 1_000_003,
                batch_size=1,
                seq_len=num_samples,
                num_features=num_features,
                num_unobserved=num_unobserved,
                **params,
            )
            break
        except Exception as exc:
            last_exc = exc
            print(
                f"[do_pfn_scm] rejected sampled SCM "
                f"seed={seed + attempt * 1_000_003} "
                f"attempt={attempt + 1}/{max_retries} "
                f"reason={type(exc).__name__}: {exc}",
                flush=True,
            )
    else:
        raise RuntimeError(f"Do-PFN sampling failed: {last_exc}") from last_exc

    n_ctx = int(num_samples * ctx_frac)
    n_query = num_samples - n_ctx

    x_obs = batch.x[:n_ctx, 0, :].cpu().numpy().astype(np.float32)
    x_int = batch.x_int[:n_query, 0, :].cpu().numpy().astype(np.float32)
    y_obs = batch.y[:n_ctx, 0, 0].cpu().numpy().astype(np.float32)
    y_int = batch.target_y[:n_query, 0, 0].cpu().numpy().astype(np.float32)

    x_full = np.concatenate([x_obs, x_int], axis=0).astype(np.float32)
    y_full = np.concatenate([y_obs, y_int], axis=0).astype(np.float32)

    y_col = np.empty(num_samples, dtype=np.float32)
    y_col[:n_ctx] = y_full[:n_ctx]
    y_col[n_ctx:] = np.nan
    x_with_y = np.concatenate([x_full, y_col[:, None]], axis=1).astype(np.float32)

    meta = dict(batch.task_meta)
    meta["n_ctx"] = n_ctx

    if int(oracle_mc) > 0:
        # Proper ground-truth CATE: E[Y|do(1),X] − E[Y|do(0),X], Monte-Carlo
        # integrated over the exogenous noise with covariates pinned per row.
        observed_values = x_full[:, 1:].astype(np.float64)  # covariate cols (N, d)
        o_rng = np.random.default_rng(int(seed) + 7_654_321)
        cate_true = monte_carlo_oracle_cate(
            equations=batch.equations,
            topo_order=batch.topo_order,
            k=int(meta["K"]),
            cov_indices=[int(c) for c in meta["x_indices"]],
            observed_values=observed_values,
            t_idx=int(meta["t_idx"]),
            y_idx=int(meta["y_idx"]),
            # get_zero_one_treatment maps low node level t1 → column 1, high
            # level t2 → column 0. Scorer does pred(1)-pred(0), so col-1 level
            # is t_low (t1) and col-0 level is t_high (t2).
            t_level_for_one=float(batch.t_low),
            t_level_for_zero=float(batch.t_high),
            sigma_exo=float(meta["sigma_exo"]),
            sigma_eps=float(meta["sigma_eps"]),
            n_mc=int(oracle_mc),
            rng=o_rng,
        )
        meta["cate_true_source"] = (
            f"monte_carlo_oracle_do1_minus_do0(n_mc={int(oracle_mc)})"
        )
    else:
        # Cheap fallback used at train time (the trainer never reads cate_true):
        # a single-draw interventional-minus-observational contrast. This is
        # NOT the CATE — request oracle_mc>0 (the eval does) for a real CATE.
        cate_true = (
            (batch.target_y[:, 0, 0] - batch.y[:, 0, 0])
            .cpu()
            .numpy()
            .astype(np.float32)
        )[:num_samples]
        meta["cate_true_source"] = "single_draw_interventional_minus_observational"

    out = {
        "X": x_with_y,
        "y": y_full[n_ctx:].astype(np.float32),
        "n_ctx": n_ctx,
        "cate_true": np.asarray(cate_true, dtype=np.float32),
        "adjacency": batch.adjacency.astype(np.int8),
        "task_meta": meta,
    }

    stats = [
        _np_stats("X", out["X"]),
        _np_stats("y", out["y"]),
        _np_stats("cate_true", out["cate_true"]),
    ]

    if _should_log_prior_accept(seed, stats):
        print(
            "[do_pfn_scm] accepted sampled SCM "
            f"seed={seed} "
            f"K={meta.get('K')} "
            f"num_features={meta.get('num_features')} "
            f"p_edge={meta.get('p_edge'):.6g} "
            f"edge_density={meta.get('edge_density'):.6g} "
            f"n_ctx={n_ctx} "
            f"sigma_exo={meta.get('sigma_exo'):.6g} "
            f"sigma_eps={meta.get('sigma_eps'):.6g} "
            f"t_idx={meta.get('t_idx')} "
            f"y_idx={meta.get('y_idx')} "
            f"stats={stats}",
            flush=True,
        )

    return out


@register_prior("do_pfn_scm")
class DoPfnSCMPrior(Prior):
    def sample(
        self,
        *,
        seed: int,
        num_samples: int = 2200,
        ctx_frac: float = 0.75,
        num_features: int = 6,
        num_unobserved: int = 1,
        K_max: int = 10,
        max_retries: int = 50,
        oracle_mc: int = 0,
        noise_dist: str = "gaussian",
        exo_dist: str = "gaussian",
        nonlins: str = "paper_text",
        **params: Any,
    ):
        params.pop("tag", None)
        return sample_do_pfn_studio_task(
            seed=seed,
            num_samples=num_samples,
            ctx_frac=ctx_frac,
            num_features=num_features,
            num_unobserved=num_unobserved,
            K_max=K_max,
            max_retries=max_retries,
            oracle_mc=oracle_mc,
            noise_dist=noise_dist,
            exo_dist=exo_dist,
            nonlins=nonlins,
            **params,
        )

    def sample_batch(
        self,
        *,
        batch_size: int,
        seed: int,
        num_samples: int = 2200,
        ctx_frac: float = 0.75,
        min_ctx: int = 10,
        vary_ctx_per_batch: bool = True,
        vary_num_features_per_batch: bool = True,
        num_features: int = 6,
        num_features_min: int = 1,
        num_features_max: int = 6,
        **params: Any,
    ):
        rng = np.random.default_rng(seed)

        if vary_ctx_per_batch:
            n_ctx = int(rng.integers(min_ctx, num_samples))
            ctx_frac = n_ctx / num_samples

        if vary_num_features_per_batch:
            num_features = int(rng.integers(num_features_min, num_features_max + 1))

        return [
            self.sample(
                seed=seed + i,
                num_samples=num_samples,
                ctx_frac=ctx_frac,
                num_features=num_features,
                **params,
            )
            for i in range(batch_size)
        ]
