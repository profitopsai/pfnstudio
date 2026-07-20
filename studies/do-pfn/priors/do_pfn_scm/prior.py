from __future__ import annotations

import json
import random
from collections.abc import Callable
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from pfnstudio_core import Prior, register_prior


def _identity(x):
    return x


def _negative(x):
    return -x


def _tag_activation(fn: Callable, spec: dict) -> Callable:
    fn._do_pfn_spec = spec
    return fn


def _simple_activation(name: str, fn: Callable, *, clamp: bool = False) -> Callable:
    def apply(x):
        out = fn(x)
        if clamp:
            out = torch.clamp(out, -1000.0, 1000.0)
        return out

    return _tag_activation(
        apply,
        {"kind": "simple", "name": name, "clamp": bool(clamp)},
    )


def activation_sampling(nonlins: str):
    """Sample one fixed activation, matching the released Do-PFN builders.

    mixed matches the released checkpoint/repository pool: square, ReLU,
    tanh, and identity. paper_text keeps only the three nonlinearities from
    the paper body for optional experiments. The sophisticated_sampling_1
    modes are the bounded modes implemented by the released repository.
    """
    simple_pool = [
        ("square", torch.square),
        ("relu", torch.relu),
        ("tanh", torch.tanh),
        ("identity", _identity),
    ]

    def choose_simple(pool):
        idx = int(np.random.choice(len(pool)))
        name, fn = pool[idx]
        return _simple_activation(name, fn)

    def make_summed():
        indices = np.random.choice(len(simple_pool), size=2, replace=False)
        chosen = [simple_pool[int(i)] for i in indices]

        def apply(x):
            out = (chosen[0][1](x) + chosen[1][1](x)) / 2.0
            return torch.clamp(out, -1000.0, 1000.0)

        return _tag_activation(
            apply,
            {
                "kind": "weighted",
                "terms": [chosen[0][0], chosen[1][0]],
                "weights": [0.5, 0.5],
                "clamp": True,
            },
        )

    def make_sophisticated_1():
        pool = [
            ("relu", nn.ReLU()),
            ("relu6", nn.ReLU6()),
            ("selu", nn.SELU()),
            ("silu", nn.SiLU()),
            ("softplus", nn.Softplus()),
            ("hardtanh", nn.Hardtanh()),
            ("sign", torch.sign),
            ("sin", torch.sin),
            ("gaussian", lambda x: torch.exp(-(x**2))),
            ("exp", torch.exp),
            ("sqrt_abs", lambda x: torch.sqrt(torch.abs(x))),
            ("unit_interval", lambda x: (torch.abs(x) < 1).float()),
            ("square", lambda x: x**2),
            ("abs", torch.abs),
        ]
        draw = np.random.rand()
        count = 1 if draw < 1.0 / 3.0 else (2 if draw < 2.0 / 3.0 else 3)
        indices = np.random.choice(len(pool), size=count, replace=False)
        chosen = [pool[int(i)] for i in np.atleast_1d(indices)]
        weights = np.ones(count, dtype=np.float64)
        if count > 1:
            weights = np.random.rand(count)
            weights /= weights.sum()

        def apply(x):
            out = sum(float(weight) * fn(x) for weight, (_, fn) in zip(weights, chosen, strict=False))
            return torch.clamp(out, -1000.0, 1000.0)

        return _tag_activation(
            apply,
            {
                "kind": "weighted",
                "terms": [name for name, _ in chosen],
                "weights": [float(w) for w in weights],
                "clamp": True,
            },
        )

    def make_sophisticated_normalized(*, rescale: bool):
        inner = make_sophisticated_1()
        scale = 1.0
        bias = 0.0
        if rescale:
            a = torch.randn(1)
            b = torch.randn(1)
            scale = float(torch.exp(2.0 * a).item())
            bias = float(b.item())

        def apply(x):
            normalized = functional.layer_norm(x, x.shape)
            if rescale:
                normalized = scale * (normalized + bias)
            return torch.clamp(inner(normalized), -1000.0, 1000.0)

        return _tag_activation(
            apply,
            {
                "kind": "normalized",
                "inner": inner._do_pfn_spec,
                "rescale": bool(rescale),
                "scale": scale,
                "bias": bias,
                "clamp": True,
            },
        )

    if nonlins == "paper_text":
        return choose_simple(simple_pool[:3])
    if nonlins in ("released_default", "default", "mixed", "post"):
        return choose_simple(simple_pool)
    if nonlins == "tanh":
        return _simple_activation("tanh", torch.tanh)
    if nonlins == "relu":
        return _simple_activation("relu", torch.relu)
    if nonlins == "sin":
        return _simple_activation("sin", torch.sin)
    if nonlins == "neg":
        return _simple_activation("negative", lambda x: -x)
    if nonlins == "id":
        return _simple_activation("identity", _identity)
    if nonlins == "elu":
        return _simple_activation("elu", functional.elu)
    if nonlins == "summed":
        return make_summed()
    if nonlins == "sophisticated_sampling_1":
        return make_sophisticated_1()
    if nonlins == "sophisticated_sampling_1_normalization":
        return make_sophisticated_normalized(rescale=False)
    if nonlins == "sophisticated_sampling_1_rescaling_normalization":
        return make_sophisticated_normalized(rescale=True)
    raise ValueError(f"Unknown nonlinearity mode: {nonlins!r}")


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
        nonlins: str = "mixed",
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
            high = vals[vals > thresh]
            t1 = low.mean() if len(low) else vals.min()
            t2 = high.mean() if len(high) else vals.max()
            if bool(torch.isclose(t1, t2)):
                raise FloatingPointError("sampled treatment levels are equal")
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


# â”€â”€ Monte-Carlo oracle CATE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The training SCM is torch-based with the exogenous noise baked into each
# equation, so it can't be re-integrated in place. But the oracle
# E[Y|do(T=high),X] - E[Y|do(T=low),X] only needs the sampled *structure*
# (graph + linear weights + noise scales), not the specific noise draws -- it
# is an expectation. So we extract the SCM into plain numpy once and Monte-
# Carlo integrate over fresh noise, pinning the observed covariates per row
# and forcing the treatment node to its low/high binarized levels.
#
# Known limitation (documented, not a bug): covariates are pinned at their
# observed values, i.e. the conditioning set is treated with do(X=x). When a
# covariate is a *descendant* of the treatment (a mediator), this blocks the
# indirect path -- exactly the estimand the model is also asked for (it
# conditions on the same covariate columns), so oracle and model stay
# consistent, but it is a controlled-direct-effect flavour of CATE, not the
# total effect. The paper's structured case studies separate these; v0.1 does
# not. See README "What this study does *not* show (yet)".


_PRIOR_WARN_ABS_MAX: float = 1.0e4
_PRIOR_ACCEPT_LOG_EVERY: int = 5000
_TARGET_NORMALIZE_EPS: float = 1.0e-6
_TARGET_CLIP: float = 100.0


def _log_prior_event(level: str, event: str, **fields: Any) -> None:
    """Emit one machine-readable line that survives remote-run log routing."""
    payload = {"level": level, "event": event, **fields}
    print(
        "[do_pfn_scm] " + json.dumps(payload, default=str, sort_keys=True),
        flush=True,
    )


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


def _torch_stats(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    finite = torch.isfinite(value)
    out: dict[str, Any] = {
        "name": name,
        "shape": tuple(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite": int(finite.sum().item()),
        "nan": int(torch.isnan(value).sum().item()),
        "inf": int(torch.isinf(value).sum().item()),
    }
    if bool(finite.any()):
        vals = value[finite]
        out.update(
            min=float(vals.min().cpu().item()),
            max=float(vals.max().cpu().item()),
            max_abs=float(vals.abs().max().cpu().item()),
        )
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


def _assert_finite_torch_sample(name: str, tensor: torch.Tensor) -> None:
    """Reject only unusable numerical samples, never large finite SCMs."""
    if tensor.numel() == 0:
        raise FloatingPointError(f"{name} is empty")
    if not torch.isfinite(tensor).all():
        bad = int((~torch.isfinite(tensor)).sum().item())
        raise FloatingPointError(f"{name} has {bad} NaN/Inf values")


def _transform_target_from_context(y_context, y_values):
    """Apply the released checkpoint's ``transform_target=True`` contract.

    Do-PFN standardizes Y from the observational context only, adds 1e-6 to
    the sample standard deviation, and clips the transformed values to
    [-100, 100].  X is normalized inside the X encoder, while this Y transform
    happens before both the Y encoder and BAR loss.
    """
    context = np.asarray(y_context, dtype=np.float64).reshape(-1)
    values = np.asarray(y_values, dtype=np.float64)
    finite = context[np.isfinite(context)]
    if finite.size == 0:
        raise FloatingPointError("Cannot transform Y without finite context targets.")

    mean = float(finite.mean())
    if finite.size <= 1:
        std = 1.0
    else:
        std = float(finite.std(ddof=1)) + _TARGET_NORMALIZE_EPS
    if not np.isfinite(std) or std < _TARGET_NORMALIZE_EPS:
        std = _TARGET_NORMALIZE_EPS

    transformed = np.clip(
        (values - mean) / std,
        -_TARGET_CLIP,
        _TARGET_CLIP,
    ).astype(np.float32)
    return transformed, mean, std


def _apply_base_nonlinearity(z, name: str):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if name == "square":
            return z * z
        if name == "tanh":
            return np.tanh(z)
        if name == "relu":
            return np.maximum(0.0, z)
        if name == "relu6":
            return np.clip(z, 0.0, 6.0)
        if name == "selu":
            alpha = 1.6732632423543772
            scale = 1.0507009873554805
            return scale * np.where(z > 0.0, z, alpha * np.expm1(z))
        if name == "silu":
            return z / (1.0 + np.exp(-np.clip(z, -80.0, 80.0)))
        if name == "softplus":
            return np.logaddexp(0.0, z)
        if name == "hardtanh":
            return np.clip(z, -1.0, 1.0)
        if name == "sign":
            return np.sign(z)
        if name == "sin":
            return np.sin(z)
        if name == "gaussian":
            return np.exp(-(z * z))
        if name == "exp":
            return np.exp(z)
        if name == "sqrt_abs":
            return np.sqrt(np.abs(z))
        if name == "unit_interval":
            return (np.abs(z) < 1.0).astype(np.float64)
        if name == "abs":
            return np.abs(z)
        if name == "negative":
            return -z
        if name == "elu":
            return np.where(z > 0.0, z, np.expm1(z))
        if name == "identity":
            return z
    raise ValueError(f"Unsupported oracle activation: {name!r}")


def _apply_nonlinearity(z, spec):
    kind = spec.get("kind")
    if kind == "simple":
        out = _apply_base_nonlinearity(z, spec["name"])
    elif kind == "weighted":
        out = np.zeros_like(z, dtype=np.float64)
        for weight, name in zip(spec["weights"], spec["terms"], strict=False):
            out += float(weight) * _apply_base_nonlinearity(z, name)
    elif kind == "normalized":
        variance = np.var(z)
        normalized = (z - np.mean(z)) / np.sqrt(variance + 1.0e-5)
        if spec.get("rescale"):
            normalized = float(spec["scale"]) * (normalized + float(spec["bias"]))
        out = _apply_nonlinearity(normalized, spec["inner"])
    else:
        raise ValueError(f"Unsupported oracle activation spec: {spec!r}")

    if spec.get("clamp"):
        out = np.clip(out, -1000.0, 1000.0)
    return out


def _extract_equations(scm, graph, k: int):
    """Pull the sampled SCM into a numpy structure indexed by node id 0..k-1.

    Each endogenous node stores its parents, weights, activation specification,
    and whether noise is applied before or after the activation.
    """
    eqs: list[dict | None] = [None] * k
    for name in graph.nodes:
        idx = _idx(name)
        fn = scm.functions.get(name)
        if fn is None:  # exogenous root -- value is pure exo noise
            eqs[idx] = {
                "parents": [],
                "weights": None,
                "activation_spec": {
                    "kind": "simple",
                    "name": "identity",
                    "clamp": False,
                },
                "post": False,
            }
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
            "activation_spec": module.activation._do_pfn_spec,
            "post": bool(module.nonlins == "post"),
        }
    if any(e is None for e in eqs):
        raise RuntimeError("failed to extract every SCM node into numpy")
    return eqs


def _build_oracle_noise(
    n: int, k: int, equations, sigma_exo: float, sigma_eps: float, rng
):
    """Root (exogenous) nodes ~ N(0, sigma_exo); non-roots ~ N(0, sigma_eps) --
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
            if eq.get("post"):
                endo[:, node] = _apply_nonlinearity(
                    linear + eps[:, node], eq["activation_spec"]
                )
            else:
                endo[:, node] = (
                    _apply_nonlinearity(linear, eq["activation_spec"]) + eps[:, node]
                )
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
    node level (t1) -> 1 and the HIGH level (t2) -> 0. The scorer computes
    `pred(col=1) - pred(col=0)`, so the oracle is
        E[Y | do(T = t_level_for_one), X] - E[Y | do(T = t_level_for_zero), X]
    with t_level_for_one = t1 (low) and t_level_for_zero = t2 (high). Getting
    this backwards silently negates every CATE metric -- see the sign check in
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
    nonlins: str = "mixed",
    max_hidden_layers: int = 0,
    invalid_policy: str = "retry",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if invalid_policy not in {"retry", "sentinel"}:
        raise ValueError(
            "invalid_policy must be 'retry' or 'sentinel', " f"got {invalid_policy!r}"
        )
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

    # Retain empty graphs as valid zero-effect tasks, matching the release.
    if graph.number_of_edges() == 0:
        scm.t_key = _choice(nodes)
        scm.y_key = _choice([n for n in nodes if n != scm.t_key])
    else:
        t_candidates = [n for n in nodes if graph.out_degree(n) > 0]
        scm.t_key = _choice(t_candidates)
        scm.y_key = _choice(list(nx.descendants(graph, scm.t_key)))

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
    x_keys = [scm.t_key, *list(np.random.choice(x_candidates, size=num_features, replace=False))]

    x_obs = torch.stack([sample_obs[key] for key in x_keys]).permute(-1, 1, 0)
    x_int = torch.stack([sample_int[key] for key in x_keys]).permute(-1, 1, 0)
    x_obs[:, :, 0] = scm.get_zero_one_treatment(x_obs[:, :, 0])
    x_int[:, :, 0] = scm.get_zero_one_treatment(x_int[:, :, 0])

    y_obs = sample_obs[scm.y_key].T.unsqueeze(-1)
    y_int = sample_int[scm.y_key].T.unsqueeze(-1)

    tensors = (x_obs, x_int, y_obs, y_int)
    is_valid = all(bool(torch.isfinite(value).all()) for value in tensors)
    if not is_valid:
        _log_prior_event(
            "warning",
            "non_finite_generated_batch",
            seed=int(seed),
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            K=int(k),
            num_features=int(num_features),
            num_unobserved=int(num_unobserved),
            nonlins=str(nonlins),
            p_edge=float(p_edge),
            sigma_exo=float(sigma_exo),
            sigma_eps=float(sigma_eps),
            invalid_policy=str(invalid_policy),
            tensors=[
                _torch_stats(name, value)
                for name, value in zip(("x_obs", "x_int", "y_obs", "y_int"), tensors, strict=False)
            ],
        )
        if invalid_policy == "sentinel":
            # Exact released-repo behavior. Use only with a trainer/evaluator
            # that skips samples whose is_valid flag is false.
            for value in tensors:
                value.fill_(-100.0)
            _log_prior_event(
                "warning",
                "sentinel_batch_returned",
                seed=int(seed),
                sentinel=-100.0,
                action="trainer_must_skip_is_valid_false",
            )
        elif invalid_policy == "retry":
            for name, value in zip(("x_obs", "x_int", "y_obs", "y_int"), tensors, strict=False):
                _assert_finite_torch_sample(name, value)
        else:
            raise ValueError(
                "invalid_policy must be 'retry' or 'sentinel', "
                f"got {invalid_policy!r}"
            )

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
    # One SCM is shared across the mini-batch, but treatment levels are
    # computed independently for each dataset/noise draw.
    t_low = scm.t1s.detach().cpu().numpy().astype(np.float64)
    t_high = scm.t2s.detach().cpu().numpy().astype(np.float64)

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
        is_valid=is_valid,
        task_meta={
            "K": k,
            "num_features": int(num_features),
            "num_unobserved": int(num_unobserved),
            "nonlins": str(nonlins),
            "invalid_policy": str(invalid_policy),
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


def _format_do_pfn_studio_task(
    *,
    batch: DoPfnBatch,
    batch_index: int,
    seed: int,
    num_samples: int,
    ctx_frac: float,
    oracle_mc: int,
):
    """Adapt one released-prior batch element to PFN Studio's dictionary API."""
    b = int(batch_index)
    n_ctx = int(num_samples * ctx_frac)
    if not 0 < n_ctx < num_samples:
        raise ValueError(f"n_ctx must be between 1 and {num_samples - 1}, got {n_ctx}")

    # The paper uses M_obs context subjects followed by M_in fresh query
    # subjects. Selecting [n_ctx:] avoids reusing context subjects as queries.
    x_obs = batch.x[:n_ctx, b, :].cpu().numpy().astype(np.float32)
    x_int = batch.x_int[n_ctx:, b, :].cpu().numpy().astype(np.float32)
    y_obs_raw = batch.y[:n_ctx, b, 0].cpu().numpy().astype(np.float32)
    y_int_raw = batch.target_y[n_ctx:, b, 0].cpu().numpy().astype(np.float32)

    # The released checkpoint was trained with transform_target=True.  Its
    # outer inference wrapper applies this same context-only normalization
    # before the internal NanHandling + Linear Y encoder.  Studio has no
    # external regression wrapper, so the prior performs the training-side
    # transform explicitly and gives the BAR loss targets on the same scale.
    y_obs, target_mean, target_std = _transform_target_from_context(
        y_obs_raw,
        y_obs_raw,
    )
    y_int, _, _ = _transform_target_from_context(y_obs_raw, y_int_raw)

    x_full = np.concatenate([x_obs, x_int], axis=0).astype(np.float32)
    np.concatenate([y_obs, y_int], axis=0).astype(np.float32)

    y_col = np.empty(num_samples, dtype=np.float32)
    y_col[:n_ctx] = y_obs
    y_col[n_ctx:] = np.nan
    x_with_y = np.concatenate([x_full, y_col[:, None]], axis=1).astype(np.float32)

    meta = dict(batch.task_meta)
    meta["n_ctx"] = n_ctx
    meta["shared_scm_batch_index"] = b
    meta["target_transform"] = "context_standardize_clip"
    meta["target_mean"] = float(target_mean)
    meta["target_std"] = float(target_std)
    meta["target_clip"] = float(_TARGET_CLIP)

    t_low = float(np.asarray(batch.t_low).reshape(-1)[b])
    t_high = float(np.asarray(batch.t_high).reshape(-1)[b])

    if int(oracle_mc) > 0:
        observed_values = x_full[:, 1:].astype(np.float64)
        o_rng = np.random.default_rng(int(seed) + 7_654_321)
        cate_true_raw = monte_carlo_oracle_cate(
            equations=batch.equations,
            topo_order=batch.topo_order,
            k=int(meta["K"]),
            cov_indices=[int(c) for c in meta["x_indices"]],
            observed_values=observed_values,
            t_idx=int(meta["t_idx"]),
            y_idx=int(meta["y_idx"]),
            # The released code maps the lower treatment level to column 1
            # and the higher level to column 0.
            t_level_for_one=t_low,
            t_level_for_zero=t_high,
            sigma_exo=float(meta["sigma_exo"]),
            sigma_eps=float(meta["sigma_eps"]),
            n_mc=int(oracle_mc),
            rng=o_rng,
        )
        meta["cate_true_source"] = (
            f"monte_carlo_oracle_do1_minus_do0(n_mc={int(oracle_mc)})"
        )
    else:
        cate_true_raw = (
            (batch.target_y[:, b, 0] - batch.y[:, b, 0])
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        meta["cate_true_source"] = "single_draw_interventional_minus_observational"

    # An affine Y transform cancels the mean in a treatment contrast.  Keep
    # both scales: normalized values are consumed by the model-facing scorer,
    # while raw values let evaluation report effects in the original units.
    cate_true = (
        np.asarray(cate_true_raw, dtype=np.float64) / float(target_std)
    ).astype(np.float32)

    out = {
        "X": x_with_y,
        "y": y_int,
        "y_raw": y_int_raw,
        "n_ctx": n_ctx,
        "cate_true": np.asarray(cate_true, dtype=np.float32),
        "cate_true_raw": np.asarray(cate_true_raw, dtype=np.float32),
        "adjacency": batch.adjacency.astype(np.int8),
        "task_meta": meta,
        "is_valid": bool(batch.is_valid),
    }

    stats = [
        _np_stats("X", out["X"]),
        _np_stats("y", out["y"]),
        _np_stats("cate_true", out["cate_true"]),
    ]
    raw_stats = [
        _np_stats("y_context_raw", y_obs_raw),
        _np_stats("y_query_raw", y_int_raw),
        _np_stats("cate_true_raw", out["cate_true_raw"]),
    ]
    if not np.isfinite(out["cate_true"]).all():
        _log_prior_event(
            "warning",
            "non_finite_oracle_cate",
            seed=int(seed),
            batch_index=int(b),
            oracle_mc=int(oracle_mc),
            K=int(meta["K"]),
            nonlins=str(meta["nonlins"]),
            t_idx=int(meta["t_idx"]),
            y_idx=int(meta["y_idx"]),
            stats=stats,
            action="evaluation_should_skip_this_task",
        )
    if _should_log_prior_accept(seed, [*stats, *raw_stats]):
        has_large_values = any(
            stat.get("max_abs", 0.0) > _PRIOR_WARN_ABS_MAX for stat in raw_stats
        )
        _log_prior_event(
            "warning" if has_large_values else "info",
            "accepted_task_large_values" if has_large_values else "accepted_task",
            seed=int(seed),
            batch_index=int(b),
            K=int(meta["K"]),
            num_features=int(meta["num_features"]),
            num_unobserved=int(meta["num_unobserved"]),
            nonlins=str(meta["nonlins"]),
            p_edge=float(meta["p_edge"]),
            edge_density=float(meta["edge_density"]),
            n_ctx=int(n_ctx),
            n_query=int(num_samples - n_ctx),
            sigma_exo=float(meta["sigma_exo"]),
            sigma_eps=float(meta["sigma_eps"]),
            t_idx=int(meta["t_idx"]),
            y_idx=int(meta["y_idx"]),
            is_valid=bool(batch.is_valid),
            stats=stats,
            raw_target_stats=raw_stats,
            target_mean=float(target_mean),
            target_std=float(target_std),
            target_clip=float(_TARGET_CLIP),
        )
    return out


def sample_do_pfn_studio_batch(
    *,
    batch_size: int,
    seed: int,
    num_samples: int = 2200,
    ctx_frac: float = 0.75,
    num_features: int = 6,
    num_unobserved: int = 1,
    K_max: int = 12,
    max_retries: int = 10,
    oracle_mc: int = 0,
    **params: Any,
):
    """Sample one SCM shared by every dataset in the mini-batch.

    This matches priors/doscm.py: the graph, mechanisms, treatment, outcome,
    and selected covariates are shared; only the sampled noise differs across
    the batch dimension.
    """
    k = int(num_features + num_unobserved + 2)
    if k > K_max:
        _log_prior_event(
            "error",
            "invalid_graph_size",
            K=int(k),
            K_max=int(K_max),
            num_features=int(num_features),
            num_unobserved=int(num_unobserved),
        )
        raise ValueError(f"K={k} exceeds K_max={K_max}")

    # The released repo writes -100 for a non-finite batch. PFN Studio has no
    # corresponding loss mask, so retrying only non-finite batches preserves
    # the intended SCM distribution without training on sentinel targets.
    last_exc = None
    for attempt in range(max_retries):
        attempt_seed = seed + attempt * 1_000_003
        try:
            batch = sample_do_pfn_torch_batch(
                seed=attempt_seed,
                batch_size=batch_size,
                seq_len=num_samples,
                num_features=num_features,
                num_unobserved=num_unobserved,
                **params,
            )
            break
        except FloatingPointError as exc:
            last_exc = exc
            _log_prior_event(
                "warning",
                "retry_non_finite_scm",
                base_seed=int(seed),
                attempt_seed=int(attempt_seed),
                attempt=int(attempt + 1),
                max_retries=int(max_retries),
                reason=str(exc),
            )
        except Exception as exc:
            _log_prior_event(
                "error",
                "unexpected_sampling_error",
                stage="sample_do_pfn_torch_batch",
                base_seed=int(seed),
                attempt_seed=int(attempt_seed),
                attempt=int(attempt + 1),
                exception_type=type(exc).__name__,
                reason=str(exc),
                batch_size=int(batch_size),
                num_samples=int(num_samples),
                num_features=int(num_features),
                num_unobserved=int(num_unobserved),
                params=sorted(params.keys()),
            )
            raise
    else:
        _log_prior_event(
            "error",
            "sampling_retries_exhausted",
            base_seed=int(seed),
            max_retries=int(max_retries),
            last_reason=str(last_exc),
        )
        raise RuntimeError(
            f"Do-PFN sampling produced only non-finite tasks: {last_exc}"
        ) from last_exc

    try:
        return [
            _format_do_pfn_studio_task(
                batch=batch,
                batch_index=i,
                seed=seed + i,
                num_samples=num_samples,
                ctx_frac=ctx_frac,
                oracle_mc=oracle_mc,
            )
            for i in range(batch_size)
        ]
    except Exception as exc:
        _log_prior_event(
            "error",
            "unexpected_formatting_error",
            stage="pfnstudio_batch_adapter",
            seed=int(seed),
            exception_type=type(exc).__name__,
            reason=str(exc),
            batch_size=int(batch_size),
            num_samples=int(num_samples),
            ctx_frac=float(ctx_frac),
            tensor_shapes={
                "x": tuple(batch.x.shape),
                "x_int": tuple(batch.x_int.shape),
                "y": tuple(batch.y.shape),
                "target_y": tuple(batch.target_y.shape),
            },
        )
        raise


def sample_do_pfn_studio_task(
    *,
    seed: int,
    num_samples: int = 2200,
    ctx_frac: float = 0.75,
    num_features: int = 6,
    num_unobserved: int = 1,
    K_max: int = 12,
    max_retries: int = 10,
    oracle_mc: int = 0,
    **params: Any,
):
    return sample_do_pfn_studio_batch(
        batch_size=1,
        seed=seed,
        num_samples=num_samples,
        ctx_frac=ctx_frac,
        num_features=num_features,
        num_unobserved=num_unobserved,
        K_max=K_max,
        max_retries=max_retries,
        oracle_mc=oracle_mc,
        **params,
    )[0]


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
        K_max: int = 12,
        max_retries: int = 10,
        oracle_mc: int = 0,
        noise_dist: str = "gaussian",
        exo_dist: str = "gaussian",
        nonlins: str = "mixed",
        invalid_policy: str = "retry",
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
            invalid_policy=invalid_policy,
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
        vary_num_unobserved_per_batch: bool = True,
        num_unobserved: int = 1,
        num_unobserved_min: int = 0,
        num_unobserved_max: int = 4,
        K_max: int = 12,
        **params: Any,
    ):
        rng = np.random.default_rng(seed)
        params.pop("tag", None)

        if vary_ctx_per_batch:
            n_ctx = int(rng.integers(min_ctx, num_samples))
            ctx_frac = n_ctx / num_samples

        if vary_num_features_per_batch:
            num_features = int(rng.integers(num_features_min, num_features_max + 1))

        if vary_num_unobserved_per_batch:
            max_allowed = int(K_max) - int(num_features) - 2
            upper = min(int(num_unobserved_max), max_allowed)
            if upper < int(num_unobserved_min):
                raise ValueError(
                    f"No valid num_unobserved for num_features={num_features}, "
                    f"K_max={K_max}"
                )
            num_unobserved = int(rng.integers(num_unobserved_min, upper + 1))

        return sample_do_pfn_studio_batch(
            batch_size=batch_size,
            seed=seed,
            num_samples=num_samples,
            ctx_frac=ctx_frac,
            num_features=num_features,
            num_unobserved=num_unobserved,
            K_max=K_max,
            **params,
        )
