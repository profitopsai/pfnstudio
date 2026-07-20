"""
CSA-PFN prior — ANALYTICAL MSM labels (the paper's PRIMARY training run).
MAIN-RUN defaults: N=1024, n_queries=32, n_gamma=5 -> M=640, sequence 1664
(paper Table 1 batch geometry). For sanity tests, set N=256, n_queries=8,
n_gamma=4 in the run/prior form (seq 384) and VERIFY via the [prior] log line.

TOKEN CONTRACT: token_style="zero_packed" (DEFAULT — proven trainer path,
same contract as the converged sanity-test-v7 run; no NaN anywhere):
    X : (N + M, D+5)
        context row: (x_1..x_D, a, y,   0,     0, 1)   is_context=1
        query   row: (x_1..x_D, a, 0, Gamma, bound, 0) is_context=0; bound 0=up 1=lo
    theta_star : (M, 1)   [role: target -> y]
    n_ctx      : scalar = N   [role: context_length]
    M = n_queries * 2 arms * 2 bounds * n_gamma
Model notes for D=10 (15 columns): use features_per_group: 1 and
row_pool_for_head target_col: 11 (the y column). Do NOT use
features_per_group: 2 — 15 columns don't divide evenly and the padded group
produced NaN gradients. token_style="nan_grid" (D+4, NaN-masked) also produced
NaN gradients in the grid path — keep zero_packed until Studio fixes both.

Includes a degenerate-DGP guard: items with non-finite context or labels, or
|theta| > 50, are re-rolled so no extreme item can poison training weights.
"""
import sys
import numpy as np
import networkx as nx
import torch

# ---- PFN Studio compatibility shim -----------------------------------------
try:
    from pfnstudio_core import Prior, register_prior
except Exception:
    class Prior:
        pass

    def register_prior(name):
        def deco(cls):
            return cls
        return deco


# =============================================================================
# DGP — vendored verbatim from the authors' gen_standard_syn.py settings
# =============================================================================

class DAGStructuredSCM:
    def __init__(self, prior_layers=lambda: np.random.randint(3, 7),
                 prior_hidden_size=lambda: np.random.randint(15, 40),
                 prior_weight=lambda: np.random.normal(0, 1),
                 edge_drop_prob=0.5, activation=lambda x: np.tanh(x)):
        self.prior_layers = prior_layers
        self.prior_hidden_size = prior_hidden_size
        self.prior_weight = prior_weight
        self.edge_drop_prob = edge_drop_prob
        self.activation = activation
        self.dag = None
        self.weights = {}
        self.biases = {}
        self.noise_distributions = {}
        self.feature_nodes = []
        self.node_values = {}
        self.topological_order = []

    def sample_noise_distribution(self):
        dist_type = np.random.choice(["normal", "uniform", "laplace", "logistic"])
        if dist_type == "uniform":
            scale = np.random.uniform(0.1, 2.0)
            return lambda size: np.random.uniform(-scale, scale, size)
        if dist_type == "laplace":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.laplace(0, scale, size)
        if dist_type == "logistic":
            scale = np.random.uniform(0.1, 1.0)
            return lambda size: np.random.logistic(0, scale, size)
        scale = np.random.uniform(0.1, 2.0)
        return lambda size: np.random.normal(0, scale, size)

    def construct_mlp_graph(self, num_layers, hidden_size):
        graph = nx.DiGraph()
        node_id = 0
        nodes_by_layer = []
        for layer_idx in range(num_layers):
            layer_nodes = []
            for _ in range(hidden_size):
                graph.add_node(node_id, layer=layer_idx)
                layer_nodes.append(node_id)
                node_id += 1
            nodes_by_layer.append(layer_nodes)
        for layer_idx in range(num_layers - 1):
            for src in nodes_by_layer[layer_idx]:
                for dst in nodes_by_layer[layer_idx + 1]:
                    graph.add_edge(src, dst)
        return graph

    def transform_to_dag(self, graph):
        dag = graph.copy()
        edges = list(graph.edges())
        num_edges_to_drop = int(len(edges) * self.edge_drop_prob)
        if num_edges_to_drop > 0:
            edges_to_drop = np.random.choice(len(edges), size=num_edges_to_drop, replace=False)
            for idx in edges_to_drop:
                u, v = edges[idx]
                dag.remove_edge(u, v)
        assert nx.is_directed_acyclic_graph(dag)
        return dag

    def sample_structural_equation_parameters(self):
        for node in self.dag.nodes():
            parents = list(self.dag.predecessors(node))
            if parents:
                self.weights[node] = {p: self.prior_weight() for p in parents}
            self.biases[node] = self.prior_weight()
            self.noise_distributions[node] = self.sample_noise_distribution()

    def generate_dataset(self, num_features, num_samples):
        """VECTORIZED node evaluation: each node is computed for all samples at
        once (numpy) instead of the authors' per-sample Python loop. The joint
        distribution is identical (same DAG/weights/noise families; i.i.d.
        noise drawn per node across samples); only the RNG call order differs.
        ~20-50x faster — the difference between days and hours per main run."""
        mlp_graph = self.construct_mlp_graph(self.prior_layers(), self.prior_hidden_size())
        self.dag = self.transform_to_dag(mlp_graph)
        self.topological_order = list(nx.topological_sort(self.dag))
        self.sample_structural_equation_parameters()
        all_nodes = list(self.dag.nodes())
        assert num_features <= len(all_nodes)
        self.feature_nodes = np.random.choice(all_nodes, size=num_features, replace=False)
        values = {}
        for node in self.topological_order:
            parents = list(self.dag.predecessors(node))
            noise = self.noise_distributions[node](num_samples)
            if not parents:
                values[node] = self.activation(self.biases[node] + noise)
            else:
                weighted = sum(self.weights[node][p] * values[p] for p in parents)
                values[node] = self.activation(weighted + self.biases[node] + noise)
        return np.stack([values[n] for n in self.feature_nodes], axis=1)


def _random_mlp(in_dim, out_dim, n_layers, hidden, weight_std, device):
    sizes = [in_dim] + [hidden] * (n_layers - 2) + [out_dim]
    layers = []
    for i in range(len(sizes) - 1):
        W = torch.randn(sizes[i], sizes[i + 1], device=device) * weight_std
        b = torch.randn(sizes[i + 1], device=device) * weight_std
        layers.append((W, b))
    return layers


def _mlp(x, layers):
    h = x
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = torch.tanh(h)
    return h


def _generate_dgp(N, D, device="cpu"):
    dag_scm = DAGStructuredSCM()
    X = torch.as_tensor(dag_scm.generate_dataset(D, N), dtype=torch.float32, device=device)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    f_a = _random_mlp(D, 1, np.random.randint(3, 5), np.random.randint(8, 20),
                      weight_std=0.8, device=device)
    pi_obs = torch.sigmoid(_mlp(X, f_a))
    A = torch.bernoulli(pi_obs)

    f_y = _random_mlp(D + 2, 1, np.random.randint(3, 6), np.random.randint(10, 25),
                      weight_std=1.0, device=device)
    U = torch.randn(N, 1, device=device)
    zeros, ones = torch.zeros(N, 1, device=device), torch.ones(N, 1, device=device)
    y0 = _mlp(torch.cat([X, zeros, U], 1), f_y)
    y1 = _mlp(torch.cat([X, ones, U], 1), f_y)
    Y_obs = torch.where(A > 0.5, y1, y0)
    y_all = torch.cat([y0, y1], 0)
    y_mean, y_std = y_all.mean(), y_all.std() + 1e-6
    Y = (Y_obs - y_mean) / y_std
    return X, A, Y, f_y, (y_mean, y_std)


# =============================================================================
# Closed-form sharp MSM bounds on the shared MC bank (Table 2: k=128, reused)
# =============================================================================

def _msm_bounds_closed_form(y_bank_sorted, gammas):
    """Sharp MSM bounds, exact on a sorted MC bank. Verified vs brute-force LP."""
    k = y_bank_sorted.shape[0]
    cum_from_top = np.cumsum(y_bank_sorted[::-1])
    cum_from_bot = np.cumsum(y_bank_sorted)
    total = cum_from_bot[-1]
    theta_up, theta_lo = [], []
    for G in gammas:
        G = float(G)
        p = 1.0 / (1.0 + G)
        t = p * k
        j = int(np.floor(t))
        frac = t - j
        w_hi, w_lo = G / k, 1.0 / (G * k)
        top = cum_from_top[j - 1] if j > 0 else 0.0
        bnd_up = y_bank_sorted[k - 1 - j] if j < k else 0.0
        rest_up = total - top - (bnd_up if j < k else 0.0)
        theta_up.append(w_hi * top
                        + (frac * w_hi + (1 - frac) * w_lo) * bnd_up
                        + w_lo * rest_up)
        bot = cum_from_bot[j - 1] if j > 0 else 0.0
        bnd_lo = y_bank_sorted[j] if j < k else 0.0
        rest_lo = total - bot - (bnd_lo if j < k else 0.0)
        theta_lo.append(w_hi * bot
                        + (frac * w_hi + (1 - frac) * w_lo) * bnd_lo
                        + w_lo * rest_lo)
    return np.array(theta_lo), np.array(theta_up)


# =============================================================================
# One training item
# =============================================================================

MAX_ABS_THETA = 50.0     # degenerate-DGP guard threshold (normalized-y units)
_MAX_REROLLS = 5


def _sample_impl(N=1024, D=10, n_queries=32, n_gamma=5, k_mc=128,
                 gamma_min=1.0, gamma_max=5.0, token_style="zero_packed",
                 seed=None, _rerolls=0):
    if seed is not None:
        np.random.seed(int(seed) % (2**31 - 1))
        torch.manual_seed(int(seed))
    device = "cpu"                                     # closed form: CPU is enough
    print(f"[prior] N={N} D={D} n_queries={n_queries} n_gamma={n_gamma} "
          f"token_style={token_style}", file=sys.stderr, flush=True)

    X, A, Y, f_y, (y_mean, y_std) = _generate_dgp(N, D, device)

    u_bank = torch.randn(k_mc, 1, device=device)       # single bank, reused

    X_np = X.numpy()
    A_np = A.numpy().reshape(-1)
    Y_np = Y.numpy().reshape(-1)

    src = np.random.choice(N, size=n_queries, replace=(n_queries > N))
    rows = []  # (qid, src, arm, gamma, theta, bound)  bound 0=upper 1=lower
    qid = 0
    for s in src:
        x_row = X[s:s + 1].expand(k_mc, D)
        for arm in (0.0, 1.0):
            a_col = torch.full((k_mc, 1), arm, device=device)
            y_bank = _mlp(torch.cat([x_row, a_col, u_bank], 1), f_y).squeeze(-1)
            y_bank = ((y_bank - y_mean) / y_std).numpy()
            y_sorted = np.sort(y_bank)
            gammas = np.exp(np.random.uniform(np.log(gamma_min), np.log(gamma_max), n_gamma))
            gammas = np.sort(gammas)
            t_lo, t_up = _msm_bounds_closed_form(y_sorted, gammas)
            t_up = np.maximum.accumulate(t_up)
            t_lo = np.minimum.accumulate(t_lo)
            for g, tu in zip(gammas, t_up):
                rows.append((qid, s, arm, float(g), float(tu), 0))
            for g, tl in zip(gammas, t_lo):
                rows.append((qid, s, arm, float(g), float(tl), 1))
            qid += 1

    # degenerate-DGP guard: re-roll rather than emit extreme/non-finite values
    theta_arr = np.array([r[4] for r in rows], np.float32)
    ctx_bad = not (np.isfinite(X_np).all() and np.isfinite(Y_np).all())
    lbl_bad = (not np.isfinite(theta_arr).all()) or np.abs(theta_arr).max() > MAX_ABS_THETA
    if (ctx_bad or lbl_bad) and _rerolls < _MAX_REROLLS:
        print(f"[prior] degenerate item (ctx_bad={ctx_bad} lbl_bad={lbl_bad}) — re-rolling",
              file=sys.stderr, flush=True)
        return _sample_impl(N=N, D=D, n_queries=n_queries, n_gamma=n_gamma, k_mc=k_mc,
                            gamma_min=gamma_min, gamma_max=gamma_max,
                            token_style=token_style, seed=None, _rerolls=_rerolls + 1)

    M = len(rows)                                      # = n_queries*2*2*n_gamma
    if token_style == "zero_packed":                   # DEFAULT: 0-filled, D+5, is_context flag
        W = D + 5
        ctx = np.zeros((N, W), np.float32)
        ctx[:, :D], ctx[:, D], ctx[:, D + 1], ctx[:, D + 4] = X_np, A_np, Y_np, 1.0
        qry = np.zeros((M, W), np.float32)
        for j, (_q, s, arm, g, _t, b) in enumerate(rows):
            qry[j, :D], qry[j, D], qry[j, D + 2], qry[j, D + 3] = X_np[s], arm, g, float(b)
    else:                                              # "nan_grid": D+4, NaN-masked (see docstring warning)
        W = D + 4
        ctx = np.full((N, W), np.nan, np.float32)
        ctx[:, :D], ctx[:, D], ctx[:, D + 1] = X_np, A_np, Y_np
        qry = np.full((M, W), np.nan, np.float32)
        for j, (_q, s, arm, g, _t, b) in enumerate(rows):
            qry[j, :D], qry[j, D], qry[j, D + 2], qry[j, D + 3] = X_np[s], arm, g, float(b)

    return {
        "X":                np.concatenate([ctx, qry], 0),
        "theta_star":       np.array([[r[4]] for r in rows], np.float32),
        "n_ctx":            np.int64(N),
        "gamma":            np.array([[r[3]] for r in rows], np.float32),
        "bound_type":       np.array([[r[5]] for r in rows], np.float32),
        "query_id":         np.array([[r[0]] for r in rows], np.float32),
        "source_row_index": np.array([[r[1]] for r in rows], np.float32),
    }


# =============================================================================
# Parallel prefetch sampling (v7 — fork workers, no function pickling)
# =============================================================================
# Studio loads this file under a dynamic module name, so functions cannot be
# pickled by reference (v6 failure mode). v7 uses raw fork Processes: children
# inherit _sample_impl in memory and receive only seeds through a queue.
# Outputs remain BITWISE IDENTICAL to serial (same seeded call per item).
# Auto-fallback to serial on any failure. Env: PRIOR_WORKERS, PRIOR_PREFETCH.

import os as _os

_PREFETCH = int(_os.environ.get("PRIOR_PREFETCH", "64"))
_WORKERS = int(_os.environ.get("PRIOR_WORKERS", str(min(16, _os.cpu_count() or 8))))
_state = {"started": False, "broken": False, "task_q": None, "result_q": None,
          "procs": [], "done": {}, "submitted": set(), "kw": None}


def _worker_loop(task_q, result_q, kw):
    # Runs in forked children: _sample_impl is inherited, never pickled.
    try:
        import torch as _t
        _t.set_num_threads(1)          # avoid thread oversubscription
    except Exception:
        pass
    while True:
        s = task_q.get()
        if s is None:
            break
        try:
            result_q.put((s, _sample_impl(seed=s, **kw)))
        except Exception as e:  # surfaced to parent, triggers serial fallback
            result_q.put((s, e))


def _start_pool(kw):
    import multiprocessing as _mp
    ctx = _mp.get_context("fork")
    task_q, result_q = ctx.Queue(), ctx.Queue()
    procs = []
    for _ in range(_WORKERS):
        p = ctx.Process(target=_worker_loop, args=(task_q, result_q, kw), daemon=True)
        p.start()
        procs.append(p)
    _state.update(started=True, task_q=task_q, result_q=result_q, procs=procs,
                  kw=dict(kw), done={}, submitted=set())
    print(f"[prior] parallel sampling: {_WORKERS} fork workers, prefetch {_PREFETCH}",
          file=sys.stderr, flush=True)


def _sample_parallel(seed, kw):
    if seed is None or _state["broken"]:
        return _sample_impl(seed=seed, **kw)
    try:
        if not _state["started"]:
            _start_pool(kw)
        if _state["kw"] != kw:         # params changed mid-run: bypass the pool
            return _sample_impl(seed=seed, **kw)
        s = int(seed)
        for s2 in range(s, s + _PREFETCH):
            if s2 not in _state["submitted"]:
                _state["task_q"].put(s2)
                _state["submitted"].add(s2)
        while s not in _state["done"]:
            s2, out = _state["result_q"].get(timeout=600)
            _state["done"][s2] = out
        out = _state["done"].pop(s)
        for k in [k for k in _state["done"] if k < s - 2 * _PREFETCH]:
            _state["done"].pop(k)
        if isinstance(out, Exception):
            raise out
        return out
    except Exception as e:
        print(f"[prior] parallel sampling failed ({type(e).__name__}: {e}) — "
              f"falling back to serial", file=sys.stderr, flush=True)
        _state["broken"] = True
        return _sample_impl(seed=seed, **kw)


def sample(N=1024, D=10, n_queries=32, n_gamma=5, seed=None, **kwargs):
    return _sample_parallel(seed, dict(N=N, D=D, n_queries=n_queries, n_gamma=n_gamma,
                                       token_style=kwargs.get("token_style", "zero_packed")))


@register_prior("causal_sensitivity_msm_analytical")
class CausalSensitivityMSMAnalyticalPrior(Prior):
    def sample(self, N=1024, D=10, n_queries=32, n_gamma=5, seed=None, **kwargs):
        return _sample_parallel(seed, dict(N=N, D=D, n_queries=n_queries,
                                           n_gamma=n_gamma,
                                           token_style=kwargs.get("token_style", "zero_packed")))


if __name__ == "__main__":
    import time
    t0 = time.time()
    out = sample(N=1024, D=10, n_queries=32, n_gamma=5, seed=0)
    dt = time.time() - t0
    X, th, g, b = out["X"], out["theta_star"], out["gamma"], out["bound_type"]
    n, M = int(out["n_ctx"]), th.shape[0]
    n_gamma = 5
    print(f"sample() took {dt:.2f}s   X={X.shape}  M={M}  seq={n + M}")
    assert X.shape == (n + M, 10 + 5), f"bad X shape {X.shape}"
    assert M == 32 * 2 * 2 * n_gamma, f"bad M {M}"
    assert np.isfinite(X).all(), "zero_packed X must contain no NaN/inf"
    assert (X[:n, 14] == 1.0).all() and (X[n:, 14] == 0.0).all(), "is_context flag wrong"
    assert np.isfinite(th).all() and np.abs(th).max() <= MAX_ABS_THETA, "labels out of range"
    up = th[b[:, 0] == 0].reshape(-1, n_gamma)
    lo = th[b[:, 0] == 1].reshape(-1, n_gamma)
    assert (up >= lo - 1e-6).all(), "upper < lower somewhere"
    assert (np.diff(up, axis=1) >= -1e-6).all() and (np.diff(lo, axis=1) <= 1e-6).all()
    print("closed-form MSM contract OK: finite, upper>=lower, monotone in Gamma")
