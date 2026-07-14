"""
CSA-PFN prior for PFN Studio  —  faithful method, author-exact scale, one self-contained file.

Same authors' pipeline (Setting_standard) as before; the ONLY change vs the
author's raw contract is the OUTPUT ARRANGEMENT, which PFN Studio's trainer
requires: a single packed X (context rows stacked on top of query rows) plus
n_ctx marking the boundary. The frontier science is byte-identical.

Studio's trainer contract (confirmed by its own geometry check):
    X      : (n_ctx + M, D+4)   context rows on top, query rows below
    y      : theta_star (M, 1)   scored at the query rows   [role: target -> y]
    n_ctx  : scalar = N          train/test boundary        [role: context_length -> n_ctx]

Packed token = (x_1..x_D, a, y, gamma, bound_flag), width D+4:
    context row (N of them):  (x, a, y,   NaN, 0)     gamma unobserved on context
    query   row (M of them):  (x, a, NaN, gamma, b)   y unobserved on query; b=0 upper, 1 lower
NaN entries become (value=0, indicator=1) inside grid_preprocessor — the faithful
analog of the authors' NaN-aware X/A/Y/Gamma encoders. Column 11 (= D+1) is the
Y slot -> row_pool_for_head target_col = 11.

Pipeline (unchanged):
  1. SCM/DGP: covariate DAG + f_A propensity + f_BNN(X,A,U) outcome, U~N(0,1).
  2. queries: n_queries context patients x both arms -> 2*n_queries groups.
  3. bounds:  authors' Lagrangian sweep, rational-quadratic spline flow over U,
              MSM divergence gamma = max(max r, 1/min r), warm-started lambdas.
  4. repair:  cumulative monotonicity repair per curve.
Author-exact numerics (Table 2): 350 steps/lambda, k_train=128, k_eval=4096,
16 bins, tail 6.0, 50-point log-uniform lambda grid over [0.08, 2.0], N=1024, d=10.

Studio UI
---------
Parameters (int):  D 10 | N 1024 | n_queries 32 | n_lambda 50
Output variables, M = 4 * n_queries * n_lambda:
    X          matrix  (N+M, D+4)          packed context+query    [-- autodetect]
    theta_star matrix  (M, 1)              bound label             [target -> y]
    n_ctx      scalar                      = N                     [context_length -> n_ctx]
    gamma / bound_type / query_id / source_row_index   (M,1) diagnostics, not trained on
"""
import math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# Flow config + small numerics (ported from the authors' frontier.py)
# =============================================================================

class FlowConfig:
    # author-exact: 16 bins (via _sample_impl), tail bound 6.0, min bin 1e-3 (Table 2)
    def __init__(self, num_bins=8, tail_bound=6.0,
                 min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative


def _inverse_softplus_target(target):
    return math.log(math.exp(target) - 1.0)


def standard_normal_logprob(u):
    return -0.5 * (math.log(2.0 * math.pi) + u.pow(2))


def batched_spline_forward(base_u, spline_params, cfg):
    """1D rational-quadratic spline flow over (B,m,S). Ported from the authors.
    Returns transformed samples u and their log-density log_p_eta."""
    B, m, S = base_u.shape
    nb = cfg.num_bins
    base_u = base_u.contiguous()
    widths = spline_params[..., :nb]
    heights = spline_params[..., nb:2 * nb]
    derivatives = spline_params[..., 2 * nb:]

    widths = F.softmax(widths, dim=-1)
    widths = cfg.min_bin_width + (1.0 - cfg.min_bin_width * nb) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), value=0.0)
    cumwidths = 2.0 * cfg.tail_bound * cumwidths - cfg.tail_bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = F.softmax(heights, dim=-1)
    heights = cfg.min_bin_height + (1.0 - cfg.min_bin_height * nb) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), value=0.0)
    cumheights = 2.0 * cfg.tail_bound * cumheights - cfg.tail_bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivatives = F.pad(derivatives, pad=(1, 1))
    bc = _inverse_softplus_target(1.0 - cfg.min_derivative)
    derivatives[..., 0] = bc
    derivatives[..., -1] = bc
    derivatives = cfg.min_derivative + F.softplus(derivatives)
    delta = heights / widths

    inside = (base_u >= -cfg.tail_bound) & (base_u <= cfg.tail_bound)
    bin_idx = torch.searchsorted(cumwidths, base_u, right=True) - 1
    bin_idx = bin_idx.clamp(min=0, max=nb - 1)

    in_cw = cumwidths.gather(-1, bin_idx)
    in_bw = widths.gather(-1, bin_idx)
    in_ch = cumheights.gather(-1, bin_idx)
    in_bh = heights.gather(-1, bin_idx)
    in_d = delta.gather(-1, bin_idx)
    in_der = derivatives[..., :-1].gather(-1, bin_idx)
    in_der1 = derivatives[..., 1:].gather(-1, bin_idx)

    theta = ((base_u - in_cw) / in_bw).clamp(0.0, 1.0)
    t1mt = theta * (1.0 - theta)
    num = in_bh * (in_d * theta.pow(2) + in_der * t1mt)
    den = in_d + (in_der + in_der1 - 2.0 * in_d) * t1mt
    u_inside = in_ch + num / den

    der_num = in_d.pow(2) * (in_der1 * theta.pow(2) + 2.0 * in_d * t1mt
                             + in_der * (1.0 - theta).pow(2))
    logabsdet_inside = torch.log(der_num) - 2.0 * torch.log(den)

    u = torch.where(inside, u_inside, base_u)
    logabsdet = torch.where(inside, logabsdet_inside, torch.zeros_like(base_u))
    log_p_eta = standard_normal_logprob(base_u) - logabsdet
    return u, log_p_eta


# =============================================================================
# Random MLPs (the structural functions f_A, f_BNN)
# =============================================================================

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


# =============================================================================
# 1. Generate one SCM / DGP   (covariate DAG + f_A + f_BNN)
# =============================================================================

class DAGStructuredSCM:
    """Authors' covariate generator, vendored VERBATIM from gen_standard_syn.py."""

    def __init__(self, prior_layers=lambda: np.random.randint(2, 6),
                 prior_hidden_size=lambda: np.random.randint(10, 50),
                 prior_weight=lambda: np.random.normal(0, 1),
                 edge_drop_prob=0.4, activation=lambda x: np.tanh(x)):
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
        layer_sizes = [hidden_size] * num_layers
        nodes_by_layer = []
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for _ in range(size):
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
        assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG after edge removal"
        return dag

    def sample_structural_equation_parameters(self):
        for node in self.dag.nodes():
            parents = list(self.dag.predecessors(node))
            if parents:
                self.weights[node] = {parent: self.prior_weight() for parent in parents}
            self.biases[node] = self.prior_weight()
            self.noise_distributions[node] = self.sample_noise_distribution()

    def evaluate_node(self, node):
        parents = list(self.dag.predecessors(node))
        if not parents:
            noise = self.noise_distributions[node](1)[0]
            return self.activation(self.biases[node] + noise)
        weighted_sum = sum(self.weights[node][p] * self.node_values[p] for p in parents)
        noise = self.noise_distributions[node](1)[0]
        return self.activation(weighted_sum + self.biases[node] + noise)

    def sample_observation(self):
        self.node_values = {}
        for node in self.topological_order:
            self.node_values[node] = self.evaluate_node(node)
        return np.array([self.node_values[n] for n in self.feature_nodes])

    def generate_dataset(self, num_features, num_samples):
        self.sampled_num_layers = self.prior_layers()
        self.sampled_hidden_size = self.prior_hidden_size()
        mlp_graph = self.construct_mlp_graph(self.sampled_num_layers, self.sampled_hidden_size)
        self.dag = self.transform_to_dag(mlp_graph)
        self.topological_order = list(nx.topological_sort(self.dag))
        self.sample_structural_equation_parameters()
        all_nodes = list(self.dag.nodes())
        assert num_features <= len(all_nodes), "Requested more features than available nodes"
        self.feature_nodes = np.random.choice(all_nodes, size=num_features, replace=False)
        dataset = np.zeros((num_samples, num_features))
        for i in range(num_samples):
            dataset[i] = self.sample_observation()
        return dataset


def _generate_dgp(N, D, device="cpu"):
    # --- covariates: authors' DAGStructuredSCM, with their generate_single_dataset settings ---
    dag_scm = DAGStructuredSCM(
        prior_layers=lambda: np.random.randint(3, 7),
        prior_hidden_size=lambda: np.random.randint(15, 40),
        prior_weight=lambda: np.random.normal(0, 1),
        edge_drop_prob=0.5,
        activation=lambda x: np.tanh(x),
    )
    X_np = dag_scm.generate_dataset(num_features=D, num_samples=N)        # (N, D) numpy
    X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)              # standardize (authors normalize per-feature)

    # --- treatment f_A: propensity -> Bernoulli (raw sigmoid, no clamp — authors') ---
    f_a = _random_mlp(D, 1, np.random.randint(3, 5), np.random.randint(8, 20),
                      weight_std=0.8, device=device)
    pi_obs = torch.sigmoid(_mlp(X, f_a))                 # (N,1)
    A = torch.bernoulli(pi_obs)

    # --- outcome f_BNN([X, A, U]); normalize Y by potential-outcome (y0,y1) stats ---
    f_y = _random_mlp(D + 1 + 1, 1, np.random.randint(3, 6), np.random.randint(10, 25),
                      weight_std=1.0, device=device)
    U = torch.randn(N, 1, device=device)
    zeros = torch.zeros(N, 1, device=device)
    ones = torch.ones(N, 1, device=device)
    y0 = _mlp(torch.cat([X, zeros, U], dim=1), f_y)      # potential outcome a=0
    y1 = _mlp(torch.cat([X, ones, U], dim=1), f_y)       # potential outcome a=1
    Y_obs = torch.where(A > 0.5, y1, y0)
    y_all = torch.cat([y0, y1], dim=0)                   # authors normalize over [y0,y1]
    y_mean, y_std = y_all.mean(), y_all.std() + 1e-6
    Y = (Y_obs - y_mean) / y_std

    return X, A, Y, pi_obs, f_y, (y_mean, y_std)


# =============================================================================
# 2. Frontier: spline flow over U, MSM divergence, Lagrangian sweep
# =============================================================================

def _f_bnn_on_u(x, a, u, f_y, y_norm):
    """f_BNN at each (B,G) for U-samples u.  x (B,G,D) a (B,G,1) u (B,G,S) -> (B,G,S)."""
    B, G, D = x.shape
    S = u.shape[-1]
    x_exp = x.unsqueeze(2).expand(B, G, S, D)
    a_exp = a.unsqueeze(2).expand(B, G, S, 1)
    u_exp = u.unsqueeze(-1)
    raw = _mlp(torch.cat([x_exp, a_exp, u_exp], dim=-1), f_y).squeeze(-1)   # (B,G,S)
    y_mean, y_std = y_norm
    return (raw - y_mean) / y_std


def _theta_gamma(sp, base_u, phi_u, x, a, pi, f_y, y_norm, cfg):
    """One forward pass over (B,G): spline flow -> mixture -> theta, and MSM gamma."""
    nu, log_p = batched_spline_forward(base_u, sp, cfg)          # (B,G,S)
    log_r = log_p - standard_normal_logprob(nu)
    r = torch.exp(log_r)                                         # authors: no clamp (flow keeps r bounded)
    gamma = torch.maximum(r.amax(dim=-1), 1.0 / r.amin(dim=-1))  # (B,G)  MSM divergence
    xi = torch.bernoulli(pi.unsqueeze(-1).expand_as(nu))         # (B,G,S)
    u_query = xi * phi_u + (1.0 - xi) * nu
    y = _f_bnn_on_u(x, a, u_query, f_y, y_norm)                  # (B,G,S)
    theta = y.mean(dim=-1)                                       # (B,G)
    return theta, gamma


class SolverConfig:
    """Authors' SolverConfig, author-exact numerics (Table 2):
    350 optimizer steps per lambda (base), k_train=128, k_eval=4096, sqrt-lr and
    inverse-sqrt-step lambda schedules with their reference/cap values."""
    def __init__(self, lr=1e-3, lr_lambda_ref=0.25, lr_lambda_min_mult=0.40,
                 max_steps=350, max_steps_max_mult=2.0, mc_train=128, mc_eval=4096):
        self.lr = lr
        self.lr_lambda_ref = lr_lambda_ref
        self.lr_lambda_min_mult = lr_lambda_min_mult
        self.max_steps = max_steps
        self.max_steps_max_mult = max_steps_max_mult
        self.mc_train = mc_train
        self.mc_eval = mc_eval


def _lr_mult_for_lambda(lam, scfg):
    """Authors' 'sqrt' lr-by-lambda schedule (smaller lr at smaller lambda)."""
    mult = math.sqrt(float(lam) / scfg.lr_lambda_ref)
    return min(1.0, max(scfg.lr_lambda_min_mult, mult))


def _max_steps_for_lambda(lam, scfg, lr_mult):
    """Authors' 'inverse_sqrt_lr' max-steps-by-lambda schedule (more steps when lr is smaller)."""
    step_mult = min(scfg.max_steps_max_mult, max(1.0, 1.0 / math.sqrt(lr_mult)))
    return int(math.ceil(scfg.max_steps * step_mult))


def _clip_spline_grad_per_query(sp_param, max_norm):
    """Authors' per-(query) spline-gradient clipping (each row clipped independently)."""
    if sp_param.grad is None:
        return
    g = sp_param.grad
    gnorm = torch.linalg.vector_norm(g, dim=-1, keepdim=True)
    coef = (max_norm / (gnorm + 1e-6)).clamp(max=1.0)
    g.mul_(coef)


def _solve_bound(sign, x_q, a_q, pi_q, f_y, y_norm, lambdas, cfg, scfg, device):
    """Authors' procedure: descending lambda sweep, WARM-STARTED from the previous
    lambda's spline params; fresh Adam per lambda; lr/max-steps scheduled by lambda;
    per-query grad clip; high-sample eval pass.
    Returns gamma, theta each (L, G)."""
    G, D = x_q.shape
    nb = cfg.num_bins
    d_sp = 2 * nb + (nb - 1)
    der_init = _inverse_softplus_target(1.0 - cfg.min_derivative)
    x = x_q.view(1, G, D)
    a = a_q.view(1, G, 1)
    pi = pi_q.view(1, G)

    sp = torch.zeros(1, G, d_sp, device=device)
    sp[..., 2 * nb:] = der_init                          # identity-ish start

    gammas, thetas = [], []
    for lam in lambdas:                                  # descending -> warm start
        lr_mult = _lr_mult_for_lambda(lam, scfg)
        eff_lr = scfg.lr * lr_mult
        eff_steps = _max_steps_for_lambda(lam, scfg, lr_mult)
        sp_p = nn.Parameter(sp.clone())                  # start from previous solution
        opt = torch.optim.Adam([sp_p], lr=eff_lr)        # fresh momentum per lambda
        for _ in range(eff_steps):
            opt.zero_grad()
            base_u = torch.randn(1, G, scfg.mc_train, device=device)
            phi_u = torch.randn(1, G, scfg.mc_train, device=device)
            theta, gamma = _theta_gamma(sp_p, base_u, phi_u, x, a, pi, f_y, y_norm, cfg)
            obj = sign * theta - float(lam) * gamma
            (-obj.mean()).backward()      # authors reduce by MEAN over queries (per_dgp_sum / batch_mean)
            _clip_spline_grad_per_query(sp_p, 1.0)
            opt.step()
        sp = sp_p.detach()                               # carry forward = warm start
        with torch.no_grad():
            base_u = torch.randn(1, G, scfg.mc_eval, device=device)
            phi_u = torch.randn(1, G, scfg.mc_eval, device=device)
            theta, gamma = _theta_gamma(sp, base_u, phi_u, x, a, pi, f_y, y_norm, cfg)
        gammas.append(gamma.squeeze(0))                  # (G,)
        thetas.append(theta.squeeze(0))
    return torch.stack(gammas, 0), torch.stack(thetas, 0)   # (L,G)


# =============================================================================
# 3. One training item  (Studio entry point)
# =============================================================================

def _sample_impl(N=1024, D=10, n_queries=32, n_lambda=5, seed=None, num_bins=16):
    if seed is not None:
        np.random.seed(int(seed) % (2**31 - 1))
        torch.manual_seed(int(seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = FlowConfig(num_bins=num_bins)      # author-exact: 16 bins, tail 6.0
    scfg = SolverConfig()                    # author-exact: 350 steps, MC 128/4096

    # (1) fake world
    X, A, Y, pi_obs, f_y, y_norm = _generate_dgp(N, D, device=device)

    # (2) query groups = n_queries patients x both arms
    src = np.random.choice(N, size=n_queries, replace=(n_queries > N))
    src_g = np.repeat(src, 2)                                   # each patient twice
    arm_g = np.tile([0, 1], n_queries).astype(np.float32)      # arm 0 and arm 1
    G = 2 * n_queries
    src_t = torch.as_tensor(src_g, device=device)
    x_q = X.index_select(0, src_t)                             # (G,D)
    a_q = torch.as_tensor(arm_g, device=device).view(G, 1)     # (G,1)
    pi_src = pi_obs.index_select(0, src_t)                     # P(A=1|x)
    pi_q = torch.where(a_q > 0.5, pi_src, 1.0 - pi_src).view(G)  # P(arm|x)

    # (3) bounds: descending lambda sweep, warm-started (50-point grid over [0.08, 2.0])
    lambdas = torch.logspace(math.log10(2.0), math.log10(0.08), n_lambda, device=device)
    g_up, t_up = _solve_bound(+1.0, x_q, a_q, pi_q, f_y, y_norm, lambdas, cfg, scfg, device)
    g_lo, t_lo = _solve_bound(-1.0, x_q, a_q, pi_q, f_y, y_norm, lambdas, cfg, scfg, device)

    # (4) assemble rows + cumulative monotonicity repair (per group, per bound)
    X_np = X.detach().cpu().numpy()
    A_np = A.detach().cpu().numpy().reshape(-1)
    Y_np = Y.detach().cpu().numpy().reshape(-1)
    L = n_lambda
    rows = []  # (query_id, src, arm, gamma, theta, bound)  bound 0=upper 1=lower

    def add_bound(g_LG, t_LG, bound_val):
        g = g_LG.detach().cpu().numpy()
        t = t_LG.detach().cpu().numpy()
        for grp in range(G):
            order = np.argsort(g[:, grp])
            gv, tv = g[order, grp], t[order, grp]
            tv = np.maximum.accumulate(tv) if bound_val == 0 else np.minimum.accumulate(tv)
            for k in range(L):
                rows.append((grp, int(src_g[grp]), float(arm_g[grp]),
                             float(gv[k]), float(tv[k]), bound_val))

    add_bound(g_up, t_up, 0)
    add_bound(g_lo, t_lo, 1)
    rows.sort(key=lambda r: (r[0], r[5]))                      # group rows by query_id

    # (5) PACK into one sequence: context rows on top, query rows below.
    #     token = (x_1..x_D, a, y_or_0, gamma_or_0, bound_flag, is_context)  width = D + 5
    #       context row: (x, a, y, 0,     0, 1)    y observed, gamma/bound masked to 0
    #       query   row: (x, a, 0, gamma, b, 0)    gamma/bound observed, y masked to 0
    #     0-filled (NO NaN) so a plain tabular_embedder works and grid_preprocessor
    #     is NOT needed. The is_context flag tells the model which of y / gamma is real.
    M = len(rows)
    W = D + 5
    Y_COL, GAMMA_COL, BOUND_COL, ISCTX_COL = D + 1, D + 2, D + 3, D + 4

    ctx = np.zeros((N, W), dtype=np.float32)
    ctx[:, :D] = X_np
    ctx[:, D] = A_np
    ctx[:, Y_COL] = Y_np
    ctx[:, ISCTX_COL] = 1.0            # gamma=0, bound=0 (masked); is_context=1

    qry = np.zeros((M, W), dtype=np.float32)
    for j, (_qid, src, arm, gam, _th, bnd) in enumerate(rows):
        qry[j, :D] = X_np[src]
        qry[j, D] = arm
        qry[j, GAMMA_COL] = gam
        qry[j, BOUND_COL] = float(bnd)   # y=0 (masked); is_context=0

    X_packed = np.concatenate([ctx, qry], axis=0).astype(np.float32)     # (N+M, D+5)
    theta_star = np.array([[r[4]] for r in rows], dtype=np.float32)      # (M, 1) target

    # diagnostics (not trained on; used by the bounds eval / calibration)
    gamma = np.array([[r[3]] for r in rows], dtype=np.float32)
    bound_type = np.array([[r[5]] for r in rows], dtype=np.float32)
    query_id = np.array([[r[0]] for r in rows], dtype=np.float32)
    source_row_index = np.array([[r[1]] for r in rows], dtype=np.float32)

    return {
        "X":                X_packed,         # (N+M, D+4)  packed context+query
        "theta_star":       theta_star,       # (M, 1)      bound label   [role target -> y]
        "n_ctx":            np.int64(N),       # scalar     boundary       [role context_length -> n_ctx]
        "gamma":            gamma,
        "bound_type":       bound_type,
        "query_id":         query_id,
        "source_row_index": source_row_index,
    }


def sample(N=1024, D=10, n_queries=32, n_lambda=5, seed=None, **kwargs):
    return _sample_impl(N=N, D=D, n_queries=n_queries, n_lambda=n_lambda, seed=seed)


@register_prior("causal_sensitivity_packed")
class CausalSensitivityPackedPrior(Prior):
    """Faithful CSA-PFN prior (Studio-packed X): SCM -> frontier -> one item.

    NOTE ON SCALE: defaults are SANITY size (N=256, n_queries=8, n_lambda=8) so a
    run that doesn't override them stays small enough to fit a 24 GB GPU.
    Studio passes prior params via the RUN's `prior.overrides`, NOT the prior
    form's min/max fields, so these code defaults are what actually run unless
    the run overrides them.
    For the AUTHOR-SCALE run, set defaults (or run overrides) to:
        N=1024, n_queries=32, n_lambda=5   -> sequence 1664 (author's Table 1 batch, fits 24 GB)
        N=1024, n_queries=32, n_lambda=50  -> sequence 7424 (full grid, needs 80 GB + small batch)
    """

    def sample(self, N=1024, D=10, n_queries=32, n_lambda=5, seed=None, **kwargs):
        return _sample_impl(N=N, D=D, n_queries=n_queries, n_lambda=n_lambda, seed=seed)


if __name__ == "__main__":
    import time
    t0 = time.time()
    out = sample(N=256, D=10, n_queries=8, n_lambda=8, seed=0)   # sanity-size self-test
    print(f"sample() took {time.time() - t0:.2f}s")
    for k, v in out.items():
        print(f"  {k:18s} {getattr(v, 'shape', '()')}  {v.dtype}")
    X = out["X"]; n = int(out["n_ctx"]); M = out["theta_star"].shape[0]; D = 10
    assert X.shape == (n + M, D + 5), f"X must be (n_ctx+M, D+5), got {X.shape}"
    assert (X[:n, D + 4] == 1.0).all(), "is_context must be 1 on context rows"
    assert (X[n:, D + 4] == 0.0).all(), "is_context must be 0 on query rows"
    assert np.isfinite(X).all(), "packed X must be 0-filled (no NaN) for the plain embedder"
    print(f"  n_ctx={n}  M={M}  X={X.shape}  packed-contract OK")