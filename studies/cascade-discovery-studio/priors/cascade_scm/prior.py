"""Cascade-SCM discovery prior — recover a feed-forward chemical-process DAG.

Ports the cascade-topology sampler from TCPFN's ``CausalTimePrior`` (v2.3
``v2.3-tennessee-class`` preset) into PFN Studio's tabular discovery shape.
Variables are partitioned into sequential layers (feeds → reactor →
separator → … → product); edges flow strictly forward with inter-layer
probability decaying by layer-gap, so most edges connect adjacent stages.
This matches Tennessee-Eastman's empirically feed-forward ground truth
(0 cycles), the structure Erdős–Rényi priors systematically miss.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


def _sample_cascade_dag(
    N: int,
    n_layers: int,
    inter_layer_edge_prob: float,
    layer_gap_decay: float,
    intra_layer_edge_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a feed-forward DAG mimicking chemical-process / TE topology.

    Returns an (N, N) binary adjacency, acyclic by construction (edges only
    flow to higher layer-ordered indices), with no self-loops.
    """
    if n_layers >= N:
        # More layers than vars — collapse to one var per layer, no edges.
        return np.zeros((N, N), dtype=np.float32)

    # Random layer-size partition via Dirichlet (α=2 → roughly uniform but
    # stochastic), each layer guaranteed ≥1 variable.
    fractions = rng.dirichlet(np.full(n_layers, 2.0))
    sizes = np.maximum(1, np.floor(fractions * N).astype(int))
    while sizes.sum() < N:
        sizes[rng.integers(0, n_layers)] += 1
    while sizes.sum() > N:
        idx = int(np.argmax(sizes))
        if sizes[idx] > 1:
            sizes[idx] -= 1
        else:
            break

    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    layers = [list(range(int(s), int(s) + int(sz))) for s, sz in zip(starts, sizes, strict=False)]

    adj = np.zeros((N, N), dtype=np.float32)
    for src_li, src_layer in enumerate(layers):
        # Inter-layer forward edges, gap-decayed.
        for dst_li in range(src_li + 1, len(layers)):
            dst_layer = layers[dst_li]
            gap = dst_li - src_li
            p = inter_layer_edge_prob * (layer_gap_decay ** (gap - 1))
            mask = rng.random((len(src_layer), len(dst_layer))) < p
            for ii, i in enumerate(src_layer):
                for jj, j in enumerate(dst_layer):
                    if mask[ii, jj]:
                        adj[i, j] = 1.0
        # Sparse intra-layer forward edges (strict sub-index order → acyclic).
        if intra_layer_edge_prob > 0 and len(src_layer) > 1:
            for ii in range(len(src_layer)):
                for jj in range(ii + 1, len(src_layer)):
                    if rng.random() < intra_layer_edge_prob:
                        adj[src_layer[ii], src_layer[jj]] = 1.0
    return adj


@register_prior("cascade_scm")
class CascadeScmPrior(Prior):
    def sample(
        self,
        *,
        seed,
        num_points=200,
        d=52,
        n_layers=4,
        inter_layer_edge_prob=0.08,
        layer_gap_decay=0.5,
        intra_layer_edge_prob=0.02,
        weight_range=(0.5, 2.0),
        noise_scale=0.3,
        **_,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        adj = _sample_cascade_dag(
            d,
            n_layers,
            inter_layer_edge_prob,
            layer_gap_decay,
            intra_layer_edge_prob,
            rng,
        )
        # Random linear weights on the sampled edges.
        signs = rng.choice([-1.0, 1.0], size=adj.shape)
        weights = (adj * rng.uniform(*weight_range, size=adj.shape) * signs).astype(np.float32)

        # Forward simulate. The cascade DAG is acyclic in index order
        # (edges only point to higher-indexed layers), so 0..d-1 is a valid
        # topological order.
        X = np.zeros((num_points, d), dtype=np.float32)
        for k in range(d):
            parents = np.where(weights[:, k] != 0)[0]
            if len(parents):
                X[:, k] = X[:, parents] @ weights[parents, k]
            X[:, k] += rng.normal(0, noise_scale, size=num_points).astype(np.float32)
        return {"X": X, "A": adj}
