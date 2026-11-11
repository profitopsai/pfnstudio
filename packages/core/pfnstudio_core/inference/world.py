"""A world model's inference runtime: predict, recommend and diagnose from its released network alone.

A world model (PFN Studio › World models) is a prior-fitted network trained on a simulator of
one kind of system, for example one AI model served on the GPUs of one server. Asked about one
real system, with whatever was measured on it, it predicts every output at other settings with
a 90% range and reads off the system's hidden factors, in one forward pass. This module answers
the questions PFN Studio's Ask answers with the released network, from an inference repo (see
``pfnstudio_core.inference``): the same request, the same answers.

A request: ``{"question": "predict" | "recommend" | "diagnose", "environment": {...},
"observations": [{"settings": {...}, "measured": {output: value}}], ...}``.

* ``environment``: what the system *is*. Raw input values, or a catalog row by table name
  (``{"hardware": "<row id>"}`` fills that table's columns).
* ``settings`` (observations, ``queries``, ``candidates``): what an operator changes, the same
  way. ``"sweep": true`` marks a row where the engine chose the input the world's
  ``engine_picks`` names and the row is its best (a max-throughput benchmark); that input then
  holds the cap it could go up to (default: the top of its range).
* predict: ``queries``, a list of settings. recommend: ``candidates`` (a list, or
  ``{"grid": {name: [values]}, "fixed": {...}}``), ``objective``, ``direction`` (max | min),
  ``constraints`` (``[{"output", "op": "<=" | ">=", "value"}]``). diagnose: nothing more.

Not here, because PFN Studio answers them with the simulator, which an inference repo does not
carry: *effect* when the network has no effect head, and diagnose's list of hidden factors no
measured output depends on (``not_reached`` is None; such a factor keeps a range close to its
prior range, so its ``narrowed`` stays near 0).
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np


class Refusal(Exception):  # noqa: N818 — what the asker sees is a refusal, not an error
    """The question leaves what this world covers; the message says how."""


class WorldInference:
    def __init__(self, root: Path, manifest: dict[str, Any], device: str = "cpu"):
        self.root, self.m, self.device = Path(root), manifest, device
        net = manifest["network"]
        self.inputs: list[str] = list(net["inputs"])
        self.outputs: list[str] = list(net["outputs"])
        self.unknowns: list[str] = list(net["unknowns"])
        self.unknown_ranges = {k: tuple(v) for k, v in net["unknown_ranges"].items()}
        self.zero_shot = bool(net.get("zero_shot", False))
        self.effect_inputs = list(net.get("effect_inputs") or [])
        self.picks = manifest.get("engine_picks") or None
        self.ranges = {i["name"]: (float(i["lo"]), float(i["hi"])) for i in manifest["inputs"]}
        self.tables = list(manifest["facts"])
        self.min_rows = int(manifest["graph"].get("min_rows", 1))
        self._graph = None

    # ── the graph ────────────────────────────────────────────────────────

    def graph(self):
        if self._graph is None:
            try:
                import torch
            except ImportError as e:
                raise Refusal('this model runs on torch: pip install "pfnstudio[torch]"') from e
            f = self.root / self.m["graph"]["file"]
            try:
                self._graph = torch.export.load(str(f)).module()
            except Exception as e:
                raise Refusal(
                    f"could not load {f.name} with torch {torch.__version__} (it was exported "
                    f"with torch {self.m.get('torch', '?')}): {type(e).__name__}: {e}"
                ) from e
        return self._graph

    def _pad(self, a: np.ndarray, n: int, fill: float) -> np.ndarray:
        if len(a) >= n:
            return a
        pad = np.full((n - len(a), *a.shape[1:]), fill, dtype=a.dtype)
        return np.concatenate([a, pad])

    def predict_arrays(
        self,
        ctx_x: np.ndarray,
        ctx_sweep: np.ndarray,
        ctx_y: np.ndarray,
        qry_x: np.ndarray,
        qry_sweep: np.ndarray,
    ) -> dict[str, Any]:
        """One system. ``ctx_x`` (rows, inputs) and ``ctx_y`` (rows, outputs; NaN = not
        measured) its measured rows, ``qry_x`` the rows to predict, columns in the manifest's
        order. Each output's median and 90% range per query row, and each hidden factor's."""
        import torch

        n_ctx, n_q = len(ctx_x), len(qry_x)
        if n_q == 0:
            raise Refusal("nothing to predict: give at least one query row")
        k = self.min_rows
        valid = self._pad(np.ones(n_ctx, dtype=bool), k, False)
        t = lambda a, dt=torch.float64: torch.as_tensor(a, dtype=dt)[None]  # noqa: E731
        with torch.no_grad():
            out = self.graph()(
                t(self._pad(ctx_x, k, 0.0)),
                t(self._pad(ctx_sweep, k, 0.0)),
                t(self._pad(ctx_y, k, np.nan)),
                t(valid, torch.bool),
                t(self._pad(qry_x, k, 0.0)),
                t(self._pad(qry_sweep, k, 0.0)),
            )
        y_med, y_lo, y_hi, u_med, u_lo, u_hi = (o[0].cpu().numpy() for o in out)
        res: dict[str, Any] = {
            o: {"median": y_med[:n_q, j], "lo": y_lo[:n_q, j], "hi": y_hi[:n_q, j]}
            for j, o in enumerate(self.outputs)
        }
        for j, n in enumerate(self.unknowns):
            res[f"unknown:{n}"] = {
                "median": float(u_med[j]),
                "lo": float(u_lo[j]),
                "hi": float(u_hi[j]),
            }
        return res

    # ── rows: catalog refs, sweeps, checks ───────────────────────────────

    def _expand(self, values: dict[str, Any]) -> dict[str, Any]:
        v = dict(values)
        for t in self.tables:
            ref = v.get(t["name"])
            if isinstance(ref, str):
                row = next((r for r in t["rows"] if str(r[t["key"]]) == ref), None)
                if row is None:
                    near = [str(r[t["key"]]) for r in t["rows"] if ref.lower() in str(r[t["key"]])]
                    raise Refusal(
                        f"no row '{ref}' in the {t['name']} table"
                        + (f" (did you mean {near[:8]}?)" if near else "")
                    )
                del v[t["name"]]
                v.update({c: row[c] for c in t["columns"]})
        return v

    def row(self, environment: dict[str, Any], settings: dict[str, Any]) -> tuple[dict, float]:
        """Every input of one row, checked, and its sweep flag."""
        s = self._expand(settings)
        sweep = 1.0 if s.pop("sweep", False) else 0.0
        if sweep:
            if not self.picks:
                raise Refusal("this world has no engine-picked input: a row cannot be a sweep")
            s.setdefault(self.picks["input"], self.picks["ceiling"])
        row = {**self._expand(environment), **s}
        missing = [n for n in self.inputs if n not in row]
        if missing:
            raise Refusal(f"these inputs are not given: {missing}")
        unknown = sorted(set(row) - set(self.inputs))
        if unknown:
            raise Refusal(f"not inputs of this world: {unknown}")
        row = {k: float(x) for k, x in row.items()}
        for k, (lo, hi) in self.ranges.items():
            if not lo <= row[k] <= hi:
                raise Refusal(f"{k} = {row[k]:g} is outside what this world covers ({lo:g}–{hi:g})")
        for t in self.tables:
            cols = t["columns"]
            if not any(
                all(abs(float(r[c]) - row[c]) <= 1e-9 * max(1.0, abs(row[c])) for c in cols)
                for r in t["rows"]
            ):
                given = ", ".join(f"{c}={row[c]:g}" for c in cols)
                raise Refusal(f"no row of the {t['name']} table has {given}")
        return row, sweep

    def _x(self, rows: list[tuple[dict, float]]) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray([[r[n] for n in self.inputs] for r, _ in rows], dtype=np.float64)
        return x.reshape(len(rows), len(self.inputs)), np.asarray([s for _, s in rows], float)

    # ── the questions ────────────────────────────────────────────────────

    def _run(self, env: dict, observations: list[dict], queries: list[dict]):
        outs = set(self.outputs)
        for o in observations:
            for k, v in o.get("measured", {}).items():
                if k not in outs:
                    raise Refusal(f"'{k}' is not an output of this world")
                if not v > 0:
                    raise Refusal(f"'{k}' = {v}: measured values must be positive")
        crows = [self.row(env, o.get("settings", {})) for o in observations]
        qrows = [self.row(env, q) for q in queries]
        cx, cs = self._x(crows)
        cy = np.asarray(
            [
                [float(o.get("measured", {}).get(k, np.nan)) for k in self.outputs]
                for o in observations
            ],
            dtype=np.float64,
        ).reshape(len(observations), len(self.outputs))
        qx, qs = self._x(qrows)
        # the context rows predicted too, in the same pass: their fitted values, for the fit
        p = self.predict_arrays(cx, cs, cy, np.concatenate([cx, qx]), np.concatenate([cs, qs]))
        n = len(observations)
        fit = []
        for j, o in enumerate(observations):
            for k, v in o.get("measured", {}).items():
                med = float(p[k]["median"][j])
                fit.append(
                    {
                        "row": j,
                        "output": k,
                        "measured": v,
                        "fitted": med,
                        "log_error": float(np.log(med / v)) if med > 0 else None,
                    }
                )
        post = {
            "engine": "network",
            "unknowns": {
                u: {
                    "p5": p[f"unknown:{u}"]["lo"],
                    "p50": p[f"unknown:{u}"]["median"],
                    "p95": p[f"unknown:{u}"]["hi"],
                    "prior_lo": self.unknown_ranges[u][0],
                    "prior_hi": self.unknown_ranges[u][1],
                }
                for u in self.unknowns
            },
            "fit": fit,
            "not_reached": None,
            "warnings": [
                "which hidden factors no measured output depends on ('not reached') needs the "
                "simulator, which PFN Studio runs: every factor is listed; one the measurements "
                "cannot see keeps a range close to its prior range"
            ],
        }
        if not observations:
            post["warnings"].append(
                "no measurements: this is the network's answer for a system like this one that "
                "nobody has measured yet; one measured run narrows it to this system"
            )
        answers = []
        for j, (q, (_, sweep)) in enumerate(zip(queries, qrows, strict=True)):
            i = n + j
            a = {
                "settings": q,
                "outputs": {
                    k: {
                        "p50": float(p[k]["median"][i]),
                        "p5": float(p[k]["lo"][i]),
                        "p95": float(p[k]["hi"][i]),
                    }
                    for k in self.outputs
                },
            }
            if sweep:
                a["note"] = (
                    "sweep row: the network learned the engine's best only for "
                    + ", ".join(self.picks["maximizes"])
                    + "; read the others with care"
                )
            answers.append(a)
        return answers, post

    def ask(self, request: dict[str, Any]) -> dict[str, Any]:
        """One question about one system (PFN Studio's Ask request; ``draws`` and ``seed`` are
        accepted and unused: the network answers in one deterministic pass)."""
        r = request
        q = r.get("question")
        if q == "effect":
            raise Refusal(
                "effect is not in this inference repo: "
                + (
                    "the network has no effect head; PFN Studio answers it with the simulator"
                    if not self.effect_inputs
                    else "this inference format does not carry the effect head yet"
                )
                + ". Predict the base and the changed settings as two queries "
                "instead (two predictions, not a paired change)"
            )
        if q not in ("predict", "recommend", "diagnose"):
            raise Refusal("question must be predict, recommend or diagnose")
        env = dict(r.get("environment", {}))
        obs = list(r.get("observations", []))
        if not obs and not self.zero_shot:
            raise Refusal("no measurements, and this network cannot answer a system with no data")
        if q == "predict":
            answers, post = self._run(env, obs, list(r.get("queries", [])))
            return {
                "question": q,
                "engine": "network",
                "posterior": post,
                "answer": {"predictions": answers},
            }
        if q == "diagnose":
            first = [obs[0]["settings"]] if obs else list(r.get("queries", []))[:1]
            if not first:
                raise Refusal("diagnose reads the measurements: give observations")
            _, post = self._run(env, obs, first)
            return {"question": q, "engine": "network", "posterior": post, "answer": diagnose(post)}
        return {"question": q, "engine": "network", **self._recommend(env, obs, r)}

    def _recommend(self, env: dict, obs: list[dict], r: dict[str, Any]) -> dict[str, Any]:
        """Rank the asker's candidates (their list: nothing invented or capped). A candidate is
        feasible only if every constraint holds at its median; each carries its 90% range."""
        objective, direction = r.get("objective"), r.get("direction", "max")
        if direction not in ("max", "min"):
            raise Refusal("direction must be max or min")
        if objective not in self.outputs:
            raise Refusal(f"'{objective}' is not an output of this world")
        ok, refused = [], []
        for c in expand_candidates(r.get("candidates", [])):
            try:
                self.row(env, c)
                ok.append(c)
            except Refusal as e:
                refused.append({"settings": c, "reason": str(e)})
        if not ok:
            raise Refusal(
                "every candidate leaves what this world covers"
                + (f" (first: {refused[0]['reason']})" if refused else ": none given")
            )
        preds, post = self._run(env, obs, ok)
        ranked = []
        for p in preds:
            o, failed = p["outputs"], []
            for c in r.get("constraints", []):
                v = o.get(c["output"], {}).get("p50")
                if v is None or not np.isfinite(v):
                    failed.append(f"{c['output']} unknown")
                elif (c["op"] == "<=" and v > c["value"]) or (c["op"] == ">=" and v < c["value"]):
                    failed.append(f"{c['output']} {v:.4g} not {c['op']} {c['value']}")
            score = o[objective]["p50"]
            ranked.append(
                {
                    **p,
                    "objective": score,
                    "fails": failed,
                    "feasible": not failed and bool(np.isfinite(score)),
                }
            )
        sign = -1 if direction == "max" else 1
        ranked.sort(
            key=lambda x: (
                not x["feasible"],
                sign * (x["objective"] if np.isfinite(x["objective"]) else -sign * np.inf),
            )
        )
        return {
            "posterior": post,
            "answer": {
                "objective": objective,
                "direction": direction,
                "ranked": ranked,
                "refused": refused,
            },
        }

    # ── the release's own score, recomputed ──────────────────────────────

    def selfcheck(self, measurements: dict[str, Any]) -> dict[str, Any]:
        """The release's score, recomputed on a measurement set (PFN Studio's format: ``rows``,
        ``outputs``, ``group_by``): each row predicted from the other rows of its group, then
        from nothing, as mean absolute percentage error. Equal to the manifest's ``scores``
        means this runtime and graph answer as the released network did."""
        rows, got = measurements["rows"], {}
        picked = self.picks["input"] if self.picks else None

        def x_of(idx):
            x = np.asarray(
                [
                    [
                        float(rows[i].get("_maximize_upto") or self.picks["ceiling"])
                        if n == picked and rows[i].get("_maximize_over") == picked
                        else float(rows[i][n])
                        for n in self.inputs
                    ]
                    for i in idx
                ],
                dtype=np.float64,
            )
            s = np.asarray(
                [1.0 if picked and rows[i].get("_maximize_over") == picked else 0.0 for i in idx]
            )
            return x.reshape(len(idx), len(self.inputs)), s

        groups = ["|".join(str(r[k]) for k in measurements["group_by"]) for r in rows]
        for out in measurements["outputs"]:
            meas = np.asarray([float(r[out]) for r in rows])
            j = self.outputs.index(out)
            loo = np.full(len(rows), np.nan)
            for g in sorted(set(groups)):
                idx = [i for i, x in enumerate(groups) if x == g]
                for i in idx:
                    c = [k for k in idx if k != i]
                    if c:
                        cx, cs = x_of(c)
                        cy = np.full((len(c), len(self.outputs)), np.nan)
                        cy[:, j] = meas[c]
                        qx, qs = x_of([i])
                        loo[i] = self.predict_arrays(cx, cs, cy, qx, qs)[out]["median"][0]
            qx, qs = x_of(range(len(rows)))
            empty = np.zeros((0, len(self.inputs)))
            zs = self.predict_arrays(empty, np.zeros(0), np.zeros((0, len(self.outputs))), qx, qs)
            ok = np.isfinite(loo)
            got[out] = {
                "network": float(np.mean(np.abs(loo[ok] - meas[ok]) / meas[ok])),
                "network_zero_shot": float(np.mean(np.abs(zs[out]["median"] - meas) / meas)),
                "rows": len(rows),
                "groups": len(set(groups)),
                "unpredicted": int((~ok).sum()),
            }
        return got


def expand_candidates(cands: Any) -> list[dict[str, Any]]:
    """Candidates as a list of settings, or ``{"grid": {name: [values]}, "fixed": {...}}``:
    every combination of the grid's values (the asker's values; none added)."""
    if isinstance(cands, dict) and "grid" in cands:
        names = list(cands["grid"])
        return [
            {**cands.get("fixed", {}), **dict(zip(names, combo, strict=True))}
            for combo in itertools.product(*[cands["grid"][n] for n in names])
        ]
    return list(cands)


def diagnose(post: dict[str, Any]) -> dict[str, Any]:
    """What about this system differs from a typical one: hidden factors the measurements moved
    furthest from the middle of their range (``shift``, in range widths), how much they narrowed,
    and the measurements fitted worst."""
    moved = []
    for name, u in post["unknowns"].items():
        width, mid = u["prior_hi"] - u["prior_lo"], (u["prior_hi"] + u["prior_lo"]) / 2
        moved.append(
            {
                "unknown": name,
                "p50": u["p50"],
                "p5": u["p5"],
                "p95": u["p95"],
                "shift": (u["p50"] - mid) / width if width else 0.0,
                "narrowed": 1.0 - (u["p95"] - u["p5"]) / (0.9 * width) if width else 0.0,
            }
        )
    moved.sort(key=lambda m: -abs(m["shift"]))
    worst = sorted(
        [f for f in post["fit"] if f["log_error"] is not None], key=lambda f: -abs(f["log_error"])
    )
    return {
        "unknowns": moved,
        "not_reached": None,
        "worst_fits": worst[:5],
        "warnings": post["warnings"],
    }


def json_safe(o: Any) -> Any:
    """JSON any parser reads: numpy → lists, NaN/inf → null."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
