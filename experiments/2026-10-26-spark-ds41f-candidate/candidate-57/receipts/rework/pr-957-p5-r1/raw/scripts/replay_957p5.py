#!/usr/bin/env python3
"""Policy replay of #957 part 5 (round 3 rebase) on candidate-54's real AdaptiveVerificationManager.

Runs in spark-ds41f:candidate-50 with candidate-54's changed files mounted over the image's vLLM
(so every vllm import is candidate-54's), loads two adaptive_verification.py files by path:
  c54: candidate-54's (= candidate-57's after 957-p2 and 957-p4; neither touches it)
  p5:  the rebased part 5
Each manager is built through its own __init__ (small CUDA buffers) with the module's
get_tp_group patched to world size 3 or 4, so the TP selection under test is the patch's own.
get_num_tokens (the host draft budget, the whole policy) then runs on steps drawn from the served
acceptance profile of #986's round-1 replay: its fitted two-component logit-normal confidence model
(theta per concurrency, seeds 0 and 1), its step sampler (N decoding requests, one 21-token prefill
at the served prompt rate), calibrated confidences (the budget sees p exactly), so expected tokens
are the admitted slots' survival. The recipe: 5 drafts, row cost 0.035, dead-rows tau 0.2 (env),
calibrated estimator (refit done), copy drafter on (16 verification steps, 5 draft steps).

Step time: candidate-50's tp3-switched screen grid (two boots), fitted
t = a + b n + c1 min(R,16) + c2 max(R-16,0) at 4K context, rows padded to the TP3 capture ladder
(max_num_seqs 8 x 6 rows: 1-4,6,8,12..32 by 4,40,48; eager above). Served step time anchored to
candidate-50's tp3-switched c8 closed loop and single-stream c1 (constant overhead, #986's model A);
#986's tp4-ring per-batch serving regression (model B) as a sensitivity only.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

DRAFTS = 5
SPEC_STEPS = 16
ROW_COST = 0.035
CTX = 150


def ladder(max_size):
    lad = [1, 2, 3, 4, 6, 8, *range(12, 33, 4), *range(40, max_size + 1, 8)]
    return [*(s for s in lad if s < max_size), max_size]


TP3_LADDER = ladder(8 * (1 + DRAFTS))      # 48
TP4_LADDER = ladder(16 * (1 + DRAFTS))     # 96


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------ #986 round 1's step sampler and confidence model
# (verbatim from candidate-47/receipts/rework/pr-986-r1/scripts/replay_budget.py)
def draw_steps(rng, m, n_mean, prefill_rate, nmax):
    lo = int(math.floor(n_mean)); frac = n_mean - lo
    n = lo + (rng.random(m) < frac).astype(int)
    n = np.clip(n, 1, nmax)
    prefill = rng.random(m) < prefill_rate
    z = rng.standard_normal((m, nmax))
    e = rng.standard_normal((m, nmax, DRAFTS))
    u = rng.random((m, nmax))
    return n, prefill, (z, u), e


def confidences(theta, z, e):
    """Two-component logit-normal: an 'easy' share pi of request-steps is shifted by delta."""
    z, u = z
    mu = np.asarray(theta[:DRAFTS]); sr, se = abs(theta[DRAFTS]), abs(theta[DRAFTS + 1])
    delta = theta[DRAFTS + 2]; pi = 1.0 / (1.0 + math.exp(-theta[DRAFTS + 3]))
    logit = mu[None, None, :] + sr * z[:, :, None] + se * e + delta * (u < pi)[:, :, None]
    return 1.0 / (1.0 + np.exp(-logit))


# ------------------------------------------------ grid
def grid_points(paths):
    shapes = {}
    for p in paths:
        g = json.load(open(p))
        for e in g["engine"]:
            s = e["shape"]
            shapes.setdefault((s["num_reqs"], s["num_tokens"], s["context_len"]), []).append(
                e["step_ms"]["mean"])
    return {k: v for k, v in sorted(shapes.items())}


def grid_fit(paths):
    pts = grid_points(paths)
    rows = [(n, r, float(np.mean(v))) for (n, r, c), v in pts.items() if c == 4096]
    X = np.array([[1, n, min(r, 16), max(r - 16, 0)] for n, r, _ in rows], float)
    y = np.array([m for *_, m in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    names = ["a_ms", "per_request_ms", "per_row_le16_ms", "per_row_above16_ms"]
    return {
        "points": [{"num_reqs": n, "rows": r, "context": c, "step_ms_boots": [round(x, 3) for x in v],
                    "mean": round(float(np.mean(v)), 3)} for (n, r, c), v in pts.items()],
        "fit": dict(zip(names, map(float, coef))),
        "implied_knee_scale": float(coef[3] / coef[2]),
        "fit_resid_ms": [round(float(x), 3) for x in resid],
        "fit_rmse_ms": float(np.sqrt(np.mean(resid ** 2))),
        "per_row_le16_over_a": float(coef[2] / coef[0]),
    }


def pad(r, lad):
    r = np.asarray(r, float)
    cap = np.array(lad, float)
    idx = np.searchsorted(cap, np.minimum(r, cap[-1]), side="left")
    return np.where(r <= cap[-1], cap[np.minimum(idx, len(cap) - 1)], r)


def t_grid(fit, n, r, lad):
    r = pad(r, lad)
    f = fit
    return f["a_ms"] + f["per_request_ms"] * n + f["per_row_le16_ms"] * np.minimum(r, 16) + \
        f["per_row_above16_ms"] * np.maximum(r - 16, 0)


def t_interp(pts, n, r, lad):
    """Piecewise-linear in padded rows between the grid's own points at this request count
    (decode steps whose request count the grid measured); NaN elsewhere."""
    out = np.full(len(n), np.nan)
    rp = pad(r, lad)
    by_n = {}
    for (nn, rr, c), v in pts.items():
        if c == 4096:
            by_n.setdefault(nn, []).append((rr, float(np.mean(v))))
    for nn, lst in by_n.items():
        lst.sort()
        xs = np.array([a for a, _ in lst], float); ys = np.array([b for _, b in lst])
        sel = (n == nn) & (rp >= xs[0]) & (rp <= xs[-1])
        out[sel] = np.interp(rp[sel], xs, ys)
    return out


# ------------------------------------------------ managers
class Arm:
    def __init__(self, name, mod, world, overrides, max_reqs):
        self.name, self.mod, self.world = name, mod, world
        mod.get_tp_group = lambda: SimpleNamespace(world_size=world)
        self.rs = SimpleNamespace(
            num_speculative_steps=SPEC_STEPS, device=torch.device("cuda"), max_num_reqs=max_reqs,
            max_num_batched_tokens=6144, req_id_to_index={},
            num_computed_tokens_np=np.full(max_reqs, CTX, np.int32),
            prefill_len=SimpleNamespace(np=np.ones(max_reqs, np.int32)),
        )
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import get_max_chunk_logits
        self.max_total_logits = get_max_chunk_logits(129280)
        self.m = mod.AdaptiveVerificationManager(
            self.rs, torch.zeros(max_reqs + 1, dtype=torch.int32, device="cuda"), 1,
            max_total_logits=self.max_total_logits, row_cost=ROW_COST, planned_confidences=True,
            num_draft_steps=DRAFTS, tp_invariant_confidence=False, calibrated_confidence=lambda: True,
        )
        self.constants_from_init = {
            "high_row_cost_scale": getattr(self.m, "high_row_cost_scale", None),
            "single_req_tile_spill_scale": getattr(self.m, "single_req_tile_spill_scale", None),
            "min_calibrated_dead_rows_tau": getattr(self.m, "min_calibrated_dead_rows_tau", None),
            "high_row_cost_knee": getattr(self.m, "high_row_cost_knee", None),
            "dead_rows_tau": self.m.dead_rows_tau,
            "row_cost": self.m.row_cost,
        }
        for k, v in overrides.items():
            setattr(self.m, k, v)
        self.overrides = overrides
        self.tau = float(self.m.effective_dead_rows_tau())

    def budget(self, conf, n_drafting, extra_req, drafts_per_req=DRAFTS, extra_tokens=21):
        """conf: (k, steps) float32 confidences of the k drafting requests."""
        k = n_drafting
        reqs = k + extra_req
        ids = [f"r{j}" for j in range(reqs)]
        self.rs.req_id_to_index = {rid: j for j, rid in enumerate(ids)}
        st = self.m._stale_confidences[self.m._stale_idx].np
        st[:reqs, :] = 1.0
        st[:k, : conf.shape[1]] = conf
        ntok = {ids[j]: 1 + drafts_per_req for j in range(k)}
        drafts = {ids[j]: list(range(1, drafts_per_req + 1)) for j in range(k)}
        if extra_req:
            ntok[ids[k]] = extra_tokens
            drafts[ids[k]] = []
        total = self.m.get_num_tokens(ntok, drafts)
        return int(self.m._batch_budget[2]), int(total)


def admitted_from_budget(S, live, budget, tau):
    m, nmax, d = S.shape
    v = np.where(live[:, :, None] & (S >= tau), S, -np.inf).reshape(m, nmax * d)
    srt = -np.sort(-v, axis=1)
    thr = np.where(budget > 0, srt[np.arange(m), np.maximum(budget - 1, 0)], np.inf)
    vv = v.reshape(m, nmax, d)
    return (vv >= thr[:, None, None]) & np.isfinite(vv)


def counters(S, admitted, live):
    live = live & admitted[:, :, 0]
    nreq = max(int(live.sum()), 1)
    cov = [float((admitted[:, :, k] & live).sum() / nreq) for k in range(DRAFTS)]
    surv = [float((S[:, :, k] * (admitted[:, :, k] & live)).sum() / nreq) for k in range(DRAFTS)]
    return cov, surv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c54", required=True)
    ap.add_argument("--p5", required=True)
    ap.add_argument("--served", required=True, help="#986 r1 served_targets.json")
    ap.add_argument("--fit", required=True, help="#986 r1 replay-seed<k>.json (fitted theta)")
    ap.add_argument("--grid3", nargs="+", required=True)
    ap.add_argument("--grid4", nargs="+", required=True)
    ap.add_argument("--tp3-c8", nargs="+", required=True)
    ap.add_argument("--tp3-c1", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    assert os.environ.get("SPARK_DS41F_DSPARK_DEAD_ROWS_TAU") == "0.2"
    c54 = load(a.c54, "av_c54")
    p5 = load(a.p5, "av_p5")
    served = json.load(open(a.served))
    fitrep = json.load(open(a.fit))
    seed = int(fitrep["seed"])
    g3 = grid_fit(a.grid3); g4 = grid_fit(a.grid4)
    pts3 = grid_points(a.grid3)
    f3 = g3["fit"]
    # served TP3 anchors (candidate-50 tp3-switched): ms per step = 1000 / steps_per_s
    sp_c8 = [json.load(open(p))["summary"]["server"]["steps_per_s"]["mean"] for p in a.tp3_c8]
    sp_c1 = json.load(open(a.tp3_c1))["summary"]["server"]["steps_per_s"]["mean"]
    served_ms = {"c1": 1000.0 / sp_c1, "c8": 1000.0 / float(np.mean(sp_c8))}
    tp3_counters = {}
    for lab, paths in (("c8", a.tp3_c8), ("c1", [a.tp3_c1])):
        cov = [float(np.mean([json.load(open(p))["summary"]["server"][f"coverage_at_{k}"]["mean"] for p in paths])) for k in range(DRAFTS)]
        sur = [float(np.mean([json.load(open(p))["summary"]["server"][f"survival_at_{k}"]["mean"] for p in paths])) for k in range(DRAFTS)]
        tok = float(np.mean([json.load(open(p))["summary"]["server"]["output_tokens_per_step"]["mean"] for p in paths]))
        tp3_counters[lab] = {"coverage": cov, "survival": sur, "output_tokens_per_step": tok}

    report = {"seed": seed, "steps": a.steps, "torch": torch.__version__,
              "modules": {"c54": a.c54, "p5": a.p5}, "grid_tp3": g3, "grid_tp4_switched": g4,
              "tp3_ladder": TP3_LADDER, "served_tp3_step_ms": served_ms, "served_tp3_counters": tp3_counters,
              "concurrency": {}, "deterministic": {}}

    # ---- concurrencies: (label, theta source, n_mean, prefill_rate, nmax, arms to run)
    cc = served["concurrency"]
    c8 = cc["c8"]; c1 = cc["c1"]; c16 = cc["c16"]
    # (label, theta source, mean decoding requests, prefill rate, max requests, model-B regression key)
    plans = [
        ("c1", "c1", c1["requests"], c1["prefill_rate"], 1, "c1"),
        ("c4", "c8", c8["requests"] / 2, c8["prefill_rate"] / 2, 4, "c8"),
        ("c4-theta-c1", "c1", c8["requests"] / 2, c8["prefill_rate"] / 2, 4, "c8"),
        ("c8", "c8", c8["requests"], c8["prefill_rate"], 8, "c8"),
        ("c16-tp4-check", "c16", c16["requests"], c16["prefill_rate"], 16, "c16"),
    ]
    arm_specs = [
        ("c54@tp3", c54, 3, {}),
        ("c54@tp4", c54, 4, {}),
        ("p5@tp3", p5, 3, {}),
        ("p5@tp4", p5, 4, {}),
        ("p5-scale-only@tp3", p5, 3, {"min_calibrated_dead_rows_tau": 0.0}),
        ("p5-floor-only@tp3", p5, 3, {"high_row_cost_scale": 0.55}),
    ]
    reg = served["token_wasted_regression"]
    for ci, (label, tkey, n_mean, prate, nmax, rkey) in enumerate(plans):
        theta = fitrep["concurrency"][tkey]["fit"]["theta"]
        rng = np.random.default_rng(7 + 1000 * seed + 17 * ci)
        n, prefill, z, e = draw_steps(rng, a.steps, n_mean, prate, nmax)
        live = np.arange(nmax)[None, :] < n[:, None]
        P = confidences(theta, z, e).astype(np.float32)
        S = np.cumprod(P.astype(np.float64), axis=2)
        base = n + 21 * prefill
        nsamp = n + prefill
        nrq = n + prefill
        arms = [Arm(nm, mod, w, ov, nmax + 1) for nm, mod, w, ov in arm_specs]
        budgets = {ar.name: np.zeros(a.steps, int) for ar in arms}
        for i in range(a.steps):
            k = int(n[i]); p = int(prefill[i])
            conf = P[i, :k, :]
            for ar in arms:
                budgets[ar.name][i] = ar.budget(conf, k, p)[0]
        res = {"theta_from": tkey, "model_B_regression_from": rkey, "n_mean": n_mean, "prefill_rate": prate, "nmax": nmax,
               "share_prefill_steps": float(prefill.mean()), "arms": {}}
        if label in ("c1", "c8"):
            T_served = served_ms[label]
        elif label.startswith("c4"):
            T_served = None
        else:
            T_served = None
        ref = budgets["c54@tp3"]
        evals = {}
        for ar in arms:
            b = budgets[ar.name]
            adm = admitted_from_budget(S, live, b, ar.tau)
            mism = int(np.sum(adm.sum((1, 2)) != b))
            E = nsamp + (S * adm).sum((1, 2))
            R = base + b
            tg = t_grid(f3, nrq, R, TP3_LADDER)
            ti = t_interp(pts3, nrq, R, TP3_LADDER)
            dec = (~prefill)
            cov, surv = counters(S, adm, live)
            padded = pad(R, TP3_LADDER).astype(int)
            hist = np.bincount(np.minimum(padded, 120), minlength=121)
            dec_tok = (E - prefill).mean(); dec_rows = (R - 21 * prefill).mean()
            evals[ar.name] = {"E": E, "R": R, "tg": tg, "ti": ti, "b": b}
            res["arms"][ar.name] = {
                "constants_from_init": ar.constants_from_init, "overrides": ar.overrides,
                "effective_dead_rows_tau": ar.tau,
                "draft_budget_mean": float(b.mean()), "rows_mean": float(R.mean()),
                "tokens_mean": float(E.mean()), "wasted_rows_mean": float((R - E).mean()),
                "coverage": cov, "survival": surv,
                "steps_budget_differs_from_c54@tp3": float(np.mean(b != ref)),
                "steps_budget_differs_from_c54@tp4": float(np.mean(b != budgets["c54@tp4"])),
                "mean_budget_delta_vs_c54@tp3": float((b - ref).mean()),
                "admitted_count_mismatch_steps": mism,
                "grid_gpu_step_ms": float(tg.mean()),
                "grid_interp_gpu_step_ms_decode_steps": float(np.nanmean(ti[dec])) if np.isfinite(ti[dec]).any() else None,
                "interp_coverage_decode_steps": float(np.mean(np.isfinite(ti[dec]))),
                "padded_rows_histogram": {str(i): int(c) for i, c in enumerate(hist) if c},
                "model_B_step_ms": float(reg[rkey]["intercept"] + reg[rkey]["ms_per_token"] * dec_tok
                                         + reg[rkey]["ms_per_wasted_row"] * (dec_rows - dec_tok)),
            }
        # overhead (model A) anchored on c54@tp3; c4 overhead interpolated in n between c1 and c8
        g0 = evals["c54@tp3"]["tg"].mean()
        res["anchor"] = {}
        if T_served is not None:
            overhead = T_served - g0
        else:
            overhead = None
        res["anchor"]["served_step_ms"] = T_served
        res["anchor"]["overhead_A_ms"] = overhead
        report["concurrency"][label] = res
        res["_evals_tmp"] = evals  # dropped before writing
        print(f"== {label}: steps {a.steps}, seed {seed}, prefill share {prefill.mean():.4f}", flush=True)
        for nm, r in res["arms"].items():
            print(f"   {nm:20s} tau {r['effective_dead_rows_tau']:.4f} budget {r['draft_budget_mean']:.4f} rows {r['rows_mean']:.4f} tok {r['tokens_mean']:.4f} "
                  f"differs {100*r['steps_budget_differs_from_c54@tp3']:.2f}% gpu {r['grid_gpu_step_ms']:.4f} mism {r['admitted_count_mismatch_steps']}", flush=True)

    # c4 overhead: linear in mean requests between the c1 and c8 anchors
    o1 = report["concurrency"]["c1"]["anchor"]["overhead_A_ms"]; o8 = report["concurrency"]["c8"]["anchor"]["overhead_A_ms"]
    n1 = served["concurrency"]["c1"]["requests"]; n8 = served["concurrency"]["c8"]["requests"]
    for label in ("c4", "c4-theta-c1"):
        r = report["concurrency"][label]
        n4 = r["n_mean"]
        o4 = o1 + (o8 - o1) * (n4 - n1) / (n8 - n1)
        r["anchor"]["overhead_A_ms"] = o4
        r["anchor"]["overhead_note"] = "no served TP3 c4 record: overhead interpolated linearly in requests between the c1 and c8 anchors"

    # predictions
    for label, r in report["concurrency"].items():
        ev = r.pop("_evals_tmp")
        base_arm = "c54@tp4" if label.startswith("c16") else "c54@tp3"
        E0 = ev[base_arm]["E"].mean(); G0 = ev[base_arm]["tg"].mean()
        O = r["anchor"]["overhead_A_ms"]
        B0 = r["arms"][base_arm]["model_B_step_ms"]
        for nm, ar in r["arms"].items():
            E1 = ev[nm]["E"].mean(); G1 = ev[nm]["tg"].mean(); B1 = ar["model_B_step_ms"]
            pr = {
                "vs": base_arm,
                "tokens_per_step_pct": 100 * (E1 / E0 - 1),
                "rows_per_step_pct": 100 * (ev[nm]["R"].mean() / ev[base_arm]["R"].mean() - 1),
                "grid_gpu_step_ms_delta": float(G1 - G0),
                "grid_gpu_step_pct": 100 * (G1 / G0 - 1),
                "throughput_grid_gpu_only_pct": 100 * ((E1 / G1) / (E0 / G0) - 1),
            }
            if O is not None:
                pr["served_step_ms_model_A"] = float(G1 + O)
                pr["served_step_pct_model_A"] = 100 * ((G1 + O) / (G0 + O) - 1)
                pr["throughput_model_A_pct"] = 100 * ((E1 / (G1 + O)) / (E0 / (G0 + O)) - 1)
            pr["throughput_model_B_tp4_regression_pct"] = 100 * ((E1 / B1) / (E0 / B0) - 1)
            ti0 = ev[base_arm]["ti"]; ti1 = ev[nm]["ti"]
            ok = np.isfinite(ti0) & np.isfinite(ti1)
            if ok.any():
                pr["interp_decode_steps_share"] = float(ok.mean())
                pr["interp_gpu_step_pct"] = 100 * (ti1[ok].mean() / ti0[ok].mean() - 1)
            # paired bootstrap of the throughput change (model A), over steps
            if O is not None:
                rng = np.random.default_rng(99 + seed)
                m = len(ev[nm]["E"]); vals = []
                for _ in range(400):
                    idx = rng.integers(0, m, m)
                    e0 = ev[base_arm]["E"][idx].mean(); e1 = ev[nm]["E"][idx].mean()
                    g0 = ev[base_arm]["tg"][idx].mean(); g1 = ev[nm]["tg"][idx].mean()
                    vals.append(100 * ((e1 / (g1 + O)) / (e0 / (g0 + O)) - 1))
                pr["throughput_model_A_pct_bootstrap95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            ar["prediction"] = pr

    # ---- deterministic cells (each arm's own __init__ at TP3 and TP4)
    cells = {
        "c8_medium": (np.tile(np.array([[0.78, 0.72, 0.71, 0.71, 0.73]], np.float32), (8, 1)), 8, DRAFTS),
        "c4_medium": (np.tile(np.array([[0.78, 0.72, 0.71, 0.71, 0.73]], np.float32), (4, 1)), 4, DRAFTS),
        "c1_typical": (np.array([[0.677, 0.556, 0.396, 0.460, 0.638]], np.float32), 1, DRAFTS),
        "c1_high": (np.array([[0.85, 0.80, 0.75, 0.70, 0.70]], np.float32), 1, DRAFTS),
        "c1_low": (np.array([[0.45, 0.40, 0.30, 0.30, 0.30]], np.float32), 1, DRAFTS),
        "c1_copy_0.95x16": (np.full((1, 16), 0.95, np.float32), 1, 16),
        "c1_copy_0.97x16": (np.full((1, 16), 0.97, np.float32), 1, 16),
        "c2_copy_0.95x16": (np.full((2, 16), 0.95, np.float32), 2, 16),
    }
    det_arms = arm_specs + [("p5-0.85-also-at-c1@tp3 (rejected composition)", p5, 3, {"single_req_tile_spill_scale": 0.85}),
                            ("c54-0.85-at-c1@tp3 (pre-#971 part 5 on c1)", c54, 3, {"high_row_cost_scale": 0.85, "single_req_tile_spill_scale": 0.85})]
    for nm, mod, w, ov in det_arms:
        ar = Arm(nm, mod, w, ov, 9)
        row = {"effective_dead_rows_tau": ar.tau, "constants_from_init": ar.constants_from_init, "overrides": ov}
        for cname, (conf, k, d) in cells.items():
            b, tot = ar.budget(conf, k, 0, drafts_per_req=d)
            row[cname] = {"draft_budget": b, "total_tokens": tot}
        report["deterministic"][nm] = row
        print("   det", nm, {c: (row[c]["draft_budget"], row[c]["total_tokens"]) for c in cells}, flush=True)

    # TP4 bit-for-bit
    tp4 = {}
    for label, r in report["concurrency"].items():
        tp4[label] = {
            "p5@tp4_differs_from_c54@tp4_steps": r["arms"]["p5@tp4"]["steps_budget_differs_from_c54@tp4"],
            "c54@tp3_differs_from_c54@tp4_steps": r["arms"]["c54@tp3"]["steps_budget_differs_from_c54@tp4"],
        }
    tp4["deterministic_equal"] = all(
        report["deterministic"]["p5@tp4"][c] == report["deterministic"]["c54@tp4"][c] for c in cells)
    report["tp4_bit_for_bit"] = tp4
    report["elapsed_s"] = time.time() - t0
    json.dump(report, open(a.out, "w"), indent=1)
    print("tp4", json.dumps(tp4), flush=True)
    print(f"done in {report['elapsed_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
