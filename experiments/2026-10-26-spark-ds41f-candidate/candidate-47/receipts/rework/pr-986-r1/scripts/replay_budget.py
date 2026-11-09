#!/usr/bin/env python3
"""Replay of the DSpark draft budget (1107's and #986's adaptive_verification.py, the real
modules) over steps drawn from a confidence model fitted to the served rows-a-step counters
of candidate-47-no986 (the 1107 budget), priced by the tp4-ring engine grid of the same
image (4 boots).

Inputs (all read-only copies):
  --grid      grid.json of the four candidate-47 / candidate-47-no986 boots
  --served    rows-a-step targets (served_targets.json, from the suite runs)
  --regress   per-batch regression of serving step time (out-batch-regress.json)
  --modules   av_1107.py av_986.py
Outputs one JSON with: the grid step-time fit, the fitted confidence model per
concurrency, and per policy: rows, tokens, wasted rows, coverage/survival by position,
the share of steps above 64 rows, the share of steps whose budget differs from 1107's,
and the throughput each step-time model predicts.

Confidence model: logit p_ik = mu_k + s_r z_i + s_e e_ik (z per request-step, e per slot),
calibrated (the budget sees p exactly), acceptance a Bernoulli chain. Expected values are
used for tokens. Steps carry N decoding requests (the served requests a step) and, with the
served prompt-token rate, one 21-token prefill (21 non-draft rows, one sampling request).
"""
import argparse
import importlib.util
import json
import math
import sys
from types import SimpleNamespace

import numpy as np

DRAFTS = 5
ROW_COST = 0.035
TAU_ENV = 0.2
CAPTURE = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- grid step-time fit
def grid_fit(paths):
    shapes = {}
    for p in paths:
        g = json.load(open(p))
        for e in g["engine"]:
            s = e["shape"]
            key = (s["num_reqs"], s["num_tokens"], s["context_len"])
            shapes.setdefault(key, []).append(e["step_ms"]["mean"])
    pts = [(n, r, c, float(np.mean(v)), float(np.std(v))) for (n, r, c), v in sorted(shapes.items())]
    # context: same (N, R) at different context lengths
    ctx = {}
    for n, r, c, m, _ in pts:
        ctx.setdefault((n, r), []).append((c, m))
    ctx_effect = {f"{n}x{r}": [(c, round(m, 3)) for c, m in sorted(v)] for (n, r), v in ctx.items() if len(v) > 1}
    # fit at 4K: t = a + bN + c1 min(R,16) + c2 clip(R-16,0,48) + J [R>64] + c3 max(R-64,0)
    rows = [(n, r, m) for n, r, c, m, _ in pts if c == 4096]
    X = np.array([[1, n, min(r, 16), min(max(r - 16, 0), 48), 1.0 * (r > 64), max(r - 64, 0)] for n, r, _ in rows], float)
    y = np.array([m for *_, m in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    names = ["a_ms", "per_request_ms", "per_row_le16_ms", "per_row_17_64_ms", "jump_above_64_ms", "per_row_above_64_ms"]
    fit = dict(zip(names, map(float, coef)))
    # the same without the >64 jump (linear in 17..96) for comparison
    X2 = np.array([[1, n, min(r, 16), max(r - 16, 0)] for n, r, _ in rows], float)
    coef2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    return {
        "points": [{"num_reqs": n, "rows": r, "context": c, "step_ms": round(m, 3), "sd_boots": round(s, 3)} for n, r, c, m, s in pts],
        "context_effect": ctx_effect,
        "fit": fit,
        "fit_resid_ms": [round(float(v), 3) for v in resid],
        "fit_rmse_ms": float(np.sqrt(np.mean(resid ** 2))),
        "linear_fit": dict(zip(["a_ms", "per_request_ms", "per_row_le16_ms", "per_row_above16_ms"], map(float, coef2))),
    }


def t_gpu(fit, n, r, pad=True):
    r = np.asarray(r, float)
    if pad:
        cap = np.array(CAPTURE, float)
        idx = np.searchsorted(cap, np.minimum(r, cap[-1]), side="left")
        r = np.where(r <= cap[-1], cap[np.minimum(idx, len(cap) - 1)], r)
    f = fit
    return (f["a_ms"] + f["per_request_ms"] * n + f["per_row_le16_ms"] * np.minimum(r, 16)
            + f["per_row_17_64_ms"] * np.clip(r - 16, 0, 48) + f["jump_above_64_ms"] * (r > 64)
            + f["per_row_above_64_ms"] * np.maximum(r - 64, 0))


# ---------------------------------------------------------------- step sampler
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


def tau_1107():
    return min(TAU_ENV, 2.0 * ROW_COST / (1.0 + 3.0 * ROW_COST))


def vector_budget_1107(S, live, base_rows, nsamp):
    """1107's get_num_tokens rule, vectorised over steps (for the fit; checked against the module)."""
    m, nmax, d = S.shape
    tau = tau_1107()
    v = np.where(live[:, :, None] & (S >= tau), S, -np.inf).reshape(m, nmax * d)
    srt = -np.sort(-v, axis=1)
    fin = np.isfinite(srt)
    cs = np.concatenate([np.zeros((m, 1)), np.cumsum(np.where(fin, srt, 0.0), axis=1)], axis=1)
    k = np.arange(nmax * d + 1)
    E = nsamp[:, None] + cs
    R = base_rows[:, None] + k[None, :]
    eff = np.minimum(R, 16) + 0.55 * np.maximum(R - 16, 0)
    U = E / (1.0 + ROW_COST * eff)
    U = np.where(k[None, :] <= fin.sum(1)[:, None], U, -np.inf)
    b = U.argmax(1)
    thr = np.where(b > 0, srt[np.arange(m), np.maximum(b - 1, 0)], np.inf)
    admitted = (v.reshape(m, nmax, d) >= thr[:, None, None]) & np.isfinite(v.reshape(m, nmax, d))
    return b, admitted


def counters(S, P, admitted, live):
    """Served-style counters per decoding request-step: coverage_k, survival_k (expected)."""
    # servebench divides by vllm:spec_decode_num_drafts, which counts request-steps that
    # verified at least one draft (coverage_at_0 == 1 by construction)
    live = live & admitted[:, :, 0]
    nreq = live.sum()
    cov = [(admitted[:, :, k] & live).sum() / nreq for k in range(DRAFTS)]
    # survival_k: accepted at k = admitted at k and all of 0..k accepted -> E = S_k * 1(admitted k)
    surv = [float((S[:, :, k] * (admitted[:, :, k] & live)).sum() / nreq) for k in range(DRAFTS)]
    return [float(c) for c in cov], surv


def fit_conf(target, rng_seed, n_mean, prefill_rate, nmax, m=6000):
    rng = np.random.default_rng(rng_seed)
    n, prefill, z, e = draw_steps(rng, m, n_mean, prefill_rate, nmax)
    live = np.arange(nmax)[None, :] < n[:, None]
    base = n + 21 * prefill
    nsamp = n + prefill
    tc = np.array(target["coverage"][1:DRAFTS]); ts = np.array(target["survival"][:DRAFTS])
    ttok = target["tokens_per_step"]

    def loss(theta):
        S = np.cumprod(confidences(theta, z, e), axis=2)
        _, adm = vector_budget_1107(S, live, base, nsamp)
        cov, surv = counters(S, None, adm, live)
        tok = float((nsamp + (S * adm).sum((1, 2))).mean())
        return float(np.sum((np.array(cov[1:]) - tc) ** 2) + np.sum((np.array(surv) - ts) ** 2)
                     + 1.0 * ((tok - ttok) / ttok) ** 2)

    # start: logits of served conditional acceptance
    cond = [ts[0]] + [ts[k] / ts[k - 1] for k in range(1, DRAFTS)]
    x0 = np.array([math.log(c / (1 - c)) for c in cond] + [1.0, 0.8, 2.0, -1.0])
    # Nelder-Mead
    dim = len(x0); simplex = [x0] + [x0 + np.eye(dim)[i] * 0.4 for i in range(dim)]
    vals = [loss(x) for x in simplex]
    for _ in range(2500):
        order = np.argsort(vals); simplex = [simplex[i] for i in order]; vals = [vals[i] for i in order]
        c = np.mean(simplex[:-1], axis=0)
        xr = c + (c - simplex[-1]); fr = loss(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - simplex[-1]); fe = loss(xe)
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (simplex[-1] - c); fc = loss(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                simplex = [simplex[0]] + [simplex[0] + 0.5 * (s - simplex[0]) for s in simplex[1:]]
                vals = [vals[0]] + [loss(s) for s in simplex[1:]]
        if max(vals) - min(vals) < 1e-10:
            break
    best = simplex[int(np.argmin(vals))]
    S = np.cumprod(confidences(best, z, e), axis=2)
    _, adm = vector_budget_1107(S, live, base, nsamp)
    cov, surv = counters(S, None, adm, live)
    tok = float((nsamp + (S * adm).sum((1, 2))).mean())
    share = float((adm[:, :, 0] & live).sum() / live.sum())
    return best, float(min(vals)), cov, surv + [tok, share]


# ---------------------------------------------------------------- real modules
def make_manager(mod, conf, ctx_len, attrs=None):
    num_reqs = conf.shape[0]
    m = mod.AdaptiveVerificationManager.__new__(mod.AdaptiveVerificationManager)
    m.num_speculative_steps = DRAFTS
    m.num_draft_steps = DRAFTS
    m._stale_confidences = [SimpleNamespace(np=conf)]
    m._stale_idx = 0
    m.req_states = SimpleNamespace(
        req_id_to_index={f"r{i}": i for i in range(num_reqs)},
        num_computed_tokens_np=np.full(num_reqs, ctx_len, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int32)),
    )
    m._max_total_logits = 1 << 20
    m.num_bonus_tokens = 1
    m.row_cost = ROW_COST
    m.dead_rows_tau = TAU_ENV
    m.calibrated_confidence = True
    m.high_row_cost_knee = mod.HIGH_ROW_COST_KNEE
    m.high_row_cost_scale = mod.HIGH_ROW_COST_SCALE
    for k, v in (attrs or {}).items():
        setattr(m, k, v)
    return m


def module_budget(mod, Sconf, n, prefill, ctx_len, attrs=None, step_cost=None):
    """Draft budget per step from the module's own get_num_tokens; returns budget array."""
    m_steps = len(n)
    out = np.zeros(m_steps, int)
    for i in range(m_steps):
        k = int(n[i]); p = bool(prefill[i])
        conf = Sconf[i, :k, :]
        reqs = k + p
        full = np.ones((reqs, DRAFTS), np.float32)
        full[:k] = conf
        mgr = make_manager(mod, full, ctx_len, attrs)
        if step_cost is not None:
            mgr._step_cost = step_cost.__get__(mgr)
        ntok = {f"r{j}": 1 + DRAFTS for j in range(k)}
        drafts = {f"r{j}": list(range(1, DRAFTS + 1)) for j in range(k)}
        if p:
            ntok[f"r{k}"] = 21
            drafts[f"r{k}"] = []
        mgr.get_num_tokens(ntok, drafts)
        out[i] = mgr._batch_budget[2]
    return out


def admitted_from_budget(S, live, budget, row_cost=ROW_COST):
    m, nmax, d = S.shape
    tau = min(TAU_ENV, 2.0 * row_cost / (1.0 + 3.0 * row_cost))
    v = np.where(live[:, :, None] & (S >= tau), S, -np.inf).reshape(m, nmax * d)
    srt = -np.sort(-v, axis=1)
    thr = np.where(budget > 0, srt[np.arange(m), np.maximum(budget - 1, 0)], np.inf)
    vv = v.reshape(m, nmax, d)
    return (vv >= thr[:, None, None]) & np.isfinite(vv)


def oracle_budget(S, live, base, nsamp, tmodel, lam0):
    """Dinkelbach: per step maximise E_k - lam t_k, lam = long-run tokens per ms."""
    m, nmax, d = S.shape
    tau = 0.0
    v = np.where(live[:, :, None] & (S >= tau), S, -np.inf).reshape(m, nmax * d)
    srt = -np.sort(-v, axis=1)
    fin = np.isfinite(srt)
    cs = np.concatenate([np.zeros((m, 1)), np.cumsum(np.where(fin, srt, 0.0), axis=1)], axis=1)
    k = np.arange(nmax * d + 1)
    E = nsamp[:, None] + cs
    R = base[:, None] + k[None, :]
    nrq = (live.sum(1) + (base - live.sum(1) > 0))[:, None]
    lam = lam0
    for _ in range(30):
        T = tmodel(nrq, R, E)
        obj = np.where(k[None, :] <= fin.sum(1)[:, None], E - lam * T, -np.inf)
        b = obj.argmax(1)
        Eb = E[np.arange(m), b]; Tb = T[np.arange(m), b]
        new = Eb.sum() / Tb.sum()
        if abs(new - lam) < 1e-12:
            break
        lam = new
    thr = np.where(b > 0, srt[np.arange(m), np.maximum(b - 1, 0)], np.inf)
    vv = v.reshape(m, nmax, d)
    return b, (vv >= thr[:, None, None]) & np.isfinite(vv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", nargs=4, required=True)
    ap.add_argument("--served", required=True)
    ap.add_argument("--regress", required=True)
    ap.add_argument("--modules", nargs=2, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0, help="offsets the fit's and the replay's random draws")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    av1107 = load(args.modules[0], "av_1107")
    av986 = load(args.modules[1], "av_986")
    served = json.load(open(args.served))
    regress = json.load(open(args.regress))
    grid = grid_fit(args.grid)
    f = grid["fit"]
    report = {"grid": grid, "seed": args.seed, "steps": args.steps, "concurrency": {}}

    # 1107 _step_cost in ms terms: unit_ms so that rc*0.55*unit = grid per-row 17..64 slope
    unit_ms = f["per_row_17_64_ms"] / (ROW_COST * 0.55)
    jump_rows = f["jump_above_64_ms"] / (unit_ms * ROW_COST)
    above_scale = f["per_row_above_64_ms"] / (unit_ms * ROW_COST)

    def grid_knee_cost(self, rows, **kw):
        """1107's two-slope cost plus the grid's own >64-row jump and slope (a 'corrected' knee)."""
        r = np.asarray(rows, np.float64)
        eff = np.minimum(r, 16) + 0.55 * np.clip(r - 16, 0, 48) + above_scale * np.maximum(r - 64, 0) + jump_rows * (r > 64)
        return 1.0 + self.row_cost * eff

    report["grid_knee_in_1107_units"] = {"unit_ms": unit_ms, "jump_rows": jump_rows, "above_64_scale": above_scale}

    variants = {
        "1107": (av1107, None, None),
        "986": (av986, None, None),
        "986-no-tile-steps": (av986, {"mma_tile_step_rows": 0.0, "single_req_tile_step_rows": 0.0}, None),
        "986-no-64-knee": (av986, {"saturated_row_cost_knee": 0, "second_wave_step_rows": 0.0}, None),
        "1107+grid-64-knee": (av1107, None, grid_knee_cost),
        # the recipe constant itself, for the 2026-10-18 check (0.02 against 0.035 served the same)
        "1107-row-cost-0.020": (av1107, {"row_cost": 0.020}, None),
        "1107-row-cost-0.050": (av1107, {"row_cost": 0.050}, None),
    }

    fits = {}
    for ci, (name, cfg) in enumerate(served["concurrency"].items()):
        nmax = cfg["nmax"]
        key = cfg.get("base", name)
        ki = list(served["concurrency"]).index(key)
        if key not in fits:
            fits[key] = fit_conf(cfg["target"], 100 + ki + 1000 * args.seed, cfg["requests"], cfg["prefill_rate"], nmax)
        best, loss, cov_fit, surv_fit = fits[key]
        rng = np.random.default_rng(7 + ki + 1000 * args.seed)
        n, prefill, z, e = draw_steps(rng, args.steps, cfg["requests"], cfg["prefill_rate"], nmax)
        live = np.arange(nmax)[None, :] < n[:, None]
        P = confidences(best, z, e)
        P = P.astype(np.float32).astype(np.float64)
        S = np.cumprod(P, axis=2)
        base = n + 21 * prefill
        nsamp = n + prefill
        vb, _ = vector_budget_1107(S, live, base, nsamp)

        # serving step-time models, anchored at the served 1107 step time
        reg = regress[key]
        t_served = reg["step_ms_mean"]

        res = {"fit": {"theta": [float(x) for x in best], "loss": loss, "coverage": cov_fit, "survival": surv_fit[:DRAFTS],
                       "tokens_per_step": surv_fit[DRAFTS], "share_request_steps_drafting": surv_fit[DRAFTS + 1],
                       "target": cfg["target"]}, "policies": {}}
        budgets = {}
        for vname, (mod, attrs, sc) in variants.items():
            budgets[vname] = module_budget(mod, P.astype(np.float32), n, prefill, cfg.get("context", 150), attrs, sc)
        res["vector_rule_matches_1107_module"] = float(np.mean(budgets["1107"] == vb))

        def evaluate(adm, b):
            Eb = nsamp + (S * adm).sum((1, 2))
            rows = base + adm.sum((1, 2))
            nrq = n + prefill
            tg = t_gpu(f, nrq, rows, pad=True)
            return Eb, rows, tg

        # anchor overheads on the 1107 policy
        adm1107 = admitted_from_budget(S, live, budgets["1107"])
        E0, R0, T0 = evaluate(adm1107, budgets["1107"])
        overhead_A = t_served - T0.mean()
        wr = reg["with_requests"]
        # model B: the per-batch serving regression on tokens and wasted rows
        ab = served["token_wasted_regression"][key]

        def tA(nrq, R, E):
            return t_gpu(f, nrq, R, pad=True) + overhead_A

        pre = prefill.astype(float)

        def tB(nrq, R, E):
            # decoding requests only, as the served counters count them: rows less the
            # prefill's 21, tokens less its one sampled token
            dec_rows = R - 21 * pre[:, None]; dec_tok = E - pre[:, None]
            return ab["intercept"] + ab["ms_per_token"] * dec_tok + ab["ms_per_wasted_row"] * (dec_rows - dec_tok)

        res["overhead_A_ms"] = float(overhead_A)
        out_pol = {}
        for vname in list(variants) + ["oracle-A", "oracle-B"]:
            if vname == "oracle-A":
                b, adm = oracle_budget(S, live, base, nsamp, tA, E0.mean() / (T0.mean() + overhead_A))
            elif vname == "oracle-B":
                b, adm = oracle_budget(S, live, base, nsamp, tB, 0.25)
            else:
                rc = (variants[vname][1] or {}).get("row_cost", ROW_COST)
                b = budgets[vname]; adm = admitted_from_budget(S, live, b, rc)
            Eb, rows, tg = evaluate(adm, b)
            cov, surv = counters(S, None, adm, live)
            tok = Eb.mean(); rw = rows.mean(); wasted = rw - tok
            thrA = tok / (tg.mean() + overhead_A)
            dec_tok = (Eb - prefill).mean(); dec_rows = (rows - 21 * prefill).mean()
            thrB = tok / (ab["intercept"] + ab["ms_per_token"] * dec_tok + ab["ms_per_wasted_row"] * (dec_rows - dec_tok))
            TA = tg.mean() + overhead_A
            TB = ab["intercept"] + ab["ms_per_token"] * dec_tok + ab["ms_per_wasted_row"] * (dec_rows - dec_tok)
            out_pol[vname] = {
                "step_ms_model_A": float(TA), "step_ms_model_B": float(TB),
                "rows": float(rw), "tokens": float(tok), "wasted": float(wasted), "yield": float(tok / rw),
                "coverage": cov, "survival": surv,
                "share_steps_rows_gt_64": float(np.mean(rows > 64)),
                "share_steps_budget_differs_from_1107": float(np.mean(b != budgets["1107"])),
                "gpu_step_ms": float(tg.mean()),
                "tok_per_ms_model_A_gpu_plus_const": float(thrA),
                "tok_per_ms_model_B_serving_regression": float(thrB),
            }
        # Model M: w A + (1 - w) B, w chosen so that row cost 0.020 against 0.035 predicts what
        # experiments/2026-10-18-dspark-linear-budget measured in serving (r6v, tp4): c16 screen
        # 240.3 -> 242.5/241.9 (+0.79%), sampled c8 180.7 -> 180.2/180.2 (-0.28%), single-stream c1
        # 71.1 -> 71.2/71.3 (+0.21%); each 'same' on its interval, so w is bracketed at +-0.5%.
        measured = {"c16": 0.79, "c8": -0.28, "c1": 0.21}[key]

        def mix_gain(pol, w):
            b0 = out_pol["1107"]
            thr = pol["tokens"] / (w * pol["step_ms_model_A"] + (1 - w) * pol["step_ms_model_B"])
            thr0 = b0["tokens"] / (w * b0["step_ms_model_A"] + (1 - w) * b0["step_ms_model_B"])
            return 100 * (thr / thr0 - 1)

        def w_for(m):
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = (lo + hi) / 2
                (lo, hi) = (mid, hi) if mix_gain(out_pol["1107-row-cost-0.020"], mid) < m else (lo, mid)
            return (lo + hi) / 2
        ws = {"central": w_for(measured), "minus_0.5": w_for(measured - 0.5), "plus_0.5": w_for(measured + 0.5)}
        res["model_M_weight_on_A"] = ws
        for wname, w in ws.items():
            def tM(nrq, R, E, w=w):
                return w * tA(nrq, R, E) + (1 - w) * tB(nrq, R, E)
            b, adm = oracle_budget(S, live, base, nsamp, tM, E0.mean() / t_served)
            Eb, rows, tg = evaluate(adm, b)
            tok = Eb.mean(); dec_tok = (Eb - prefill).mean(); dec_rows = (rows - 21 * prefill).mean()
            TA = tg.mean() + overhead_A
            TB = ab["intercept"] + ab["ms_per_token"] * dec_tok + ab["ms_per_wasted_row"] * (dec_rows - dec_tok)
            out_pol[f"oracle-M-{wname}"] = {"step_ms_model_A": float(TA), "step_ms_model_B": float(TB),
                                            "rows": float(rows.mean()), "tokens": float(tok), "wasted": float(rows.mean() - tok),
                                            "yield": float(tok / rows.mean()), "share_steps_rows_gt_64": float(np.mean(rows > 64)),
                                            "share_steps_budget_differs_from_1107": float(np.mean(b != budgets["1107"])),
                                            "gpu_step_ms": float(tg.mean()),
                                            "tok_per_ms_model_A_gpu_plus_const": float(tok / TA),
                                            "tok_per_ms_model_B_serving_regression": float(tok / TB), "weight_on_A": w}
        for vname, p in out_pol.items():
            base_p = out_pol["1107"]
            p["vs_1107"] = {
                "rows_pct": 100 * (p["rows"] / base_p["rows"] - 1),
                "tokens_pct": 100 * (p["tokens"] / base_p["tokens"] - 1),
                "wasted_pct": 100 * (p["wasted"] / base_p["wasted"] - 1) if base_p["wasted"] else None,
                "throughput_model_A_pct": 100 * (p["tok_per_ms_model_A_gpu_plus_const"] / base_p["tok_per_ms_model_A_gpu_plus_const"] - 1),
                "throughput_model_B_pct": 100 * (p["tok_per_ms_model_B_serving_regression"] / base_p["tok_per_ms_model_B_serving_regression"] - 1),
                "throughput_model_M_pct": {wn: mix_gain(p, w) for wn, w in ws.items()},
            }
        rows_hist = np.bincount(np.minimum((base + adm1107.sum((1, 2))).astype(int), 120), minlength=121)
        res["rows_histogram_1107"] = {str(i): int(c) for i, c in enumerate(rows_hist) if c}
        res["policies"] = out_pol
        report["concurrency"][name] = res
        print(f"== {name}: fit loss {loss:.2e}; vector rule == module on {res['vector_rule_matches_1107_module']*100:.2f}% of steps; overhead_A {overhead_A:.1f} ms")
        print(f"   fit cov {np.round(cov_fit,3).tolist()} vs {cfg['target']['coverage']}")
        print(f"   fit surv {np.round(surv_fit[:DRAFTS],3).tolist()} vs {np.round(cfg['target']['survival'],3).tolist()}; "
              f"tokens {surv_fit[DRAFTS]:.3f} vs {cfg['target']['tokens_per_step']:.3f}; request-steps drafting {surv_fit[DRAFTS+1]:.3f}")
        for vname, p in out_pol.items():
            d = p["vs_1107"]
            print(f"   {vname:20s} rows {p['rows']:6.2f} ({d['rows_pct']:+.2f}%) tok {p['tokens']:6.3f} ({d['tokens_pct']:+.2f}%) "
                  f">64 {p['share_steps_rows_gt_64']*100:5.2f}% differs {p['share_steps_budget_differs_from_1107']*100:5.2f}% "
                  f"A {d['throughput_model_A_pct']:+.2f}% B {d['throughput_model_B_pct']:+.2f}% "
                  f"M {d['throughput_model_M_pct']['minus_0.5']:+.2f}/{d['throughput_model_M_pct']['central']:+.2f}/{d['throughput_model_M_pct']['plus_0.5']:+.2f}%")
        print(f"   model M weight on A: {ws}")
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
