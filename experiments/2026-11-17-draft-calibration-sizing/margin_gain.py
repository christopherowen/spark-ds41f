"""The one margin term serving adds to the confidence head's logit, from the cached
features (``size_calibration.py features``).

``python margin_gain.py <out> [<other out>]`` prints, for the turns in ``<out>``:
  forms     the serving form (shared slope, an intercept a position) on ``z`` alone and
            on ``z`` with one margin term (log1p, linear, capped, square root), on
            ``z + g * log1p(margin)`` at fixed gains ``g``, and with the four margin terms
            a position of ``followup.py``: discrimination and the policy's throughput
            gain over the serving form (first half fits, second half scores);
  warm      the raw slope and intercepts of ``sigmoid(w * (z + g * log1p(margin)) +
            b_pos)`` at g = 0 and 2, which the estimator's start for the margin feature
            is derived from;
and with ``<other out>`` (another boot's turns):
  transfer  the form fitted on ``<out>`` applied unchanged to ``<other out>``, with and
            without the margin at g = 2.
"""
import json
import os
import sys

import numpy as np
import torch

import size_calibration as sc

N = sc.N
RHOS = (0.09, 0.11, 0.15, 0.23, 0.5, 0.75)
TAUS = np.concatenate([[0.0], np.geomspace(0.01, 0.95, 80)])


def load(out):
    f = dict(np.load(os.path.join(out, "features.npz")))
    f["z"] = f["z"].astype(np.float64)
    f["m"] = np.maximum(f["margin"].astype(np.float64), 0)
    f["lm"] = np.log1p(f["m"])
    return f


def one(f, extra):
    return np.concatenate([f["z"][:, None]] + [e[:, None] for e in extra] + [sc.onehot(f["pos"])[:, 1:]], axis=1)


def blocks_of(f):
    blocks = f["block"]
    nb = int(blocks.max()) + 1
    cap = np.bincount(blocks[f["valid"]], minlength=nb)
    run = np.bincount(blocks[f["valid"] & f["cond"] & f["y"]], minlength=nb)
    bturn = np.zeros(nb, dtype=np.int64)
    bturn[blocks] = f["turn"]
    return nb, cap, run, bturn


def forms(f):
    valid, cond, y, pos, turn = f["valid"], f["cond"], f["y"], f["pos"], f["turn"]
    n_turns = int(turn.max()) + 1
    split = n_turns // 2
    first = turn < split
    fit_rows, all_rows = np.flatnonzero(cond & first), np.flatnonzero(valid)
    z, m, lm = f["z"], f["m"], f["lm"]
    sets = {"serving": one(f, []), "z + c log1p(m)": one(f, [lm]), "z + c m": one(f, [m]),
            "z + c min(m, 4)": one(f, [np.minimum(m, 4)]), "z + c sqrt(m)": one(f, [np.sqrt(m)]),
            "z, margin terms a position": sc.by_pos([z, m, lm, z * lm], pos)}
    for g in (0.5, 1.0, 1.5, 2.0, 2.5):
        sets[f"g = {g}"] = np.concatenate([(z + g * lm)[:, None], sc.onehot(pos)[:, 1:]], axis=1)
    logits, coef = {}, {}
    for k, x in sets.items():
        logits[k] = sc.fit_logistic(x[fit_rows], y[fit_rows])(x[all_rows])
        if x.shape[1] == 6:  # the fitted logit is linear in the raw columns
            w = np.linalg.lstsq(np.concatenate([x[all_rows], np.ones((all_rows.size, 1))], axis=1), logits[k], rcond=None)[0]
            coef[k] = {"slope": round(float(w[0]), 3), "term": round(float(w[1]), 3), "gain": round(float(w[1] / w[0]), 3)}
    test_c = cond[all_rows] & ~first[all_rows]
    yc = y[all_rows][test_c]
    res = {"fitted_gain": coef,
           "auc": {k: round(sc.auc(v[test_c], yc), 4) for k, v in logits.items()},
           "logloss": {k: round(sc.logloss(sc.sigmoid(v[test_c]), yc), 4) for k, v in logits.items()}}
    nb, cap, run, bturn = blocks_of(f)
    keep = cap > 0
    tr = np.arange(n_turns) < split
    te = ~tr
    steps = np.bincount(bturn[keep], minlength=n_turns).astype(np.float64)
    per = {}
    for k, lg in logits.items():
        pb = np.full((nb, N), np.nan)
        pb[f["block"][all_rows], pos[all_rows]] = sc.sigmoid(lg)
        per[k] = sc.grid(pb[keep], cap[keep], run[keep], bturn[keep], n_turns, TAUS)
    gains = {}
    for rho in RHOS:
        thr = {}
        for k, (tok, rows) in per.items():
            i = int(np.argmax(tok[:, tr].sum(1) / (steps[tr].sum() + rho * rows[:, tr].sum(1))))
            thr[k] = tok[i, te].sum() / (steps[te].sum() + rho * rows[i, te].sum())
        gains[str(rho)] = {k: round(100 * (v / thr["serving"] - 1), 2) for k, v in thr.items() if k != "serving"}
    res["gain_pct"] = gains
    return res


def warm(f):
    cond, y, pos = f["cond"], f["y"], f["pos"]
    rows = np.flatnonzero(cond)
    res = {"positions": int(rows.size),
           "mean_log1p_margin_by_pos": [round(float(f["lm"][cond & (pos == i)].mean()), 3) for i in range(N)]}
    for g in (0.0, 2.0):
        x = torch.from_numpy(f["z"][rows] + g * f["lm"][rows])
        p = torch.from_numpy(pos[rows].astype(np.int64))
        yt = torch.from_numpy(y[rows].astype(np.float64))
        w = torch.ones(1, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(N, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=500, tolerance_grad=1e-10, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(w * x + b[p], yt)
            loss.backward()
            return loss

        opt.step(closure)
        res[f"g = {g}"] = {"slope": round(float(w.detach()), 3), "intercepts": [round(float(v), 3) for v in b.detach()]}
    return res


def transfer(a, b):
    nb, cap, run, _ = blocks_of(b)
    keep = cap > 0
    rows_b = np.flatnonzero(b["valid"])
    res = {"steps": int(keep.sum())}
    thr = {}
    for g in (0.0, 2.0):
        def design(f):
            return np.concatenate([(f["z"] + g * f["lm"])[:, None], sc.onehot(f["pos"])[:, 1:]], axis=1)
        lg = sc.fit_logistic(design(a)[a["cond"]], a["y"][a["cond"]])(design(b)[rows_b])
        pb = np.full((nb, N), np.nan)
        pb[b["block"][rows_b], b["pos"][rows_b]] = sc.sigmoid(lg)
        tok, rws = sc.grid(pb[keep], cap[keep], run[keep], np.zeros(int(keep.sum()), dtype=np.int64), 1, TAUS)
        thr[g] = (tok[:, 0], rws[:, 0])
        c = b["cond"][rows_b]
        res[f"g = {g}"] = {"auc": round(sc.auc(lg[c], b["y"][rows_b][c]), 4),
                           "logloss": round(sc.logloss(sc.sigmoid(lg[c]), b["y"][rows_b][c]), 4)}
    n = keep.sum()
    # the threshold here is the best for each on these turns: an upper reading for both
    res["gain_pct_at_each_best_threshold"] = {
        str(rho): round(100 * ((thr[2.0][0] / (n + rho * thr[2.0][1])).max() / (thr[0.0][0] / (n + rho * thr[0.0][1])).max() - 1), 2)
        for rho in RHOS}
    return res


if __name__ == "__main__":
    a = load(sys.argv[1])
    out = {"forms": forms(a), "warm": warm(a)}
    if len(sys.argv) > 2:
        out["transfer"] = transfer(a, load(sys.argv[2]))
    print(json.dumps(out, indent=1))
