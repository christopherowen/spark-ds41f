"""Follow-ups on the cached features: (1) the top-2 margin alone, the draft feature the
step already holds; (2) what follows a step whose five drafts were all accepted."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import size_calibration as sc

out = sys.argv[1]
f = dict(np.load(os.path.join(out, "features.npz")))
N = sc.N
valid, cond, y, pos, turn, blocks = f["valid"], f["cond"], f["y"], f["pos"], f["turn"], f["block"]
n_turns = int(turn.max()) + 1
split = n_turns // 2
first = turn < split
fit_rows, all_rows = np.flatnonzero(cond & first), np.flatnonzero(valid)
z, m, lp = f["z"].astype(np.float64), f["margin"].astype(np.float64), f["lpmax"].astype(np.float64)
lm = np.log1p(m)
sets = {
    "serving": np.concatenate([z[:, None], sc.onehot(pos)[:, 1:]], axis=1),
    "margin": sc.by_pos([m, lm], pos),
    "z+margin": sc.by_pos([z, m, lm, z * lm], pos),
    "z+lpmax": sc.by_pos([z, lp, lp * lp, z * lp], pos),
    "z+draft": sc.by_pos([z, m, lm, lp, lp * lp, z * lp], pos),
}
logits = {k: sc.fit_logistic(x[fit_rows], y[fit_rows])(x[all_rows]) for k, x in sets.items()}
test_c = cond[all_rows] & ~first[all_rows]
yc = y[all_rows][test_c]
res = {"discrimination": {k: {"auc": sc.auc(v[test_c], yc), "logloss": sc.logloss(sc.sigmoid(v[test_c]), yc)} for k, v in logits.items()}}

n_blocks = int(blocks.max()) + 1
cap = np.bincount(blocks[valid], minlength=n_blocks)
run = np.bincount(blocks[valid & cond & y], minlength=n_blocks)
bturn = np.zeros(n_blocks, dtype=np.int64); bturn[blocks] = turn
keep = cap > 0
taus = np.concatenate([[0.0], np.geomspace(0.01, 0.95, 80)])
tr = np.arange(n_turns) < split; te = ~tr
steps = np.bincount(bturn[keep], minlength=n_turns).astype(np.float64)
rng = np.random.default_rng(0)
tt = np.flatnonzero(te); boots = rng.integers(0, tt.size, size=(1000, tt.size))
per = {}
for k, lg in logits.items():
    pb = np.full((n_blocks, N), np.nan); pb[blocks[all_rows], pos[all_rows]] = sc.sigmoid(lg)
    per[k] = sc.grid(pb[keep], cap[keep], run[keep], bturn[keep], n_turns, taus)
pol = {}
for rho in (0.09, 0.11, 0.15, 0.23, 0.5, 0.75):
    row = {}; ref = None
    for k, (tok, rows) in per.items():
        i = int(np.argmax(tok[:, tr].sum(1) / (steps[tr].sum() + rho * rows[:, tr].sum(1))))
        tk, rw, st = tok[i, te], rows[i, te], steps[te]
        if k == "serving": ref = (tk, rw, st)
        row[k] = (tk, rw, st)
    rtk, rrw, rst = ref
    base = rtk.sum() / (rst.sum() + rho * rrw.sum())
    pol[str(rho)] = {}
    for k, (tk, rw, st) in row.items():
        g = (tk[boots].sum(1) / (st[boots].sum(1) + rho * rw[boots].sum(1))) / (rtk[boots].sum(1) / (rst[boots].sum(1) + rho * rrw[boots].sum(1))) - 1
        pol[str(rho)][k] = [round(100 * (tk.sum() / (st.sum() + rho * rw.sum()) / base - 1), 2), round(100 * np.percentile(g, 2.5), 2), round(100 * np.percentile(g, 97.5), 2)]
res["policy_gain_pct"] = pol

# what follows a fully accepted step (next kept step of the same turn)
b = np.flatnonzero(keep)
nxt_same = (bturn[b[1:]] == bturn[b[:-1]])
full_prev = (run[b[:-1]] == N) & (cap[b[:-1]] == N) & nxt_same
any_prev = nxt_same
res["next_step"] = {
    "steps_all_five_accepted_share": float(((run[b] == N) & (cap[b] == N)).mean()),
    "next_run_mean_after_full": float(run[b[1:]][full_prev].mean()),
    "next_run_mean_overall": float(run[b[1:]][any_prev].mean()),
    "next_run_hist_after_full": (np.bincount(run[b[1:]][full_prev], minlength=N + 1) / full_prev.sum()).round(3).tolist(),
    "next_first_draft_accept_after_full": float((run[b[1:]][full_prev] >= 1).mean()),
    "next_first_draft_accept_overall": float((run[b[1:]][any_prev] >= 1).mean()),
    "pairs": int(full_prev.sum()),
}
json.dump(res, open(os.path.join(out, "followup.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
