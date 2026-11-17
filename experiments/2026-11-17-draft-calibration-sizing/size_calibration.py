"""Offline sizing of the draft-acceptance calibration lever on dumped turns.

Question: how much decode throughput is left in *predicting which drafted rows the
target will accept*, at the drafts serving already makes?

Reads the turns the turn-learning observe stage dumped (hidden head rows, the drafts,
the committed labels) and the checkpoint's drafter tensors. Nothing here changes a
draft: every predictor only decides how many of a step's drafted rows to verify.

Stage ``features`` (one pass over the turns, cached):
  per drafted position: the confidence head's logit z (the feature serving calibrates),
  the draft distribution's top-1/top-2 margin and log p_max (full-vocabulary Markov
  logits, FP32), whether the draft is the committed token, and the head's input row.

Stage ``study`` (time-ordered split: first half of the turns fits, second half tests):
  predictors of P(draft i accepted | drafts before it accepted):
    serving   sigmoid(w * z + b_pos): the online estimator's form (shared slope,
              per-position intercepts), fitted on the first half;
    posslope  per-position slope and intercept on z;
    draft     margin / log p_max only (no confidence head);
    z+draft   z with the margin and log p_max, per position;
    mlp       a small MLP on the same scalar features;
    probe     the confidence head's linear form refitted on this traffic (its 5,376
              inputs), per-position intercepts;
    probe+draft  the refitted head stacked with the draft features;
    oracle    the label itself.
  policy: verify the leading rows whose predicted survival (running product) is at
  least tau; tokens = 1 + accepted among them; cost = 1 + rho * rows verified, rho the
  marginal row cost over the fixed step cost. tau is chosen on the first half for each
  predictor and rho, and scored on the second half. Throughput is tokens over cost.
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
from safetensors import safe_open

N = 5


def device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tensors(ckpt: str, device: str):
    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(ckpt, index["head.weight"]), "pt", device="cpu") as f:
        head = f.get_tensor("head.weight").float().to(device)
    out = {"head": head}
    names = {
        "norm": "mtp.2.norm.weight",
        "w1": "mtp.2.markov_head.embed.weight",
        "w2": "mtp.2.markov_head.head.weight",
        "conf": "mtp.2.confidence_head.proj.weight",
    }
    # ``drafter-heads.safetensors``: these four tensors copied out of their shard (so a
    # host without the whole snapshot can run this); otherwise read the shard itself.
    extract = os.path.join(ckpt, "drafter-heads.safetensors")
    for key, name in names.items():
        src = extract if os.path.exists(extract) else os.path.join(ckpt, index[name])
        with safe_open(src, "pt", device="cpu") as f:
            out[key] = f.get_tensor(name).float().to(device)
    config = json.load(open(os.path.join(ckpt, "config.json")))
    eps = float(config.get("text_config", config).get("rms_norm_eps", 1e-6))
    return out, eps


def rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def features(args) -> None:
    dev = device_name()
    t, eps = load_tensors(args.ckpt, dev)
    paths = sorted(glob.glob(os.path.join(args.turns, args.pattern)))
    if args.limit:
        paths = paths[: args.limit]
    print(json.dumps({"device": dev, "turn_files": len(paths), "eps": eps,
                      "conf_shape": list(t["conf"].shape)}), flush=True)
    cols = {k: [] for k in ("turn", "block", "pos", "z", "zn", "margin", "lpmax", "y",
                            "valid", "cond", "active", "replay")}
    rows_x = []
    block0 = 0
    started = time.time()
    for ti, path in enumerate(paths):
        rec = torch.load(path, map_location="cpu", weights_only=True)
        hidden = rec["hidden"].float()
        ids = rec["ids"].long()
        labels = rec["labels"].long()
        meta = rec["meta"].long()
        kept = hidden.shape[0]
        if ti == 0:
            print(json.dumps({"first": os.path.basename(path), "hidden": list(hidden.shape),
                              "ids": list(ids.shape), "labels": list(labels.shape),
                              "meta": list(meta.shape), "meta_head": meta[:3].tolist()}), flush=True)
        if kept == 0:
            continue
        active = meta[:, 1].clamp(0, N)
        pos = torch.arange(N).expand(kept, N)
        # ``ids`` is the anchor token then the committed tokens; ``drafts`` is what the
        # step drafted. Serving's chain reads the anchor, then its own drafts.
        drafts = rec["drafts"].long()
        prev = torch.cat([ids[:, :1], drafts[:, : N - 1]], dim=1).clamp(min=0)
        y = (drafts == labels) & (labels >= 0)
        valid = ((labels >= 0) & (pos < active[:, None])).int().cumprod(dim=1).bool()
        before = torch.cat([torch.ones(kept, 1, dtype=torch.bool), (y & valid)[:, :-1]], dim=1)
        cond = valid & before.int().cumprod(dim=1).bool()

        h = hidden.to(dev)
        emb = t["w1"][prev.to(dev)]
        z = (torch.cat([h, emb], dim=-1) @ t["conf"].t()).squeeze(-1)
        normed = rms_norm(h, t["norm"], eps)
        zn = (torch.cat([normed, emb], dim=-1) @ t["conf"].t()).squeeze(-1)
        flat_n = normed.reshape(-1, normed.shape[-1])
        flat_e = emb.reshape(-1, emb.shape[-1])
        margin = torch.empty(kept * N, device=dev)
        lpmax = torch.empty(kept * N, device=dev)
        top = torch.empty(kept * N, dtype=torch.long, device=dev)
        for s in range(0, kept * N, args.chunk):
            e = min(s + args.chunk, kept * N)
            logits = flat_n[s:e] @ t["head"].t() + flat_e[s:e] @ t["w2"].t()
            v, i = logits.topk(2, dim=-1)
            margin[s:e] = v[:, 0] - v[:, 1]
            lpmax[s:e] = v[:, 0] - torch.logsumexp(logits, dim=-1)
            top[s:e] = i[:, 0]
        replay = top.view(kept, N).cpu() == drafts

        cols["turn"].append(np.full(kept * N, ti, dtype=np.int32))
        cols["block"].append((block0 + torch.arange(kept)[:, None].expand(kept, N)).reshape(-1).numpy().astype(np.int32))
        cols["pos"].append(pos.reshape(-1).numpy().astype(np.int8))
        cols["z"].append(z.reshape(-1).cpu().numpy())
        cols["zn"].append(zn.reshape(-1).cpu().numpy())
        cols["margin"].append(margin.cpu().numpy())
        cols["lpmax"].append(lpmax.cpu().numpy())
        cols["y"].append(y.reshape(-1).numpy())
        cols["valid"].append(valid.reshape(-1).numpy())
        cols["cond"].append(cond.reshape(-1).numpy())
        cols["active"].append(active[:, None].expand(kept, N).reshape(-1).numpy().astype(np.int8))
        cols["replay"].append(replay.reshape(-1).numpy())
        rows_x.append(torch.cat([hidden, emb.cpu()], dim=-1).reshape(kept * N, -1).half().numpy())
        block0 += kept
        if (ti + 1) % 50 == 0 or ti + 1 == len(paths):
            print(json.dumps({"turns": ti + 1, "blocks": block0,
                              "seconds": round(time.time() - started, 1)}), flush=True)
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "features.npz"), **{k: np.concatenate(v) for k, v in cols.items()})
    np.save(os.path.join(args.out, "x.npy"), np.concatenate(rows_x))
    print(json.dumps({"written": args.out, "blocks": block0}), flush=True)


# ---------------------------------------------------------------- study


def auc(score: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    ranks = (end - (counts - 1) / 2.0)[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-4):
    """Standardized logistic regression by L-BFGS (float64); returns a predictor."""
    mu, sd = x.mean(0), x.std(0) + 1e-9
    xs = torch.from_numpy((x - mu) / sd).double()
    yt = torch.from_numpy(y.astype(np.float64))
    w = torch.zeros(xs.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=300, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xs @ w + b, yt) + l2 * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    wn, bn = w.detach().numpy(), float(b.detach())

    def predict_logit(xq: np.ndarray) -> np.ndarray:
        return ((xq - mu) / sd) @ wn + bn

    return predict_logit


def onehot(pos: np.ndarray) -> np.ndarray:
    return (pos[:, None] == np.arange(N)[None, :]).astype(np.float64)


def by_pos(cols: list[np.ndarray], pos: np.ndarray) -> np.ndarray:
    """Each column interacted with the position one-hot, plus the one-hot itself."""
    oh = onehot(pos)
    parts = [oh[:, 1:]]
    for c in cols:
        parts.append(oh * c[:, None])
    return np.concatenate(parts, axis=1)


def scalar_sets(f: dict) -> dict:
    z, m, lp, pos = f["z"].astype(np.float64), f["margin"].astype(np.float64), f["lpmax"].astype(np.float64), f["pos"]
    lm = np.log1p(m)
    return {
        "serving": np.concatenate([z[:, None], onehot(pos)[:, 1:]], axis=1),
        "posslope": by_pos([z], pos),
        "draft": by_pos([m, lm, lp, lp * lp], pos),
        "z+draft": by_pos([z, m, lm, lp, lp * lp, z * lp], pos),
    }


def fit_mlp(x, y, xq, dev, epochs=40, seed=0):
    torch.manual_seed(seed)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    xt = torch.from_numpy(((x - mu) / sd).astype(np.float32)).to(dev)
    yt = torch.from_numpy(y.astype(np.float32)).to(dev)
    n = xt.shape[0]
    perm = torch.randperm(n, device=dev)
    hold, train = perm[: n // 10], perm[n // 10 :]
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 64), torch.nn.SiLU(),
                              torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state = math.inf, None
    for _ in range(epochs):
        net.train()
        order = train[torch.randperm(train.numel(), device=dev)]
        for s in range(0, order.numel(), 4096):
            idx = order[s : s + 4096]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[idx]).squeeze(-1), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(torch.nn.functional.binary_cross_entropy_with_logits(net(xt[hold]).squeeze(-1), yt[hold]))
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        q = torch.from_numpy(((xq - mu) / sd).astype(np.float32)).to(dev)
        return torch.cat([net(q[s : s + 65536]).squeeze(-1) for s in range(0, q.shape[0], 65536)]).cpu().numpy().astype(np.float64)


def fit_probe(x_mm, z, rows_fit, y_fit, pos_fit, rows_all, pos_all, dev, epochs=12, seed=0):
    """The confidence head's linear form over its own inputs, refitted on this traffic as
    a residual over its logit z (starts at the head itself; AdamW, held-out early stop),
    with per-position intercepts. Returns logits for ``rows_all``."""
    torch.manual_seed(seed)
    sample = np.sort(np.random.default_rng(seed).choice(rows_fit, size=min(60000, rows_fit.size), replace=False))
    xs = torch.from_numpy(x_mm[sample].astype(np.float32))
    mu, sd = xs.mean(0).to(dev), (xs.std(0) + 1e-6).to(dev)
    del xs
    xt = torch.from_numpy(x_mm[rows_fit]).to(dev)  # float16 on device
    yt = torch.from_numpy(y_fit.astype(np.float32)).to(dev)
    pt = torch.from_numpy(pos_fit.astype(np.int64)).to(dev)
    zt = torch.from_numpy(z[rows_fit].astype(np.float32)).to(dev)
    n = xt.shape[0]
    perm = torch.randperm(n, device=dev)
    hold, train = perm[: n // 10], perm[n // 10 :]
    w = torch.zeros(xt.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(N, device=dev, requires_grad=True)
    sl = torch.ones(1, device=dev, requires_grad=True)
    opt = torch.optim.AdamW([{"params": [w], "weight_decay": args_probe_wd, "lr": 1e-4},
                             {"params": [b, sl], "weight_decay": 0.0, "lr": 1e-2}])

    def logits(idx_x, idx_p, idx_z):
        return sl * idx_z + ((idx_x.float() - mu) / sd) @ w + b[idx_p]

    best, best_w, best_b = math.inf, None, None
    for _ in range(epochs):
        order = train[torch.randperm(train.numel(), device=dev)]
        for s in range(0, order.numel(), 4096):
            idx = order[s : s + 4096]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits(xt[idx], pt[idx], zt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val = sum(float(torch.nn.functional.binary_cross_entropy_with_logits(
                logits(xt[hold[s : s + 16384]], pt[hold[s : s + 16384]], zt[hold[s : s + 16384]]), yt[hold[s : s + 16384]], reduction="sum"))
                for s in range(0, hold.numel(), 16384)) / max(hold.numel(), 1)
        if val < best:
            best, best_w, best_b, best_s = val, w.detach().clone(), b.detach().clone(), sl.detach().clone()
    del xt
    out = np.empty(rows_all.size, dtype=np.float64)
    pa = torch.from_numpy(pos_all.astype(np.int64)).to(dev)
    za = torch.from_numpy(z[rows_all].astype(np.float32)).to(dev)
    with torch.no_grad():
        for s in range(0, rows_all.size, 32768):
            xq = torch.from_numpy(x_mm[rows_all[s : s + 32768]]).to(dev)
            out[s : s + 32768] = (best_s * za[s : s + 32768] + ((xq.float() - mu) / sd) @ best_w + best_b[pa[s : s + 32768]]).cpu().numpy()
    return out, best


args_probe_wd = 1e-2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def grid(p_blocks: np.ndarray, cap: np.ndarray, run: np.ndarray, turn: np.ndarray, n_turns: int, taus: np.ndarray):
    """Per-turn tokens and verified rows for each tau. p_blocks [B, N] (NaN past cap)."""
    surv = np.cumprod(np.where(np.isnan(p_blocks), 0.0, p_blocks), axis=1)
    inside = np.arange(N)[None, :] < cap[:, None]
    tok = np.zeros((taus.size, n_turns))
    rows = np.zeros((taus.size, n_turns))
    for k, tau in enumerate(taus):
        d = ((surv >= tau) & inside).sum(axis=1)
        a = np.minimum(run, d)
        tok[k] = np.bincount(turn, weights=1.0 + a, minlength=n_turns)
        rows[k] = np.bincount(turn, weights=d, minlength=n_turns)
    return tok, rows


def study(args) -> None:
    dev = device_name()
    f = dict(np.load(os.path.join(args.out, "features.npz")))
    x_mm = np.load(os.path.join(args.out, "x.npy"), mmap_mode="r")
    n_turns = int(f["turn"].max()) + 1
    split = n_turns // 2
    valid, cond, y, pos, turn = f["valid"], f["cond"], f["y"], f["pos"], f["turn"]
    first = turn < split
    report = {"turns": n_turns, "split_turn": split, "blocks": int(f["block"].max()) + 1,
              "positions_valid": int(valid.sum()), "positions_conditional": int(cond.sum()),
              "replay_agreement": float(f["replay"][valid].mean()),
              "active_hist": np.bincount(f["active"][pos == 0], minlength=N + 1).tolist()}

    # sanity: which input the confidence head reads (pre-norm rows or normed rows)
    report["sanity_auc_conditional"] = {"z_prenorm": auc(f["z"][cond], y[cond]), "z_normed": auc(f["zn"][cond], y[cond])}
    if report["sanity_auc_conditional"]["z_normed"] > report["sanity_auc_conditional"]["z_prenorm"] + 0.02:
        f["z"] = f["zn"]
        report["z_used"] = "normed"
    else:
        report["z_used"] = "prenorm"
    report["raw_head"] = {
        "mean_prob_by_pos": [float(sigmoid(f["z"][cond & (pos == i)]).mean()) for i in range(N)],
        "accept_by_pos": [float(y[cond & (pos == i)].mean()) for i in range(N)],
        "n_by_pos": [int((cond & (pos == i)).sum()) for i in range(N)],
    }

    fit_rows = np.flatnonzero(cond & first)
    all_rows = np.flatnonzero(valid)
    sets = scalar_sets(f)
    logits = {}
    for name, xs in sets.items():
        predict = fit_logistic(xs[fit_rows], y[fit_rows])
        logits[name] = predict(xs[all_rows])
    scal = np.stack([f["z"], f["margin"], np.log1p(f["margin"]), f["lpmax"]], axis=1).astype(np.float64)
    scal = np.concatenate([scal, onehot(pos)], axis=1)
    logits["mlp"] = fit_mlp(scal[fit_rows], y[fit_rows], scal[all_rows], dev)

    if not args.no_probe:
        # cross-fitted stack: the probe is fitted on the first quarter and stacked with the
        # draft features on the second quarter; the plain probe is fitted on the whole half.
        probe_all, val = fit_probe(x_mm, f["z"], fit_rows, y[fit_rows], pos[fit_rows], all_rows, pos[all_rows], dev)
        logits["probe"] = probe_all
        report["probe_holdout_logloss"] = val
        quarter = turn < split // 2
        rows_a = np.flatnonzero(cond & quarter)
        probe_a, _ = fit_probe(x_mm, f["z"], rows_a, y[rows_a], pos[rows_a], all_rows, pos[all_rows], dev, seed=1)
        full = np.full(valid.shape, np.nan)
        full[all_rows] = probe_a
        m, lp = f["margin"].astype(np.float64), f["lpmax"].astype(np.float64)
        stack = by_pos([full, m, np.log1p(m), lp, lp * lp], pos)
        rows_b = np.flatnonzero(cond & first & ~quarter)
        predict = fit_logistic(stack[rows_b], y[rows_b])
        logits["probe+draft"] = predict(stack[all_rows])

    # discrimination on the held-out half, conditional population
    test_c = cond[all_rows] & ~first[all_rows]
    yc = y[all_rows][test_c]
    pc = pos[all_rows][test_c]
    disc = {}
    for name, lg in logits.items():
        p = sigmoid(lg[test_c])
        disc[name] = {"auc": auc(lg[test_c], yc), "logloss": logloss(p, yc),
                      "auc_by_pos": [auc(lg[test_c][pc == i], yc[pc == i]) for i in range(N)]}
    base_rate = np.array([yc[pc == i].mean() for i in range(N)])
    disc["position_only"] = {"auc": auc(base_rate[pc], yc), "logloss": logloss(base_rate[pc], yc)}
    report["discrimination_test"] = disc

    # policy value
    blocks = f["block"]
    n_blocks = int(blocks.max()) + 1
    cap = np.bincount(blocks[valid], minlength=n_blocks)
    run = np.bincount(blocks[valid & cond & y], minlength=n_blocks)  # leading accepted drafts
    bturn = np.zeros(n_blocks, dtype=np.int64)
    bturn[blocks] = turn
    keep = cap > 0
    taus = np.concatenate([[0.0], np.geomspace(0.01, 0.95, 80)])
    tr = np.arange(n_turns) < split
    te = ~tr
    per = {}
    for name, lg in list(logits.items()) + [("oracle", None)]:
        pb = np.full((n_blocks, N), np.nan)
        if lg is None:
            pb[blocks[all_rows], pos[all_rows]] = np.where(y[all_rows], 1.0, 0.0)
        else:
            pb[blocks[all_rows], pos[all_rows]] = sigmoid(lg)
        per[name] = grid(pb[keep], cap[keep], run[keep], bturn[keep], n_turns, taus)
    sel = keep & te[bturn]
    report["run_hist_test"] = (np.bincount(run[sel], minlength=N + 1) / sel.sum()).tolist()
    report["all_drafts_accepted_share_test"] = float((run[sel] == cap[sel]).mean())
    report["cap5_share_test"] = float((cap[sel] == N).mean())
    n_test_blocks = int((keep & te[bturn]).sum())
    report["test_blocks"] = n_test_blocks
    report["test"] = {
        "accepted_per_block_at_full_depth": float(run[keep & te[bturn]].mean()),
        "drafted_rows_per_block": float(cap[keep & te[bturn]].mean()),
    }
    rng = np.random.default_rng(0)
    test_turns = np.flatnonzero(te)
    boots = rng.integers(0, test_turns.size, size=(args.boot, test_turns.size))
    policy = {}
    for rho in [float(r) for r in args.rhos.split(",")]:
        steps_turn = np.bincount(bturn[keep], minlength=n_turns).astype(np.float64)
        out = {}
        ref = None
        for name, (tok, rows) in per.items():
            t_train = tok[:, tr].sum(1) / (steps_turn[tr].sum() + rho * rows[:, tr].sum(1))
            k = int(np.argmax(t_train))
            tk, rw = tok[k, te], rows[k, te]
            st = steps_turn[te]
            t_test = tk.sum() / (st.sum() + rho * rw.sum())
            out[name] = {"tau": float(taus[k]), "tokens_per_step": float(tk.sum() / st.sum()),
                         "rows_per_step": float(rw.sum() / st.sum()),
                         "wasted_rows_per_step": float((rw.sum() - (tk.sum() - st.sum())) / st.sum()),
                         "throughput": float(t_test)}
            if name == "serving":
                ref = (tk, rw, st)
        # baselines that need no predictor
        k_all = 0  # tau = 0: every drafted row
        tok_all, rows_all_ = per["serving"][0][k_all, te], per["serving"][1][k_all, te]
        out["verify_all_drafted"] = {"throughput": float(tok_all.sum() / (steps_turn[te].sum() + rho * rows_all_.sum())),
                                     "rows_per_step": float(rows_all_.sum() / steps_turn[te].sum()),
                                     "tokens_per_step": float(tok_all.sum() / steps_turn[te].sum())}
        out["no_drafts"] = {"throughput": 1.0}
        base = out["serving"]["throughput"]
        rtk, rrw, rst = ref
        for name, (tok, rows) in per.items():
            k = int(np.where(taus == out[name]["tau"])[0][0])
            tk, rw = tok[k, te], rows[k, te]
            num = tk[boots].sum(1) / (rst[boots].sum(1) + rho * rw[boots].sum(1))
            den = rtk[boots].sum(1) / (rst[boots].sum(1) + rho * rrw[boots].sum(1))
            g = num / den - 1.0
            out[name]["gain_vs_serving_pct"] = float(100 * (out[name]["throughput"] / base - 1))
            out[name]["gain_ci95_pct"] = [float(100 * np.percentile(g, 2.5)), float(100 * np.percentile(g, 97.5))]
        policy[f"rho={rho}"] = out
    report["policy_test"] = policy
    path = os.path.join(args.out, "study.json")
    json.dump(report, open(path, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["features", "study"])
    ap.add_argument("--turns")
    ap.add_argument("--pattern", default="20261117T020420-*.pt")
    ap.add_argument("--ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--rhos", default="0.09,0.11,0.15,0.23,0.5,0.75")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    features(a) if a.stage == "features" else study(a)
