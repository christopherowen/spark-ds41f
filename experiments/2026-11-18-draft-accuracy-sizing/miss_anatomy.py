#!/usr/bin/env python3
"""Where the DSpark drafter's misses are, on dumped turns that carry the request's tokens.

usage: miss_anatomy.py features --turns DIR --ckpt DIR --out DIR
       miss_anatomy.py study --out DIR

A step's five hidden rows do not depend on what it drafts: draft i is the argmax of
``norm(h_i) @ head + embed(previous token) @ markov``, the previous token being the
anchor or the step's own draft. While every earlier draft is right the previous token is
the committed one, so a step's accepted run is the leading run of right answers under the
committed tokens ("teacher forcing"), and any other scoring of the same rows can be
replayed exactly: tokens a step = 1 + that run.

``features`` writes, for every drafted position of every DSpark step (teacher forced):
the top-1 token and its logit, the committed token's rank and logit, the logit gap to the
runner-up, the same with the Markov term left out, and what the request's own earlier
tokens say: for the longest k of 1..12 at which the k tokens before this position occur
earlier in the row, the token that followed the most recent occurrence, that k, and the
logit the drafter gives it.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2026-11-17-draft-calibration-sizing"))
import size_calibration as sc  # noqa: E402

N = 5
KS = (1, 2, 3, 4, 6, 8, 12)
MULT = np.uint64(1000003)


def ngram_predictions(tok: np.ndarray):
    """For every index q of the row: (k, token) of the longest k in KS whose k tokens
    before q occurred earlier, the token being the one after the most recent occurrence;
    k = 0 where none did."""
    n = len(tok)
    best_k = np.zeros(n, dtype=np.int16)
    best_t = np.full(n, -1, dtype=np.int64)
    t64 = tok.astype(np.uint64) + np.uint64(1)
    for k in KS:
        if n <= k:
            continue
        # h[q] hashes tok[q-k:q], defined for q >= k
        h = np.zeros(n + 1, dtype=np.uint64)
        acc = np.zeros(n - k + 1, dtype=np.uint64)
        for j in range(k):
            acc = acc * MULT + t64[j : n - k + 1 + j]
        h = acc  # acc[s] hashes tok[s:s+k]; a query at q uses s = q-k, its continuation is tok[s+k]
        starts = np.arange(n - k + 1)
        order = np.lexsort((starts, h))
        hs, ss = h[order], starts[order]
        same = np.zeros(len(order), dtype=bool)
        same[1:] = hs[1:] == hs[:-1]
        prev_start = np.where(same, np.concatenate(([0], ss[:-1])), -1)
        q = ss + k                       # the index this k-gram predicts
        ok = (prev_start >= 0) & (q < n)
        # verify the match (hash collisions) on the first and last token
        ps = prev_start[ok]
        qq = q[ok]
        good = (tok[ps] == tok[qq - k]) & (tok[ps + k - 1] == tok[qq - 1])
        qq, ps = qq[good], ps[good]
        best_k[qq] = k
        best_t[qq] = tok[ps + k]
    return best_k, best_t


def features(args) -> None:
    dev = sc.device_name()
    t, eps = sc.load_tensors(args.ckpt, dev)
    paths = sorted(glob.glob(os.path.join(args.turns, "*.pt")))
    cols = {k: [] for k in ("turn", "step", "pos", "anchor", "active", "label", "top", "top_logit", "gap", "label_rank",
                            "label_logit", "top_h", "ng_k", "ng_tok", "ng_logit", "seen", "drafted")}
    skipped = 0
    started = time.time()
    for ti, path in enumerate(paths):
        rec = torch.load(path, map_location="cpu", weights_only=True)
        if "tokens" not in rec or rec["hidden"].shape[0] == 0:
            skipped += 1
            continue
        tok = rec["tokens"].long().numpy()
        meta = rec["meta"].long().numpy()
        ids, labels, drafts = rec["ids"].long(), rec["labels"].long(), rec["drafts"].long()
        anchor = meta[:, 0]
        keep = (anchor >= 0) & (anchor + N < len(tok)) & (meta[:, 1] > 0)
        keep &= tok[np.clip(anchor, 0, len(tok) - 1)] == meta[:, 2]
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            skipped += 1
            continue
        lab = np.stack([tok[anchor[idx] + 1 + i] for i in range(N)], axis=1)
        # the dump's labels must be the row's tokens wherever the dump has them
        dl = labels[idx].numpy()
        agree = ((dl == lab) | (dl < 0)).all(axis=1)
        idx, lab = idx[agree], lab[agree]
        kept = len(idx)
        if kept == 0:
            skipped += 1
            continue
        best_k, best_t = ngram_predictions(tok)
        q = anchor[idx][:, None] + 1 + np.arange(N)[None, :]
        ng_k, ng_tok = best_k[q], best_t[q]
        first_seen = np.full(int(tok.max()) + 2, len(tok), dtype=np.int64)
        np.minimum.at(first_seen, tok, np.arange(len(tok)))
        seen = first_seen[lab] < q

        h = rec["hidden"][idx].float().to(dev)
        prev = torch.cat([torch.from_numpy(tok[anchor[idx]])[:, None], torch.from_numpy(lab[:, : N - 1])], dim=1)
        emb = t["w1"][prev.to(dev)]
        normed = sc.rms_norm(h, t["norm"], eps).reshape(kept * N, -1)
        emb = emb.reshape(kept * N, -1)
        labt = torch.from_numpy(lab).reshape(-1).to(dev)
        ngt = torch.from_numpy(np.where(ng_tok < 0, 0, ng_tok)).reshape(-1).to(dev)
        out = {k: [] for k in ("top", "top_logit", "gap", "label_rank", "label_logit", "top_h", "ng_logit")}
        for s in range(0, kept * N, args.chunk):
            e = min(s + args.chunk, kept * N)
            base = normed[s:e] @ t["head"].t()
            logits = base + emb[s:e] @ t["w2"].t()
            v, i = logits.topk(2, dim=-1)
            ll = logits.gather(1, labt[s:e, None]).squeeze(1)
            out["top"].append(i[:, 0].cpu())
            out["top_logit"].append(v[:, 0].cpu())
            out["gap"].append((v[:, 0] - v[:, 1]).cpu())
            out["label_logit"].append(ll.cpu())
            out["label_rank"].append((logits > ll[:, None]).sum(dim=1).cpu())
            out["top_h"].append(base.argmax(dim=-1).cpu())
            out["ng_logit"].append(logits.gather(1, ngt[s:e, None]).squeeze(1).cpu())
        cols["turn"].append(np.full(kept * N, ti, dtype=np.int32))
        cols["step"].append(np.repeat(idx.astype(np.int32), N))
        cols["pos"].append(np.tile(np.arange(N, dtype=np.int8), kept))
        cols["anchor"].append(np.repeat(anchor[idx].astype(np.int32), N))
        cols["active"].append(np.repeat(np.clip(meta[idx, 1], 0, N).astype(np.int8), N))
        cols["label"].append(lab.reshape(-1).astype(np.int32))
        cols["ng_k"].append(ng_k.reshape(-1).astype(np.int8))
        cols["ng_tok"].append(ng_tok.reshape(-1).astype(np.int32))
        cols["seen"].append(seen.reshape(-1))
        cols["drafted"].append(drafts[idx].numpy().reshape(-1).astype(np.int32))
        for k, v in out.items():
            cols[k].append(torch.cat(v).numpy())
        if (ti + 1) % 50 == 0 or ti + 1 == len(paths):
            print(json.dumps({"turns": ti + 1, "steps": int(sum(len(c) for c in cols["pos"]) // N), "skipped": skipped,
                              "seconds": round(time.time() - started, 1)}), flush=True)
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "anatomy.npz"), **{k: np.concatenate(v) for k, v in cols.items()})


def run_tokens(right: np.ndarray) -> float:
    """Mean tokens a step: 1 + the leading run of right drafts (rows are steps)."""
    return float(1 + np.cumprod(right, axis=1).sum(axis=1).mean())


def study(args) -> None:
    d = np.load(os.path.join(args.out, "anatomy.npz"))
    steps = len(d["pos"]) // N
    g = lambda k: d[k].reshape(steps, N)
    label, top, rank, gap = g("label"), g("top"), g("label_rank"), g("gap")
    right = top == label
    out = {"steps": steps, "turns": int(len(np.unique(d["turn"])))}
    served = g("drafted") == label
    out["replay_agrees_with_serving"] = float((g("drafted") == top).mean())
    out["tokens_a_step"] = {"replayed": run_tokens(right), "served_drafts": run_tokens(served),
                            "markov_term_left_out": run_tokens(g("top_h") == label)}
    out["right_by_depth_teacher_forced"] = right.mean(axis=0).round(4).tolist()
    alive = np.concatenate([np.ones((steps, 1), dtype=bool), np.cumprod(right, axis=1)[:, :-1].astype(bool)], axis=1)
    out["right_by_depth_given_earlier_right"] = [float(right[alive[:, i], i].mean()) for i in range(N)]
    # the committed token's rank at the miss that ends a step's run
    first_miss = alive & ~right
    r = rank[first_miss]
    out["first_miss"] = {"count": int(first_miss.sum()), "share_of_steps": float(first_miss.any(axis=1).mean()),
                         "rank": {"2": float((r == 1).mean()), "3-5": float(((r >= 2) & (r <= 4)).mean()),
                                  "6-20": float(((r >= 5) & (r <= 19)).mean()), "over 20": float((r >= 20).mean())},
                         "gap_quartiles_at_miss": np.percentile(gap[first_miss], [25, 50, 75]).round(3).tolist(),
                         "gap_quartiles_at_hit": np.percentile(gap[alive & right], [25, 50, 75]).round(3).tolist(),
                         "committed_token_seen_earlier_in_request": float(g("seen")[first_miss].mean())}
    # ceilings: tokens a step if the committed token were drafted whenever it ranks in the top k
    out["ceiling_tokens_a_step_if_right_within_top"] = {str(k): run_tokens(rank < k) for k in (1, 2, 3, 5, 20)}
    # what the request's own earlier tokens say
    ng_k, ng_tok, ng_logit, top_logit = g("ng_k"), g("ng_tok"), g("ng_logit"), g("top_logit")
    ng_right = (ng_tok == label) & (ng_k > 0)
    rows = {}
    for k in KS:
        m = ng_k >= k
        rows[str(k)] = {"share_of_live_positions": float(m[alive].mean()), "ngram_right": float(ng_right[m & alive].mean()),
                        "drafter_right": float(right[m & alive].mean()),
                        "ngram_right_where_drafter_wrong": float(ng_right[m & alive & ~right].mean()),
                        "ngram_wrong_where_drafter_right": float((~ng_right)[m & alive & right].mean()),
                        "first_misses_ngram_would_fix": float((first_miss & m & ng_right).sum() / first_miss.sum())}
    out["ngram_match_at_least"] = rows
    # replay: draft the n-gram's token when the match is at least kmin tokens and the drafter's
    # own top-1 leads it by less than beta logits; kmin and beta chosen on the first half of the turns
    turn = g("turn")[:, 0]
    half = np.median(np.unique(turn))
    fit, test = turn <= half, turn > half
    lead = top_logit - ng_logit
    best = None
    grid = {}
    for kmin in KS:
        for beta in (0.5, 1, 2, 3, 4, 6, 8, 12, 1e9):
            switch = (ng_k >= kmin) & (lead < beta)
            new_right = np.where(switch, ng_tok == label, right)
            a, b = run_tokens(new_right[fit]), run_tokens(new_right[test])
            grid[f"{kmin}/{beta:g}"] = [round(a, 4), round(b, 4)]
            if best is None or a > best[0]:
                best = (a, b, kmin, beta, float(switch[test].mean()))
    out["rerank_by_request_ngram"] = {"baseline_fit_test": [run_tokens(right[fit]), run_tokens(right[test])],
                                      "best_on_fit": {"kmin": best[2], "beta": best[3], "tokens_fit": best[0], "tokens_test": best[1],
                                                      "positions_switched_test": best[4]}, "grid_fit_test": grid}
    # a runner-up leaf: one more verified row whose input is the runner-up at one depth. It adds a
    # token when the step's first miss is at that depth and the committed token is the runner-up.
    run = np.cumprod(right, axis=1).sum(axis=1)
    miss_at = np.where(run < N, run, -1)
    second = (rank[np.arange(steps), np.clip(miss_at, 0, N - 1)] == 1) & (miss_at >= 0)
    base = 1 + run.mean()

    def leaf(pick):
        hit, rows = float(((pick == miss_at) & second).mean()), float((pick >= 0).mean())
        return {"rows_a_step": rows, "tokens_added": hit,
                "throughput_at_row_cost": {str(c): float((base + hit) / base / (1 + c * rows) - 1) for c in (0.035, 0.07)}}

    out["runner_up_leaf"] = {"runner_up_right_at_first_miss": float(second[miss_at >= 0].mean()),
                             "oracle_depth": leaf(np.where(second, miss_at, -1)),
                             "smallest_gap_of_five": leaf(gap.argmin(axis=1)),
                             "smallest_gap_of_first_two": leaf(gap[:, :2].argmin(axis=1)),
                             **{f"first_depth_with_gap_under_{g:g}": leaf(np.where((gap < g).any(axis=1), (gap < g).argmax(axis=1), -1))
                                for g in (0.5, 1, 2, 3)}}
    json.dump(out, open(os.path.join(args.out, "anatomy.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rerank_by_request_ngram"}, indent=1))
    print(json.dumps({k: v for k, v in out["rerank_by_request_ngram"].items() if k != "grid_fit_test"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("features", "study"))
    ap.add_argument("--turns")
    ap.add_argument("--ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=2048)
    a = ap.parse_args()
    features(a) if a.stage == "features" else study(a)
