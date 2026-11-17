"""Offline sizing of drafts past the five-draft block, on the same dumped turns.

A DSpark step drafts five tokens. When the target accepts all five the step ends there
(the target's own next token is the sixth). What would drafts for positions six and
later be worth?

The ring keeps a row for every decode step of a request (its last 256), whichever
drafter's proposal the step verified, at ``count % capacity``: rows are put in anchor
order here, and the distance to the next anchor is the number of tokens the step
committed. A step that committed more than six verified a copy proposal
(``CopyDrafter``, up to 16 drafts), not DSpark's five.

Stage ``features`` (one pass over the turns): each turn's steps in order, the tokens
each committed, the committed stream by position, and for every step whose five DSpark
drafts were right the sources of a later draft that need no new drafter:
  row5     the Markov chain continued on the fifth row's base logits (the fifth draft
           as the chain's input, then its own drafts);
  markov   the Markov bias alone (no backbone row);
  lookup   the continuation of the most recent earlier occurrence of the last n tokens
           (n = 6, 4, 3) in this turn's committed output (``local``), or also in the
           outputs of the turns that finished before it (``lookup``: a stand-in for
           the prompt, which the dumps do not hold; it reads other sessions' turns too).
Only tokens committed or drafted before the step's verification are read. Position five
is also replayed from the fourth row, where the real fifth draft is known, to show what
a row one position stale costs.

Stage ``study``: how steps end (a miss, the survival cut, the five-draft block, a copy),
what follows a step that ended at the block, the sources' acceptance there, and the
value of a block of 5 + m drafts, at acceptance q of each added draft given the ones
before it were accepted and at the measured sources:
  ceiling     no step ends at the block (any length, every added draft right, free rows);
  merge       a step is saved only where added drafts carry a step to the end of a later
              one: a run of block-ended steps and the step that ends it are re-tiled
              with steps of 6 + e tokens, e the accepted added drafts (a run a copy step
              ends gets no credit); the later steps end where they did;
  next-step   (measured sources) a step is saved where the added drafts the stream
              accepted reach the end of the step after it;
  stationary  every accepted added draft is a token the later steps need not produce, at
              the average step yield (the later steps yield what they did; this
              over-credits, and passes the ceiling for long perfect blocks).
Cost is 1 + rho * rows verified a step. An added row is verified where five are and the
predicted survival to it (the serving form's survival to the fifth draft times the added
drafts' acceptance) is at least a threshold, chosen on the first half of the turns for
each case and rho (never adding a row is one of the choices) and scored on the second.
"""

import argparse
import glob
import json
import os
import time
from functools import lru_cache

import numpy as np
import torch

import size_calibration as sc

N = sc.N
M = 5  # added drafts measured
NGRAMS = (6, 4, 3)
TAU = 0.2  # serving's dead-row survival threshold
LOOKAHEAD = 16  # the copy drafter's longest proposal


def features(args) -> None:
    dev = sc.device_name()
    t, eps = sc.load_tensors(args.ckpt, dev)
    paths = sorted(glob.glob(os.path.join(args.turns, args.pattern)))
    if args.limit:
        paths = paths[: args.limit]
    names = ("turn", "run", "cap", "delta", "next", "consistent", "truth", "row5", "markov", "lookup",
             "lookup_n", "local", "local_n", "stale5_ok", "real5_ok", "stale5_valid")
    cols = {k: [] for k in names}
    seen: dict[tuple, int] = {}  # n-gram -> next token, over the turns already finished
    base_row = 0
    started = time.time()
    for ti, path in enumerate(paths):
        rec = torch.load(path, map_location="cpu", weights_only=True)
        hidden = rec["hidden"].float()
        kept = hidden.shape[0]
        if kept == 0:
            continue
        ids, labels, drafts = rec["ids"].long(), rec["labels"].long(), rec["drafts"].long()
        meta = rec["meta"].long()
        anchor_pos = meta[:, 0].numpy()
        active = meta[:, 1].clamp(0, N)
        pos = torch.arange(N).expand(kept, N)
        valid = ((labels >= 0) & (pos < active[:, None])).int().cumprod(dim=1).bool()
        y = (drafts == labels) & valid
        cap = valid.sum(1).numpy()
        run = y.int().cumprod(dim=1).sum(1).numpy()

        order = np.argsort(anchor_pos, kind="stable")
        nxt = np.full(kept, -1, dtype=np.int64)
        nxt[order[:-1]] = order[1:]
        delta = np.zeros(kept, dtype=np.int64)
        delta[order[:-1]] = anchor_pos[order[1:]] - anchor_pos[order[:-1]]
        broken = delta > LOOKAHEAD + 1  # not one step's tokens: a row out of sequence
        nxt[broken] = -1
        delta[broken] = 0
        # the next anchor is a token this step committed: checked where a label covers it
        consistent = np.ones(kept, dtype=bool)
        for k in order[:-1]:
            d = int(delta[k])
            if d < 1:
                consistent[k] = False
            elif d <= cap[k]:
                consistent[k] = int(labels[k, d - 1]) == int(ids[nxt[k], 0])

        stream: dict[int, int] = {}
        for k in range(kept):
            stream[int(anchor_pos[k])] = int(ids[k, 0])
            for j in range(int(cap[k])):
                stream[int(anchor_pos[k]) + 1 + j] = int(labels[k, j])
        truth = np.full((kept, M), -1, dtype=np.int64)
        for k in range(kept):
            for j in range(M):
                truth[k, j] = stream.get(int(anchor_pos[k]) + N + 1 + j, -1)

        full_mask = (cap == N) & (run == N)
        full = np.flatnonzero(full_mask)
        row5 = np.full((kept, M), -1, dtype=np.int64)
        markov = np.full((kept, M), -1, dtype=np.int64)
        if full.size:
            h = sc.rms_norm(hidden[full, N - 1].to(dev), t["norm"], eps)
            for s in range(0, full.size, args.chunk):
                rows = full[s : s + args.chunk]
                base = h[s : s + args.chunk] @ t["head"].t()
                prev = drafts[rows, N - 1].to(dev)
                prev_m = prev.clone()
                for j in range(M):
                    prev = (base + t["w1"][prev] @ t["w2"].t()).argmax(-1)
                    row5[rows, j] = prev.cpu().numpy()
                    prev_m = (t["w1"][prev_m] @ t["w2"].t()).argmax(-1)
                    markov[rows, j] = prev_m.cpu().numpy()

        four = np.flatnonzero((cap == N) & (run >= N - 1))
        stale5 = np.full(kept, -1, dtype=np.int64)
        if four.size:
            h4 = sc.rms_norm(hidden[four, N - 2].to(dev), t["norm"], eps)
            for s in range(0, four.size, args.chunk):
                rows = four[s : s + args.chunk]
                logits = h4[s : s + args.chunk] @ t["head"].t() + t["w1"][drafts[rows, N - 2].to(dev)] @ t["w2"].t()
                stale5[rows] = logits.argmax(-1).cpu().numpy()
        stale5_valid = np.zeros(kept, dtype=bool)
        stale5_valid[four] = True
        lab5 = labels[:, N - 1].numpy()

        lookup = np.full((kept, M), -1, dtype=np.int64)
        lookup_n = np.zeros(kept, dtype=np.int8)
        local_chain = np.full((kept, M), -1, dtype=np.int64)
        local_n = np.zeros(kept, dtype=np.int8)
        local: dict[tuple, int] = {}  # n-gram -> next token, in this turn's committed output
        indexed = int(anchor_pos[order[0]]) - 1  # pairs (e, e + 1) are indexed for e <= indexed
        width = max(NGRAMS)
        for k in order:
            if cap[k] < N:
                continue
            p = int(anchor_pos[k])
            for e in range(indexed + 1, p):
                nxt_tok = stream.get(e + 1)
                if nxt_tok is None:
                    continue
                for n in NGRAMS:
                    ctx = tuple(stream.get(q, -2) for q in range(e - n + 1, e + 1))
                    if -2 not in ctx:
                        local[ctx] = nxt_tok
            indexed = max(indexed, p - 1)
            # what the step knows before verification: the stream to its anchor, its drafts
            known = [stream.get(q, -2) for q in range(p - width + N + 1, p + 1)] + drafts[k].tolist()
            for out, out_n, tables in ((lookup, lookup_n, (local, seen)), (local_chain, local_n, (local,))):
                tail = list(known)
                for j in range(M):
                    guess = None
                    for n in NGRAMS:
                        ctx = tuple(tail[-n:])
                        if -2 in ctx:
                            continue
                        for table in tables:
                            guess = table.get(ctx)
                            if guess is not None:
                                break
                        if guess is not None:
                            if j == 0:
                                out_n[k] = n
                            break
                    if guess is None:
                        break
                    out[k, j] = guess
                    tail.append(guess)
        for q in sorted(stream):
            if q + 1 in stream:
                for n in NGRAMS:
                    ctx = tuple(stream.get(r, -2) for r in range(q - n + 1, q + 1))
                    if -2 not in ctx:
                        seen[ctx] = stream[q + 1]

        cols["turn"].append(np.full(kept, ti, dtype=np.int32))
        cols["run"].append(run.astype(np.int8))
        cols["cap"].append(cap.astype(np.int8))
        cols["delta"].append(delta.astype(np.int32))
        cols["next"].append(np.where(nxt >= 0, nxt + base_row, -1))
        cols["consistent"].append(consistent)
        cols["truth"].append(truth)
        cols["row5"].append(row5)
        cols["markov"].append(markov)
        cols["lookup"].append(lookup)
        cols["lookup_n"].append(lookup_n)
        cols["local"].append(local_chain)
        cols["local_n"].append(local_n)
        cols["stale5_ok"].append(stale5 == lab5)
        cols["real5_ok"].append(drafts[:, N - 1].numpy() == lab5)
        cols["stale5_valid"].append(stale5_valid)
        base_row += kept
        if (ti + 1) % 200 == 0 or ti + 1 == len(paths):
            print(json.dumps({"turns": ti + 1, "steps": base_row, "seconds": round(time.time() - started, 1)}), flush=True)
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "longer.npz"), **{k: np.concatenate(v) for k, v in cols.items()})
    print(json.dumps({"written": os.path.join(args.out, "longer.npz"), "steps": base_row}), flush=True)


# ---------------------------------------------------------------- study

CLASSES = ("miss", "cut", "block", "copy-short", "copy")


def classify(run, delta):
    """How a step ended, from the tokens it committed and DSpark's right leading drafts."""
    c = np.full(run.shape, -1, dtype=np.int8)
    acc = delta - 1
    c[(delta >= 1) & (delta <= N) & (acc == run)] = 0  # the next draft was wrong
    c[(delta >= 1) & (delta <= N) & (acc < run)] = 1  # right drafts left unverified
    c[(delta == N + 1) & (run == N)] = 2  # ended at the five-draft block
    c[(delta >= 1) & (delta <= N + 1) & (acc > run)] = 3  # more than DSpark's drafts gave: a copy
    c[delta > N + 1] = 4  # a copy proposal
    return c


def chain_accept(drafts, truth):
    """Leading added drafts that match the stream, and by depth: [share right given the
    ones before were right and the truth is known, n]."""
    ok = (drafts == truth) & (truth >= 0)
    lead = ok.astype(np.int64).cumprod(axis=1)
    by_depth = []
    alive = np.ones(drafts.shape[0], dtype=bool)
    for j in range(drafts.shape[1]):
        pop = alive & (truth[:, j] >= 0)
        by_depth.append([float(ok[pop, j].mean()) if pop.any() else None, int(pop.sum())])
        alive = alive & ok[:, j]
    return lead.sum(axis=1), by_depth


def tile_steps(pe: tuple):
    """Expected steps to cover L tokens with steps of 6 + e tokens, e the accepted added
    drafts (``pe[e]`` its distribution). ``hard``: the L-th token is the target's
    correction, so a step that reaches it ends there. ``easy``: steps started inside the
    L tokens (what a last step covers past them is not credited)."""

    @lru_cache(maxsize=None)
    def hard(L: int) -> float:
        if L <= N + 1:
            return 1.0
        return sum(p * (1.0 if N + 1 + e >= L else 1.0 + hard(L - N - 1 - e)) for e, p in enumerate(pe) if p > 0)

    @lru_cache(maxsize=None)
    def easy(L: int) -> float:
        if L <= 0:
            return 0.0
        return sum(p * (1.0 + easy(L - N - 1 - e)) for e, p in enumerate(pe) if p > 0)

    return hard, easy


def study(args) -> None:
    f = dict(np.load(os.path.join(args.out, "features.npz")))
    g = dict(np.load(os.path.join(args.out, "longer.npz")))
    n_steps = g["run"].size
    assert int(f["block"].max()) + 1 == n_steps
    run, cap, delta, nxt, turn = g["run"].astype(np.int64), g["cap"].astype(np.int64), g["delta"].astype(np.int64), g["next"], g["turn"]
    n_turns = int(turn.max()) + 1
    has_next = nxt >= 0
    cls = classify(run, delta)
    report = {"steps": int(n_steps), "turns": n_turns, "steps_with_next": int(has_next.sum()),
              "next_anchor_matches_label": float(g["consistent"][has_next].mean()),
              "delta_hist": np.bincount(np.clip(delta[has_next], 0, 18), minlength=19).tolist()}
    s = has_next & (cls >= 0)
    report["unclassified"] = int((has_next & (cls < 0)).sum())
    tokens = delta[s].sum()
    report["ends"] = {name: {"steps_share": float((cls[s] == i).mean()),
                             "tokens_share": float(delta[s][cls[s] == i].sum() / tokens),
                             "tokens_per_step": float(delta[s][cls[s] == i].mean())}
                      for i, name in enumerate(CLASSES)}
    report["tokens_per_step"] = float(delta[s].mean())
    allright = s & (run == N) & (cap == N)
    report["five_right"] = {"steps_share": float(allright.sum() / s.sum()),
                            "then": {name: float((cls[allright] == i).mean()) for i, name in enumerate(CLASSES)}}

    # what follows a step that ended at the block, and (for comparison) any step
    def after(mask):
        k = np.flatnonzero(mask & has_next)
        j = nxt[k]
        j = j[has_next[j] & (cls[j] >= 0)]
        return {"pairs": int(j.size),
                "next_end": {name: float((cls[j] == i).mean()) for i, name in enumerate(CLASSES)},
                "next_tokens": float(delta[j].mean()),
                "next_dspark_first_right": float((run[j] >= 1).mean()),
                "next_dspark_right_mean": float(run[j].mean()),
                "next_five_right": float((run[j] == N).mean())}
    block = s & (cls == 2)
    report["after_block_end"] = after(block)
    report["after_five_right"] = after(allright)
    report["after_any"] = after(s)
    for r in range(N):
        report[f"after_miss_at_{r + 1}"] = after(s & (cls == 0) & (run == r))

    # runs of consecutive block-ended steps
    runs = []  # (turn, length, the class of the step that ends it, the tokens it committed)
    prev_block = np.zeros(n_steps, dtype=bool)
    prev_block[nxt[block & has_next]] = True
    for k in np.flatnonzero(block & ~prev_block):
        c, j = 0, k
        while j >= 0 and block[j]:
            c += 1
            j = nxt[j]
        term = (int(cls[j]), int(delta[j])) if j >= 0 and s[j] else (-1, 0)
        runs.append((int(turn[k]), c, term[0], term[1]))
    runs = np.array(runs, dtype=np.int64).reshape(-1, 4)
    report["block_runs"] = {"runs": int(runs.shape[0]), "length_hist": np.bincount(np.clip(runs[:, 1], 0, 8), minlength=9).tolist(),
                            "terminator": {name: float((runs[:, 2] == i).mean()) for i, name in enumerate(CLASSES)}}

    # the sources at the positions after a block-ended step
    truth_b = g["truth"][block]
    t0 = truth_b[:, 0]
    lead = {}
    src = {}
    for name in ("row5", "markov", "lookup", "local"):
        lead[name], by_depth = chain_accept(g[name][block], truth_b)
        src[name] = {"by_depth": by_depth, "accepted_mean": float(lead[name].mean()),
                     "offered_first": float((g[name][block][:, 0] >= 0).mean())}
    for name in ("lookup", "local"):
        ln = g[name + "_n"][block]
        first = g[name][block][:, 0]
        src[name]["first_by_n"] = {str(n): {"share": float((ln == n).mean()), "right": float((first[ln == n] == t0[ln == n]).mean()),
                                            "accepted_mean": float(lead[name][ln == n].mean())} for n in NGRAMS}
    r5 = g["row5"][block][:, 0]
    lk = g["lookup"][block][:, 0]
    agree = r5 == lk
    src["row5_and_lookup"] = {"agree_share": float(agree.mean()), "right_given_agree": float((r5[agree] == t0[agree]).mean()),
                              "either_right": float(((r5 == t0) | (lk == t0)).mean())}
    v = g["stale5_valid"]
    src["position5_from_row4"] = {"n": int(v.sum()), "stale_right": float(g["stale5_ok"][v].mean()), "real_right": float(g["real5_ok"][v].mean())}
    report["sources_after_block_end"] = src

    # rows verified: the serving form at serving's threshold on DSpark steps, the
    # accepted drafts on copy steps; a block-ended step verified five
    valid, cond, y, pos, blocks = f["valid"], f["cond"], f["y"], f["pos"], f["block"]
    split = n_turns // 2
    first = f["turn"] < split
    xs = np.concatenate([f["z"].astype(np.float64)[:, None], sc.onehot(pos)[:, 1:]], axis=1)
    rows_fit, rows_all = np.flatnonzero(cond & first), np.flatnonzero(valid)
    lg = sc.fit_logistic(xs[rows_fit], y[rows_fit])(xs[rows_all])
    pb = np.zeros((n_steps, N))
    pb[blocks[rows_all], pos[rows_all]] = sc.sigmoid(lg)
    surv = np.cumprod(pb, axis=1)
    d = ((surv >= TAU) & (np.arange(N)[None, :] < cap[:, None])).sum(axis=1)
    d = np.where(cls == 4, delta - 1, d)
    d = np.where(cls == 2, N, np.maximum(d, np.minimum(delta - 1, N)))
    five = (d >= N) & (cls != 4)
    surv5 = np.where(cap == N, surv[:, N - 1], 0.0)
    te = s & (turn >= split)
    report["test"] = {"steps": int(te.sum()), "tokens_per_step": float(delta[te].mean()), "rows_per_step": float(d[te].mean()),
                      "block_end_share": float(block[te].mean()), "five_verified_share": float(five[te].mean()),
                      "copy_steps_share": float((cls[te] == 4).mean()),
                      "block_end_given_five_verified": float(block[te & five].mean())}

    def per_turn(w, mask):
        return np.bincount(turn[mask], weights=np.asarray(w, dtype=np.float64)[mask], minlength=n_turns)
    ones = np.ones(n_steps)
    S, R, T, NB = per_turn(ones, s), per_turn(d, s), per_turn(delta, s), per_turn(ones, block)
    halves = {"fit": np.arange(n_turns) < split, "test": np.arange(n_turns) >= split}
    test_turns = np.flatnonzero(halves["test"])
    rng = np.random.default_rng(0)
    boots = test_turns[rng.integers(0, test_turns.size, size=(args.boot, test_turns.size))]
    rhos = [float(r) for r in args.rhos.split(",")]
    thresholds = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def gain(idx, saved, extra_tok, extra_rows, rho):
        base = T[idx].sum(-1) / (S[idx].sum(-1) + rho * R[idx].sum(-1))
        new = (T[idx].sum(-1) + extra_tok[idx].sum(-1)) / (S[idx].sum(-1) - saved[idx].sum(-1) + rho * (R[idx].sum(-1) + extra_rows[idx].sum(-1)))
        return 100 * (new / base - 1)

    def merged(e, x_max):
        """Steps saved a turn by re-tiling the block runs, each half with its own
        distribution of the accepted added drafts on block-ended steps."""
        saved = np.zeros(n_turns)
        for half in halves.values():
            hb = block & half[turn]
            if not hb.any():
                continue
            hard, easy = tile_steps(tuple((np.bincount(e[hb].round().astype(np.int64), minlength=x_max + 1) / hb.sum()).tolist())
                                    if e.dtype.kind == "i" else tuple(e_dist(hb)))
            for tn, c, tc, tr in runs[half[runs[:, 0]]]:
                saved[tn] += (c + 1) - hard(6 * c + tr) if tc in (0, 1, 3) else c - easy(6 * c)
        return saved

    zero = np.zeros(n_turns)
    fit_turns = np.flatnonzero(halves["fit"])
    value = {"ceiling": {}}
    for rho in rhos:
        b = gain(boots, NB, zero, zero, rho)
        value["ceiling"][str(rho)] = [round(float(gain(test_turns, NB, zero, zero, rho)), 2), round(float(np.percentile(b, 2.5)), 2), round(float(np.percentile(b, 97.5)), 2)]

    def score(cases):
        """cases: threshold -> {reading: (saved, extra_tok, extra_rows)} per turn. The
        threshold is chosen on the fit half for each reading and rho."""
        out = {}
        for reading in next(iter(cases.values())):
            out[reading] = {}
            for rho in rhos:
                best, best_gain = None, 0.0  # never adding a row
                for th, by in cases.items():
                    gfit = float(gain(fit_turns, *by[reading], rho))
                    if gfit > best_gain:
                        best, best_gain = th, gfit
                if best is None:
                    out[reading][str(rho)] = {"threshold": None, "gain": 0.0, "ci": [0.0, 0.0], "added_rows_per_step": 0.0}
                    continue
                parts = cases[best][reading]
                b = gain(boots, *parts, rho)
                out[reading][str(rho)] = {"threshold": best, "gain": round(float(gain(test_turns, *parts, rho)), 2),
                                          "ci": [round(float(np.percentile(b, 2.5)), 2), round(float(np.percentile(b, 97.5)), 2)],
                                          "added_rows_per_step": round(float(parts[2][test_turns].sum() / S[test_turns].sum()), 4)}
        return out

    # a drafter whose added drafts are each right with probability q
    for q in (0.5, 0.67, 0.8, 0.885, 1.0):
        for m in (1, 2, 3, 5):
            cases = {}
            for th in thresholds:
                x = np.where(five, sum((surv5 * q ** j >= th).astype(np.int64) for j in range(1, m + 1)), 0)
                geo = np.array([sum(q ** j for j in range(1, n + 1)) for n in range(m + 1)])

                def e_dist(hb, x=x, q=q, m=m):
                    pe = np.zeros(m + 1)
                    for n, cnt in enumerate(np.bincount(x[hb], minlength=m + 1)):
                        for e in range(n + 1):
                            pe[e] += cnt * ((q ** e) * (1 - q) if e < n else q ** n)
                    return pe / hb.sum()
                saved = merged(np.zeros(1), m)
                cases[th] = {"merge": (saved, zero, per_turn(x, s)),
                             "stationary": (zero, per_turn(geo[x], block), per_turn(x, s))}
            value[f"q={q},m={m}"] = score(cases)

    # the measured sources: what the stream accepted of their added drafts
    bi = np.flatnonzero(block)
    fit_b = halves["fit"][turn[bi]]
    measured = {"row5": ("row5", None), "lookup": ("lookup", 0), "lookup6": ("lookup", 6), "local": ("local", 0), "local6": ("local", 6)}
    for name, (chain, level) in measured.items():
        if level is None:
            offered = np.ones(n_steps, dtype=bool)
        else:
            offered = (g[chain + "_n"] == level) if level else (g[chain + "_n"] > 0)
        ob = offered[bi]
        _, depth_fit = chain_accept(g[chain][bi][ob & fit_b], truth_b[ob & fit_b])
        cum = np.cumprod([x[0] if x[0] is not None else 0.0 for x in depth_fit])
        for m in (1, 2, 3, 5):
            cases = {}
            for th in thresholds:
                x = np.where(five & offered, sum((surv5 * cum[j] >= th).astype(np.int64) for j in range(m)), 0)
                e = np.zeros(n_steps, dtype=np.int64)
                e[bi] = np.minimum(lead[chain], x[bi])
                nd = np.where(has_next, delta[np.maximum(nxt, 0)], 0)
                direct = per_turn((nd >= 1) & (nd <= e) & s[np.maximum(nxt, 0)], block)
                rows_x = per_turn(x, s)
                cases[th] = {"merge": (merged(e, m), zero, rows_x), "next_step": (direct, zero, rows_x),
                             "stationary": (zero, per_turn(e, block), rows_x)}
            value[f"{name},m={m}"] = score(cases)
            value[f"{name},m={m}"]["acceptance_by_depth_fit"] = [round(float(c), 3) for c in (x[0] or 0.0 for x in depth_fit)][:m]
            value[f"{name},m={m}"]["offered_share_of_block_ends"] = float(ob.mean())
    report["value_pct"] = value
    json.dump(report, open(os.path.join(args.out, "longer.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


def summary(args) -> None:
    r = json.load(open(os.path.join(args.out, "longer.json")))
    pct = lambda x: f"{100 * x:.1f}%"
    print(f"steps {r['steps_with_next']} with a next step in {r['turns']} turns; tokens a step {r['tokens_per_step']:.3f}; "
          f"next anchor equals the label it lands on {pct(r['next_anchor_matches_label'])}")
    print("\nhow a step ended         steps   tokens  tokens/step")
    for name, e in r["ends"].items():
        print(f"  {name:12s}        {pct(e['steps_share']):>7s} {pct(e['tokens_share']):>7s}  {e['tokens_per_step']:6.2f}")
    fr = r["five_right"]
    print(f"\nall five DSpark drafts right: {pct(fr['steps_share'])} of steps; then " + ", ".join(f"{k} {pct(v)}" for k, v in fr["then"].items() if v))
    print("\nthe step after ...        first draft right  drafts right  five right  tokens  ends at block  copy")
    for key in ("after_any", "after_miss_at_1", "after_miss_at_5", "after_five_right", "after_block_end"):
        a = r[key]
        print(f"  {key[6:]:22s} {a['next_dspark_first_right']:8.3f} {a['next_dspark_right_mean']:14.2f} {a['next_five_right']:11.3f} {a['next_tokens']:8.2f} "
              f"{pct(a['next_end']['block']):>10s} {pct(a['next_end']['copy'] + a['next_end']['copy-short']):>8s}   ({a['pairs']} pairs)")
    br = r["block_runs"]
    print(f"\nruns of block-ended steps: {br['runs']}; lengths 1..8+ {br['length_hist'][1:]}; ended by " + ", ".join(f"{k} {pct(v)}" for k, v in br["terminator"].items() if v))
    src = r["sources_after_block_end"]
    print("\nadded drafts after a block-ended step: right given the ones before were right, by depth (n at depth 6); accepted a step")
    for name in ("row5", "markov", "lookup", "local"):
        e = src[name]
        print(f"  {name:8s} " + " ".join("  -  " if x[0] is None else f"{x[0]:.3f}" for x in e["by_depth"]) + f"  ({e['by_depth'][0][1]}); {e['accepted_mean']:.2f}; offered {pct(e['offered_first'])}")
        for n, x in e.get("first_by_n", {}).items():
            print(f"      matched {n} tokens: {pct(x['share'])} of block ends, first right {x['right']:.3f}, accepted {x['accepted_mean']:.2f}")
    x = src["row5_and_lookup"]
    print(f"  row5 and lookup agree on the sixth at {pct(x['agree_share'])}, right {x['right_given_agree']:.3f} there; either right {x['either_right']:.3f}")
    x = src["position5_from_row4"]
    print(f"  position five from the fourth row: right {x['stale_right']:.3f} against the real fifth draft's {x['real_right']:.3f} ({x['n']} steps)")
    t = r["test"]
    print(f"\nsecond half: {t['steps']} steps, tokens a step {t['tokens_per_step']:.3f}, rows a step {t['rows_per_step']:.3f}, block-ended {pct(t['block_end_share'])}, "
          f"five verified {pct(t['five_verified_share'])} ({pct(t['block_end_given_five_verified'])} of them block-ended), copy steps {pct(t['copy_steps_share'])}")
    v = r["value_pct"]
    rhos = list(v["ceiling"])
    print("\nthroughput gain, % (second half), by rho: " + " ".join(rhos))
    print("  ceiling                 " + " ".join(f"{v['ceiling'][k][0]:+6.2f}" for k in rhos) + f"   [{v['ceiling']['0.11'][1]}, {v['ceiling']['0.11'][2]}] at 0.11")
    for key, x in v.items():
        if key == "ceiling":
            continue
        for reading in ("merge", "next_step", "stationary"):
            if reading in x:
                c = x[reading]["0.11"]
                print(f"  {key:14s} {reading:10s}" + " ".join(f"{x[reading][k]['gain']:+6.2f}" for k in rhos) + f"   {c['ci']} at 0.11, threshold {c['threshold']}, rows +{c['added_rows_per_step']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["features", "study", "summary"])
    ap.add_argument("--turns")
    ap.add_argument("--pattern", default="20261117T020420-*.pt")
    ap.add_argument("--ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--rhos", default="0.09,0.11,0.15,0.23,0.5,0.75")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    {"features": features, "study": study, "summary": summary}[a.stage](a)
