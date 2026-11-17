"""Replay of the copy drafter's rule on dumped turns: what a shorter minimum match would
add to the steps serving took.

Reads the turns of the turn-learning observe stage (``VLLM_DSPARK_TURN_LEARNING_DUMP``:
every decode step's anchor, DSpark drafts and committed labels, in ring order) and the
cached features of experiments/2026-11-17-draft-calibration-sizing (``features.npz``, for
the rows a DSpark step verifies). A turn's history is what the dump holds of it: the
committed output (``tokens`` too, the request's whole token row, when the dump has it;
without it the prompt is unknown and so are the tokens inside a copy step past its
fifth, which are holes no match crosses).

The rule (vllm/v1/worker/gpu/spec_decode/tilelang_copy.py, copy_drafts.py): a request
whose context ends with the same ``min_match`` tokens (or 2, 4, ... times as many) as an
earlier position of its own history is offered that position's continuation in place of
DSpark's drafts: the longest level wins, then a source with the full lookahead (16), then
the latest. A first proposal offers ``initial`` (8) drafts, doubled after a proposal
accepted in full and reset after a rejection; it is taken when ``offered * estimate`` is
at least DSpark's survival yield, the estimate ``(hits + 19) / (trials + 20)`` over
counts decayed by 0.9. Serving's ``min_match`` is 16. In a context past 32K tokens
serving scans the newest 16K and the first 16K densely and the rest only for 256-token
matches (``window``: sources in those two ranges only; without it, the whole history).

``stats``: at every step's anchor, the longest match in the history (4 to 64 tokens), how
many of its next 16 tokens the stream then commits, what the step did, and whether a
16-token match lies in serving's scan.
``simulate``: the turn walked from its first anchor with a given ``min_match``. A step that
serving copied is taken as it was. Elsewhere a match proposes by the rule and commits its
accepted drafts and the target's token; without one, DSpark's step is the dumped one (from
an anchor the dump does not have, the rest of the dumped step it falls in). A step costs
``1 + rho * rows verified``: a copy's offered drafts, a DSpark step's rows from the serving
form at serving's threshold.
"""
import argparse
import bisect
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2026-11-17-draft-calibration-sizing"))
import size_calibration as sc  # noqa: E402

N = 5
LOOKAHEAD = 16
INITIAL = 8
PRIOR_HITS, PRIOR_TRIALS, DECAY = 19.0, 20.0, 0.9
KEY = 4  # the shortest context indexed
KEYS = (16, 8, 4)  # a query uses the longest key its minimum match allows
MAX_MATCH = 64
WINDOW = 16384  # serving's dense scans: the newest and the first tokens
CANDIDATES = 48
TAU = 0.2
HOLE = -9


class Turn:
    """A turn's steps in anchor order and its history as an array by position."""

    def __init__(self, rec, surv):
        ids, labels, drafts = rec["ids"].long().numpy(), rec["labels"].long().numpy(), rec["drafts"].long().numpy()
        meta = rec["meta"].long().numpy()
        anchor = meta[:, 0]
        active = np.clip(meta[:, 1], 0, N)
        valid = np.cumprod((labels >= 0) & (np.arange(N)[None, :] < active[:, None]), axis=1).astype(bool)
        cap = valid.sum(1)
        run = np.cumprod((drafts == labels) & valid, axis=1).sum(1)
        order = np.argsort(anchor, kind="stable")
        self.anchor, self.cap, self.run = anchor[order], cap[order], run[order]
        d = ((surv[order] >= TAU) & (np.arange(N)[None, :] < self.cap[:, None])).sum(1)
        self.yield_ = surv[order].sum(1)
        delta = np.zeros(len(order), dtype=np.int64)
        delta[:-1] = np.diff(self.anchor)
        delta[(delta > LOOKAHEAD + 1) | (delta < 1)] = 0  # not one step's tokens
        self.delta = delta
        self.rows = np.where(delta == N + 1, N, np.maximum(d, np.minimum(np.maximum(delta - 1, 0), N)))
        full = rec.get("tokens")
        if full is not None:  # the request's whole token row: prompt and output
            self.base = 0
            self.tok = full.long().numpy().copy()
        else:
            self.base = int(self.anchor[0])
            top = int((self.anchor + np.maximum(self.cap, 0)).max())
            self.tok = np.full(top - self.base + 1, HOLE, dtype=np.int64)
            for k in range(len(order)):
                p = int(self.anchor[k]) - self.base
                self.tok[p] = ids[order[k], 0]
                self.tok[p + 1 : p + 1 + self.cap[k]] = labels[order[k], : self.cap[k]]
        self._index: dict[int, dict[tuple, list[int]]] = {}

    def index(self, width):
        """Ends of every ``width``-token run of the history, by its tokens."""
        if width not in self._index:
            tok = self.tok.tolist()
            table: dict[tuple, list[int]] = {}
            for e in range(width - 1, len(tok)):
                key = tuple(tok[e - width + 1 : e + 1])
                if HOLE not in key:
                    table.setdefault(key, []).append(e)
            self._index[width] = table
        return self._index[width]

    def match(self, pos, min_match, window=False):
        """The rule's source for the context ending at ``pos`` (array index): (match
        length, source end, drafts it can offer), or None."""
        width = next(k for k in KEYS if k <= min_match)
        if pos < width - 1:
            return None
        tok = self.tok
        key = tuple(tok[pos - width + 1 : pos + 1].tolist())
        if HOLE in key:
            return None
        ends = self.index(width).get(key, ())
        ends = ends[: bisect.bisect_left(ends, pos)]  # a source ends before the context
        if window and pos + self.base + 1 > 2 * WINDOW:
            ends = [e for e in ends if e + self.base < WINDOW or e > pos - WINDOW]
        best = None
        for e in reversed(ends[-CANDIDATES:]):
            n = width
            while n < MAX_MATCH and e - n >= 0 and tok[e - n] == tok[pos - n] and tok[e - n] != HOLE:
                n += 1
            if n < min_match:
                continue
            level = min_match
            while level * 2 <= n:
                level *= 2
            avail = 0
            while avail < LOOKAHEAD and e + 1 + avail <= pos and tok[e + 1 + avail] != HOLE:
                avail += 1
            if avail == 0:
                continue
            rank = (level, avail == LOOKAHEAD, avail if avail < LOOKAHEAD else 0, e)
            if best is None or rank > best[0]:
                best = (rank, n, e, avail)
        return None if best is None else best[1:]

    def accepted(self, pos, source, offered):
        """Leading drafts of the source's continuation that the stream commits after
        ``pos`` (a hole or the end of the dump ends the count)."""
        tok = self.tok
        a = 0
        while a < offered and pos + 1 + a < len(tok) and tok[pos + 1 + a] != HOLE and tok[source + 1 + a] == tok[pos + 1 + a]:
            a += 1
        return a


def turns(args):
    f = dict(np.load(os.path.join(args.features, "features.npz")))
    valid, cond, y, pos, turn = f["valid"], f["cond"], f["y"], f["pos"], f["turn"]
    n_turns = int(turn.max()) + 1
    xs = np.concatenate([f["z"].astype(np.float64)[:, None], sc.onehot(pos)[:, 1:]], axis=1)
    fit, rows = np.flatnonzero(cond & (turn < n_turns // 2)), np.flatnonzero(valid)
    pb = np.zeros((int(f["block"].max()) + 1, N))
    pb[f["block"][rows], pos[rows]] = sc.sigmoid(sc.fit_logistic(xs[fit], y[fit])(xs[rows]))
    surv = np.cumprod(pb, axis=1)
    paths = sorted(glob.glob(os.path.join(args.turns, args.pattern)))
    base = 0
    for ti, path in enumerate(paths):
        rec = torch.load(path, map_location="cpu", weights_only=True)
        kept = rec["hidden"].shape[0]
        if kept == 0:
            continue
        if kept >= 2:
            yield ti, Turn(rec, surv[base : base + kept])
        base += kept
    assert base == surv.shape[0], (base, surv.shape)


def stats(all_turns):
    rows = []  # (match length, offerable, accepted of 16, step tokens, run, in serving's windows)
    for t in all_turns:
        for k in np.flatnonzero(t.delta > 0):
            pos = int(t.anchor[k]) - t.base
            m = t.match(pos, KEY)
            if m is None:
                rows.append((0, 0, 0, t.delta[k], t.run[k], 0))
            else:
                n, e, avail = m
                seen = t.match(pos, 16, window=True) is not None if n >= 16 else False
                rows.append((n, avail, t.accepted(pos, e, avail), t.delta[k], t.run[k], int(seen)))
    r = np.array(rows)
    out = {"steps": int(len(r)), "tokens_per_step": float(r[:, 3].mean()), "copy_steps_share": float((r[:, 3] > N + 1).mean()),
           "by_match": {}}
    groups = [(f"{lo}-{hi}", (r[:, 0] >= lo) & (r[:, 0] <= hi)) for lo, hi in ((0, 0), (4, 5), (6, 7), (8, 11), (12, 15), (16, 31), (32, 64))]
    long = r[:, 0] >= 16
    groups += [("16+, in serving's scan", long & (r[:, 5] == 1)), ("16+, outside serving's scan", long & (r[:, 5] == 0))]
    for name, sel in groups:
        if not sel.any():
            continue
        s = r[sel]
        matched = s[:, 0].max() > 0
        out["by_match"][name] = {
            "steps_share": float(sel.mean()),
            "served_as_copy": float((s[:, 3] > N + 1).mean()),
            "step_tokens": float(s[:, 3].mean()),
            "offerable": float(s[:, 1].mean()),
            "first_right": float((s[:, 2] >= 1).mean()) if matched else None,
            "accepted_of_offerable": float(s[:, 2].mean()) if matched else None,
            "accepted_of_first_8": float(np.minimum(s[:, 2], 8).mean()) if matched else None,
            "copy_of_8_beats_step": float((np.minimum(s[:, 2], 8) + 1 > s[:, 3]).mean()) if matched else None,
        }
    return out


def simulate(t, min_match, initial=INITIAL, window=False):
    """(tokens, steps, rows, copy steps added, their drafts offered, accepted)."""
    tokens = steps = rows = added = offered_sum = accepted_sum = 0
    live = np.flatnonzero(t.delta > 0)
    if live.size == 0:
        return np.zeros(6)
    doublings, hits, trials = 0, 0.0, 0.0
    k = int(live[0])
    pos = int(t.anchor[k])
    end = int(t.anchor[-1])
    while pos < end:
        while k + 1 < len(t.anchor) and t.anchor[k + 1] <= pos:
            k += 1
        delta = int(t.delta[k])
        if delta == 0:  # a break in the dump: resume at the next anchor
            k += 1
            pos = int(t.anchor[k])
            doublings = 0
            continue
        a_k = int(t.anchor[k])
        left = a_k + delta - pos
        served_copy = delta > N + 1
        steps += 1
        if served_copy:  # as serving took it (from inside it: the rest of it)
            tokens += left
            rows += min(LOOKAHEAD, max(INITIAL, left - 1)) if pos == a_k else min(LOOKAHEAD, left)
            pos += left
            doublings = min(doublings + 1, 1)
            continue
        m = t.match(pos - t.base, min_match, window) if min_match < 10**6 else None
        if m is not None:
            n, e, avail = m
            cap = min(LOOKAHEAD, initial << doublings)
            offered = min(cap, avail)
            estimate = (hits + PRIOR_HITS) / (trials + PRIOR_TRIALS)
            if offered * estimate >= t.yield_[k]:
                acc = t.accepted(pos - t.base, e, offered)
                acc = min(acc, max(end - pos - 1, 0))
                hits, trials = hits * DECAY + acc, trials * DECAY + min(offered, acc + 1)
                doublings = min(doublings + 1, 1) if acc == offered else 0
                tokens += acc + 1
                rows += offered
                added += 1
                offered_sum += offered
                accepted_sum += acc
                pos += acc + 1
                continue
        doublings = 0
        tokens += left
        rows += t.rows[k] if pos == a_k else min(t.rows[k], left)
        pos += left
    return np.array([tokens, steps, rows, added, offered_sum, accepted_sum], dtype=np.float64)


def study(args):
    all_turns = [t for _, t in turns(args)]
    out = {"turns": len(all_turns), "with_tokens": int(sum(t.base == 0 for t in all_turns)),
           "history_tokens_median": float(np.median([len(t.tok) for t in all_turns])), "stats": stats(all_turns)}
    variants = {"serving (as dumped)": dict(min_match=10**6), "16, serving's scan": dict(min_match=16, window=True),
                "16": dict(min_match=16), "12": dict(min_match=12), "8": dict(min_match=8), "6": dict(min_match=6),
                "4": dict(min_match=4), "8, first offer 4": dict(min_match=8, initial=4),
                "6, first offer 4": dict(min_match=6, initial=4), "4, first offer 4": dict(min_match=4, initial=4)}
    per = {name: [simulate(t, **kw) for t in all_turns] for name, kw in variants.items()}
    per = {k: np.array(v) for k, v in per.items()}
    n = len(per["serving (as dumped)"])
    boots = np.random.default_rng(0).integers(0, n, size=(args.boot, n))
    base = per["serving (as dumped)"]
    sim = {}
    for name, v in per.items():
        tot = v.sum(0)
        row = {"tokens_per_step": round(tot[0] / tot[1], 4), "rows_per_step": round(tot[2] / tot[1], 4), "steps": int(tot[1]),
               "copy_steps_added_share": round(tot[3] / tot[1], 4),
               "added_offered": round(tot[4] / max(tot[3], 1), 2), "added_accepted": round(tot[5] / max(tot[3], 1), 2), "gain_pct": {}}
        for rho in [float(r) for r in args.rhos.split(",")]:
            def thr(x, idx=None):
                x = x if idx is None else x[idx]
                s = x.sum(-2)
                return s[..., 0] / (s[..., 1] + rho * s[..., 2])
            g = 100 * (thr(v, boots) / thr(base, boots) - 1)
            row["gain_pct"][str(rho)] = [round(float(100 * (thr(v) / thr(base) - 1)), 2), round(float(np.percentile(g, 2.5)), 2), round(float(np.percentile(g, 97.5)), 2)]
        sim[name] = row
    out["simulate"] = sim
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--pattern", default="20261117T020420-*.pt")
    ap.add_argument("--features", required=True, help="directory of size_calibration.py's features.npz for these turns")
    ap.add_argument("--rhos", default="0.09,0.11,0.15,0.23,0.5")
    ap.add_argument("--boot", type=int, default=1000)
    study(ap.parse_args())
