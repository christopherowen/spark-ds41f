#!/usr/bin/env python3
"""The first committed position, collective and rank where two runs of one prompt differ.

usage: first_divergence.py BOOT_DIR [--pair first-vs-each|all] [--json OUT]

BOOT_DIR holds one boot's collected traces (collect_traces.py): every rank's
rows-*.bin (the row trace) and steps-*.jsonl. Requests served alone (one request in
each of their steps) are grouped by their prompt (the token ids of their prefill
step, from position 0: prefix-cache hits are left out); a prompt served alone more
than once (the gate's reference and again, in both runs) is compared pairwise, by
default its first occurrence (the first run's reference) against each later one.

A row is committed when its position is below the request's next step's first
position (the verified drafts and the bonus row; rejected and dead rows are not).
For every committed position present in both runs, in order, with equal token ids:
every collective's input digest (each rank's partial) and output digest, then the
final hidden state's digest, on every rank. The first difference names the stage:
  - an input differs on some ranks while all earlier digests of the row are equal:
    the computation between the previous collective and this one, on those ranks
    (or its cached state: the KV, index keys, compressor state of earlier rows);
  - every rank's input is equal and the output differs: the collective itself;
  - every collective is equal and the final digest differs: after the last one;
  - everything committed is equal but the tokens differ: the head or the sampler.
The steps holding that position in both runs are printed (rows, positions, token ids,
graph mode), which shows how the two schedules differed there.
"""
import argparse
import glob
import json
import os
import pickle
import struct
import sys


def records(path):
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) < 4:
                return
            (size,) = struct.unpack("<I", head)
            data = f.read(size)
            if len(data) < size:
                return
            yield pickle.loads(data)


def rank_of(path):
    name = os.path.basename(path)
    return int(name.split("-tp")[1].split("-")[0])


def requests_of(path):
    """req_id -> list of step records (time order), for requests alone in their steps."""
    by_req = {}
    for rec in records(path):
        if len(rec.get("req_ids", [])) != 1 or "ring" not in rec:
            continue
        by_req.setdefault(rec["req_ids"][0], []).append(rec)
    for steps in by_req.values():
        steps.sort(key=lambda r: r["t"])
    return by_req


def committed(steps):
    """position -> (step index, row, token id) of the request's committed rows."""
    out = {}
    for i, step in enumerate(steps[:-1]):
        nxt = int(steps[i + 1]["positions"][0])
        real = int(step["num_tokens"])
        for row in range(real):
            p = int(step["positions"][row])
            if p < nxt:
                out[p] = (i, row, int(step["ids"][row]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("boot")
    parser.add_argument("--pair", default="first-vs-each", choices=("first-vs-each", "all"))
    parser.add_argument("--json")
    args = parser.parse_args()
    files = sorted(glob.glob(os.path.join(args.boot, "**", "rows-*.bin"), recursive=True))
    if not files:
        raise SystemExit(f"{args.boot}: no rows-*.bin")
    ranks = {}
    for path in files:
        ranks.setdefault(rank_of(path), {}).update(requests_of(path))
    rank_ids = sorted(ranks)
    base = ranks[rank_ids[0]]
    prompts = {}
    for req, steps in base.items():
        first = steps[0]
        if int(first["positions"][0]) != 0:
            continue
        key = tuple(int(x) for x in first["ids"][: int(first["num_tokens"])])
        prompts.setdefault(key, []).append((steps[0]["t"], req))
    report = {"boot": args.boot, "ranks": rank_ids, "requests_alone": len(base), "prompts": [], "differences": []}
    for key, occ in prompts.items():
        occ.sort()
        reqs = [r for _, r in occ]
        if len(reqs) < 2:
            continue
        pairs = [(reqs[0], r) for r in reqs[1:]] if args.pair == "first-vs-each" else \
            [(reqs[i], reqs[j]) for i in range(len(reqs)) for j in range(i + 1, len(reqs))]
        report["prompts"].append({"prompt_tokens": len(key), "occurrences": len(reqs)})
        for ra, rb in pairs:
            ca = {r: committed(ranks[r][ra]) for r in rank_ids if ra in ranks[r]}
            cb = {r: committed(ranks[r][rb]) for r in rank_ids if rb in ranks[r]}
            positions = sorted(set(ca[rank_ids[0]]) & set(cb[rank_ids[0]]))
            found = None
            for p in positions:
                ia, rowa, ta = ca[rank_ids[0]][p]
                ib, rowb, tb = cb[rank_ids[0]][p]
                if ta != tb:
                    found = {"position": p, "kind": "token", "detail": "token ids differ here first; every "
                             "committed row before it was equal, so the logits of the row before differed "
                             "(head or sampler)", "token_a": ta, "token_b": tb}
                    break
                calls = None
                for rank in rank_ids:
                    sa = ranks[rank][ra][ca[rank][p][0]]
                    sb = ranks[rank][rb][cb[rank][p][0]]
                    n = min(sa["calls"], sb["calls"])
                    calls = n if calls is None else min(calls, n)
                diffs = []
                for k in range(calls or 0):
                    ins, outs = [], []
                    for rank in rank_ids:
                        sa = ranks[rank][ra][ca[rank][p][0]]
                        sb = ranks[rank][rb][cb[rank][p][0]]
                        ra_row, rb_row = ca[rank][p][1], cb[rank][p][1]
                        if int(sa["ring"][k][0][ra_row]) != int(sb["ring"][k][0][rb_row]):
                            ins.append(rank)
                        if int(sa["ring"][k][1][ra_row]) != int(sb["ring"][k][1][rb_row]):
                            outs.append(rank)
                    if ins or outs:
                        layout = (ranks[rank_ids[0]][ra][ia].get("layout") or [None] * (k + 1))
                        diffs.append({"call": k, "input_differs_on_ranks": ins, "output_differs_on_ranks": outs,
                                      "call_layout": layout[k] if k < len(layout) else None,
                                      "all_reduce_ordinal": sum(1 for c in layout[:k + 1] if c and c[0] == 1)})
                        break
                finals = [rank for rank in rank_ids
                          if int(ranks[rank][ra][ca[rank][p][0]]["final"][ca[rank][p][1]])
                          != int(ranks[rank][rb][cb[rank][p][0]]["final"][cb[rank][p][1]])]
                if diffs or finals:
                    d = diffs[0] if diffs else None
                    if d and d["input_differs_on_ranks"]:
                        stage = (f"computation before collective {d['call']} on ranks {d['input_differs_on_ranks']} "
                                 "(its inputs from earlier collectives of this row are equal)")
                    elif d:
                        stage = f"collective {d['call']} itself: equal inputs on every rank, other output"
                    else:
                        stage = "after the last collective (final mHC, norm): final digest differs"
                    found = {"position": p, "kind": "digest", "first": d, "final_differs_on_ranks": finals,
                             "stage": stage}
                    break
            if found:
                p = found["position"]
                ctx = {}
                for label, req, cmt in (("a", ra, ca), ("b", rb, cb)):
                    step = ranks[rank_ids[0]][req][cmt[rank_ids[0]][p][0]]
                    ctx[label] = {"rows": int(step["num_tokens"]), "padded": step.get("padded"),
                                  "positions": [int(x) for x in step["positions"][: int(step["num_tokens"])]],
                                  "ids": [int(x) for x in step["ids"][: int(step["num_tokens"])]],
                                  "cg_mode": step.get("cg_mode")}
                found["steps"] = ctx
                report["differences"].append({"prompt_tokens": len(key), "pair": [ra, rb], **found})
    text = json.dumps(report, indent=1, default=str)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text + "\n")
    print(text[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
