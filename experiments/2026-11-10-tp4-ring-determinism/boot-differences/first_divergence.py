#!/usr/bin/env python3
"""The first committed position, collective and rank where two runs of one prompt differ.

usage: first_divergence.py BOOT_DIR [--pair first-vs-each|all] [--across OTHER_BOOT_DIR] [--json OUT]

BOOT_DIR holds one boot's collected traces (collect_traces.py): every rank's
rows-*.bin (the row trace) and steps-*.jsonl. Requests served alone (one request in
each of their steps) are grouped by their prompt (the token ids of their prefill
step, from position 0: prefix-cache hits are left out); a prompt served alone more
than once (the gate's reference and again, in both runs) is compared pairwise, by
default its first occurrence (the first run's reference) against each later one.
With --across, each prompt's i-th occurrence in BOOT_DIR is compared with its i-th in
OTHER_BOOT_DIR instead: the same request on two boots, so a target difference that
leaves the tokens alone (which the gate cannot see without logprobs) shows too.

A row is committed when its position is below the request's next step's first
position (the verified drafts and the bonus row; rejected and dead rows are not).
For every committed position present in both runs, in order, with equal token ids,
every tap of the forward in order, on every rank (kind: 1 all-reduce, 2/3 all-gather,
4 compressor values -> latents, 5 the ratio-2 ring carry of a step's first row,
6 indexer top-k, 7 attention output before WO, 8 Engram table rows -> output), then
the final hidden state. The first differing tap names the stage and its layer:
  - Engram rows (8, input): the rows the disk reader or its cache delivered;
  - compressor (4): input equal and latents differ = the pooling with its carry; the
    carry itself is kind 5 (first row of a step only);
  - indexer top-k (6) with equal earlier taps: scores or selection, or the index
    keys cached for earlier positions;
  - attention output (7) with equal compressor and top-k taps: sparse MLA, or the
    KV, compressed KV it reads (written by earlier rows: see the history printed);
  - an all-reduce input differing on some ranks: the computation after the previous
    tap on those ranks; equal inputs with another output: the collective itself
    (the input's digest is taken after the collective: an input changed while it
    ran also shows here);
  - with SPARK_DS41F_ROW_TRACE_MOE, inside a MoE layer: router ids/weights (20),
    the input's MXFP8 cast (24), the shared expert (21: equal input, other output =
    the shared expert, which runs on the aux stream), the rank's output (22: equal
    20, 21, 24 = routed experts or their combine), the output at the reduction's
    entry (23: equal 22 = changed between the experts and the reduction), and an
    all-reduce input differing with 23 equal = changed during the pre-reduce hook
    (L2 prefetch issue) or the collective;
  - only the final digest: after the last collective (final mHC, norm);
  - everything equal but the tokens: the head or the sampler.
The steps holding that position in both runs are printed (rows, positions, token ids,
graph mode), and every computation of the positions just before it (step, row, token,
committed or not), which shows how the two schedules differed there.
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
        if len(rec.get("req_ids", [])) != 1 or ("ring" not in rec and "ring_bytes" not in rec):
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


KINDS = {1: "all-reduce", 2: "all-gather (last dim)", 3: "all-gather (dim 0)",
         4: "compressor (values -> latents)", 5: "compressor ring carry (first row)",
         6: "indexer top-k", 7: "attention output (before WO)", 8: "Engram (table rows -> output)",
         20: "MoE router (top-k ids -> weights)", 21: "MoE shared expert (input -> output)",
         22: "MoE (input -> this rank's output)", 23: "MoE output at the reduction's entry",
         24: "MoE input MXFP8 cast (values -> scales)"}
# Taps that follow their layer's attention output: the layer of the last kind 7.
AFTER_ATTENTION = {1, 2, 3, 20, 21, 22, 23, 24}


def tap(step, k, j, row):
    """Digest j (0 input, 1 output) of tap k for the step's row, or None when the tap
    is not row-aligned with it (a per-request carry for another row, a dim-0 gather)."""
    layout = step.get("layout") or []
    entry = layout[k] if k < len(layout) else None
    rows = (entry[1] if j == 0 else entry[2]) if entry else step["rows"]
    if rows == step["rows"]:
        index = row
    elif rows == 1 and row == 0:
        index = 0
    else:
        return None
    if "ring_bytes" in step:  # [calls, 2, ring_width] little-endian int64
        width = step["ring_width"]
        if k >= step["calls"] or index >= width:
            return None
        return struct.unpack_from("<q", step["ring_bytes"], ((k * 2 + j) * width + index) * 8)[0]
    ring = step["ring"]  # the first format: nested lists
    if k >= len(ring) or index >= len(ring[k][j]):
        return None
    return int(ring[k][j][index])


def layer_of(layout, k):
    """The decoder layer of tap k: attention outputs (kind 7) counted before it; a
    collective or MoE tap belongs to the last attention before it (-1: the embedding)."""
    count = sum(1 for entry in layout[:k] if entry and entry[0] == 7)
    kind = layout[k][0] if k < len(layout) and layout[k] else None
    return count - 1 if kind in AFTER_ATTENTION else count


def history(steps, positions):
    """Every computation of these positions in a request's steps (step, row, token,
    committed), in time order."""
    out = {p: [] for p in positions}
    for i, step in enumerate(steps):
        nxt = int(steps[i + 1]["positions"][0]) if i + 1 < len(steps) else None
        for row in range(int(step["num_tokens"])):
            p = int(step["positions"][row])
            if p in out:
                out[p].append({"step": i, "row": row, "token": int(step["ids"][row]),
                               "committed": nxt is not None and p < nxt,
                               "step_rows": int(step["num_tokens"])})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("boot")
    parser.add_argument("--pair", default="first-vs-each", choices=("first-vs-each", "all"))
    parser.add_argument("--across")
    parser.add_argument("--json")
    args = parser.parse_args()

    def load_boot(directory, prefix=""):
        files = sorted(path for path in glob.glob(os.path.join(directory, "**", "rows-*.bin"), recursive=True)
                       # the drafter's traces (rows-dctx-*, rows-dquery-*): diff_draft_traces.py
                       if os.path.basename(path).split("-")[1] not in ("dctx", "dquery"))
        if not files:
            raise SystemExit(f"{directory}: no rows-*.bin")
        out = {}
        for path in files:
            for req, steps in requests_of(path).items():
                out.setdefault(rank_of(path), {})[prefix + req] = steps
        return out

    def prompts_of(reqs):
        prompts = {}
        for req, steps in reqs.items():
            first = steps[0]
            if int(first["positions"][0]) != 0:
                continue
            key = tuple(int(x) for x in first["ids"][: int(first["num_tokens"])])
            prompts.setdefault(key, []).append((steps[0]["t"], req))
        return {key: [r for _, r in sorted(occ)] for key, occ in prompts.items()}

    ranks = load_boot(args.boot)
    other_prompts = None
    if args.across:
        other = load_boot(args.across, "B:")
        for rank, reqs in other.items():
            ranks.setdefault(rank, {}).update(reqs)
        other_prompts = prompts_of(other[min(other)])
    rank_ids = sorted(ranks)
    base = {r: s for r, s in ranks[rank_ids[0]].items() if not r.startswith("B:")}
    prompts = prompts_of(base)
    report = {"boot": args.boot, "across": args.across, "ranks": rank_ids, "requests_alone": len(base),
              "prompts": [], "differences": []}
    for key, reqs in prompts.items():
        if other_prompts is not None:
            theirs = other_prompts.get(key, [])
            pairs = list(zip(reqs, theirs))
        elif len(reqs) < 2:
            continue
        else:
            pairs = [(reqs[0], r) for r in reqs[1:]] if args.pair == "first-vs-each" else \
                [(reqs[i], reqs[j]) for i in range(len(reqs)) for j in range(i + 1, len(reqs))]
        report["prompts"].append({"prompt_tokens": len(key), "occurrences": len(reqs), "pairs": len(pairs)})
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
                layout = ranks[rank_ids[0]][ra][ia].get("layout") or []
                for k in range(calls or 0):
                    ins, outs = [], []
                    for rank in rank_ids:
                        sa = ranks[rank][ra][ca[rank][p][0]]
                        sb = ranks[rank][rb][cb[rank][p][0]]
                        ra_row, rb_row = ca[rank][p][1], cb[rank][p][1]
                        for j, bucket in ((0, ins), (1, outs)):
                            x, y = tap(sa, k, j, ra_row), tap(sb, k, j, rb_row)
                            if x is not None and y is not None and x != y:
                                bucket.append(rank)
                    if ins or outs:
                        entry = layout[k] if k < len(layout) else None
                        diffs.append({"call": k, "kind": KINDS.get(entry[0]) if entry else None,
                                      "layer": layer_of(layout, k) if entry else None,
                                      "input_differs_on_ranks": ins, "output_differs_on_ranks": outs,
                                      "call_layout": entry})
                        break
                finals = [rank for rank in rank_ids
                          if int(ranks[rank][ra][ca[rank][p][0]]["final"][ca[rank][p][1]])
                          != int(ranks[rank][rb][cb[rank][p][0]]["final"][cb[rank][p][1]])]
                if diffs or finals:
                    d = diffs[0] if diffs else None
                    if d and d["input_differs_on_ranks"]:
                        stage = (f"the input of tap {d['call']} ({d['kind']}, layer {d['layer']}) differs on ranks "
                                 f"{d['input_differs_on_ranks']}, every earlier tap of the row equal: the computation "
                                 "before it, or the cached state it reads")
                    elif d:
                        stage = (f"tap {d['call']} ({d['kind']}, layer {d['layer']}) gives another output from equal "
                                 f"inputs on ranks {d['output_differs_on_ranks']}")
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
                near = [q for q in range(max(0, p - 4), p + 1)]
                found["history"] = {label: history(ranks[rank_ids[0]][req], near)
                                    for label, req in (("a", ra), ("b", rb))}
                report["differences"].append({"prompt_tokens": len(key), "pair": [ra, rb], **found})
    text = json.dumps(report, indent=1, default=str)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text + "\n")
    print(text[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
