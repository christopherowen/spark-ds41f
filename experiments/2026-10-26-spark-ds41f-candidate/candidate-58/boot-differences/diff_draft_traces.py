#!/usr/bin/env python3
"""The first drafter tap where two boots differ, with the rank and row it differs on.

usage: diff_draft_traces.py BOOT_A_DIR BOOT_B_DIR [--json OUT] [--limit 20]

Each directory holds one boot's collected traces (collect_traces.py) from a profile
with SPARK_DS41F_ROW_TRACE_DRAFTER=1: every rank's rows-dctx-*.bin (the drafter's
context preparation) and rows-dquery-*.bin (its query forward, sampling and
confidence head), one record per engine step, written after the step's drafter ran
(a step without a drafter forward has no calls). Request IDs differ between boots,
so steps are matched by their requests' first-seen order and each request's steps
by their order within it (the c1 workload sends the same prompts in the same order).

For every aligned step, the context preparation then the query forward, on every
rank: the row count, the positions and token ids, then each tap in order, its input
then its output (kinds: 1 all-reduce, 2/3 all-gather, 7 attention output before WO,
10 the target's auxiliary states copied in, 11 the combined context states, 12 the
head hidden state, 13 the confidence head's hidden input -> calibrated probability,
14 its Markov embeddings, 15 a draft layer's raw context KV, 16 its slots, 17 the
confidence rows' request slots, 18 the estimator's slope -> intercepts). A tap row is
real when it belongs to the step's live rows (context: the target's tokens; query:
requests x (1 + drafts)); padded rows are compared too but reported separately, since
a CUDA graph's padding holds whatever its buffers held. Reading the first difference:
  - an input that differs on some ranks only, every earlier tap equal: the
    computation between the previous tap and this one on those ranks, or state it
    reads (the context KV written by earlier steps: compare the earlier steps' 15/16);
  - equal inputs, another output: the operation itself (a collective's output from
    equal inputs on every rank would be the fabric);
  - 13 with equal 13 input, 14, 17 and 18: the confidence kernel itself;
  - only padded rows: the padding (harmless unless a later real row differs).
"""
import argparse
import glob
import json
import os
import pickle
import struct
import sys

KINDS = {1: "all-reduce", 2: "all-gather (last dim)", 3: "all-gather (dim 0)", 7: "attention output (before WO)",
         10: "context: target aux states", 11: "context: combined states", 12: "query: head hidden",
         13: "confidence: hidden -> probability", 14: "confidence: Markov embeddings",
         15: "context: layer raw KV", 16: "context: layer slots", 17: "confidence: request slots",
         18: "estimator: slope -> intercepts", 20: "MoE router (top-k ids -> weights)",
         21: "MoE shared expert (input -> output)", 22: "MoE (input -> this rank's output)",
         23: "MoE output at the reduction's entry", 24: "MoE input MXFP8 cast (values -> scales)"}
TRACES = ("rows-dctx", "rows-dquery")


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
    return int(os.path.basename(path).split("-tp")[1].split("-")[0])


def load(directory):
    """trace -> rank -> list of step records in time order."""
    out = {}
    for trace in TRACES:
        files = sorted(glob.glob(os.path.join(directory, "**", f"{trace}-*.bin"), recursive=True))
        for path in files:
            steps = out.setdefault(trace, {}).setdefault(rank_of(path), [])
            steps.extend(records(path))
        for steps in out.get(trace, {}).values():
            steps.sort(key=lambda r: r["t"])
    if not out:
        raise SystemExit(f"{directory}: no rows-dctx-*.bin or rows-dquery-*.bin")
    return out


def keyed(steps):
    """(first-seen indices of the step's requests, the first request's step index) -> record."""
    names, counts, out = {}, {}, {}
    for rec in steps:
        reqs = rec.get("req_ids") or []
        for r in reqs:
            names.setdefault(r, len(names))
        ids = tuple(names[r] for r in reqs)
        lead = ids[0] if ids else -1
        index = counts.get(lead, 0)
        counts[lead] = index + 1
        out[(ids, index)] = rec
    return out


def real_rows(trace, rec):
    if trace == "rows-dctx":
        return int(rec.get("num_tokens", 0))
    per = rec.get("query_per_req")
    return int(rec.get("num_reqs", 0)) * int(per) if per else int(rec.get("rows", 0))


def digest(rec, k, j, index):
    width = rec.get("ring_width", 0)
    if k >= rec.get("calls", 0) or index >= width:
        return None
    return struct.unpack_from("<q", rec["ring_bytes"], ((k * 2 + j) * width + index) * 8)[0]


def layer_of(layout, k):
    count = sum(1 for entry in layout[:k] if entry and entry[0] == 7)
    kind = layout[k][0] if k < len(layout) and layout[k] else None
    return count - 1 if kind in (1, 2, 3, 20, 21, 22, 23, 24) else count


def compare_step(trace, ra, rb):
    """Differences between two boots' records of one step on one rank, in tap order:
    (call, j, rows that differ, real?) and header differences."""
    out = []
    for name in ("calls", "rows"):
        if ra.get(name) != rb.get(name):
            out.append({"header": name, "a": ra.get(name), "b": rb.get(name)})
    if out or not ra.get("calls"):
        return out
    rows = int(ra["rows"])
    real = min(real_rows(trace, ra), rows)
    for name in ("positions", "ids"):
        xa, xb = ra.get(name) or [], rb.get(name) or []
        bad = [i for i in range(min(len(xa), len(xb))) if xa[i] != xb[i]]
        if bad:
            out.append({"header": name, "rows": bad, "real": any(i < real for i in bad)})
    layout = ra.get("layout") or []
    for k in range(int(ra["calls"])):
        entry = layout[k] if k < len(layout) else None
        for j in (0, 1):
            tap_rows = (entry[1] if j == 0 else entry[2]) if entry else rows
            n = min(tap_rows, rows)
            # Taps with the step's row count have its padding; others (the confidence
            # rows, the estimator) are all real.
            live = real if tap_rows == rows else n
            bad = [i for i in range(n) if digest(ra, k, j, i) != digest(rb, k, j, i)]
            if bad:
                out.append({"call": k, "j": "input" if j == 0 else "output",
                            "kind": KINDS.get(entry[0]) if entry else None, "kind_id": entry[0] if entry else None,
                            "layer": layer_of(layout, k) if entry else None, "rows": bad,
                            "real_rows": [i for i in bad if i < live], "live": live, "tap_rows": tap_rows,
                            "call_layout": entry})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--json")
    parser.add_argument("--limit", type=int, default=20, help="differing steps listed in full")
    args = parser.parse_args()
    boots = {"a": load(args.a), "b": load(args.b)}
    report = {"a": args.a, "b": args.b, "traces": {}, "first_real": None, "first_any": None, "steps": []}
    # The aligned steps of both traces, in the order the drafter ran them.
    aligned = {}
    for trace in TRACES:
        ranks = sorted(set(boots["a"].get(trace, {})) & set(boots["b"].get(trace, {})))
        if not ranks:
            continue
        ka = {r: keyed(boots["a"][trace][r]) for r in ranks}
        kb = {r: keyed(boots["b"][trace][r]) for r in ranks}
        lead = ranks[0]
        order = [key for key in sorted(ka[lead], key=lambda key: ka[lead][key]["t"]) if key in kb[lead]]
        report["traces"][trace] = {"ranks": ranks, "steps_a": len(ka[lead]), "steps_b": len(kb[lead]),
                                   "aligned": len(order), "differing_steps": 0, "differing_real_steps": 0,
                                   "first_kinds": {}}
        for position, key in enumerate(order):
            aligned.setdefault(key, {"index": position, "t": ka[lead][key]["t"]})[trace] = (ka, kb, ranks)
    for key in sorted(aligned, key=lambda key: aligned[key]["t"]):
        entry = aligned[key]
        for trace in TRACES:  # the context preparation ran before the query forward
            if trace not in entry:
                continue
            ka, kb, ranks = entry[trace]
            per_rank = {r: compare_step(trace, ka[r][key], kb[r][key]) for r in ranks if key in ka[r] and key in kb[r]}
            per_rank = {r: d for r, d in per_rank.items() if d}
            if not per_rank:
                continue
            info = report["traces"][trace]
            info["differing_steps"] += 1

            def is_real(d):
                return bool(d.get("real_rows")) or bool(d.get("real")) or ("header" in d and "rows" not in d)

            def first_of(select):
                best = None
                for r, diffs in per_rank.items():
                    for d in diffs:
                        if select(d):
                            order = (d.get("call", -1), 0 if d.get("j") == "input" else 1, r)
                            if best is None or order < best[0]:
                                best = (order, r, d)
                            break
                return best

            def item_of(found, real):
                _, rank, head = found
                rec = ka[rank][key]
                return {"trace": trace, "key": [list(key[0]), key[1]], "index": entry["index"],
                        "rows": rec.get("rows"), "num_tokens": rec.get("num_tokens"), "num_reqs": rec.get("num_reqs"),
                        "positions": (rec.get("positions") or [])[: rec.get("rows") or 0], "real": real,
                        "first": {**head, "rank": rank},
                        "ranks_differing_at_first_call": sorted(
                            r for r, diffs in per_rank.items()
                            if any(d.get("call") == head.get("call") and d.get("j") == head.get("j")
                                   and d.get("header") == head.get("header") for d in diffs)),
                        "per_rank": {r: d[:6] for r, d in per_rank.items()}}

            any_first = first_of(lambda d: True)
            real_first = first_of(is_real)
            if real_first is not None:
                info["differing_real_steps"] += 1
            head = (real_first or any_first)[2]
            label = head.get("kind") or head.get("header")
            info["first_kinds"][label] = info["first_kinds"].get(label, 0) + 1
            item = item_of(real_first or any_first, real_first is not None)
            if report["first_any"] is None:
                report["first_any"] = item_of(any_first, real_first is not None)
            if real_first is not None and report["first_real"] is None:
                report["first_real"] = item
            if len(report["steps"]) < args.limit:
                report["steps"].append(item)
    if report["first_real"] is None:
        report["verdict"] = ("no real drafter row differs between the boots" if report["first_any"] is None
                             else "only padded drafter rows differ")
    else:
        f = report["first_real"]["first"]
        where = f.get("kind") or f.get("header")
        report["verdict"] = (f"first real difference: {report['first_real']['trace']} step {report['first_real']['key']}, "
                             f"call {f.get('call')} ({where}, layer {f.get('layer')}) {f.get('j', '')} on rank "
                             f"{f['rank']}; ranks at that call: {report['first_real']['ranks_differing_at_first_call']}")
    text = json.dumps(report, indent=1, default=str)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text + "\n")
    print(text[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
