#!/usr/bin/env python3
"""Per-chunk Engram waits of cold prefills, every rank and Engram layer of a boot together.

The debug timeline (SPARK3_ENGRAM_TIMELINE=1) writes one JSON line per step and disk table
to <host>-tp<rank>-layer<id>.jsonl, appending across boots; each boot's reader sequence
restarts at 1. Steps line up across files by sequence within a boot.

A prefill request is a run of steps whose row count is a whole chunk's (the commonest
large count), ended by its last, shorter chunk. For each request and chunk index the report gives
the GPU's wait for the chunk's rows (the largest over ranks and layers: the ranks meet in
every layer's collectives), the reader's submit-to-publish time and the step period
(submission to the next submission).

usage: timeline_chunks.py --from-line N --tz Europe/Berlin FILE... [--json OUT]
"""

import argparse
import json
import os
import statistics
import time
from collections import defaultdict


def boots(records):
    out, current = [], []
    for record in records:
        if current and record["sequence"] <= current[-1]["sequence"]:
            out.append(current)
            current = []
        current.append(record)
    if current:
        out.append(current)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--from-line", type=int, required=True, help="first new line (1-based)")
    parser.add_argument("--offset", type=float, required=True, help="unix - monotonic seconds (dgx1)")
    parser.add_argument("--json")
    args = parser.parse_args()

    # boot index -> sequence -> list of (file, record)
    steps: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for path in args.files:
        with open(path) as handle:
            lines = handle.readlines()[args.from_line - 1:]
        for b, boot in enumerate(boots([json.loads(line) for line in lines if line.strip()])):
            for record in boot:
                steps[b][record["sequence"]].append((os.path.basename(path), record))

    report = {}
    for b in sorted(steps):
        seqs = sorted(steps[b])
        dgx1 = [r for s in seqs for f, r in steps[b][s] if f.startswith("dgx1") and "layer1." in f]
        large = [r["count"] for r in dgx1 if r["count"] >= 10000]
        full = max(set(large), key=large.count)  # a whole chunk's rows (the commonest)
        requests, current = [], []
        for s in seqs:
            group = steps[b][s]
            count = group[0][1]["count"]
            if any(r["count"] != count for _, r in group):
                raise SystemExit(f"boot {b} seq {s}: ranks disagree on the row count")
            if (count < full // 4 or count > full) and not current:
                continue  # warm-up and decode steps
            current.append((s, group))
            if count < full:
                requests.append(current)
                current = []
        if current:
            requests.append(current)
        first = dgx1[0]["submitted"]
        print(f"\n== boot {b}: {len(seqs)} steps from "
              f"{time.strftime('%H:%M:%S', time.localtime(first / 1e9 + args.offset))}, "
              f"{len(requests)} prefill requests, whole chunk {full} rows")
        rows = []
        for i, request in enumerate(requests):
            waits = [max(r["gpu_wait_ns"] for _, r in g) / 1e6 for _, g in request]
            by_file = defaultdict(float)
            for _, g in request[1:]:
                for f, r in g:
                    by_file[f] += r["gpu_wait_ns"] / 1e6
            d1 = [dict(g)[next(f for f, _ in g if f.startswith("dgx1") and "layer1." in f)] for _, g in request]
            periods = [(y["submitted"] - x["submitted"]) / 1e6 for x, y in zip(d1, d1[1:])]
            reads = [(x["published"] - x["submitted"]) / 1e6 for x in d1]
            tokens = sum(x["count"] for x in d1) // 24
            print(f"  req {i}: {len(request)} chunks, ~{tokens} tokens; max-rank wait per chunk: "
                  f"c1 {waits[0]:.1f}, c2 {waits[1] if len(waits) > 1 else 0:.1f}, "
                  f"c3-N sum {sum(waits[2:]):.1f} ms; chunks 2-N sum {sum(waits[1:]):.1f} ms; "
                  f"submit->publish median {statistics.median(reads):.0f} ms; "
                  f"chunk period median {statistics.median(periods[1:-1]) if len(periods) > 2 else 0:.0f} ms, "
                  f"c1->c2 {periods[0] if periods else 0:.0f} ms")
            print("         chunks 2-N wait by file: " + ", ".join(
                f"{f.replace('.jsonl', '')} {v:.1f}" for f, v in sorted(by_file.items())))
            rows.append({"chunks": len(request), "tokens": tokens, "wait_ms": waits,
                         "period_ms": periods, "submit_to_publish_ms": reads,
                         "wait_by_file_chunks_2_n": dict(by_file)})
        report[f"boot{b}"] = rows
    if args.json:
        with open(args.json, "w") as out:
            json.dump(report, out, indent=1)


if __name__ == "__main__":
    main()
