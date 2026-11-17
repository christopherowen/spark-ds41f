#!/usr/bin/env python3
"""The byte floor and the busy and idle split of a traced candidate-61 step, with
experiments/2026-10-26-c1-100tps/floor.py on today's kernel names.

usage: floor_trace.py TRACES [--expert-rate OUTPUT.txt]

``--expert-rate`` may hold ``{node}``: each rank is then read with its own node's fit.

``TRACES`` holds ``decode-trace/<node>/*.json.gz`` (one trace a rank, as a lab profile job
leaves them). The record's ``window`` does the work; three things differ from 1022's build:

- a forward starts with ``vocab_packed_embedding_kernel`` (the record's mark was
  ``vocab_embedding_kernel``): renamed here, with the Markov head's screen and refine
  kernels, so that ``vocab_packed`` is the target's LM head alone again;
- the drafter's single-stream head is read through its 8-bit screen
  (``vocab_screen_greedy_partials_kernel``): one byte a weight, and a stream;
- ``gemm_mxfp8_rope_mxfp4_kernel`` streams its projection.

The expert-rate fit (FC1 and FC2 against the experts read) defaults to the record's dgx4
bundle of 1022's kernels; ``bundles/expert-rate`` is the same measurement on candidate-61's.
"""
import argparse
import glob
import importlib.util
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, "..", "2026-10-26-c1-100tps")
spec = importlib.util.spec_from_file_location("floor", os.path.join(REC, "floor.py"))
fl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl)

fl.MARK = "forward_embedding"
fl.STREAM = fl.STREAM + ("vocab_screen", "gemm_mxfp8_rope")
SCREEN = 32320 * 5120  # the 8-bit screening copy of the drafter's head shard
fl.DENSE = dict(fl.DENSE, **{"drafter LM head": SCREEN})
RENAME = {"vocab_packed_embedding_kernel": "forward_embedding"}
FAMILIES = (("routed experts", ("moe_grouped",)), ("L2 prefetch (side stream)", ("l2_prefetch",)),
            ("dense projections", ("gemm_",)), ("collectives", fl.COLLECTIVE), ("mHC", ("mhc_",)),
            ("GLP steer", ("glp_steer",)), ("LM heads and screens", ("vocab_packed", "vocab_screen")),
            ("attention, indexer, cache", fl.ATTN + ("attention_front",)))


def load(path):
    ks = []
    for ts, dur, name, tid, grid in fl.load(path):
        if name.startswith("vocab_packed_markov"):
            name = "markov" + name[len("vocab_packed_markov"):]
        ks.append((ts, dur, RENAME.get(name, name), tid, grid))
    return ks


def families(ks, steps):
    """Kernel time a step by family, over the steps' span (kernels overlap across streams)."""
    a, b = steps[0][2], steps[-1][3]
    acc, other = defaultdict(float), 0.0
    for ts, dur, name, _, _ in ks:
        if not a <= ts < b:
            continue
        for label, prefixes in FAMILIES:
            if name.startswith(prefixes):
                acc[label] += dur
                break
        else:
            acc["everything else"] += dur
    return {k: round(v / len(steps) / 1e3, 3) for k, v in sorted(acc.items(), key=lambda x: -x[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("--expert-rate", default=os.path.join(REC, "runs", "expert-rate-dgx4-2.txt"))
    args = ap.parse_args()
    out = {"source": args.traces, "expert_rate": args.expert_rate, "ranks": {}}
    workload = os.path.join(args.traces, "decode.json")
    if os.path.exists(workload):
        out["workload"] = [json.loads(line) for line in open(workload) if line.startswith("{")]
    for folder in sorted(glob.glob(os.path.join(args.traces, "decode-trace", "*"))):
        ks = load(sorted(glob.glob(os.path.join(folder, "*.json.gz")))[0])
        steps = fl.steps(ks)[1:]
        fits = fl.expert_fits(args.expert_rate.format(node=os.path.basename(folder)))
        w = fl.window(ks, 1, None, fits)
        w["fit_us (launch, an expert)"] = {k: [round(v, 3) for v in fits[6][k]] for k in ("fc1", "fc2")}
        w["kernel_ms_by_family"] = families(ks, steps)
        spans = sorted((b - a) / 1e3 for _, _, a, b in steps)
        w["step_ms_median"] = round(statistics.median(spans), 3)
        out["ranks"][os.path.basename(folder)] = w
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
