#!/usr/bin/env python3
"""A traced boot's stream kernels rank by rank: the mean time a call of each kernel that
streams weights, over the decode steps of every rank's trace.

usage: rank_compare.py TRACES [TRACES ...]

Each ``TRACES`` holds ``decode-trace/<node>/*.json.gz``. The ranks of one boot run the same
kernels on the same rows in lockstep, so a rank whose memory streams faster shows it in
these kernels' times and nowhere else. FC1 and FC2 are the target's routed-expert GEMMs
(grids 9 x 36 and 40 x 36), whose time follows the experts the step reads, the same on
every rank.
"""
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

import floor_trace as ft

KERNELS = (("FC1 (target routed experts)", lambda k: k[2].startswith("moe_grouped") and k[4][:2] == (9, 36)),
           ("FC2 (target routed experts)", lambda k: k[2].startswith("moe_grouped") and k[4][:2] == (40, 36)),
           ("target LM head (248 MB)", lambda k: k[2] == "vocab_packed_projection_kernel"),
           ("drafter head screen (165 MB)", lambda k: k[2] == "vocab_screen_greedy_partials_kernel"),
           ("dense projection (gemm_mxfp8)", lambda k: k[2] == "gemm_mxfp8_kernel"),
           ("L2 prefetch", lambda k: k[2] == "l2_prefetch_kernel"),
           ("all-reduce", lambda k: k[2].startswith("oneshot_allreduce")))


def main(roots):
    out = {}
    for root in roots:
        table = {}
        for folder in sorted(glob.glob(os.path.join(root, "decode-trace", "*"))):
            ks = ft.load(sorted(glob.glob(os.path.join(folder, "*.json.gz")))[0])
            steps = ft.fl.steps(ks)[1:]
            six = [(t, d, a, b) for t, d, a, b in steps if any(k[4][:2] == (9, 36) for k in t)]
            acc = defaultdict(list)
            for t, d, a, b in six:
                for k in t + d:
                    for name, match in KERNELS:
                        if match(k):
                            acc[name].append(k[1])
                            break
            table[os.path.basename(folder)] = {"steps": len(six), "step_ms": round(statistics.mean((b - a) for _, _, a, b in six) / 1e3, 3),
                                               **{name: round(statistics.mean(v), 2) for name, v in acc.items()},
                                               "ms_a_step": {name: round(sum(v) / len(six) / 1e3, 3) for name, v in acc.items()}}
        out[os.path.basename(os.path.normpath(root))] = table
        nodes = list(table)
        print(f"\n{root}: mean us a call ({table[nodes[0]]['steps']} six-row steps)")
        print(f"  {'':32s}" + "".join(f"{n:>10s}" for n in nodes) + "   last against the others")
        for name in ["step_ms"] + [k for k, _ in KERNELS]:
            vals = [table[n][name] for n in nodes]
            others = statistics.mean(vals[:-1])
            print(f"  {name:32s}" + "".join(f"{v:10.2f}" for v in vals) + f"   {100 * (vals[-1] / others - 1):+6.1f}%")
        print("  ms a step on " + nodes[0] + ": " + ", ".join(f"{k} {v}" for k, v in table[nodes[0]]["ms_a_step"].items()))
    json.dump(out, sys.stdout if False else open("receipts/rank-compare.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
