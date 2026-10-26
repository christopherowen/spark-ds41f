#!/usr/bin/env python3
"""Write a kernel-local bundle that times DS4.1 decode steps kernel by kernel and phase by phase.

usage: step_bundle.py VLLM_REPO REF BUNDLE_DIR [--tp 1|3|4] [--requests N] [--tokens N]
                      [--ctas N] [--iters N] [--skip REGEX] [--image IMAGE --base COMMIT]

REF is a vLLM ref that carries the IKET debug mode: vllm/models/deepseek_v4_1/tilelang/iket.py,
the VLLM_DS41_IKET variables in vllm/envs.py and the hook in vllm/v1/worker/gpu_worker.py
(cherry-pick vLLM branch iket-debug's commits onto your branch). Every package file REF
changes from BASE, the image's vLLM tree, is mounted over it.

The bundle runs harness/iket_step.py on one GPU: the target's six layers that keep every
attention role and the DSpark drafter, at one TP4 (ring) or TP3 (triangle) rank's shapes
(``--tp``; 1 is the whole model's), with dummy weights and the production recipe's
kernels and CUDA graphs. ``--requests`` concurrent requests of ``--tokens`` tokens each
are measured after a warm request. Its steps:
1. on under run-iket: four CTAs a launch and the middle iteration of each loop record
   their ranges; every warp's lifetime is recorded;
2. the records against the dry run's per-warp count (trace_counts.py);
3. nsys of off with the graphs;
4. the measured requests' timeline and per-kernel, per-phase tables (iket_table.py
   --between iket_step_marker_kernel), with nsys durations beside them;
5. the nsys timeline (nsys_window.py).

On a test node, from a spark-ds41f checkout:

    python3 scripts/lab.py kernel-local BUNDLE_DIR --out OUT

OUT/output.txt holds the tables. The run's files are under the node's compile cache,
~/.cache/spark-lab/compile/iket/<bundle>-<UTC stamp>/: trace/ (run-iket's output; its
.pftrace opens in https://ui.perfetto.dev), table.json, nsys-graphs.sqlite. The node
needs DS4.1-Flash-1017's config in ~/.cache/huggingface/hub and Nsight Systems 2025.3.2
in /opt/nvidia/nsight-systems; a run takes about 15 minutes and 75 GB of memory.
"""

import argparse
import subprocess
from pathlib import Path

from make_bundles import BASE, IMAGE, step_spec, write_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("ref")
    parser.add_argument("bundle")
    parser.add_argument("--tp", type=int, choices=(1, 3, 4), default=4)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--ctas", type=int, default=4)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--skip", default="^(vision_|topk_topp)")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--base", default=BASE, help="the image's vLLM tree")
    args = parser.parse_args()
    iket = "vllm/models/deepseek_v4_1/tilelang/iket.py"
    if subprocess.run(["git", "-C", args.repo, "cat-file", "-e", f"{args.ref}:{iket}"],
                      capture_output=True).returncode:
        raise SystemExit(f"{args.ref} has no {iket}: cherry-pick vLLM branch iket-debug's commits")
    bundle = Path(args.bundle).resolve()
    spec = step_spec(args.ref, args.tp, args.requests, args.tokens, args.ctas, args.iters, args.skip)
    print(write_bundle(args.repo, bundle.name, spec, bundle, args.image, args.base))


if __name__ == "__main__":
    main()
