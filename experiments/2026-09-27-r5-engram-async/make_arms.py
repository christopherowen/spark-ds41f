#!/usr/bin/env python3
"""Derive this experiment's arm config from config/cluster.json.

usage: experiments/2026-09-27-r5-engram-async/make_arms.py   (from the deployment checkout)
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
base = json.loads((ROOT / "config/cluster.json").read_text())

cfg = copy.deepcopy(base)
cfg["container"]["image"] = "vllm-ds41f-kkref:04c30fa98e79-r5c"
cfg["container"]["expected_labels"] = {
    "local.spark3.vllm.tree": "f250542a825081fa1d0d579146366d6d1686cbc4",
    "local.spark3.b12x.tree": "f77d175f0605b1cedae57caf8f00d535198318c2",
    "local.spark3.nccl.tree": "47687d2a75b06fdff1b752dbf08bb87f12ca98bb",
}
args = cfg["serve_args"]
# Thinking is already the default when a request names neither key; the
# explicit default overrode a client's enable_thinking=false.
i = args.index("--default-chat-template-kwargs")
del args[i:i + 2]
environment = cfg["environment"]
# Patch 0002's asynchronous rows (SPARK3_ENGRAM_ASYNC=1, already set) replace
# the base's Engram overlap, which must be off.
environment["VLLM_DS41_ENGRAM_OVERLAP"] = "0"
environment["VLLM_CACHE_DIR"] = environment["VLLM_CACHE_ROOT"] = "/cache/kkref/jit/vllm-r5c"
cfg["environment"] = dict(sorted(environment.items()))
(E / "cluster-r5c.json").write_text(json.dumps(cfg, indent=2) + "\n")

# Five drafts, the drafter's trained block, with graphs for eight streams at
# 1 + 5 rows each; adaptive verification as promoted (cost scale 2.0).
k5 = copy.deepcopy(cfg)
args = k5["serve_args"]
i = args.index("--speculative-config")
speculative = json.loads(args[i + 1])
speculative["num_speculative_tokens"] = 5
args[i + 1] = json.dumps(speculative, separators=(",", ":"))
args[args.index("--max-cudagraph-capture-size") + 1] = "48"
i = args.index("--compilation-config")
compilation = json.loads(args[i + 1])
compilation["cudagraph_capture_sizes"] += [40, 48]
args[i + 1] = json.dumps(compilation, separators=(",", ":"))
k5["environment"]["VLLM_CACHE_DIR"] = k5["environment"]["VLLM_CACHE_ROOT"] = "/cache/kkref/jit/vllm-r5c-k5"
(E / "cluster-r5c-k5.json").write_text(json.dumps(k5, indent=2) + "\n")
