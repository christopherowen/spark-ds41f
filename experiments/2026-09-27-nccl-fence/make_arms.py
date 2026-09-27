#!/usr/bin/env python3
"""Derive this experiment's arm config from config/cluster.json.

usage: experiments/2026-09-27-nccl-fence/make_arms.py   (from the deployment checkout)
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
base = json.loads((ROOT / "config/cluster.json").read_text())

cfg = copy.deepcopy(base)
cfg["container"]["image"] = "vllm-ds41f-kkref:01f1b874c774-r4c"
cfg["container"]["expected_labels"] = {
    "local.spark3.vllm.tree": "ee3a0fd4ab59883e4a9e15dfb101b42b09e47d00",
    "local.spark3.b12x.tree": "d661de3161ff474a92646974584d6f5eb1958f75",
    "local.spark3.nccl.tree": "47687d2a75b06fdff1b752dbf08bb87f12ca98bb",
}
args = cfg["serve_args"]
# Thinking is already the default when a request names neither key; the
# explicit default overrode a client's enable_thinking=false.
i = args.index("--default-chat-template-kwargs")
del args[i:i + 2]
cfg["environment"]["VLLM_CACHE_DIR"] = cfg["environment"]["VLLM_CACHE_ROOT"] = "/cache/kkref/jit/vllm-r4a"
(E / "cluster-r4c.json").write_text(json.dumps(cfg, indent=2) + "\n")
