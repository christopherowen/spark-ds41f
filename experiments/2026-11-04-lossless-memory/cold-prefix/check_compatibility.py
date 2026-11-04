#!/usr/bin/env python3
"""Audit the pinned model's cold-storage prerequisites without starting CUDA."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def audit(repo, tree):
    hashes = {}

    def read(path):
        raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{tree}:{path}"])
        hashes[path] = hashlib.sha256(raw).hexdigest()
        return ast.parse(raw, filename=path)

    config = read("vllm/config/vllm.py")
    method = next(n for n in ast.walk(config) if isinstance(n, ast.FunctionDef)
                  and n.name == "use_request_boundary_checkpoints")
    supported = set()
    for node in ast.walk(method):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute)
                and node.left.attr == "model_type"):
            for operation, comparator in zip(node.ops, node.comparators):
                if isinstance(operation, ast.In) and isinstance(comparator, (ast.Tuple, ast.List)):
                    supported.update(v.value for v in comparator.elts
                                     if isinstance(v, ast.Constant) and isinstance(v.value, str))
    model = read("vllm/transformers_utils/configs/deepseek_v41.py")
    model_type = next(n.value.value for n in ast.walk(model) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "model_type" for t in n.targets)
                      and isinstance(n.value, ast.Constant))
    attention = read("vllm/models/deepseek_v4_1/attention.py")
    private_windows = any(isinstance(n, ast.keyword) and n.arg == "prefix_cache_enabled"
                          and isinstance(n.value, ast.UnaryOp) and isinstance(n.value.op, ast.Not)
                          for n in ast.walk(attention))
    compressor = read("vllm/models/deepseek_v4_1/compressor.py")
    circular_state = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                         and n.func.id == "CircularBufferSpec" for n in ast.walk(compressor))
    shared = read("vllm/v1/kv_offload/cpu/shared_offload_region.py")
    host_local = any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                     and n.value.startswith("/dev/shm/vllm_offload_") for n in ast.walk(shared))
    tier = read("vllm/v1/kv_offload/tiering/spec.py")
    scheduler_buffer = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                           and n.func.id == "SharedOffloadRegion"
                           and any(k.arg == "rank" and isinstance(k.value, ast.Constant)
                                   and k.value.value is None for k in n.keywords)
                           for n in ast.walk(tier))
    if not supported or not all((private_windows, circular_state, host_local, scheduler_buffer)):
        raise ValueError("source contract differs; review the new source before drawing a conclusion")
    return {"schema_version": 1, "scope": "static compatibility audit; no performance measurement",
            "vllm_tree": tree, "model_type": model_type,
            "generic_boundary_supported_model_types": sorted(supported),
            "generic_boundary_supports_this_model": model_type in supported,
            "private_attention_windows": private_windows, "compressor_circular_state": circular_state,
            "filesystem_tier_uses_host_local_scheduler_buffer": host_local and scheduler_buffer,
            "source_sha256": hashes,
            "serving_candidate_qualified": False,
            "decision": "defer serving integration until model-specific and per-rank transfer gates pass",
            "required_gates": [
                "Define complete target/draft capture, or prove ordinary prefix replay rebuilds every private state.",
                "Export only committed written spans of complete cacheable blocks; exclude allocation padding.",
                "Prove every node's shard is persisted and restored; a head-local mmap is not that proof.",
                "Reserve destinations and wait for every rank before publishing an external prefix hit.",
                "Verify byte identity, continuation equivalence and TTFT after forced GPU/CPU cache eviction."
            ], "command": sys.argv}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads((HERE.parent / "base.json").read_text())
    result = audit(args.repo, base["prepared_vllm_tree"])
    with args.out.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
