#!/usr/bin/env python3
"""CPU-only execution of c37's actual budget/compaction methods for 16 vs 5.

Requires NumPy, not torch/CUDA. --source-file names the exact candidate-37
adaptive_verification.py (or '-' reads it from stdin). Only the two selected
method bodies and the literal exploration constant are compiled from that
hash-checked source; none of the serving module's imports or GPU code run.

The finite 0.95 confidences match c37's existing host-limit unit fixture. They
are representative inputs, NOT a snapshot of the failing serving step. This
demonstrates the host-layout consequence, not a NaN producer or GPU offsets.
"""

import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace


SOURCE_COMMIT = "5b7cfa37fee42f96bfafbe09b7d30e8c0eda8575"
SOURCE_SHA256 = "851eebdde10371bcf4c3d2c5a0d86a68ee917c70cd6d86f12e8386164eb11295"


def host_methods(source, filename, np):
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("source is not the exact candidate-37 adaptive_verification.py")
    module = ast.parse(source.decode(), filename=filename)
    manager = next(node for node in module.body
                   if isinstance(node, ast.ClassDef) and node.name == "AdaptiveVerificationManager")
    methods = [node for node in manager.body
               if isinstance(node, ast.FunctionDef) and node.name in {"get_num_tokens", "compact_batch"}]
    constant = next(node for node in module.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "EXPLORE_AFTER_IDLE_STEPS"
                            for target in node.targets))
    assert len(methods) == 2
    assert isinstance(constant.value, ast.Constant)
    namespace = {"np": np, "logger": logging.getLogger(__name__)}
    selected = ast.Module(body=[constant, *methods], type_ignores=[])
    exec(compile(selected, filename, "exec"), namespace)
    return namespace["get_num_tokens"], namespace["compact_batch"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", required=True)
    args = parser.parse_args()
    source = sys.stdin.buffer.read() if args.source_file == "-" else Path(args.source_file).read_bytes()
    import numpy as np

    get_num_tokens, compact_batch = host_methods(source, args.source_file, np)
    cases = []
    for host_limit in (16, 5, None):
        manager = SimpleNamespace(
            num_speculative_steps=16, num_draft_steps=5,
            _slot_draft_limits=None if host_limit is None else np.array([host_limit], dtype=np.int32),
            _stale_confidences=[SimpleNamespace(np=np.full((1, 16), 0.95, dtype=np.float32))],
            _stale_idx=0, _max_total_logits=1024, num_bonus_tokens=1, row_cost=0.02,
            req_states=SimpleNamespace(
                req_id_to_index={"request": 0},
                num_computed_tokens_np=np.array([64], dtype=np.int32),
                prefill_len=SimpleNamespace(np=np.array([32], dtype=np.int32)),
            ),
        )
        tokens = get_num_tokens(manager, {"request": 17}, {"request": [-1] * 16})
        compacted, cpu_logits_upper_bound = compact_batch(
            manager, np.array([16], dtype=np.int32), np.array([17], dtype=np.int32),
            np.array([0, 17], dtype=np.int32),
        )
        expected = 6 if host_limit == 5 else 17
        assert tokens == expected and compacted.tolist() == [expected]
        assert manager._batch_budget[2] == expected - 1
        cases.append({
            "host_limit": host_limit, "scheduled_drafts": 16,
            "eligible_drafts": manager._batch_budget[0]["request"],
            "budget": manager._batch_budget[2], "num_tokens_for_dispatch": tokens,
            "compacted_counts": compacted.tolist(),
            "host_query_prefix": [0, *np.cumsum(compacted).tolist()],
            "cpu_logits_upper_bound": cpu_logits_upper_bound.tolist(),
        })
    print(json.dumps({
        "source_commit": SOURCE_COMMIT, "source_sha256": SOURCE_SHA256,
        "numpy_version": np.__version__, "cases": cases,
        "gpu_executed": False,
        "scope": "Actual c37 host budget and compaction bodies only; no graph dispatch, GPU metadata, collective or model forward.",
        "evidence_limit": "Representative finite confidences; no NaN is inserted or inferred, and actual failing-step confidences were not captured.",
    }, indent=2))


if __name__ == "__main__":
    main()
