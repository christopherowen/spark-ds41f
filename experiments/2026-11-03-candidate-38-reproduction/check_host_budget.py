#!/usr/bin/env python3
"""Execute pinned c37/c38 host budget and compaction bodies only, on CPU."""
import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

import numpy as np

PINS = {
    "candidate37": ("5b7cfa37fee42f96bfafbe09b7d30e8c0eda8575", "851eebdde10371bcf4c3d2c5a0d86a68ee917c70cd6d86f12e8386164eb11295"),
    "candidate38": ("1f33102547e4da5c64712390604695d1abd577c4", "54c2e186cdc17aea950d3f8d6549b1f463cc780c836dac75f6c33eaf9d45c7d1"),
}
SOURCE = "vllm/v1/worker/gpu/spec_decode/adaptive_verification.py"


def methods(repo, name):
    commit, expected_hash = PINS[name]
    data = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{SOURCE}"])
    assert hashlib.sha256(data).hexdigest() == expected_hash
    module = ast.parse(data, filename=f"{commit}:{SOURCE}")
    manager = next(n for n in module.body if isinstance(n, ast.ClassDef)
                   and n.name == "AdaptiveVerificationManager")
    selected = [n for n in manager.body if isinstance(n, ast.FunctionDef)
                and n.name in {"get_num_tokens", "compact_batch"}]
    constant = next(n for n in module.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "EXPLORE_AFTER_IDLE_STEPS"
                            for t in n.targets))
    assert len(selected) == 2 and isinstance(constant.value, ast.Constant)
    namespace = {"np": np, "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[constant, *selected], type_ignores=[]),
                 f"{commit}:{SOURCE}", "exec"), namespace)
    return namespace["get_num_tokens"], namespace["compact_batch"]


def fail_if_read():
    raise AssertionError("the removed rank-local limit callback was read")


def one_rank(functions, scheduled, slots, confidence, old_limits):
    req_ids = [f"request-{i}" for i in range(len(scheduled))]
    max_reqs = max(slots) + 1
    manager = SimpleNamespace(
        num_speculative_steps=16, num_draft_steps=5,
        _stale_confidences=[SimpleNamespace(np=np.full((max_reqs, 16), confidence, dtype=np.float32))],
        _stale_idx=0, _max_total_logits=1024, num_bonus_tokens=1, row_cost=0.02,
        req_states=SimpleNamespace(
            req_id_to_index=dict(zip(req_ids, slots, strict=True)),
            num_computed_tokens_np=np.full(max_reqs, 64, dtype=np.int32),
            prefill_len=SimpleNamespace(np=np.full(max_reqs, 32, dtype=np.int32)),
        ),
    )
    if old_limits != "absent":
        manager._slot_draft_limits = (fail_if_read if old_limits == "must-not-read"
                                     else np.full(max_reqs, old_limits, dtype=np.int32))
    drafts = np.array(scheduled, dtype=np.int32)
    counts = drafts + 1
    tokens = functions[0](manager, dict(zip(req_ids, counts.tolist(), strict=True)),
                          dict(zip(req_ids, [[-1] * n for n in scheduled], strict=True)))
    compacted, logits_upper = functions[1](manager, drafts, counts,
                                          np.array([0, *np.cumsum(counts)], dtype=np.int32))
    return {
        "eligible": manager._batch_budget[0],
        "draft_budget": manager._batch_budget[2],
        "dispatch_input_count": tokens,
        "compacted_counts": compacted.tolist(),
        "cpu_query_prefix": [0, *np.cumsum(compacted).tolist()],
        "cpu_logits_upper_bound": logits_upper.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = methods(args.source_repo, "candidate37")
    current = methods(args.source_repo, "candidate38")
    mirrors = [16, 5, 16]
    negative_control = [one_rank(old, [16], [0], .95, limit) for limit in mirrors]
    assert [r["dispatch_input_count"] for r in negative_control] == [17, 6, 17]
    cases = []
    for name, scheduled, slots, confidence in [
        ("original-wide-budget", [16], [0], .95),
        ("narrow-budget", [5], [0], .95),
        ("no-drafts", [0], [0], .95),
        ("mixed-width-reordered-slots", [16, 5], [2, 0], .95),
        ("low-confidence-budget", [16, 5], [2, 0], .20),
    ]:
        ranks = [one_rank(current, scheduled, slots, confidence, limit) for limit in mirrors]
        assert ranks[0] == ranks[1] == ranks[2]
        assert one_rank(current, scheduled, slots, confidence, "absent") == ranks[0]
        assert one_rank(current, scheduled, slots, confidence, "must-not-read") == ranks[0]
        if name == "original-wide-budget":
            assert [r["dispatch_input_count"] for r in ranks] == [17, 17, 17]
        cases.append({"name": name, "scheduled_drafts": scheduled,
                      "slots": slots, "planned_confidence": confidence,
                      "rank_results": ranks, "absent_and_forbidden_callback_checks": "pass"})
    result = {
        "source_repo": str(args.source_repo.resolve()),
        "source_path": SOURCE,
        "source_pins": {name: {"commit": pin, "sha256": digest} for name, (pin, digest) in PINS.items()},
        "python": platform.python_version(), "numpy": np.__version__,
        "old_rank_limit_mirrors": mirrors,
        "candidate37_negative_control": negative_control,
        "candidate38_cases": cases,
        "scope": "Actual pinned host get_num_tokens and compact_batch bodies; simulated rank inputs on CPU. No serving module imports, GPU, graph selection/enqueue, device offsets, collectives, or model forward.",
        "evidence_limit": "Representative finite confidence inputs, not captured production confidences. Reproduces the old host layout mismatch and checks its removal; does not reproduce or rule out a full-model NaN.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"candidate37_counts": [r["dispatch_input_count"] for r in negative_control],
                      "candidate38_counts": [r["dispatch_input_count"] for r in cases[0]["rank_results"]],
                      "candidate38_cases_passed": len(cases), "gpu_executed": False,
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
