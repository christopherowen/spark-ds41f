#!/usr/bin/env python3
"""Write the budget sweep's arms: candidate 6 at tp3-triangle (rc6/rc-tp3.json, 8,192)
at each long-prefill budget, every other setting resolved from that arm, plus the
diagnostic memory trace (memory/overlays: the release's gpu_worker.py with an allocator
sampler thread, phase marks and pool summaries; serving behaviour unchanged) and the
workspace manager's slot log (VLLM_DEBUG_WORKSPACE=1).

An arm overlays a resolved configuration, so a budget is a whole serve_args list (as
rc6/rc-tp3-6144.json is): this script takes the base arm's list and changes only the
budget and its threshold (budget - streams x 6 rows), the rule scripts/recipe.py applies.

usage (from the repository root): experiments/2026-10-26-tp3-budget-memory/make_arms.py
"""
import importlib.machinery
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent.relative_to(ROOT)
BASE = "experiments/2026-10-25-spark-ds41f-candidate/rc6/rc-tp3.json"
BUDGETS = (6144, 8192, 12288)
OVERLAY = "memory/overlays/vllm_v1_worker_gpu_worker.py"
WORKER = "/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu_worker.py"


def spark():
    loader = importlib.machinery.SourceFileLoader("spark_cli", str(ROOT / "bin/spark"))
    spec = importlib.util.spec_from_loader("spark_cli", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def with_value(args: list[str], flag: str, value: int) -> list[str]:
    out = list(args)
    at = out.index(flag)
    out[at + 1] = str(value)
    return out


def main() -> None:
    base = spark().read_cluster(BASE)
    args = base["serve_args"]
    seqs = int(args[args.index("--max-num-seqs") + 1])
    rows = seqs * (1 + 5)
    assert int(args[args.index("--max-num-batched-tokens") + 1]) == 8192
    assert int(args[args.index("--long-prefill-token-threshold") + 1]) == 8192 - rows
    mounts = [m for m in base["container"]["mounts"] if m[1] == "/cache"]
    assert len(mounts) == 1, base["container"]["mounts"]
    for budget in BUDGETS:
        serve = with_value(args, "--max-num-batched-tokens", budget)
        serve = with_value(serve, "--long-prefill-token-threshold", budget - rows)
        arm = {
            "notes": (
                f"Candidate 6 at tp3-triangle (rc6/rc-tp3.json) at a {budget:,}-token long-prefill "
                f"budget (threshold {budget - rows:,}), everything else unchanged, with the "
                "diagnostic memory trace of experiments/2026-10-26-tp3-budget-memory/memory "
                "(allocator totals every 0.1 s, start-up phase marks, pool summaries; serving "
                "behaviour unchanged) and the workspace manager's slot log."
            ),
            "extends": {"config": BASE},
            "set": {
                "serve_args": serve,
                "container": {
                    "mounts": [
                        *mounts,
                        [f"{{home}}/projects/spark-ds41f/{HERE}/{OVERLAY}", WORKER, "ro"],
                    ]
                },
                "environment": {
                    "SPARK_DS41F_MEMORY_TRACE": f"/cache/tp3-budget-memory/b{budget}",
                    "SPARK_DS41F_MEMORY_TRACE_SECONDS": "2700",
                    "SPARK_DS41F_MEMORY_TRACE_POOLS_AFTER": "600",
                    "VLLM_DEBUG_WORKSPACE": "1",
                },
            },
        }
        path = ROOT / HERE / f"budget-{budget}-tp3.json"
        path.write_text(json.dumps(arm, indent=1) + "\n")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
