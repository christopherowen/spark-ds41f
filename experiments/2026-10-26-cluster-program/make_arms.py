"""Write the program's arms from the recipe (config/cluster.json at tp4-ring).

- arm-1025.json: spark-ds41f-1025, the recipe at the ring, unchanged.
- arm-verify-all.json: 1025 with every one of the five drafts verified (adaptive
  verification off, its row cost dropped: the config refuses a row cost without it).
  A measurement arm: temperature-0 text does not depend on the verification policy.
- prof-1025.json: 1025 under the torch profiler without Python stacks or shapes (stacks
  inflate host gaps), with VLLM_SERVER_DEV_MODE=1 so the engine grid runs on the same boot.

Run from the repository root: python3 experiments/2026-10-26-cluster-program/make_arms.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import recipe  # noqa: E402

EXPERIMENT = "experiments/2026-10-26-cluster-program"
PROFILES = "/cache/spark-ds41f/profiles/cluster-program"


def resolved() -> dict:
    return recipe.resolve(json.loads((ROOT / "config/cluster.json").read_text()), "tp4-ring")


def argument(args: list[str], flag: str) -> str:
    if args.count(flag) != 1:
        raise SystemExit(f"the recipe passes {flag} {args.count(flag)} times")
    return args[args.index(flag) + 1]


def write(name: str, arm: dict) -> None:
    (HERE / name).write_text(json.dumps(arm, indent=1) + "\n")
    print(name)


def main() -> None:
    args = resolved()["serve_args"]
    write("arm-1025.json", {"notes": "spark-ds41f-1025: the recipe at the ring, unchanged.",
                            "extends": {"config": "config/cluster.json", "shape": "tp4-ring"}})
    speculative = json.loads(argument(args, "--speculative-config"))
    if not speculative.get("enable_adaptive_verification"):
        raise SystemExit("the recipe already verifies every draft")
    speculative["enable_adaptive_verification"] = False
    speculative.pop("adaptive_verification_row_cost", None)
    write("arm-verify-all.json", {
        "notes": "Measurement arm, not a candidate: spark-ds41f-1025 with every one of the five drafts "
                 "verified (adaptive verification off and its row cost dropped). Accepted drafts by "
                 "position over drafts are the drafter's survival on the text; temperature-0 text does "
                 "not depend on the verification policy.",
        "extends": {"config": f"{EXPERIMENT}/arm-1025.json"},
        "serve_options": {"--speculative-config": json.dumps(speculative, separators=(",", ":"))}})
    if "--profiler-config" in args:
        raise SystemExit("the recipe already passes --profiler-config")
    profiler = {"profiler": "torch", "torch_profiler_dir": PROFILES, "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": False, "ignore_frontend": True,
                "torch_profiler_use_gzip": True, "torch_profiler_dump_cuda_time_total": False}
    write("prof-1025.json", {
        "notes": "spark-ds41f-1025 under the torch profiler (every rank; no Python stacks or shapes, "
                 "which inflate host gaps; no summary table at /stop_profile), with "
                 "VLLM_SERVER_DEV_MODE=1 for the engine grid on the same boot. serve_args are the "
                 "recipe's resolved list at tp4-ring with --profiler-config appended.",
        "extends": {"config": f"{EXPERIMENT}/arm-1025.json"},
        "set": {"serve_args": [*args, "--profiler-config", json.dumps(profiler)],
                "environment": {"VLLM_SERVER_DEV_MODE": "1"}}})


if __name__ == "__main__":
    main()
