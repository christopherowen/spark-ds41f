"""Write the validation arms from the recipe (config/cluster.json).

- production-{tp4,tp3}.json: the recipe at the shape, unchanged.
- prefill-yield-{tp4,tp3}.json: production with `--prefill-yield short --prefill-yield-limit-s 10`,
  vLLM's scheduler, scheduler config and engine arguments mounted from overlay/ (the
  spark-ds41f-1025 image's tree, e0ccc837d, with vllm/spark-ds41f-1025/*.patch).
- slots-24-tp4.json / slots-12-tp3.json: more sequences, with the budget raised by their rows so
  the long-prefill threshold, and with it every chunk boundary, stays the recipe's.
- retention-2048-tp3.json: the triangle's prefix-cache retention interval 2,048 instead of 512.

Run from the repository root: python3 experiments/2026-10-26-concurrency-stalls/make_arms.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import recipe  # noqa: E402

EXPERIMENT = "experiments/2026-10-26-concurrency-stalls"
IMAGE_VLLM = "/opt/spark-ds41f/candidate/vllm"
OVERLAY = ("vllm/v1/core/sched/scheduler.py", "vllm/config/scheduler.py", "vllm/engine/arg_utils.py")
SHAPES = {"tp4": "tp4-ring", "tp3": "tp3-triangle"}


def base(shape: str) -> dict:
    return {"extends": {"config": "config/cluster.json", "shape": shape}}


def resolved(shape: str) -> dict:
    cluster = json.loads((ROOT / "config/cluster.json").read_text())
    return recipe.resolve(cluster, shape)


def prefill_yield(shape: str) -> dict:
    cluster = resolved(shape)
    mounts = [m for m in copy.deepcopy(cluster["container"]["mounts"]) if m[1] == "/cache"]
    mounts += [[f"{{checkout}}/{EXPERIMENT}/overlay/vllm/{path}", f"{IMAGE_VLLM}/{path}", "ro"] for path in OVERLAY]
    arm = base(shape)
    arm["notes"] = ("spark-ds41f-1025 with prefill yield (vLLM e0ccc837d + vllm/spark-ds41f-1025/0001-0002, "
                    "mounted from overlay/): a long prompt's aligned chunk gives its step to waiting prompts "
                    "that finish within one chunk and to held first tokens; 10 s limit.")
    arm["set"] = {"serve_args": cluster["serve_args"] + ["--prefill-yield", "short", "--prefill-yield-limit-s", "10"],
                  "container": {"mounts": mounts}}
    return arm


def slots(shape: str, seqs: int) -> dict:
    cluster = json.loads((ROOT / "config/cluster.json").read_text())
    entry = cluster["shapes"][shape]
    sizing = entry["sizing"]
    drafts = int(cluster["drafts"])
    threshold = int(sizing["max_num_batched_tokens"]) - int(sizing["max_num_seqs"]) * (1 + drafts)
    rows = seqs * (1 + drafts)
    arm = base(shape)
    arm["notes"] = (f"spark-ds41f-1025 with {seqs} sequences; budget {threshold + rows:,} so the long-prefill "
                    f"threshold stays {threshold:,} (chunk boundaries unchanged); graphs to {rows} rows.")
    resolved_args = resolved(shape)["serve_args"]
    compilation = json.loads(resolved_args[resolved_args.index("--compilation-config") + 1])
    compilation["cudagraph_capture_sizes"] = recipe.capture_sizes(rows)
    arm["serve_options"] = {
        "--max-num-seqs": str(seqs),
        "--max-num-batched-tokens": str(threshold + rows),
        "--max-cudagraph-capture-size": str(rows),
        "--compilation-config": json.dumps(compilation, separators=(",", ":")),
    }
    return arm


def arms() -> dict:
    out = {}
    for tag, shape in SHAPES.items():
        production = base(shape)
        production["notes"] = "spark-ds41f-1025, the recipe at this shape, unchanged."
        out[f"production-{tag}.json"] = production
        out[f"prefill-yield-{tag}.json"] = prefill_yield(shape)
    out["slots-24-tp4.json"] = slots("tp4-ring", 24)
    out["slots-12-tp3.json"] = slots("tp3-triangle", 12)
    retention = base("tp3-triangle")
    retention["notes"] = "spark-ds41f-1025 on the triangle with prefix-cache retention every 2,048 tokens (not 512)."
    retention["serve_options"] = {"--prefix-cache-retention-interval": "2048"}
    out["retention-2048-tp3.json"] = retention
    return out


def main() -> int:
    for name, arm in arms().items():
        (HERE / name).write_text(json.dumps(arm, indent=1) + "\n")
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
