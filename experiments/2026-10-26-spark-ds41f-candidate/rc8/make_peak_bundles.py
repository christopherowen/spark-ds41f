#!/usr/bin/env python3
"""Candidate 7 against candidate 8 on one GPU: PyTorch's peak memory at each shape's prefill
budget, and the three-request batch that found 0533's bug.

usage: make_peak_bundles.py IKET_STEP_PY

Writes bundles/moe-peak-next7 and bundles/moe-peak-next8, the same steps on each image.
IKET_STEP_PY is the IKET record's step harness (spark-ds41f #610,
experiments/2026-10-26-iket-debug/harness/iket_step.py): DS4.1-Flash-1017's config with
dummy weights, six layers at one rank's TP4 or TP3 shapes, at TP1, its engine in the
process with the production backends, DSpark and graphs. It is copied in unchanged.

Steps, each a fresh process (peak.py runs the harness, then prints PyTorch's peak
allocated and reserved memory, which vLLM's profile run at max_num_batched_tokens sets):
1. TP4 shape, a 16,384-token budget (threshold 16,288), one request;
2. TP3 shape, an 8,192-token budget (threshold 8,144), one request;
3. TP4 shape, three requests, eager: next-7 asserts on non-finite gate logits (the
   routed output in the workspace arena), next-8 must finish (0533).
At TP1 no all-reduce copies the MoE output and nothing runs sequence-parallel, so the
output 0533 allocates is written here; at TP3 and TP4 prefill it is allocated beside the
exchange's slots. Either way it is one hidden-sized BF16 tensor a layer call.
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = {"next7": "spark-ds41f:19f2c20ed4d6-next-7", "next8": "spark-ds41f:19f2c20ed4d6-next-8"}
ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    # The image's own path (its vLLM and B12X trees) after the bundle's: without it the
    # base image's vLLM in dist-packages is imported.
    "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x",
    "MAX_JOBS": "8",
    "VLLM_PLUGINS": "",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_DS41_HEAD_DTYPE": "float32",
    "VLLM_DS41_PACKED_BF16_LM_HEAD": "1",
    "VLLM_L2_PREFETCH_KERNELS": "tilelang",
    "VLLM_L2_PREFETCH_GRID": "2",
    "SPARK_DS41F_DSPARK_DEAD_ROWS_TAU": "0.2",
    "SPARK_DS41F_DSPARK_MAIN_PROJ_TP": "1",
    "SPARK_DS41F_DSPARK_VOCAB_PARALLEL": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.6",
}
STEPS = [
    ("TP4 shape, 16,384 budget, c1", ["--tp-shape", "4", "--batched", "16384", "--chunk", "16288"], "tp4-16384"),
    ("TP3 shape, 8,192 budget, c1", ["--tp-shape", "3", "--batched", "8192", "--chunk", "8144"], "tp3-8192"),
    ("TP4 shape, c3, eager", ["--tp-shape", "4", "--requests", "3", "--eager"], "tp4-c3-eager"),
]
PEAK = '''#!/usr/bin/env python3
"""The step harness (iket_step.py) in this process, then PyTorch's peak memory.

usage: peak.py OUT.json iket_step.py-arguments...
"""
import json
import sys
import traceback

import torch

import iket_step

out = sys.argv[1]
sys.argv = ["iket_step.py", *sys.argv[2:]]
error = None
try:
    iket_step.main()
except BaseException as exc:  # an assertion is a result here: record it, then fail
    error = f"{type(exc).__name__}: {exc}"[:400]
    traceback.print_exc()
gib = 2**30
stats = {"summary": "peak memory", "argv": sys.argv[1:], "error": error,
         "max_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 3),
         "max_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 3)}
open(out, "w").write(json.dumps(stats) + "\\n")
print(json.dumps(stats), flush=True)
sys.exit(1 if error else 0)
'''
RUN = '''#!/usr/bin/env python3
"""Run steps.json in order, each with its timeout; print every step's exit status and fail
if any step failed. {python} is this interpreter, {out} the output root."""
import json
import os
import subprocess
import sys
import time

OUT = f"/c/moe-peak/{os.environ['PEAK_BUNDLE']}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
os.makedirs(OUT)
print(f"=== output {OUT}", flush=True)
codes = []
for label, argv, timeout in json.load(open("/b/steps.json")):
    argv = [a.replace("{python}", sys.executable).replace("{out}", OUT) for a in argv]
    print(f"=== {label}", flush=True)
    started = time.monotonic()
    try:
        code = subprocess.run(argv, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    print(f"=== {label}: exit {code} in {time.monotonic() - started:.0f} s", flush=True)
    codes.append((label, code))
print(json.dumps({"output": OUT, "steps": codes}), flush=True)
sys.exit(0 if all(code == 0 for _, code in codes) else 1)
'''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    harness = Path(sys.argv[1])
    for name, image in IMAGES.items():
        bundle = HERE / "bundles" / f"moe-peak-{name}"
        if bundle.exists():
            raise SystemExit(f"{bundle} exists")
        bundle.mkdir(parents=True)
        shutil.copyfile(harness, bundle / "iket_step.py")
        (bundle / "peak.py").write_text(PEAK, encoding="utf-8")
        (bundle / "run.py").write_text(RUN, encoding="utf-8")
        steps = [[label, ["{python}", "/b/peak.py", f"{{out}}/{stem}.json", "off", *args,
                          "--out", f"{{out}}/{stem}"], 2400] for label, args, stem in STEPS]
        (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n", encoding="utf-8")
        candidate = {
            "description": (f"Candidate {name[-1]}'s image on one GPU: PyTorch's peak memory at TP4's "
                            "16,384 and TP3's 8,192 prefill budgets, and the three-request batch "
                            "of 0533's bug (make_peak_bundles.py)."),
            "image": image,
            "env": {**ENV, "PEAK_BUNDLE": f"moe-peak-{name}",
                    "TILELANG_CACHE_DIR": f"/c/tilelang-moe-peak-{name}",
                    "TRITON_CACHE_DIR": f"/c/triton-moe-peak-{name}",
                    "VLLM_CACHE_ROOT": f"/c/vllm-moe-peak-{name}"},
            "workdir": "vllm",
            "argv": ["/b/run.py"],
            "verdict": "status",
            "mounts": [["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"],
                       ["peak.py", "/b/peak.py"], ["iket_step.py", "/b/iket_step.py"],
                       ["~/.cache/huggingface/hub", "/hf/hub"]],
        }
        (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {bundle.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
