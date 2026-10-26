#!/usr/bin/env python3
"""Candidate 9 against candidates 7 and 8 on one GPU: PyTorch's peak memory at each shape's
prefill budget, and outputs bit for bit (run 3 of rc8's measurement).

usage: make_peak_bundles.py IKET_STEP_PY

Writes bundles/moe-peak3-next7, -next8 and -next9, the same steps on each image, from
../rc8/make_peak_bundles.py's harness wrapper (peak.py), runner, environment and run-2
harness change (max_model_len from PEAK_MAX_MODEL_LEN, 32,768). IKET_STEP_PY is #610's step
harness (experiments/2026-10-26-iket-debug/harness/iket_step.py) at 63fca56c, the version
runs 1 and 2 copied (later versions take the recipe's sizing).

Steps, each a fresh process:
1-4. TP4 shape at a 16,384-token budget and TP3 shape at 8,192, one request, twice each:
   the peak allocated and reserved memory (vLLM's profile run at the budget sets them);
5. TP4 shape, one request with the graphs, --filled (the harness's deterministic fill of
   tensors the dummy engine reads unwritten), 16 tokens;
6. TP4 shape, three requests, eager, --filled: next-7 asserts on non-finite gate logits
   (0533's bug); next-8 and next-9 finish.
Steps 5 and 6 write the measured request's token ids and top-5 log probabilities
(result.json); compare_results.py compares them between images.
"""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rc8_peak", HERE.parent / "rc8/make_peak_bundles.py")
rc8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc8)

IMAGES = {
    "next7": "spark-ds41f:19f2c20ed4d6-next-7",
    "next8": "spark-ds41f:19f2c20ed4d6-next-8",
    "next9": "spark-ds41f:19f2c20ed4d6-next-9",
}
STEPS = (
    [(f"TP4 shape, 16,384 budget, c1, repeat {n}", rc8.TP4, f"tp4-16384-{n}") for n in (1, 2)]
    + [(f"TP3 shape, 8,192 budget, c1, repeat {n}", rc8.TP3, f"tp3-8192-{n}") for n in (1, 2)]
    + [("TP4 shape, c1, graphs, filled, 16 tokens",
        ["--tp-shape", "4", "--filled", "--tokens", "16"], "tp4-c1-filled"),
       ("TP4 shape, c3, eager, filled", ["--tp-shape", "4", "--requests", "3", "--eager", "--filled"],
        "tp4-c3-eager-filled")]
)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    source = Path(sys.argv[1]).read_text(encoding="utf-8")
    if source.count(rc8.MAX_MODEL_LEN[0]) != 1:
        raise SystemExit("the harness does not set max_model_len=8192 once")
    source = source.replace(*rc8.MAX_MODEL_LEN)
    for name, image in IMAGES.items():
        bundle = HERE / "bundles" / f"moe-peak3-{name}"
        if bundle.exists():
            raise SystemExit(f"{bundle} exists")
        bundle.mkdir(parents=True)
        (bundle / "iket_step.py").write_text(source, encoding="utf-8")
        (bundle / "peak.py").write_text(rc8.PEAK, encoding="utf-8")
        (bundle / "run.py").write_text(rc8.RUN, encoding="utf-8")
        steps = [[label, ["{python}", "/b/peak.py", f"{{out}}/{stem}.json", "off", *args,
                          "--out", f"{{out}}/{stem}"], 2400] for label, args, stem in STEPS]
        (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n", encoding="utf-8")
        candidate = {
            "description": (f"{image} on one GPU: PyTorch's peak memory at TP4's 16,384 and TP3's "
                            "8,192 prefill budgets, and two filled runs whose outputs compare "
                            "between images (rc9/make_peak_bundles.py, run 3)."),
            "image": image,
            "env": {**rc8.ENV, "PEAK_BUNDLE": f"moe-peak3-{name}", "PEAK_MAX_MODEL_LEN": "32768",
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
