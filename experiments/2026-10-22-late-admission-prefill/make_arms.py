#!/usr/bin/env python3
"""Write the arms and run specs of the late-admission prefill fix (vLLM 0226).

usage: make_arms.py [--staged DIR]

Arms, each extending the spark-ds41f-1022 candidate at the recipe's tp3-triangle:
- fix-1022.json: the candidate with overlay/ (0226's three changed sources) mounted over
  the image's vLLM tree (c282fa51, 0226's base);
- cur-1022.json: the candidate as built (image -1022-2);
- la-off-1022.json: the candidate with late admission off (VLLM_LATE_ADMISSION_LAYER unset),
  the most any change to late admission could recover;
- sink-off-1022.json: the candidate with sink rejection off (VLLM_SINK_REJECTION=0): the
  detector's forward cost and the held first token;
- r6x.json: production r6x at the triangle.

Specs:
- ab-tp3.json: fix, current, fix again; each cold 32K/131K prefill (3 repeats), the traced
  TTFT replay of experiments/2026-10-20-ttft-gap and token_determinism_text.py --wide with and
  without log probabilities;
- attribution-tp3.json: cold 32K/131K prefill only (3 repeats) on late admission off, sink
  rejection off and r6x, queued behind ab-tp3.json in the same window, to split the
  candidate's per-chunk cost among its pieces.

With --staged DIR (repository-relative, e.g. results/private/lab/la-prefill) the files go
to DIR with the overlay mounts and the experiment pointing into DIR, to run before this
experiment is merged once DIR (with overlay/) is on dgx1, dgx2 and dgx3.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = "experiments/2026-10-22-late-admission-prefill"
CANDIDATE = {"config": "experiments/2026-10-22-spark-ds41f-1022/spark-ds41f-1022-tp3.json"}
R6X = {"config": "experiments/2026-10-19-r6x-tp3-1m/r6x-tp3-1m-1017.json"}
FILES = ("vllm/v1/engine/core.py", "vllm/v1/core/sched/interface.py",
         "vllm/v1/core/sched/scheduler.py")
PREFILL = ["--compare", "none", "--suites", "prefill", "--prefill-text", "source",
           "--prefill-sizes", "32768,131072", "--prefill-repeats", "3"]
DETERMINISM = "experiments/2026-10-12-r6g/token_determinism_text.py"
EXTRAS = [
    [DETERMINISM, ["--tokens", "192", "--long-lines", "700", "--wide"], "determinism"],
    [DETERMINISM, ["--tokens", "192", "--long-lines", "700", "--wide", "--no-logprobs"], "greedy"],
    ["experiments/2026-10-20-ttft-gap/ttft_trace.py",
     ["--repo", ".", "--output", "results/private/lab/{run}-trace-{label}.json", "--analyze"],
     "trace"],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged")
    a = ap.parse_args()
    base = a.staged or EXPERIMENT
    out = ROOT / a.staged if a.staged else HERE
    out.mkdir(parents=True, exist_ok=True)
    mounts = [["{home}/projects/spark-ds41f/cache", "/cache", "rw"]] + [
        [f"{{home}}/projects/spark-ds41f/{base}/overlay/{f}", f"/opt/spark3/candidate/vllm/{f}", "ro"]
        for f in FILES]
    arms = {
        "fix-1022.json": {
            "notes": "spark-ds41f-1022 with vLLM 0226 (overlay/ over the image's tree c282fa51): the "
                     "step after a continuing prompt chunk is decided at once; late admission otherwise.",
            "extends": CANDIDATE, "set": {"container": {"mounts": mounts}}},
        "cur-1022.json": {
            "notes": "spark-ds41f-1022 as built (image -1022-2): the A/B's reference.",
            "extends": CANDIDATE},
        "la-off-1022.json": {
            "notes": "spark-ds41f-1022 with late admission off: the step after each is decided at once.",
            "extends": CANDIDATE, "unset": ["environment.VLLM_LATE_ADMISSION_LAYER"]},
        "sink-off-1022.json": {
            "notes": "spark-ds41f-1022 with sink detection and rejection off (VLLM_SINK_REJECTION=0).",
            "extends": CANDIDATE, "set": {"environment": {"VLLM_SINK_REJECTION": "0"}}},
        "r6x.json": {"notes": "production r6x at the triangle.", "extends": R6X},
    }
    ab_job = {"kind": "measure", "bench": True, "bench_args": PREFILL, "bracket": False,
              "extras": EXTRAS,
              "arms": [{"config": "fix-1022.json", "label": "fix"},
                       {"config": "cur-1022.json", "label": "cur"},
                       {"config": "fix-1022.json", "label": "fix-again"}]}
    attribution_job = {"kind": "measure", "bench": True, "bench_args": PREFILL, "bracket": False,
                       "extras": [],
                       "arms": [{"config": "la-off-1022.json", "label": "la-off"},
                                {"config": "sink-off-1022.json", "label": "sink-off"},
                                {"config": "r6x.json", "label": "r6x"}]}
    specs = {
        "ab-tp3.json": {"experiment": base, "run": "la-prefill-ab",
                        "jobs": [{"kind": "sync"}, ab_job]},
        "attribution-tp3.json": {"experiment": base, "run": "la-prefill-attribution",
                                 "jobs": [{"kind": "sync"}, attribution_job]},
    }
    for name, body in {**arms, **specs}.items():
        (out / name).write_text(json.dumps(body, indent=1) + "\n")
        print(out / name)


if __name__ == "__main__":
    main()
