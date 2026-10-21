#!/usr/bin/env python3
"""Write the late-admission arms and the screen spec.

usage: make_arms.py LAYER [--staged DIR]

- late-tp3.json: r6x on the triangle (the recipe at tp3-triangle) with overlay/ mounted
  over the image's vLLM sources and VLLM_LATE_ADMISSION_LAYER=LAYER;
- r6x-tp3.json: the recipe at tp3-triangle, unchanged (the reference);
- screen-tp3.json: candidate then reference, each measured with the traced replay of
  experiments/2026-10-20-ttft-gap, servebench c4 and c8 (30 s warm-up, 120 s) and the
  determinism text check.

With --staged DIR (repository-relative, e.g. results/private/lab/late-admission) the files
go to DIR with every path pointing into DIR: overlay mounts, scripts and the experiment.
That copy runs before the experiment is merged, once DIR (with overlay/ and the ttft-gap
scripts) is on every node.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT = "experiments/2026-10-21-late-admission"
TTFT = "experiments/2026-10-20-ttft-gap"
FILES = ("vllm/envs.py", "vllm/v1/engine/core.py", "vllm/v1/worker/gpu/model_runner.py",
         "vllm/v1/worker/gpu/late_admission.py")
EXTRA_BODY = json.dumps({"chat_template_kwargs": {"thinking": False}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", type=int)
    ap.add_argument("--staged")
    a = ap.parse_args()
    base = a.staged or EXPERIMENT
    scripts = a.staged or TTFT
    out = HERE.parents[1] / a.staged if a.staged else HERE
    out.mkdir(parents=True, exist_ok=True)
    recipe = {"config": "config/cluster.json", "shape": "tp3-triangle"}
    mounts = [["{home}/projects/spark-ds41f/cache", "/cache", "rw"]] + [
        [f"{{home}}/projects/spark-ds41f/{base}/overlay/{f}", f"/opt/spark3/candidate/vllm/{f}", "ro"]
        for f in FILES]
    late = {"notes": f"r6x on the triangle at 1M with vLLM 0222 (overlay/, vLLM 91eaaa59e on r6x 7316c7f9f, "
                     f"the image's tree): late admission at decoder layer {a.layer}.",
            "extends": recipe,
            "set": {"container": {"mounts": mounts},
                    "environment": {"VLLM_LATE_ADMISSION_LAYER": str(a.layer)}}}
    reference = {"notes": "r6x on the triangle at 1M, unchanged: the late-admission screen's reference.",
                 "extends": recipe, "set": {}}

    def servebench(streams):
        return ["scripts/servebench_run.py",
                ["--mode", "closed", "--concurrency", str(streams), "--warmup", "30", "--duration", "120",
                 "--extra-body", EXTRA_BODY, "--label", "{label}",
                 "--output", f"{base}/c{streams}-{{run}}-{{label}}.json"], f"c{streams}"]

    # Where async scheduling could still pay off: one stream (the shortest step, so the
    # largest share of host time) and long prefills (the Engram reader reads a chunk's
    # rows while the step before it runs).
    suite = {"name": "late-admission-screen", "version": 1,
             "description": "One stream in a closed loop (30 s warm-up, 120 s, thinking off) and cold "
                            "real-text prefill at 32K and 131K, three repeats.",
             "steps": [{"id": "c1", "command": "run",
                        "args": {"mode": "closed", "concurrency": 1, "warmup": 30, "duration": 120,
                                 "extra_body": {"chat_template_kwargs": {"thinking": False}}}},
                       {"id": "prefill", "command": "prefill",
                        "args": {"sizes": [32768, 131072], "repeats": 3}}]}
    # The ceiling A/B's synchronous-scheduling arm (a path with "/" is repository-relative).
    async_off = ("results/private/lab/ttft-gap/async-off.json" if a.staged
                 else f"{TTFT}/lab/async-off.json")
    screen = {"experiment": base, "run": f"late-admission-l{a.layer}-screen", "jobs": [
        {"kind": "sync"},
        {"kind": "measure", "bench": False, "bracket": False,
         "extras": [[f"{scripts}/ttft_trace.py",
                     ["--repo", ".", "--output", f"{base}/trace-{{run}}-{{label}}.json", "--analyze"], "trace"],
                    servebench(4), servebench(8),
                    ["scripts/servebench_suite.py",
                     ["--suite", f"{base}/screen-suite.json", "--out", f"{base}/suite-{{run}}-{{label}}",
                      "--label", "{label}"], "suite"],
                    ["experiments/2026-10-12-r6g/token_determinism_text.py",
                     ["--tokens", "192", "--long-lines", "700", "--wide"], "determinism"]],
         "arms": [{"config": "late-tp3.json", "label": f"late-l{a.layer}"},
                  {"config": "r6x-tp3.json", "label": "r6x"},
                  {"config": async_off, "label": "async-off"}]}]}
    for name, content in (("late-tp3.json", late), ("r6x-tp3.json", reference),
                          ("screen-suite.json", suite), ("screen-tp3.json", screen)):
        (out / name).write_text(json.dumps(content, indent=1) + "\n")
        print(out / name)


if __name__ == "__main__":
    main()
