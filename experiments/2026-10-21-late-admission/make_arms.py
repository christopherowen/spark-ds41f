#!/usr/bin/env python3
"""Write the late-admission arms and the screen spec.

usage: make_arms.py LAYER [--kv-bytes N] [--staged DIR]

- late-tp3.json: r6x on the triangle (the recipe at tp3-triangle) with overlay/ mounted
  over the image's vLLM sources and VLLM_LATE_ADMISSION_LAYER=LAYER;
- r6x-tp3.json: the recipe at tp3-triangle (the reference);
- async-off-tp3.json: the recipe with synchronous scheduling (--no-async-scheduling and
  --disable-cascade-attn, which async speculative decoding sets by itself);
- screen-suite.json and screen-tp3.json: the three arms, candidate first, each measured with
  the traced replay of experiments/2026-10-20-ttft-gap, servebench c4 and c8 (30 s warm-up,
  120 s), one stream and cold 32K/131K prefill (the suite), and the determinism text check.

--kv-bytes N gives every arm the recipe's serve arguments with that KV pool. dgx1's 5 GiB
startup guard stopped production-sized lab boots at 4.99 GiB on 2026-10-21; the screen's
short prompts and 131K prefills do not depend on the pool's size, and all three arms share it.

With --staged DIR (repository-relative, e.g. results/private/lab/late-admission) the files
go to DIR with every path pointing into DIR: overlay mounts, scripts and the experiment.
That copy runs before the experiment is merged, once DIR (with overlay/ and the ttft-gap
scripts) is on every node.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = "experiments/2026-10-21-late-admission"
TTFT = "experiments/2026-10-20-ttft-gap"
FILES = ("vllm/envs.py", "vllm/v1/engine/core.py", "vllm/v1/worker/gpu/model_runner.py",
         "vllm/v1/worker/gpu/late_admission.py")
EXTRA_BODY = json.dumps({"chat_template_kwargs": {"thinking": False}})


def recipe_args(kv_bytes: int | None) -> list[str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import recipe

    args = recipe.resolve(json.loads((ROOT / "config/cluster.json").read_text()), "tp3-triangle")[
        "serve_args"]
    if kv_bytes is not None:
        args[args.index("--kv-cache-memory-bytes") + 1] = str(kv_bytes)
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", type=int)
    ap.add_argument("--kv-bytes", type=int)
    ap.add_argument("--staged")
    a = ap.parse_args()
    base = a.staged or EXPERIMENT
    scripts = a.staged or TTFT
    out = ROOT / a.staged if a.staged else HERE
    out.mkdir(parents=True, exist_ok=True)
    recipe = {"config": "config/cluster.json", "shape": "tp3-triangle"}
    args = recipe_args(a.kv_bytes)
    sync = [("--no-async-scheduling" if arg == "--async-scheduling" else arg) for arg in args]
    sync.insert(sync.index("--no-async-scheduling") + 1, "--disable-cascade-attn")
    sized = {"serve_args": args} if a.kv_bytes is not None else {}
    pool = f" KV pool {a.kv_bytes / 2**30:g} GiB." if a.kv_bytes is not None else ""
    mounts = [["{home}/projects/spark-ds41f/cache", "/cache", "rw"]] + [
        [f"{{home}}/projects/spark-ds41f/{base}/overlay/{f}", f"/opt/spark3/candidate/vllm/{f}", "ro"]
        for f in FILES]
    arms = {
        "late-tp3.json": {
            "notes": f"r6x on the triangle at 1M with vLLM 0222 (overlay/, vLLM 91eaaa59e on r6x 7316c7f9f, "
                     f"the image's tree): late admission at decoder layer {a.layer}.{pool}",
            "extends": recipe,
            "set": {**sized, "container": {"mounts": mounts},
                    "environment": {"VLLM_LATE_ADMISSION_LAYER": str(a.layer)}}},
        "r6x-tp3.json": {
            "notes": f"r6x on the triangle at 1M: the late-admission screen's reference.{pool}",
            "extends": recipe, "set": sized},
        "async-off-tp3.json": {
            "notes": "r6x on the triangle at 1M with synchronous scheduling: --no-async-scheduling (async is "
                     "otherwise on by default for DSpark) and --disable-cascade-attn (which async speculative "
                     f"decoding sets by itself).{pool}",
            "extends": recipe, "set": {"serve_args": sync}},
    }

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
                  {"config": "async-off-tp3.json", "label": "async-off"}]}]}
    for name, content in (*arms.items(), ("screen-suite.json", suite), ("screen-tp3.json", screen)):
        (out / name).write_text(json.dumps(content, indent=1) + "\n")
        print(out / name)


if __name__ == "__main__":
    main()
