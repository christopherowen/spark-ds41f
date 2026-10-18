#!/usr/bin/env python3
"""Apple's generation judge for speculative decoding, across our arms (a lab extras script).

usage: judge_rsi.py URL --side NAME [--lib DIR] [--prompts FILE] [--judge]

The judge (the benchmark's optimize-task library `_verifier_lib/generation_judge.py`,
outside this repository) asks whether a speculative decoder changes what the target model
would emit: each arm's server generates 32 prompts x 256 tokens, greedy and sampled
(temperature 1, seeded), and a server of the same model with speculation off
teacher-forces every arm's tokens.
- J1 counts greedy tokens more than 0.5 nats below the target's own top-1; a candidate
  fails when it exceeds the pristine arm's count by more than
  max(0.005 * n, 4 * sqrt(c + p + 1)).
- J2 compares the sampled sets' top-1 calibration; |z| beyond 5 fails.
(The thresholds are the benchmark's DSpark task's: INFB_GEN_JUDGE_*.)

`--side NAME` probes the booted arm under NAME. With `--judge` the booted arm must be
the speculation-off server: it probes itself as the pristine side, teacher-forces every
probed side and writes the verdict of each candidate against it. Outputs land in
results/private/apple-rsi/judge/.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "private" / "apple-rsi" / "judge"
WORKLOAD = "dba1"
THRESHOLDS = {"INFB_GEN_JUDGE_J1_EPS": "0.5", "INFB_GEN_JUDGE_J1_MAX_EXCESS": "0.005",
              "INFB_GEN_JUDGE_J1_Z": "4", "INFB_GEN_JUDGE_J2_Z": "5"}


def library(path: Path):
    spec = importlib.util.spec_from_file_location("apple_generation_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--side", required=True)
    ap.add_argument("--judge", action="store_true")
    task = "~/spark-lab/apple-rsi-src/optimize_dsv4_flash_4xh100_dspark_sglang_open/tests"
    ap.add_argument("--lib", default=f"{task}/_verifier_lib/generation_judge.py")
    ap.add_argument("--prompts", default=f"{task}/correctness_prompts.json")
    ap.add_argument("--port", type=int, help="the arm's port, when it is not the URL's (lab passes :8000)")
    args = ap.parse_args()
    os.environ.update({k: os.environ.get(k, v) for k, v in THRESHOLDS.items()})
    gj = library(Path(os.path.expanduser(args.lib)))
    port = args.port or urlparse(args.url).port or 80
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = gj.load_prompts(Path(os.path.expanduser(args.prompts)))
    seed_side = "pristine" if args.judge else "cand"
    rec = gj.probe(engine="vllm", port=port, side=seed_side, prompts=prompts)
    rec["arm"] = args.side
    (OUT / f"probe-{args.side}.json").write_text(json.dumps(rec))
    print(json.dumps({"probe": args.side, "elapsed_s": rec.get("elapsed_s"), "model": rec.get("model")}),
          flush=True)
    if not args.judge:
        return 0
    probes = {p.stem.removeprefix("probe-"): json.loads(p.read_text()) for p in sorted(OUT.glob("probe-*.json"))}
    judged = gj.judge(engine="vllm", port=port, probes=probes)["arms"]
    (OUT / "judged.json").write_text(json.dumps(judged))
    report = {}
    for name in probes:
        if name == args.side:
            continue
        summary = gj.summarise({"arms": {"cand": judged[name], "pristine": judged[args.side]}})
        verdict = gj.gate({WORKLOAD: summary}, [WORKLOAD], speculative=True)
        report[name] = verdict
        e = verdict["per_workload"][WORKLOAD]
        print(json.dumps({"candidate": name, "pristine": args.side, "failed": verdict["failed"],
                          "reason": verdict.get("reason"), "failures": verdict.get("failures"),
                          "j1_excess": e.get("j1_excess"), "j1_bar": e.get("j1_bar"),
                          "j1_cand": (e.get("j1") or {}).get("cand"),
                          "j1_pristine": (e.get("j1") or {}).get("pristine"),
                          "j2_cal": e.get("j2_cal"), "judge_sane": e.get("judge_sane")}, default=str),
              flush=True)
    (OUT / "verdicts.json").write_text(json.dumps(report, indent=1, default=str))
    return 1 if any(v["failed"] for v in report.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
