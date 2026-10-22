#!/usr/bin/env python3
"""The A/B's verdict lines: cold prefill paired by prompt, traced TTFT, determinism hashes.

usage: ab_table.py RUN REFERENCE LABEL [LABEL...]   (run from the repository root on dgx1)

An arm is LABEL (of RUN) or OTHER_RUN:LABEL, to compare arms of two runs in one window.

Prefill: each arm's bench prompts are the same prompts in the same order (the bench's seeded
generator), so each prompt's TTFT is compared with the reference's, and the per-size mean of
those ratios is reported beside the bench's own tok/s summary.
"""
import json
import statistics
import sys
from pathlib import Path

HASHES = {"determinism": ("62313cb5", "9742ac2c"), "greedy": ("4d143977", "12fd8257")}


def summary_line(path: Path):
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "summary" in record:
            return record
    return None


def split(run: str, arm: str) -> tuple[str, str]:
    return tuple(arm.split(":", 1)) if ":" in arm else (run, arm)


def prefill(run: str, arm: str) -> dict:
    run, label = split(run, arm)
    bench = json.loads(Path(f"results/private/bench/{run}-{label}/bench.json").read_text())
    return bench["suites"]["prefill"]


def main():
    run, reference, *labels = sys.argv[1:]
    arms = [reference, *labels]
    pre = {label: prefill(run, label) for label in arms}
    print(f"cold prefill (tok/s, bench summary; per-prompt TTFT against {reference})")
    for size in pre[reference]["points"]:
        ref = pre[reference]["samples"][size]
        line = [f"  {size:>6}:"]
        for label in arms:
            tps = pre[label]["points"][size]["prefill_tps"]
            line.append(f"{label} {tps['mean']:.1f} ±{tps.get('ci95_pct', 0):.1f}%")
            if label != reference:
                ratios = [x["ttft_s"] / y["ttft_s"] - 1
                          for x, y in zip(pre[label]["samples"][size], ref)
                          if x["prompt_tokens"] == y["prompt_tokens"]]
                line.append(f"(TTFT {100 * statistics.mean(ratios):+.2f}%, per prompt "
                            + " ".join(f"{100 * r:+.1f}" for r in ratios) + ")")
        print(" ".join(line))
    print("traced TTFT (ms): isolated mean; c4 p50; c8 p50; c4/c8 plain step p50; foreign")
    for label in arms:
        arm_run, arm_label = split(run, label)
        record = summary_line(Path(f"results/private/determinism/{arm_run}/trace-{arm_label}.jsonl"))
        if record is None:
            print(f"  {label}: no trace summary")
            continue
        s = record["summary"]
        print(f"  {label}: {s['isolated_ttft_mean']:.1f}; {s['c4']['ttft_p50']:.1f}; "
              f"{s['c8']['ttft_p50']:.1f}; {s['c4']['plain_step_p50']:.1f} / "
              f"{s['c8']['plain_step_p50']:.1f}; {s['c4']['foreign_requests']:.0f} / "
              f"{s['c8']['foreign_requests']:.0f}")
    print("determinism (identical; reference / long hash, expected r6x's)")
    for kind, expected in HASHES.items():
        for label in arms:
            arm_run, arm_label = split(run, label)
            record = summary_line(
                Path(f"results/private/determinism/{arm_run}/{kind}-{arm_label}.jsonl"))
            if record is None:
                print(f"  {kind} {label}: no summary")
                continue
            got = (record["reference_sha256"][:8], record["long_sha256"][:8])
            diffs = sum(v["token_diffs"] + v["logprob_diffs"] for v in record["scenarios"].values())
            verdict = "pass" if record["identical"] and got == expected and diffs == 0 else "FAIL"
            print(f"  {kind} {label}: identical={record['identical']} diffs={diffs} "
                  f"{got[0]} / {got[1]} {verdict}")


if __name__ == "__main__":
    main()
