#!/usr/bin/env python3
"""Decode step time by context depth, per boot.

usage: tables_arms.py RUN LABEL...

Reads results/private/determinism/RUN/decode-depth-LABEL.jsonl (profile_decode.py's lines).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
run, labels = sys.argv[1], sys.argv[2:]
for label in labels:
    path = ROOT / "results/private/determinism" / run / f"decode-depth-{label}.jsonl"
    if not path.exists():
        print(f"{label}: no output")
        continue
    lines = []
    for raw in path.read_text().splitlines():
        try:
            lines.append(json.loads(raw))
        except ValueError:
            pass
    prompt = {l["depth"]: l["usage"].get("prompt_tokens") for l in lines if l.get("phase") == "fill"}
    print(f"{label}")
    print("  depth label  prompt tokens  phase            rows  step ms   steps  tokens/step")
    for line in lines:
        if line.get("phase") == "real":
            print(f"  {line['depth']:>11,}  {prompt.get(line['depth']) or 0:>13,}  real repeat {line['repeat']}    -     "
                  f"{line['step_ms']:>7.3f}  {line['steps']:>5.0f}  {line['tokens_per_step']:>6.3f}")
        elif line.get("phase") == "grid":
            for row in line["grid"]:
                step = row["step_ms"]
                forward = (row.get("forward_ms") or {}).get("mean")
                drafter = (row.get("drafter_ms") or {}).get("mean")
                split = f"  forward {forward:.3f} drafter {drafter:.3f}" if forward and drafter else ""
                print(f"  {line['depth']:>11,}  {'':>13}  grid @{line['context_len']:<9,}  {row['shape']['num_tokens']:>2}    "
                      f"{step['mean']:>7.3f}  {step.get('n', 0):>5}{split}")
