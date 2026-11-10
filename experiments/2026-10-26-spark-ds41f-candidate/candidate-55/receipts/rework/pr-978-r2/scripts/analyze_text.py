#!/usr/bin/env python3
"""Summarize serving_text_bench.py: the candidate arm against candidate-54 per workload, for the
submission (generate() -> request encoded) and round-trip medians. Faster (slower) beyond the
interval: every candidate boot's median below (above) every base boot's. Identity: round 0's
prompt-token digests equal the base's.
Usage: analyze_text.py <out.json> --base <json>... --cand <json>...
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--base", nargs="+", required=True)
ap.add_argument("--cand", nargs="+", required=True)
args = ap.parse_args()
base = [json.loads(Path(p).read_text()) for p in args.base]
cand = [json.loads(Path(p).read_text()) for p in args.cand]


def verdict(b, c):
    if max(c) < min(b):
        return "faster beyond the interval"
    if min(c) > max(b):
        return "slower beyond the interval"
    return "within the interval"


out = {"base_files": args.base, "cand_files": args.cand, "workloads": {}}
for w in base[0]["workloads"]:
    row = {}
    for key in ("submit_median_ms", "roundtrip_median_ms"):
        b = [r["workloads"][w][key] for r in base]
        c = [r["workloads"][w][key] for r in cand]
        row[key] = {"base": b, "cand": c, "ratio_cand_over_base": round(sum(c) / sum(b), 3),
                    "verdict": verdict(b, c)}
    ref = base[0]["workloads"][w]["identity_round0"]
    row["prompt_token_ids_equal_base"] = all(r["workloads"][w]["identity_round0"] == ref
                                             for r in base + cand)
    out["workloads"][w] = row
Path(args.out).write_text(json.dumps(out, indent=1))
for w, v in out["workloads"].items():
    s, t = v["submit_median_ms"], v["roundtrip_median_ms"]
    print(f"{w:16} submit base {s['base']} cand {s['cand']} x{s['ratio_cand_over_base']} {s['verdict']}; "
          f"round trip base {t['base']} cand {t['cand']} x{t['ratio_cand_over_base']} {t['verdict']}; "
          f"identity {v['prompt_token_ids_equal_base']}")
