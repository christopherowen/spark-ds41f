#!/usr/bin/env python3
"""Assemble #978 round 2's result.json from the raw summaries (laptop, after the runs).
Usage: build_result.py <rework dir> <tests.json> <apply-check.txt> <patch sha256> <commits.json>"""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
tests = json.loads(Path(sys.argv[2]).read_text())
apply_check = Path(sys.argv[3]).read_text()
patch_sha = sys.argv[4]
commits = json.loads(Path(sys.argv[5]).read_text())
raw = D / "raw"


def load(p):
    return json.loads((raw / p).read_text())


def cells(summary):
    out = {}
    for w, v in summary["workloads"].items():
        out[w] = {
            "turn1": {"base_ms": v["turn1"]["base_ms"], "r2_ms": v["turn1"]["cand_ms"],
                      "r2_over_base": v["turn1"]["ratio_cand_over_base"], "verdict": v["turn1"]["verdict"]},
            "turn2plus": {"base_ms": v["turn2plus"]["base_ms"], "r2_ms": v["turn2plus"]["cand_ms"],
                          "speedup": v["turn2plus"]["speedup"], "verdict": v["turn2plus"]["verdict"]},
            "request_bytes_turn1": v["turn1_bytes"],
            "request_bytes_turn2plus": v["turn2plus_bytes"],
            "stripped_items_r2": v["stripped_items"]["cand"],
            "prompt_token_ids_and_mm_hashes_equal_base": v["prompt_token_ids_and_mm_hashes_equal_base"],
        }
    return out


def text_cells(summary):
    return {w: {"submit_ms": {"base": v["submit_median_ms"]["base"], "r2": v["submit_median_ms"]["cand"],
                              "verdict": v["submit_median_ms"]["verdict"]},
                "round_trip_ms": {"base": v["roundtrip_median_ms"]["base"], "r2": v["roundtrip_median_ms"]["cand"],
                                  "verdict": v["roundtrip_median_ms"]["verdict"]},
                "prompt_token_ids_equal_base": v["prompt_token_ids_equal_base"]}
            for w, v in summary["workloads"].items()}


def stages(path, workload, turn="turn1"):
    d = load(path)["workloads"][workload]
    return {"total_ms": d[f"{turn}_total_ms" if turn == "turn1" else "turn2plus_total_ms"],
            **{s: {k: x.get(k) for k in ("wall_ms", "thread_cpu_ms", "sys", "minflt", "little_core_share")}
               for s, x in d[turn].items() if s in ("load", "mat", "hash", "hf", "proc")}}


mark = load("mark/summary.json")
text = load("text/summary.json")
prodenv = load("serving-supplementary/prodenv/summary.json")

result = json.loads((D / "result-skeleton.json").read_text())
result["serving_path_cells"]["mark"]["cells"] = cells(mark)
result["serving_path_cells"]["text_only"]["cells"] = text_cells(text)
result["serving_path_cells"]["supplementary_recipe_environment"]["cells"] = cells(prodenv)
jf = result["jpeg_finding"]["evidence"]
jf["1302x1302_jpeg_1img_turn2plus_stages (diag1, fast path off)"] = {
    a: stages(f"jpeg-bisect/diag1/d1-{a}.json", "1302x1302_jpeg_1img", "turn2plus") for a in ("base", "r1", "img", "h")}
jf["1302x1302_jpeg_1img_turn1_stages (diag2, with fault counters)"] = {
    a: stages(f"jpeg-bisect/diag2/d2-{a}.json", "1302x1302_jpeg_1img") for a in ("base", "r1", "r1st", "img", "imgst", "base2")}
jf["1920x1080_png_1img_turn1_stages_final (stage-diag-screen, r2 final)"] = {
    a: stages(f"serving-supplementary/stage-diag-screen/d7-{a}.json", "1920x1080_png_1img") for a in ("base", "r2f", "r2fb", "baseb")}
jf["1920x1080_png_1img_turn1_stages_recipe_environment (prodenv-diag)"] = {
    a: stages(f"serving-supplementary/prodenv-diag/pd-{a}.json", "1920x1080_png_1img") for a in ("base-1", "r2f-1", "r2f-2", "base-2")}
result["tests"]["runs"] = tests
result["patch"]["sha256"] = patch_sha
result["patch"].update(commits)
result["patch"]["apply_check"] = apply_check.strip().splitlines()
(D / "result.json").write_text(json.dumps(result, indent=1) + "\n")
print("wrote", D / "result.json")
