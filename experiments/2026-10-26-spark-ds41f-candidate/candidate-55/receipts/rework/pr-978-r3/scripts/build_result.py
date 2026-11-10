#!/usr/bin/env python3
"""Builds result.json for #978 round 3 from raw/ (copied from sp8) and the narrative below.
Usage: build_result.py <deliverable dir>"""
import hashlib
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
R = D / "raw"


def load(p):
    return json.loads((R / p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


replay = load("replay/summary.json")
controls = {c: load(f"replay/replay-control-{c}.json") for c in ("nohit", "late")}
ref = load("replay/c54-1.json")
mark = load("mark/summary.json")
text = load("text/summary.json")
prodenv = load("prodenv/summary.json")
prodenv2 = load("prodenv2/summary.json")
pre = {k: load(f"superseded/{k}/summary.json") for k in ("r3pre-mark", "r3pre-text", "r3pre-prodenv", "r3pre-text2")}
counts = load("tests/test-counts.json")
history = load("gate-history.json")
apply_check = (R / "apply-check.txt").read_text()
patch = next(D.glob("0793-*.patch"))


def cells(summary):
    out = {}
    for w, v in summary["workloads"].items():
        out[w] = {
            "turn1": {"base_ms": v["turn1"]["base_ms"], "r3_ms": v["turn1"]["cand_ms"],
                      "ratio": v["turn1"]["ratio_cand_over_base"], "verdict": v["turn1"]["verdict"]},
            "turn2plus": {"base_ms": v["turn2plus"]["base_ms"], "r3_ms": v["turn2plus"]["cand_ms"],
                          "speedup": v["turn2plus"]["speedup"], "verdict": v["turn2plus"]["verdict"]},
            "turn2plus_bytes": v["turn2plus_bytes"], "stripped_items": v["stripped_items"],
            "prompt_token_ids_and_mm_hashes_equal_base": v["prompt_token_ids_and_mm_hashes_equal_base"],
        }
    return out


def verdicts(summary, keys=("turn1", "turn2plus")):
    vs = [v[k]["verdict"] for v in summary["workloads"].values() for k in keys]
    return {x: vs.count(x) for x in sorted(set(vs))}


def text_cells(summary):
    return {w: {k: {"base_ms": v[k]["base"], "r3_ms": v[k]["cand"], "ratio": v[k]["ratio_cand_over_base"],
                    "verdict": v[k]["verdict"]} for k in ("submit_median_ms", "roundtrip_median_ms")}
            | {"prompt_token_ids_equal_base": v["prompt_token_ids_equal_base"]}
            for w, v in summary["workloads"].items()}


NARRATIVE = json.loads((D / "scripts" / "result-narrative.json").read_text())

result = {
    "pr": 978,
    "label": "978-r3",
    "kind": "perf (host)",
    "round": "candidate-55 round 3 (the owner's exception of 2026-11-10 beyond two reworks), patch 0793",
    "date": "2026-11-10",
    **NARRATIVE["head"],
    "dynamic_evidence": {
        **NARRATIVE["dynamic"],
        "runs": {k: v for k, v in replay.items() if k not in ("engine", "by_scenario")},
        "engine_as_built": replay["engine"],
        "prefix_cache_hits_by_scenario": replay["by_scenario"],
        "per_request_prefill_on_candidate_54": {
            k: v["prefill"] for k, v in ref["per_request"].items()
            if k in ("reference-2", "cached-2-a", "cached-2-b", "again-2", "again_salted-2",
                     "long_reference", "mixed-long", "again_salted-long")},
        "first_steps_on_candidate_54": [x for x in ref["trace"] if x.get("new")][:3],
        "positive_controls": {
            c: {"trace_differs_from_candidate_54": r["trace_sha256"] != ref["trace_sha256"],
                "outputs_differ": r["outputs_sha256"] != ref["outputs_sha256"],
                "frontend_requests_equal": r["frontend_requests_sha256"] == ref["frontend_requests_sha256"],
                "again_salted": r["by_scenario"]["again_salted"], "steps": r["steps"]}
            for c, r in controls.items()},
    },
    "gate_history": {
        "what": history["what"], "origin_main": history["commit"], "runs": history["runs"],
        "runs_with_a_difference": history["runs_with_a_difference"],
        "differing": [{"file": r["file"], "requests": r["differing"]} for r in history["rows"] if r["differing"]],
        "file": "raw/gate-history.json",
    },
    "serving_path_cells": {
        **NARRATIVE["bench"],
        "mark": {"verdict_counts": verdicts(mark), "mm_prefix_fastpath": mark["mm_prefix_fastpath"],
                 "cells": cells(mark), "session": "raw/mark/session.log"},
        "prodenv": {"verdict_counts": verdicts(prodenv), "cells": cells(prodenv),
                    "session": "raw/prodenv/session.log"},
        "prodenv_repeat": {"verdict_counts": verdicts(prodenv2), "cells": cells(prodenv2),
                           "session": "raw/prodenv2/session.log"},
        "text": {"verdict_counts": verdicts(text, ("submit_median_ms", "roundtrip_median_ms")),
                 "cells": text_cells(text), "session": "raw/text/session.log"},
        "first_round_3_tree (r3-pre: the handler read req.mm_features directly; image paths the same)": {
            "mark": {"verdict_counts": verdicts(pre["r3pre-mark"]), "cells": cells(pre["r3pre-mark"])},
            "prodenv": {"verdict_counts": verdicts(pre["r3pre-prodenv"]), "cells": cells(pre["r3pre-prodenv"])},
            "text": {"verdict_counts": verdicts(pre["r3pre-text"], ("submit_median_ms", "roundtrip_median_ms")),
                     "cells": text_cells(pre["r3pre-text"])},
            "text_repeat_4_boots": {"verdict_counts": verdicts(pre["r3pre-text2"], ("submit_median_ms", "roundtrip_median_ms")),
                                    "cells": text_cells(pre["r3pre-text2"])},
            "files": "raw/superseded/r3pre-*"},
    },
    "tests": {**NARRATIVE["tests"], "counts": counts},
    "patch": {
        **NARRATIVE["patch"],
        "file": patch.name, "sha256": sha(patch),
        "apply_check": "raw/apply-check.txt",
        "apply_check_lines": [l for l in apply_check.splitlines() if not l.startswith("./")][:60],
    },
    "overlay": {
        "mount_root": "/opt/spark-ds41f/candidate/vllm (overlay/<path> -> /opt/spark-ds41f/candidate/vllm/<path>)",
        "files": {str(p.relative_to(D / "overlay")): sha(p) for p in sorted((D / "overlay").rglob("*")) if p.is_file()},
        "mounted_by_the_profile": "the 9 non-test files, from experiments/2026-10-26-spark-ds41f-candidate/candidate-55/978r3/overlay/",
    },
    **NARRATIVE["tail"],
}
(D / "result.json").write_text(json.dumps(result, indent=1) + "\n")
print("written", D / "result.json")
