#!/usr/bin/env python3
"""Sensitivity input: #986 round 1's own confidence-model fit (fit_conf, imported unchanged from its
replay_budget.py) run against the TP3 served counters of candidate-50's tp3-switched screen (c8: two
boots, c1: single-stream) instead of tp4-ring's. Request counts and prompt rates stay #986's.
Writes a file shaped like #986's replay-seed<k>.json (concurrency.<c>.fit.theta) for replay_957p5.py.
"""
import argparse
import importlib.util
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay986", required=True)
    ap.add_argument("--served", required=True)
    ap.add_argument("--fit986", required=True)
    ap.add_argument("--tp3-c8", nargs="+", required=True)
    ap.add_argument("--tp3-c1", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("rb986", a.replay986)
    rb = importlib.util.module_from_spec(spec); sys.modules["rb986"] = rb; spec.loader.exec_module(rb)
    served = json.load(open(a.served))
    fit986 = json.load(open(a.fit986))
    out = {"seed": a.seed, "source": "fit_conf of #986 r1 on candidate-50 tp3-switched served counters", "concurrency": {}}
    for label, paths in (("c1", [a.tp3_c1]), ("c8", a.tp3_c8)):
        srv = [json.load(open(p))["summary"]["server"] for p in paths]
        target = {
            "coverage": [float(np.mean([s[f"coverage_at_{k}"]["mean"] for s in srv])) for k in range(5)],
            "survival": [float(np.mean([s[f"survival_at_{k}"]["mean"] for s in srv])) for k in range(5)],
            "tokens_per_step": float(np.mean([s["output_tokens_per_step"]["mean"] for s in srv])),
        }
        cfg = served["concurrency"][label]
        ki = list(served["concurrency"]).index(label)
        best, loss, cov, surv = rb.fit_conf(target, 100 + ki + 1000 * a.seed, cfg["requests"], cfg["prefill_rate"], cfg["nmax"])
        out["concurrency"][label] = {"fit": {"theta": [float(x) for x in best], "loss": loss, "coverage": cov,
                                             "survival": surv[:5], "tokens_per_step": surv[5], "target": target}}
        print(label, "loss", loss, "cov", np.round(cov, 3).tolist(), "target", np.round(target["coverage"], 3).tolist(),
              "surv", np.round(surv[:5], 3).tolist(), "target", np.round(target["survival"], 3).tolist(),
              "tok", round(surv[5], 3), "target", round(target["tokens_per_step"], 3), flush=True)
    out["concurrency"]["c16"] = fit986["concurrency"]["c16"]  # only the TP4 equality check reads it
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
