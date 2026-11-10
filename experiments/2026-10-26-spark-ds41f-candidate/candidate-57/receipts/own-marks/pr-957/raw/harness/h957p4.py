"""#957 part 4 (0798): DSpark confidence and its online calibration, one draft step.

usage: python3 /h/h957p4.py --arm base|c57p4 --boot K --out FILE

The PR's bench_confidence_calibration workload (screen_tp3_decode.py at e464d364): hidden
5120, Markov rank 256, 5 draft steps, an FP32 head weight, OnlineAcceptanceEstimator
.for_confidence_head(16, 5), here with slope 1.3 and intercepts -0.5..0.5 so the
calibration is not the identity. Each arm times the call its own DSparkSpeculator makes:
candidate-54 runs TileLangConfidenceHead.probability then calibrate_confidence_probs
(two kernels); 0798 runs probability_and_calibrate (one). The draft step runs inside the
DSpark draft CUDA graph in serving, so CUDA-graph replay (20 calls a graph) decides; CUPTI
kernel time and eager timing are beside it. The candidate arm also runs the two-step path
in process and compares the calibrated probabilities, features and predictions.
"""

import argparse
import json
from types import SimpleNamespace

import torch

from omlib import eager_us, env_info, gpu_warmup, graph_hot_us, h, kernel_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--boot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cand = args.arm != "base"
    from vllm.models.deepseek_v4_1.tilelang import confidence
    from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import OnlineAcceptanceEstimator

    device = torch.device("cuda", 0)
    warm = gpu_warmup()
    hidden_size, rank, steps = 5120, 256, 5
    gen = torch.Generator(device=device).manual_seed(99)
    h_buf = torch.randn((96, hidden_size), generator=gen, device=device, dtype=torch.bfloat16)
    e_buf = torch.randn((96, rank), generator=gen, device=device, dtype=torch.bfloat16)
    weight = (torch.randn((hidden_size + rank,), generator=gen, device=device) * 0.01).float()
    head = confidence.TileLangConfidenceHead.__new__(confidence.TileLangConfidenceHead)
    torch.nn.Module.__init__(head)
    head.proj = SimpleNamespace(weight=weight.view(1, -1))
    head.hidden_size, head.markov_rank = hidden_size, rank

    def estimator():
        est = OnlineAcceptanceEstimator.for_confidence_head(16, steps, device)
        est.slope.fill_(1.3)
        est.intercepts.copy_(torch.linspace(-0.5, 0.5, steps, device=device))
        return est

    out = {"arm": args.arm, "boot": args.boot, "env": env_info(), "gpu_warmup_ms": warm, "cells": {}}
    for num_reqs in (1, 2, 4, 8, 16):
        rows = num_reqs * steps
        hh, ee = h_buf[:rows], e_buf[:rows]
        idx_map = torch.arange(num_reqs, dtype=torch.int32, device=device).repeat_interleave(steps).contiguous()
        est_two, est_fused = estimator(), estimator()
        p_two = torch.empty((num_reqs, steps), dtype=torch.float32, device=device)
        p_fused = torch.empty((num_reqs, steps), dtype=torch.float32, device=device)

        def two_step():
            head.probability(hh, ee, out=p_two.view(-1))
            est_two.calibrate_confidence_probs(p_two, idx_map)

        def fused():
            head.probability_and_calibrate(hh, ee, p_fused.view(-1), est_fused.slope, est_fused.intercepts,
                                           idx_map, est_fused.features, est_fused.predictions)

        two_step()
        torch.cuda.synchronize()
        cell = {"rows": rows, "two_step_hash": h(p_two), "two_step_features_hash": h(est_two.features),
                "two_step_predictions_hash": h(est_two.predictions)}
        serving = fused if cand else two_step
        if cand:
            fused()
            torch.cuda.synchronize()
            d = (p_fused - p_two).abs()
            ulp = (p_fused.view(torch.int32) - p_two.view(torch.int32)).abs()
            cell["fused_vs_two_step"] = {
                "max_abs": float(d.max()), "max_ulp": int(ulp.max()), "equal": bool(torch.equal(p_fused, p_two)),
                "features_max_abs": float((est_fused.features - est_two.features).abs().max()),
                "predictions_max_abs": float((est_fused.predictions - est_two.predictions).abs().max()),
                "prob_range": [float(p_two.min()), float(p_two.max())],
            }
            cell["fused_hash"] = h(p_fused)
        cell["serving_call"] = "probability_and_calibrate" if cand else "probability + calibrate_confidence_probs"
        cell["graph_hot"] = graph_hot_us(serving)
        k = kernel_us(serving)
        cell["kernel"] = {"per_call_us": k["per_call_us"], "launches": k["kernels"]}
        cell["eager_us"] = eager_us(serving)
        if cand:
            cell["two_step_in_process"] = {"graph_hot": graph_hot_us(two_step),
                                           "kernel_us": kernel_us(two_step)["per_call_us"],
                                           "eager_us": eager_us(two_step)}
        out["cells"][f"r{num_reqs}"] = cell
        print(f"r{num_reqs}", cell["serving_call"], cell["graph_hot"]["median_us"], cell["kernel"]["per_call_us"],
              cell["eager_us"], cell.get("fused_vs_two_step"), flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
