#!/usr/bin/env python3
"""In-process A/B/C step cost of the six files #980 changes.

stepcost.py runs each arm in its own process, so its comparison carries
process-to-process variance (about 1% on the greedy total, and a bimodal
copy-event wait in record_confidences). This loads the three arms' copies of the
changed modules side by side in ONE process (1107's files, candidate-48's =
original #980, r1 = the correction), builds each arm's objects from its own
module copies, and calls them interleaved (order rotated every iteration), so
process state cancels. Each call is timed alone on an idle GPU: synchronize,
host clock, call, host clock (host_us: the Python and launch cost the patch can
change), then synchronize and read CUDA events around the call (gpu_us).

TP4 is emulated as PR #980's screen does it. Shapes: the engine grid's decode
shapes (num_reqs x rows per request).

Natural A/A controls: r1's adaptive_verification.py is byte-identical to 1107's
(record_conf r1 vs 1107), and r1's other five files are byte-identical to
candidate-48's (every other component, r1 vs c48).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

GRID = [(r, 1) for r in (1, 4, 8, 16)] + [(r, 2) for r in (1, 4, 8, 16)] + [
    (r, 5) for r in (1, 2, 4, 8, 12, 16)
]
MAX_REQS, STEPS, VOCAB, WORLD = 16, 5, 129280, 4
ARMS = ("1107", "c48", "r1")
REL = {
    "states": "vllm/v1/worker/gpu/sample/states.py",
    "sampler": "vllm/v1/worker/gpu/sample/sampler.py",
    "ae": "vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py",
    "av": "vllm/v1/worker/gpu/spec_decode/adaptive_verification.py",
}
CANON = {
    "states": "vllm.v1.worker.gpu.sample.states",
    "ae": "vllm.v1.worker.gpu.spec_decode.acceptance_estimator",
}


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def trimmed(values: list[float]) -> list[float]:
    """Drop slow outliers beyond median + 5 MAD (servebench's rule)."""
    med = statistics.median(values)
    mad = statistics.median(abs(v - med) for v in values) or 1e-9
    return [v for v in values if v <= med + 5 * mad]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding <arm>/vllm/... copies")
    ap.add_argument("--boot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    import vllm.distributed.parallel_state as ps
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.states import RequestState

    ps.model_parallel_is_initialized = lambda: True
    ps.get_tensor_model_parallel_world_size = lambda: WORLD
    bcast = {"n": 0}

    def fake_broadcast(t, src=0):
        bcast["n"] += 1
        return t

    tp_group = SimpleNamespace(
        world_size=WORLD,
        broadcast=fake_broadcast,
        device_communicator=SimpleNamespace(rank_order_reduce_scatter=True),
    )

    mods = {}
    for arm in ARMS:
        base = os.path.join(args.root, arm)
        m = {}
        saved = {k: sys.modules.get(v) for k, v in CANON.items()}
        try:
            m["states"] = load(f"ab_{arm}_states", os.path.join(base, REL["states"]))
            sys.modules[CANON["states"]] = m["states"]
            m["sampler"] = load(f"ab_{arm}_sampler", os.path.join(base, REL["sampler"]))
            m["ae"] = load(f"ab_{arm}_ae", os.path.join(base, REL["ae"]))
            sys.modules[CANON["ae"]] = m["ae"]
            m["av"] = load(f"ab_{arm}_av", os.path.join(base, REL["av"]))
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(CANON[k], None)
                else:
                    sys.modules[CANON[k]] = v
        m["av"].get_tp_group = lambda: tp_group
        assert m["sampler"].SamplingStates is m["states"].SamplingStates
        mods[arm] = m

    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    t_end = time.time() + 3.0
    while time.time() < t_end:
        a @ a
    torch.cuda.synchronize()
    del a

    req_states = RequestState(
        max_num_reqs=MAX_REQS, max_model_len=8192, max_num_batched_tokens=8192,
        num_speculative_steps=STEPS, vocab_size=VOCAB, device=device,
    )
    gen = torch.Generator(device=device).manual_seed(1234)
    conf_src = torch.rand((MAX_REQS, STEPS), generator=gen, device=device) * 0.9 + 0.05
    planned_src = conf_src * 0.95
    dead_cut = torch.full((MAX_REQS,), STEPS, dtype=torch.int32, device=device)
    draft_logits = torch.randn(
        (MAX_REQS, VOCAB), generator=gen, device=device
    ).to(torch.bfloat16)
    target_logits = torch.randn((MAX_REQS * (STEPS + 1), VOCAB), generator=gen, device=device)
    vp_max = torch.randn((MAX_REQS, WORLD), generator=gen, device=device)
    vp_sum = torch.rand((MAX_REQS, WORLD), generator=gen, device=device) * 100 + 1
    draft_step = torch.tensor(1, dtype=torch.int32, device=device)

    objs = {}
    for arm in ARMS:
        m = mods[arm]

        def make_sampler(params, m=m):
            s = m["sampler"].Sampler(
                max_num_reqs=MAX_REQS, vocab_size=VOCAB, device=device,
                req_states=req_states, num_speculative_tokens=STEPS + 1,
            )
            for i in range(MAX_REQS):
                s.add_request(i, 16, params)
            s.apply_staged_writes()
            return s

        est = m["ae"].OnlineAcceptanceEstimator.for_confidence_head(MAX_REQS, STEPS, device)
        req_ns = SimpleNamespace(
            num_speculative_steps=STEPS, device=device, max_num_reqs=MAX_REQS,
            max_num_batched_tokens=8192,
            req_id_to_index={f"r{i}": i for i in range(MAX_REQS)},
            num_computed_tokens_np=np.full(MAX_REQS, 4096, dtype=np.int32),
            prefill_len=SimpleNamespace(np=np.full(MAX_REQS, 16, dtype=np.int32)),
        )
        mgr = m["av"].AdaptiveVerificationManager(
            req_ns, torch.zeros(MAX_REQS + 1, dtype=torch.int32, device=device), 1,
            max_total_logits=1 << 20, row_cost=0.035, planned_confidences=True,
            num_draft_steps=STEPS, tp_invariant_confidence=True,
            calibrated_confidence=lambda est=est: est._refits > 0,
        )
        objs[arm] = SimpleNamespace(
            greedy=make_sampler(SamplingParams(temperature=0.0)),
            sampled=make_sampler(SamplingParams(temperature=0.8, top_p=0.95, top_k=50, seed=None)),
            est=est,
            est_draft=m["ae"].OnlineAcceptanceEstimator(MAX_REQS, STEPS, device),
            mgr=mgr,
            draft_conf=torch.zeros((MAX_REQS, STEPS), dtype=torch.float32, device=device),
            ae=m["ae"],
        )

    def shape_inputs(num_reqs, depth):
        rows = 1 + depth
        idx_np = np.arange(num_reqs, dtype=np.int32)
        idx = torch.arange(num_reqs, dtype=torch.int32, device=device)
        n = num_reqs * rows
        exp = torch.from_numpy(np.repeat(np.arange(num_reqs), rows).astype(np.int32)).to(device)
        num_sampled = torch.from_numpy((np.arange(num_reqs) % rows + 1).astype(np.int32)).to(device)
        return SimpleNamespace(
            idx_np=idx_np, idx=idx, exp=exp,
            pos=torch.arange(n, dtype=torch.int64, device=device) + 4096,
            input_ids=torch.ones(n, dtype=torch.int32, device=device),
            local_pos=torch.from_numpy(np.tile(np.arange(rows, dtype=np.int32), num_reqs)).to(device),
            num_sampled=num_sampled,
            num_rejected=(depth + 1 - num_sampled).clamp_min(0).to(torch.int32),
            conf={arm: conf_src[:num_reqs].clone() for arm in ARMS},
            planned=planned_src[:num_reqs].clone(),
            batch=SimpleNamespace(num_reqs=num_reqs, idx_mapping=idx, idx_mapping_np=idx_np,
                                  has_prefill=False),
            ntok={f"r{i}": 1 + depth for i in range(num_reqs)},
            drafts={f"r{i}": list(range(1, depth + 1)) for i in range(num_reqs)},
            logits=target_logits[:n], dlogits=draft_logits[:num_reqs],
            vmax=vp_max[:num_reqs], vsum=vp_sum[:num_reqs],
        )

    def comp_fns(arm, x):
        o = objs[arm]

        def greedy_check():
            s = o.greedy
            st = s.sampling_states
            return bool(np.all(st.temperature.np[x.idx_np] == 0.0)
                        and not np.any(s.needs_logits_processing[x.idx_np])
                        and s.get_logprobs_dims(x.idx_np) is None)

        def est_step():
            o.est.step(x.idx, x.num_sampled, x.num_rejected, dead_row_cut=dead_cut)

        def est_calibrate():
            o.est.calibrate_confidence_probs(x.conf[arm], x.idx)

        def record_conf():
            o.mgr.record_confidences(x.conf[arm], x.batch, planned=x.planned)

        def get_num_tokens():
            o.mgr.get_num_tokens(x.ntok, x.drafts)

        def sample_greedy():
            o.greedy.sample(x.logits, x.exp, x.idx, x.idx_np, x.pos, x.input_ids, x.local_pos)

        def sample_sampled():
            o.sampled.sample(x.logits, x.exp, x.idx, x.idx_np, x.pos, x.input_ids, x.local_pos)

        def draft_predict():
            o.est_draft.predict(x.dlogits, x.idx, draft_step, o.draft_conf,
                                o.greedy.sampling_states.temperature.gpu)

        def draft_vp_predict():
            e = o.est_draft
            if hasattr(e, "predict_from_stats"):
                e.predict_from_stats(x.vmax, x.vsum, x.idx, draft_step, o.draft_conf)
                return
            from vllm.utils.math_utils import next_power_of_2

            num_tokens, nparts = x.vmax.shape
            o.ae._predict_kernel[(num_tokens,)](
                e.features, e.features.stride(0), e.predictions, e.predictions.stride(0),
                o.draft_conf, o.draft_conf.stride(0), e.slope, e.intercepts,
                x.vmax, x.vmax.stride(0), x.vsum, x.vsum.stride(0),
                x.idx, x.idx.stride(0), draft_step, num_tokens, nparts,
                per_token_step=draft_step.dim() > 0,
                NUM_SPECULATIVE_STEPS=e.num_speculative_steps,
                MAX_LOG_ODDS=o.ae._MAX_LOG_ODDS,
                PADDED_VOCAB_NUM_BLOCKS=next_power_of_2(nparts),
            )

        def greedy_total():
            greedy_check()
            est_step()
            est_calibrate()
            record_conf()
            get_num_tokens()

        return {
            "greedy_check": greedy_check, "est_step": est_step,
            "est_calibrate": est_calibrate, "record_conf": record_conf,
            "get_num_tokens": get_num_tokens, "greedy_total": greedy_total,
            "sample_greedy": sample_greedy, "sample_sampled": sample_sampled,
            "draft_predict": draft_predict, "draft_vp_predict": draft_vp_predict,
        }

    names = list(comp_fns(ARMS[0], shape_inputs(1, 1)).keys())
    orders = [ARMS[i:] + ARMS[:i] for i in range(len(ARMS))]
    orders += [tuple(reversed(o)) for o in orders]
    results = []
    for num_reqs, depth in GRID:
        x = shape_inputs(num_reqs, depth)
        fns = {arm: comp_fns(arm, x) for arm in ARMS}
        for name in names:
            calls = {arm: fns[arm][name] for arm in ARMS}
            for _ in range(args.warmup):
                for arm in ARMS:
                    calls[arm]()
            torch.cuda.synchronize()
            host = {arm: [] for arm in ARMS}
            gpu = {arm: [] for arm in ARMS}
            ev = {arm: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for arm in ARMS}
            for i in range(args.iters):
                for arm in orders[i % len(orders)]:
                    s, e = ev[arm]
                    torch.cuda.synchronize()
                    s.record()
                    t0 = time.perf_counter_ns()
                    calls[arm]()
                    t1 = time.perf_counter_ns()
                    e.record()
                    e.synchronize()
                    host[arm].append((t1 - t0) / 1000.0)
                    gpu[arm].append(s.elapsed_time(e) * 1000.0)
            row = {"component": name, "num_reqs": num_reqs, "rows": 1 + depth}
            for arm in ARMS:
                h = trimmed(host[arm])
                g = trimmed(gpu[arm])
                row[arm] = {
                    "host_us_mean": round(statistics.fmean(h), 4),
                    "host_us_median": round(statistics.median(h), 4),
                    "host_us_se": round(statistics.stdev(h) / math.sqrt(len(h)), 4),
                    "gpu_us_mean": round(statistics.fmean(g), 4),
                    "gpu_us_se": round(statistics.stdev(g) / math.sqrt(len(g)), 4),
                    "n": len(h),
                }
            for arm in ("c48", "r1"):
                d = row[arm]["host_us_mean"] - row["1107"]["host_us_mean"]
                se = math.hypot(row[arm]["host_us_se"], row["1107"]["host_us_se"])
                row[f"{arm}_minus_1107_host_us"] = round(d, 4)
                row[f"{arm}_minus_1107_host_ci95_us"] = round(1.96 * se, 4)
                dg = row[arm]["gpu_us_mean"] - row["1107"]["gpu_us_mean"]
                seg = math.hypot(row[arm]["gpu_us_se"], row["1107"]["gpu_us_se"])
                row[f"{arm}_minus_1107_gpu_us"] = round(dg, 4)
                row[f"{arm}_minus_1107_gpu_ci95_us"] = round(1.96 * seg, 4)
            results.append(row)
            print(json.dumps({k: row[k] for k in row if k.endswith("_us") or k in ("component", "num_reqs", "rows")}), flush=True)

    out = {
        "boot": args.boot, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iters": args.iters, "warmup": args.warmup, "world_emulated": WORLD,
        "broadcast_calls": bcast["n"],
        "sampler_use_flashinfer": {a: bool(objs[a].sampled.use_flashinfer) for a in ARMS},
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
