#!/usr/bin/env python3
"""Single-GPU step cost of the worker's sampler and spec-decode post-step path.

Runs the real classes of the six files #980 changes (whichever copies are mounted
into the image) at the engine grid's decode shapes, with TP4 emulated the way
PR #980's own screen does it (a fake TP group of world size 4 whose device
communicator is rank-ordered, and parallel_state reporting TP 4), so every
TP > 1 branch the patch adds is taken. One process is one boot of one arm.

Components, in the model runner's order for one greedy DSpark decode step
(production recipe: DSpark, TileLang confidence head, adaptive verification,
copy drafts, greedy drafts, greedy verification):
  greedy_check      the sampler state reads _verifies_greedily makes
  est_step          OnlineAcceptanceEstimator.step (observe_verification)
  est_calibrate     OnlineAcceptanceEstimator.calibrate_confidence_probs (propose)
  record_conf       AdaptiveVerificationManager.record_confidences
  get_num_tokens    AdaptiveVerificationManager.get_num_tokens (next schedule)
  greedy_total      the five above back to back, one decode step
Other paths, for completeness (not the greedy decode path):
  sample_greedy     Sampler.sample on an all-greedy batch (non-spec rows)
  sample_sampled    Sampler.sample, temperature 0.8, top-k 50, top-p 0.95, unseeded
  draft_predict     OnlineAcceptanceEstimator.predict, one draft step
                    (DraftModel/DFlash drafters without a confidence head)
  draft_vp_predict  the DFlash vocab-parallel predictor from per-rank stats
                    (predict_from_stats, or 1107's inline launch)

Per component and shape: warmup calls, then rounds of iters calls; per round the
host time to issue the calls and the wall time to the end of their GPU work,
both per call. Reported: the median and min over rounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
MAX_REQS = 16
STEPS = 5
VOCAB = 129280
WORLD = 4
FILES = (
    "vllm/v1/worker/gpu/sample/sampler.py",
    "vllm/v1/worker/gpu/sample/states.py",
    "vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py",
    "vllm/v1/worker/gpu/spec_decode/adaptive_verification.py",
    "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py",
    "vllm/v1/worker/gpu/spec_decode/speculator.py",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--boot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--trace-out", default=None)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    import vllm
    import vllm.distributed.parallel_state as ps
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode import adaptive_verification as av_mod
    from vllm.v1.worker.gpu.spec_decode import acceptance_estimator as ae_mod
    from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
        OnlineAcceptanceEstimator,
    )
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        AdaptiveVerificationManager,
    )
    from vllm.v1.worker.gpu.states import RequestState
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator  # noqa: F401
    import vllm.v1.worker.gpu.spec_decode.speculator  # noqa: F401

    root = os.path.dirname(os.path.dirname(vllm.__file__))
    file_sha = {}
    for rel in FILES:
        with open(os.path.join(root, rel), "rb") as f:
            file_sha[rel] = hashlib.sha256(f.read()).hexdigest()[:16]

    # TP4 as PR #980's screen emulates it.
    bcast = {"n": 0}

    def fake_broadcast(tensor, src=0):
        bcast["n"] += 1
        return tensor

    tp_group = SimpleNamespace(
        world_size=WORLD,
        broadcast=fake_broadcast,
        device_communicator=SimpleNamespace(rank_order_reduce_scatter=True),
    )
    av_mod.get_tp_group = lambda: tp_group
    ps.model_parallel_is_initialized = lambda: True
    ps.get_tensor_model_parallel_world_size = lambda: WORLD

    # Warm the GPU clocks.
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    t_end = time.time() + 3.0
    while time.time() < t_end:
        a @ a
    torch.cuda.synchronize()
    del a

    req_states = RequestState(
        max_num_reqs=MAX_REQS,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        num_speculative_steps=STEPS,
        vocab_size=VOCAB,
        device=device,
    )

    def make_sampler(params):
        s = Sampler(
            max_num_reqs=MAX_REQS,
            vocab_size=VOCAB,
            device=device,
            req_states=req_states,
            num_speculative_tokens=STEPS + 1,
        )
        for i in range(MAX_REQS):
            s.add_request(i, 16, params)
        s.apply_staged_writes()
        return s

    greedy_sampler = make_sampler(SamplingParams(temperature=0.0))
    sampled_sampler = make_sampler(
        SamplingParams(temperature=0.8, top_p=0.95, top_k=50, seed=None)
    )

    est = OnlineAcceptanceEstimator.for_confidence_head(MAX_REQS, STEPS, device)
    est_draft = OnlineAcceptanceEstimator(MAX_REQS, STEPS, device)

    req_ns = SimpleNamespace(
        num_speculative_steps=STEPS,
        device=device,
        max_num_reqs=MAX_REQS,
        max_num_batched_tokens=8192,
        req_id_to_index={f"r{i}": i for i in range(MAX_REQS)},
        num_computed_tokens_np=np.full(MAX_REQS, 4096, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.full(MAX_REQS, 16, dtype=np.int32)),
    )
    mgr = AdaptiveVerificationManager(
        req_ns,
        torch.zeros(MAX_REQS + 1, dtype=torch.int32, device=device),
        1,
        max_total_logits=1 << 20,
        row_cost=0.035,
        planned_confidences=True,
        num_draft_steps=STEPS,
        tp_invariant_confidence=True,
        calibrated_confidence=lambda: est._refits > 0,
    )

    gen = torch.Generator(device=device).manual_seed(1234)
    conf_src = torch.rand((MAX_REQS, STEPS), generator=gen, device=device) * 0.9 + 0.05
    planned_src = conf_src * 0.95
    dead_cut = torch.full((MAX_REQS,), STEPS, dtype=torch.int32, device=device)
    draft_conf = torch.zeros((MAX_REQS, STEPS), dtype=torch.float32, device=device)
    draft_logits = torch.randn(
        (MAX_REQS, VOCAB), generator=gen, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    target_logits = torch.randn(
        (MAX_REQS * (STEPS + 1), VOCAB), generator=gen, device=device
    )
    vp_max = torch.randn((MAX_REQS, WORLD), generator=gen, device=device)
    vp_sum = torch.rand((MAX_REQS, WORLD), generator=gen, device=device) * 100 + 1
    draft_step = torch.tensor(1, dtype=torch.int32, device=device)

    def shape_inputs(num_reqs, depth):
        rows = 1 + depth
        idx_np = np.arange(num_reqs, dtype=np.int32)
        idx = torch.arange(num_reqs, dtype=torch.int32, device=device)
        n = num_reqs * rows
        exp_np = np.repeat(np.arange(num_reqs), rows)
        exp = torch.from_numpy(exp_np.astype(np.int32)).to(device)
        pos = torch.arange(n, dtype=torch.int64, device=device) + 4096
        input_ids = torch.ones(n, dtype=torch.int32, device=device)
        local_pos = torch.from_numpy(
            np.tile(np.arange(rows, dtype=np.int32), num_reqs)
        ).to(device)
        num_sampled = torch.from_numpy(
            (np.arange(num_reqs) % rows + 1).astype(np.int32)
        ).to(device)
        num_rejected = (depth + 1 - num_sampled).clamp_min(0).to(torch.int32)
        conf = conf_src[:num_reqs].clone()
        planned = planned_src[:num_reqs].clone()
        batch = SimpleNamespace(
            num_reqs=num_reqs, idx_mapping=idx, idx_mapping_np=idx_np, has_prefill=False
        )
        req_ids = [f"r{i}" for i in range(num_reqs)]
        ntok = {rid: 1 + depth for rid in req_ids}
        drafts = {rid: list(range(1, depth + 1)) for rid in req_ids}
        return SimpleNamespace(
            rows=rows, n=n, idx_np=idx_np, idx=idx, exp=exp, pos=pos,
            input_ids=input_ids, local_pos=local_pos, num_sampled=num_sampled,
            num_rejected=num_rejected, conf=conf, planned=planned, batch=batch,
            ntok=ntok, drafts=drafts, logits=target_logits[:n],
            dlogits=draft_logits[:num_reqs], vmax=vp_max[:num_reqs],
            vsum=vp_sum[:num_reqs],
        )

    def greedy_check(x):
        s = greedy_sampler
        st = s.sampling_states
        return bool(
            np.all(st.temperature.np[x.idx_np] == 0.0)
            and not np.any(s.needs_logits_processing[x.idx_np])
            and s.get_logprobs_dims(x.idx_np) is None
        )

    def est_step(x):
        est.step(x.idx, x.num_sampled, x.num_rejected, dead_row_cut=dead_cut)

    def est_calibrate(x):
        x.conf.copy_(conf_src[: x.conf.shape[0]])
        est.calibrate_confidence_probs(x.conf, x.idx)

    def record_conf(x):
        mgr.record_confidences(x.conf, x.batch, planned=x.planned)

    def get_num_tokens(x):
        mgr.get_num_tokens(x.ntok, x.drafts)

    def greedy_total(x):
        greedy_check(x)
        est_step(x)
        est_calibrate(x)
        record_conf(x)
        get_num_tokens(x)

    def sample_with(sampler, x):
        sampler.sample(
            x.logits, x.exp, x.idx, x.idx_np, x.pos, x.input_ids, x.local_pos
        )

    def sample_greedy(x):
        sample_with(greedy_sampler, x)

    def sample_sampled(x):
        sample_with(sampled_sampler, x)

    def draft_predict(x):
        est_draft.predict(
            x.dlogits, x.idx, draft_step, draft_conf,
            greedy_sampler.sampling_states.temperature.gpu,
        )

    has_from_stats = hasattr(est_draft, "predict_from_stats")

    def draft_vp_predict(x):
        if has_from_stats:
            est_draft.predict_from_stats(x.vmax, x.vsum, x.idx, draft_step, draft_conf)
            return
        # 1107's DFlashSpeculator.sample_draft inline launch.
        from vllm.utils.math_utils import next_power_of_2

        num_tokens, nparts = x.vmax.shape
        ae_mod._predict_kernel[(num_tokens,)](
            est_draft.features, est_draft.features.stride(0),
            est_draft.predictions, est_draft.predictions.stride(0),
            draft_conf, draft_conf.stride(0),
            est_draft.slope, est_draft.intercepts,
            x.vmax, x.vmax.stride(0), x.vsum, x.vsum.stride(0),
            x.idx, x.idx.stride(0), draft_step, num_tokens, nparts,
            per_token_step=draft_step.dim() > 0,
            NUM_SPECULATIVE_STEPS=est_draft.num_speculative_steps,
            MAX_LOG_ODDS=ae_mod._MAX_LOG_ODDS,
            PADDED_VOCAB_NUM_BLOCKS=next_power_of_2(nparts),
        )

    components = [
        ("greedy_check", greedy_check),
        ("est_step", est_step),
        ("est_calibrate", est_calibrate),
        ("record_conf", record_conf),
        ("get_num_tokens", get_num_tokens),
        ("greedy_total", greedy_total),
        ("sample_greedy", sample_greedy),
        ("sample_sampled", sample_sampled),
        ("draft_predict", draft_predict),
        ("draft_vp_predict", draft_vp_predict),
    ]

    # Optional: which lines of the six files each component executes.
    if args.trace_out:
        traced = {}
        x = shape_inputs(8, 5)
        for name, fn in components:
            for _ in range(3):
                fn(x)
            torch.cuda.synchronize()
            hits: dict[str, set[int]] = {}

            def tracer(frame, event, arg):
                f = frame.f_code.co_filename
                for rel in FILES:
                    if f.endswith(rel):
                        if event == "line":
                            hits.setdefault(rel, set()).add(frame.f_lineno)
                        return tracer
                return tracer

            sys.settrace(tracer)
            try:
                fn(x)
            finally:
                sys.settrace(None)
            torch.cuda.synchronize()
            traced[name] = {k: sorted(v) for k, v in hits.items()}
        with open(args.trace_out, "w") as f:
            json.dump({"arm": args.arm, "files": file_sha, "lines": traced}, f)

    results = []
    errors = {}
    for num_reqs, depth in GRID:
        x = shape_inputs(num_reqs, depth)
        for name, fn in components:
            key = f"{name}"
            try:
                for _ in range(args.warmup):
                    fn(x)
                torch.cuda.synchronize()
                host, wall = [], []
                for _ in range(args.rounds):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter_ns()
                    for _ in range(args.iters):
                        fn(x)
                    t1 = time.perf_counter_ns()
                    torch.cuda.synchronize()
                    t2 = time.perf_counter_ns()
                    host.append((t1 - t0) / args.iters / 1000.0)
                    wall.append((t2 - t0) / args.iters / 1000.0)
                results.append({
                    "component": key, "num_reqs": num_reqs, "depth": depth,
                    "rows": 1 + depth,
                    "host_us_median": round(statistics.median(host), 3),
                    "host_us_min": round(min(host), 3),
                    "wall_us_median": round(statistics.median(wall), 3),
                    "wall_us_min": round(min(wall), 3),
                    "wall_us_rounds": [round(v, 3) for v in wall],
                })
            except Exception as exc:  # record and continue
                errors[f"{key}@{num_reqs}x{depth}"] = f"{type(exc).__name__}: {exc}"[:500]

    out = {
        "arm": args.arm,
        "boot": args.boot,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files_sha256_16": file_sha,
        "world_emulated": WORLD,
        "max_num_reqs": MAX_REQS,
        "num_speculative_steps": STEPS,
        "vocab": VOCAB,
        "rounds": args.rounds,
        "iters": args.iters,
        "warmup": args.warmup,
        "sampler_use_flashinfer": bool(sampled_sampler.use_flashinfer),
        "has_tp_lockstep_sampling": hasattr(sampled_sampler, "_tp_lockstep_sampling"),
        "broadcast_calls": bcast["n"],
        "est_refits": est._refits,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "results": results,
        "errors": errors,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in out if k != "results"}, indent=1))


if __name__ == "__main__":
    main()
