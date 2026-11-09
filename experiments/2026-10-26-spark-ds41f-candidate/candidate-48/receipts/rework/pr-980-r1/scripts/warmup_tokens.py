#!/usr/bin/env python3
"""Which tokens the startup sampler warmup leaves behind, per arm.

The model runner's warmup (vllm/v1/worker/gpu/warmup.py) runs one real step of
up to max_num_reqs requests with SamplingParams.for_sampler_warmup()
(temperature 0.9, top-k 50, top-p 0.9, unseeded) in slots 0..n-1. Their sampled
tokens stay in req_states.last_sampled_tokens, which every later dummy step,
including each engine grid step, hands the drafter (model_runner._dummy_run ->
speculator.propose(last_sampled=...) -> prepare_dflash_inputs), and the DSpark
draft layers route MoE on them.

This replays that sampling decision in a fresh process seeded as the worker is
(set_random_seed(0)), with TP4 emulated: the same 16 slots, the warmup's
temperature / top-k / top-p, fixed logits. It prints the per-slot seeds and the
sampled tokens, so two boots of one arm can be compared with each other and with
another arm's.
"""

import argparse
import hashlib
import json
import random

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--boot", required=True)
    args = ap.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    import vllm.distributed.parallel_state as ps
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    ps.model_parallel_is_initialized = lambda: True
    ps.get_tensor_model_parallel_world_size = lambda: 4
    n, vocab = 16, 129280
    req_states = RequestState(n, 8192, 8192, 5, vocab, device)
    sampler = Sampler(n, vocab, device, req_states, num_speculative_tokens=6)
    params = SamplingParams(temperature=0.9, top_k=50, top_p=0.9)
    for i in range(n):
        sampler.add_request(i, 16, params)
    sampler.apply_staged_writes()
    gen = torch.Generator(device=device).manual_seed(42)
    logits = torch.randn((n, vocab), generator=gen, device=device)
    idx_np = np.arange(n, dtype=np.int32)
    idx = torch.from_numpy(idx_np).to(device)
    pos = torch.full((n,), 16, dtype=torch.int64, device=device)
    tokens = []
    for step in range(3):
        sampled, _ = sampler.sample(
            logits, idx, idx, idx_np, pos + step,
            torch.ones(n, dtype=torch.int32, device=device),
            torch.zeros(n, dtype=torch.int32, device=device),
        )
        tokens.append(sampled.view(-1).tolist())
    seeds = sampler.sampling_states.seeds.np[:n].tolist()
    print("WARMUP_TOKENS " + json.dumps({
        "arm": args.arm, "boot": args.boot,
        "flashinfer_path": bool(sampler.use_flashinfer
                                and not getattr(sampler, "_tp_lockstep_sampling", lambda: False)()),
        "seeds_sha256_16": hashlib.sha256(json.dumps(seeds).encode()).hexdigest()[:16],
        "first_step_tokens": tokens[0],
        "tokens_sha256_16": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16],
    }))


if __name__ == "__main__":
    main()
