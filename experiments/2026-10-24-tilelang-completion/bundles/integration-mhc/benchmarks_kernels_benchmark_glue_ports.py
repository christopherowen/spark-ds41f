# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired per-call timing of the last torch glue the DS4.1 serving step launched
against the TileLang launches that replaced it, at serving shapes:

- ``topk_topp``: the sampler's per-row top-k and top-p, torch indexing of the
  UVA sampling state against ``SamplingStates.get_top_k_top_p``'s row gathers,
  for one, 16 and 64 requests of 1 + 5 rows;
- ``alternates``: the sink's alternate logprobs and ranks
  (``SinkSampler.alternate_logprobs_of``), torch's index math, gather, clamps,
  per-alternate log-softmax and column copies against one copy launch, one
  log-softmax launch and the rank launches, FP32 processed and BF16 raw logits at
  DS4.1's vocabulary (raw rows TP3's padded stride), one and 16 requests;
- ``policy``: the sink policy's recovered logprobs and ranks, torch's two zero
  fills and the policy against the policy alone (it writes the zeros now);
- ``masks``: a prefill forward's image-token and routing masks for TileKernels'
  gate, torch's two comparisons, conjunction and negation against one launch.

Each case is measured on eight fresh inputs; on each, CUDA graphs of 100 calls of
both run back to back, alternating which goes first, and the case's 95% interval
comes from the eight per-input differences. Before timing, each pair's outputs are
compared bit for bit. The gate (``--gate``) is that no port is slower (the
interval's lower bound above zero).
"""

import argparse
import json
import math
import statistics
from collections.abc import Callable

import numpy as np
import torch

from vllm.models.deepseek_v4_1.tilelang import router
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.sink import NUM_SINK_EVENT_FIELDS, SinkConfig
from vllm.v1.worker.gpu.sample import tilelang_sampling
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.sink import SinkAlternates, SinkSampler
from vllm.v1.worker.gpu.sample.states import SamplingStates

DEVICE = torch.device("cuda")
CALLS = 100
VOCAB = 129280
TP3_STRIDE = 129408
DRAFT_ROWS = 6  # 1 + 5 drafted rows a request
MAX_REQS = 64
LO = 129264  # the target's image sentinel token
SINK = SinkConfig(column=2455, threshold=85000.0, rejection=True, max_retries=3)

SHAPES = {
    "topk_topp": {"1 request": 1, "16 requests": 16, "64 requests": 64},
    "alternates": {"1 request": 1, "16 requests": 16},
    "policy": {"1 request": 1, "16 requests": 16, "64 requests": 64},
    "masks": {"2K prefill": 2048, "8K prefill": 8192, "16K prefill": 16384},
}


def graph(function: Callable[[], None]) -> torch.cuda.CUDAGraph:
    """A CUDA graph of ``CALLS`` calls of ``function``, after one warm-up call."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        function()
    torch.cuda.current_stream().wait_stream(side)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for _ in range(CALLS):
            function()
    return capture


def elapsed(capture: torch.cuda.CUDAGraph) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    capture.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / CALLS


def _cu(requests: int) -> torch.Tensor:
    return torch.arange(
        0, (requests + 1) * DRAFT_ROWS, DRAFT_ROWS, dtype=torch.int32, device=DEVICE
    )


# --- the torch glue the ports replace -------------------------------------------


def torch_alternates(sink, logits, cu, num_sampled, alternates):
    """``SinkSampler.alternate_logprobs_of`` before the port."""
    num_reqs = alternates.shape[0]
    rows = cu[:num_reqs].to(torch.int64) + (num_sampled.to(torch.int64) - 1).clamp(
        min=0
    )
    row_logits = logits[rows]
    logprobs = torch.empty(
        (num_reqs, sink.num_alternates), dtype=torch.float32, device=DEVICE
    )
    ranks = torch.empty(
        (num_reqs, sink.num_alternates), dtype=torch.int64, device=DEVICE
    )
    for i in range(sink.num_alternates):
        tokens = alternates[:, i].clamp(min=0)
        scores = compute_topk_scores(row_logits, 0, tokens, None)
        logprobs[:, i] = scores.logprobs[:, 0]
        ranks[:, i] = scores.selected_token_ranks
    return logprobs, ranks


def torch_masks(input_ids, is_padding, count):
    """The router's image-token mask and routing mask before the port."""
    return (input_ids >= LO) & (input_ids < LO + count), ~is_padding


# --- the cases ---------------------------------------------------------------------


def case(kind: str, size: int, seed: int, dtype: torch.dtype):
    """(torch call, TileLang call, outputs to compare) on fresh inputs."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    rng = np.random.default_rng(seed)
    out: dict = {}
    if kind == "topk_topp":
        states = SamplingStates(MAX_REQS, VOCAB)
        for slot in range(MAX_REQS):
            states.add_request(
                slot,
                SamplingParams(
                    top_k=int(rng.choice([20, 50, VOCAB])),
                    top_p=float(rng.choice([0.9, 0.95, 1.0])),
                ),
            )
        states.add_request(0, SamplingParams(top_k=20, top_p=0.9))
        states.apply_staged_writes()
        slots = torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:size]
        rows = slots.repeat_interleave(DRAFT_ROWS)
        slots_np = np.concatenate([[0], slots[1:].cpu().numpy()])
        rows[:DRAFT_ROWS] = 0

        def reference():
            out["ref"] = (states.top_k.gpu[rows], states.top_p.gpu[rows])

        def candidate():
            out["new"] = states.get_top_k_top_p(rows, slots_np)

        return reference, candidate, out
    if kind in ("alternates", "policy"):
        sink = SinkSampler(SINK, MAX_REQS, 5120, VOCAB, 5, DEVICE, False)
        cu = _cu(size)
        num_sampled = torch.randint(
            0, DRAFT_ROWS + 1, (size,), generator=gen, device=DEVICE, dtype=torch.int32
        )
        alternates = torch.randint(0, VOCAB, (size, 3), generator=gen, device=DEVICE)
        alternates[::5, 2] = -1
    if kind == "alternates":
        stride = VOCAB if dtype == torch.float32 else TP3_STRIDE
        storage = torch.randn(size * DRAFT_ROWS, stride, generator=gen, device=DEVICE)
        logits = storage.to(dtype)[:, :VOCAB]

        def reference():
            out["ref"] = torch_alternates(sink, logits, cu, num_sampled, alternates)

        def candidate():
            new = sink.alternate_logprobs_of(logits, cu, num_sampled, alternates, False)
            out["new"] = (new.logprobs, new.ranks)

        return reference, candidate, out
    if kind == "policy":
        width = DRAFT_ROWS
        sampled = torch.randint(0, VOCAB, (size, width), generator=gen, device=DEVICE)
        flags = (torch.rand(size * width, generator=gen, device=DEVICE) < 0.2).to(
            torch.int8
        )
        idx = torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:size]
        decoding = torch.ones(size, dtype=torch.int8, device=DEVICE)
        drafts = torch.randint(
            0, VOCAB, (size * width,), generator=gen, device=DEVICE, dtype=torch.int32
        )
        values = torch.rand(size * width, generator=gen, device=DEVICE) * 1e5
        new = SinkAlternates(
            alternates,
            torch.randn(size, 3, generator=gen, device=DEVICE),
            torch.randint(1, 99, (size, 3), generator=gen, device=DEVICE),
        )
        total_len = torch.full((MAX_REQS,), 40, dtype=torch.int32, device=DEVICE)
        bin_counts = torch.ones((MAX_REQS, VOCAB), dtype=torch.int32, device=DEVICE)
        # Every call starts from the same state: the policy's per-slot state and the
        # step's counts are restored outside the timed graphs, and with draft zero
        # never flagged here no call changes them.
        flags.view(size, width)[:, 0] = 0
        events = torch.empty(
            (size, NUM_SINK_EVENT_FIELDS), dtype=torch.int32, device=DEVICE
        )
        tokens = torch.empty((size, width + 1), dtype=torch.int64, device=DEVICE)
        num_out = torch.empty(size, dtype=torch.int32, device=DEVICE)
        num_rejected = torch.zeros_like(num_sampled)

        def launch(recovered_logprobs, recovered_ranks):
            tilelang_sampling.sink_policy(
                sampled,
                num_sampled,
                num_rejected,
                tokens,
                num_out,
                events,
                recovered_logprobs,
                recovered_ranks,
                cu,
                idx,
                decoding,
                flags,
                values,
                drafts,
                new.tokens,
                new.logprobs,
                new.ranks,
                sink.alternates,
                sink.alternate_logprobs,
                sink.alternate_ranks,
                sink.tries,
                total_len,
                bin_counts,
                sink.counts,
                sink.num_alternates,
                True,
                True,
                NUM_SINK_EVENT_FIELDS,
            )

        def reference():
            lp = torch.zeros(size, dtype=torch.float32, device=DEVICE)
            rk = torch.zeros(size, dtype=torch.int64, device=DEVICE)
            launch(lp, rk)
            out["ref"] = (lp, rk)

        def candidate():
            lp = torch.empty(size, dtype=torch.float32, device=DEVICE)
            rk = torch.empty(size, dtype=torch.int64, device=DEVICE)
            launch(lp, rk)
            out["new"] = (lp, rk)

        return reference, candidate, out
    # The masks of a prefill forward of ``size`` rows, its last rows padding.
    input_ids = torch.randint(
        0, VOCAB, (size,), generator=gen, device=DEVICE, dtype=torch.int32
    )
    input_ids[::9] = LO
    is_padding = torch.zeros(size, dtype=torch.bool, device=DEVICE)
    is_padding[size - 3 :] = True

    def reference():
        out["ref"] = torch_masks(input_ids, is_padding, 1)

    def candidate():
        out["new"] = router._masks(input_ids, LO, 1, is_padding)

    return reference, candidate, out


def same(a, b) -> bool:
    return all(
        torch.equal(
            x.view(torch.int32) if x.dtype == torch.float32 else x,
            y.view(torch.int32) if y.dtype == torch.float32 else y,
        )
        for x, y in zip(a, b)
    )


def measure(kind: str, size: int, samples: int, dtype: torch.dtype) -> dict:
    ours, theirs = [], []
    for sample in range(samples):
        reference, candidate, out = case(kind, size, 7919 + sample, dtype)
        reference()
        candidate()
        torch.accelerator.synchronize()
        if not same(out["ref"], out["new"]):
            raise SystemExit(f"{kind} at {size}: outputs differ from the reference")
        a, b = graph(candidate), graph(reference)
        order = (a, b) if sample % 2 == 0 else (b, a)
        times = {id(g): elapsed(g) for g in order}
        ours.append(times[id(a)])
        theirs.append(times[id(b)])
        del a, b, order
    paired = [x - y for x, y in zip(ours, theirs)]
    mean = statistics.fmean(paired)
    half = 2.365 * statistics.stdev(paired) / math.sqrt(len(paired))
    return {
        "kernel": kind,
        "size": size,
        "dtype": str(dtype).removeprefix("torch."),
        "tilelang_us": statistics.median(ours),
        "torch_us": statistics.median(theirs),
        "difference_us": mean,
        "interval_us": [mean - half, mean + half],
        "slower": mean - half > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--kinds", default=",".join(SHAPES))
    args = parser.parse_args()
    results = []
    for kind in args.kinds.split(","):
        dtypes = (
            (torch.float32, torch.bfloat16)
            if kind == "alternates"
            else (torch.float32,)
        )
        for dtype in dtypes:
            for name, size in SHAPES[kind].items():
                result = measure(kind, size, args.samples, dtype)
                result["shape"] = name
                results.append(result)
                print(json.dumps(result), flush=True)
    slower = [
        f"{r['kernel']} ({r['dtype']}) at {r['shape']}" for r in results if r["slower"]
    ]
    print(json.dumps({"summary": "glue ports", "slower": slower}), flush=True)
    if args.gate and slower:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
