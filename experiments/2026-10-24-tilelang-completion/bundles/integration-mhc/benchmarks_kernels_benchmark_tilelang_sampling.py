# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TileLang sampling kernels against the Triton kernels they replace, per call
(and the Gumbel finish against the torch argmax and gather it replaced, reported in
the "triton_us" column).

Each kernel runs at the shapes a DeepSeek V4.1 deployment launches it (vocabulary
129280; tensor-parallel shards of 32320; five drafts, so six rows per request;
three sink alternates). For each shape, eight samples on fresh inputs: a CUDA
graph of 100 calls of each implementation, replayed as a pair whose order
alternates. One JSON line per kernel and shape with both medians, the mean
difference and its 95% interval. ``--gate`` exits non-zero when a shape is slower
than Triton by more than 1% at the interval's lower end.
"""

import argparse
import json
import math
import statistics
from collections.abc import Callable

import torch
import triton

from vllm.v1.worker.gpu.sample import gumbel, sink
from vllm.v1.worker.gpu.sample import tilelang_sampling as tls
from vllm.v1.worker.gpu.spec_decode import greedy_verification
from vllm.v1.worker.gpu.spec_decode.dspark import greedy

DEVICE = torch.device("cuda")
VOCAB = 129280
SHARD = VOCAB // 4
WIDTH = 6
ALTERNATES = 3
SAMPLES = 8
CALLS = 100


def graph(function: Callable[[], None]) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for _ in range(CALLS):
            function()
    return capture


def elapsed_us(capture: torch.cuda.CUDAGraph) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS


def measure(name: str, shape: str, make: Callable[[int], tuple]) -> dict:
    """``make(sample)`` returns ``(ours, theirs)`` closures over fresh inputs."""
    ours_us, theirs_us, differences = [], [], []
    for sample in range(SAMPLES):
        ours, theirs = make(sample)
        a, b = graph(ours), graph(theirs)
        pair = (a, b) if sample % 2 == 0 else (b, a)
        times = {id(c): elapsed_us(c) for c in pair}
        ours_us.append(times[id(a)])
        theirs_us.append(times[id(b)])
        differences.append(times[id(a)] - times[id(b)])
        del a, b
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(SAMPLES)
    theirs_median = statistics.median(theirs_us)
    row = {
        "kernel": name,
        "shape": shape,
        "tilelang_us": round(statistics.median(ours_us), 3),
        "triton_us": round(theirs_median, 3),
        "ratio": round(statistics.median(ours_us) / theirs_median, 4),
        "difference_us": round(mean, 3),
        "interval_us": [round(mean - half, 3), round(mean + half, 3)],
        "slower": (mean - half) > 0.01 * theirs_median,
    }
    print(json.dumps(row), flush=True)
    return row


# The Triton kernels, launched as greedy.py and greedy_verification.py did.


def triton_markov_pair(base, bias, vocab_start, pairs, values, indices):
    rows, vocab = base.shape
    parts = triton.cdiv(vocab, greedy._BLOCK)
    greedy._partial_argmax[(rows, parts)](
        base,
        bias,
        values,
        indices,
        base.stride(0),
        bias.stride(0),
        vocab,
        parts,
        greedy._BLOCK,
    )
    greedy._finish_argmax_pair[(rows,)](
        values, indices, pairs, vocab_start, parts, triton.next_power_of_2(parts)
    )


def triton_shard_pair(logits, vocab, vocab_start, pairs, values, indices):
    rows = logits.shape[0]
    parts = triton.cdiv(vocab, greedy._BLOCK)
    greedy._partial_argmax_logits[(rows, parts)](
        logits, values, indices, logits.stride(0), vocab, parts, greedy._BLOCK
    )
    greedy._finish_argmax_pair[(rows,)](
        values, indices, pairs, vocab_start, parts, triton.next_power_of_2(parts)
    )


def triton_select(pairs, output):
    rows, width = pairs.shape
    ranks = width // 2
    greedy._select_first_argmax[(rows,)](
        pairs, output, output.stride(0), ranks, triton.next_power_of_2(ranks)
    )


def triton_markov(base, bias, output, values, indices):
    rows, vocab = base.shape
    parts = triton.cdiv(vocab, greedy._BLOCK)
    greedy._partial_argmax[(rows, parts)](
        base,
        bias,
        values,
        indices,
        base.stride(0),
        bias.stride(0),
        vocab,
        parts,
        greedy._BLOCK,
    )
    greedy._finish_argmax[(rows,)](
        values, indices, output, output.stride(0), parts, triton.next_power_of_2(parts)
    )


def triton_greedy_rejection(target, drafts, cu, steps):
    reqs = cu.shape[0] - 1
    sampled = drafts.new_empty(reqs, steps + 1, dtype=torch.int64)
    num_sampled = sampled.new_empty(reqs, dtype=torch.int32)
    greedy_verification._greedy_rejection_kernel[(reqs,)](
        sampled, sampled.stride(0), num_sampled, target, drafts, cu, num_warps=1
    )
    return sampled, num_sampled


def generator(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def gumbel_case(tokens: int, dtype: torch.dtype):
    blocks = triton.cdiv(VOCAB, tls.GUMBEL_BLOCK)

    def make(sample: int):
        g = generator(sample)
        logits = (torch.randn(tokens, VOCAB, generator=g, device=DEVICE) * 4).to(dtype)
        mapping = torch.arange(tokens, device=DEVICE)
        temperature = torch.ones(tokens, device=DEVICE)
        seeds = torch.randint(0, 1 << 62, (tokens,), generator=g, device=DEVICE)
        pos = torch.randint(0, 1 << 20, (tokens,), generator=g, device=DEVICE)
        argmax = torch.empty(tokens, blocks, dtype=torch.int64, device=DEVICE)
        best = torch.empty(tokens, blocks, device=DEVICE)
        kernel = tls._kernel(
            tls.gumbel_sample_kernel,
            *tls.gumbel_sample_key(
                dtype, torch.int64, False, False, run=tls._triton_run(logits)
            ),
        )

        def ours():
            kernel(
                argmax,
                best,
                logits,
                mapping,
                seeds,
                pos,
                temperature,
                best.unsqueeze(1),
                pos,
            )

        def theirs():
            gumbel._gumbel_sample_kernel[(tokens, blocks)](
                argmax,
                argmax.stride(0),
                best,
                best.stride(0),
                None,
                0,
                0,
                None,
                logits,
                logits.stride(0),
                mapping,
                seeds,
                pos,
                temperature,
                VOCAB,
                BLOCK_SIZE=tls.GUMBEL_BLOCK,
                IS_DRAFTING=False,
                APPLY_TEMPERATURE=False,
                USE_FP64=False,
                PER_TOKEN_COL=False,
            )

        return ours, theirs

    return make


def gumbel_finish_case(tokens: int):
    """Each token's id from its blocks' argmax and maximum: one warp a token against
    torch's argmax over the blocks and gather of its argmax."""
    blocks = triton.cdiv(VOCAB, tls.GUMBEL_BLOCK)

    def make(sample: int):
        g = generator(sample)
        best = torch.randn(tokens, blocks, generator=g, device=DEVICE)
        argmax = torch.randint(0, VOCAB, (tokens, blocks), generator=g, device=DEVICE)

        def theirs():
            max_block_idx = best.argmax(dim=-1, keepdim=True)
            argmax.gather(dim=-1, index=max_block_idx).view(-1)

        return (lambda: tls.gumbel_finish(argmax, best)), theirs

    return make


def temperature_case(tokens: int):
    def make(sample: int):
        g = generator(sample)
        logits = torch.randn(tokens, VOCAB, generator=g, device=DEVICE)
        mapping = torch.arange(tokens, device=DEVICE)
        temperature = torch.full((tokens,), 0.7, device=DEVICE)
        return (
            lambda: tls.apply_temperature(logits, mapping, temperature),
            lambda: gumbel._temperature_kernel[(tokens, triton.cdiv(VOCAB, 8192))](
                logits, logits.stride(0), mapping, temperature, VOCAB, BLOCK_SIZE=8192
            ),
        )

    return make


def _verification_batch(reqs: int, g: torch.Generator):
    cu = torch.arange(0, (reqs + 1) * WIDTH, WIDTH, dtype=torch.int32, device=DEVICE)
    num_sampled = torch.randint(1, WIDTH + 1, (reqs,), generator=g, device=DEVICE).to(
        torch.int32
    )
    drafts = torch.randint(
        0, VOCAB, (reqs * WIDTH + 1,), generator=g, device=DEVICE
    ).to(torch.int32)
    sampled = torch.randint(0, VOCAB, (reqs, WIDTH), generator=g, device=DEVICE)
    mapping = torch.arange(reqs, device=DEVICE)
    return cu, num_sampled, drafts, sampled, mapping


def candidates_case(reqs: int, temperature: float):
    candidates = ALTERNATES + 1
    blocks = triton.cdiv(VOCAB, sink._CANDIDATE_BLOCK_SIZE)

    def make(sample: int):
        g = generator(sample)
        cu, num_sampled, drafts, sampled, mapping = _verification_batch(reqs, g)
        logits = (torch.randn(reqs * WIDTH, VOCAB, generator=g, device=DEVICE) * 4).to(
            torch.bfloat16
        )
        temp = torch.full((reqs,), temperature, device=DEVICE)
        seeds = torch.randint(0, 1 << 62, (reqs,), generator=g, device=DEVICE)
        pos = torch.randint(0, 1 << 20, (reqs * WIDTH,), generator=g, device=DEVICE)
        values = torch.empty(reqs, blocks * candidates, device=DEVICE)
        indices = torch.empty(
            reqs, blocks * candidates, dtype=torch.int64, device=DEVICE
        )
        kernel = tls._kernel(
            tls.sink_candidates_kernel,
            *tls.sink_candidates_key(
                torch.bfloat16, torch.int64, candidates, True, tls._triton_run(logits)
            ),
        )

        def ours():
            kernel(
                values,
                indices,
                logits,
                cu,
                num_sampled,
                sampled,
                drafts,
                mapping,
                temp,
                seeds,
                pos,
            )

        def theirs():
            sink._sink_candidates_kernel[(reqs, blocks)](
                values,
                indices,
                blocks * candidates,
                logits,
                logits.stride(0),
                VOCAB,
                cu,
                num_sampled,
                sampled,
                sampled.stride(0),
                drafts,
                mapping,
                temp,
                seeds,
                pos,
                BLOCK_SIZE=sink._CANDIDATE_BLOCK_SIZE,
                NUM_CANDIDATES=candidates,
                USE_FP64=False,
                MASK_REJECTED_DRAFT=True,
            )

        return ours, theirs

    return make


def policy_case(reqs: int):
    def make(sample: int):
        g = generator(sample)
        cu, num_sampled, drafts, sampled, mapping = _verification_batch(reqs, g)
        rows = reqs * WIDTH
        num_rejected = torch.zeros(reqs, dtype=torch.int32, device=DEVICE)
        decoding = torch.ones(reqs, dtype=torch.int8, device=DEVICE)
        flags = (torch.rand(rows, generator=g, device=DEVICE) < 0.05).to(torch.int8)
        values = torch.rand(rows, generator=g, device=DEVICE) * 2e5
        new = torch.randint(0, VOCAB, (reqs, ALTERNATES), generator=g, device=DEVICE)
        alternates = torch.randint(
            0, VOCAB, (reqs, ALTERNATES), generator=g, device=DEVICE
        )
        alt_lp = torch.zeros(reqs, ALTERNATES, device=DEVICE)
        alt_rk = torch.zeros(reqs, ALTERNATES, dtype=torch.int64, device=DEVICE)
        tries = torch.zeros(reqs, dtype=torch.int32, device=DEVICE)
        total_len = torch.full((reqs,), 1000, dtype=torch.int32, device=DEVICE)
        bins = torch.ones(reqs, VOCAB, dtype=torch.int32, device=DEVICE)
        counts = torch.zeros(3, dtype=torch.int64, device=DEVICE)
        events = torch.empty(reqs, 6, dtype=torch.int32, device=DEVICE)
        tokens = torch.empty(reqs, WIDTH + 1, dtype=torch.int64, device=DEVICE)
        num_out = torch.empty(reqs, dtype=torch.int32, device=DEVICE)
        # The step's state is restored by the kernels' own writes being idempotent
        # here: decoding rows rarely flagged, tries reset each call.
        args = (
            sampled,
            num_sampled,
            num_rejected,
            tokens,
            num_out,
            events,
            alt_lp,
            alt_rk,
            cu,
            mapping,
            decoding,
            flags,
            values,
            drafts,
            new,
            alt_lp,
            alt_rk,
            alternates,
            alt_lp,
            alt_rk,
            tries,
            total_len,
            bins,
        )

        def ours():
            tls.sink_policy(*args, counts, ALTERNATES, True, False, 6)

        def theirs():
            sink._sink_policy_kernel[(reqs,)](
                *args,
                bins.stride(0),
                counts,
                NUM_ALTERNATES=ALTERNATES,
                REJECTION=True,
                LOGPROBS=False,
                SAMPLED_WIDTH=WIDTH,
                PADDED_WIDTH=triton.next_power_of_2(WIDTH + 1),
                PADDED_ALTERNATES=triton.next_power_of_2(ALTERNATES),
                NUM_EVENT_FIELDS=6,
            )

        return ours, theirs

    return make


def markov_pair_case(rows: int):
    def make(sample: int):
        g = generator(sample)
        base = (torch.randn(rows, WIDTH - 1, SHARD, generator=g, device=DEVICE)).to(
            torch.bfloat16
        )[:, 2]
        bias = torch.randn(rows, SHARD, generator=g, device=DEVICE).to(torch.bfloat16)
        pairs = torch.empty(rows, 2, device=DEVICE)
        shape = greedy.scratch_shape(rows, SHARD)
        values = torch.empty(shape, device=DEVICE)
        indices = torch.empty(shape, dtype=torch.int32, device=DEVICE)
        return (
            lambda: tls.greedy_markov_shard_pair(
                base, bias, SHARD, pairs, values, indices
            ),
            lambda: triton_markov_pair(base, bias, SHARD, pairs, values, indices),
        )

    return make


def shard_pair_case(rows: int):
    def make(sample: int):
        g = generator(sample)
        logits = torch.randn(rows, SHARD + 64, generator=g, device=DEVICE).to(
            torch.bfloat16
        )
        pairs = torch.empty(rows, 2, device=DEVICE)
        shape = greedy.scratch_shape(rows, SHARD)
        values = torch.empty(shape, device=DEVICE)
        indices = torch.empty(shape, dtype=torch.int32, device=DEVICE)
        return (
            lambda: tls.greedy_shard_pair(logits, SHARD, SHARD, pairs, values, indices),
            lambda: triton_shard_pair(logits, SHARD, SHARD, pairs, values, indices),
        )

    return make


def select_case(rows: int):
    def make(sample: int):
        g = generator(sample)
        values = torch.randn(rows, 4, generator=g, device=DEVICE)
        index = torch.randint(0, VOCAB, (rows, 4), generator=g, device=DEVICE).float()
        pairs = torch.stack([values, index], dim=-1).view(rows, 8)
        out = torch.empty(rows, dtype=torch.int64, device=DEVICE)
        return (
            lambda: tls.select_first_argmax(pairs, out),
            lambda: triton_select(pairs, out),
        )

    return make


def markov_full_case(rows: int):
    def make(sample: int):
        g = generator(sample)
        base = torch.randn(rows, VOCAB, generator=g, device=DEVICE).to(torch.bfloat16)
        bias = torch.randn(rows, VOCAB, generator=g, device=DEVICE).to(torch.bfloat16)
        out = torch.empty(rows, WIDTH, dtype=torch.int64, device=DEVICE)[:, 1]
        shape = greedy.scratch_shape(rows, VOCAB)
        values = torch.empty(shape, device=DEVICE)
        indices = torch.empty(shape, dtype=torch.int32, device=DEVICE)
        return (
            lambda: tls.sample_greedy_markov(base, bias, out, values, indices),
            lambda: triton_markov(base, bias, out, values, indices),
        )

    return make


def rejection_case(reqs: int):
    """Greedy verification at TP4 from the gathered 16-byte pair rows, one launch,
    against Triton's selection over the packed pairs and its verification."""
    ranks = 4

    def make(sample: int):
        g = generator(sample)
        cu, _, drafts, _, _ = _verification_batch(reqs, g)
        rows = reqs * WIDTH
        # Each row's first maximum is mostly the draft that verifies it.
        values = torch.randn(rows, ranks, generator=g, device=DEVICE)
        index = torch.randint(0, VOCAB, (rows, ranks), generator=g, device=DEVICE)
        best = values.argmax(-1)
        index[torch.arange(rows, device=DEVICE), best] = drafts[1:].long()
        index[::4, :] += 1
        full = torch.zeros(rows, ranks, 4, device=DEVICE)
        full[..., 0], full[..., 1] = values, index.float()
        pairs = full.view(rows, 4 * ranks)
        packed = full[..., :2].contiguous().view(rows, 2 * ranks)
        target = torch.empty(rows, dtype=torch.int64, device=DEVICE)

        def theirs():
            triton_select(packed, target)
            triton_greedy_rejection(target, drafts, cu, WIDTH - 1)

        ours = tls.greedy_rejection_sample(pairs, drafts, cu, WIDTH - 1)
        theirs()
        expected = triton_greedy_rejection(target, drafts, cu, WIDTH - 1)
        assert torch.equal(ours[1], expected[1])
        for r in range(reqs):
            n = int(expected[1][r])
            assert torch.equal(ours[0][r, :n], expected[0][r, :n])
        return (
            lambda: tls.greedy_rejection_sample(pairs, drafts, cu, WIDTH - 1),
            theirs,
        )

    return make


CASES = {
    "gumbel_sample": [
        (f"{t} tokens {str(d)[6:]}", gumbel_case(t, d))
        for d in (torch.bfloat16, torch.float32)
        for t in (1, 6, 16, 96)
    ],
    "gumbel_finish": [(f"{t} tokens", gumbel_finish_case(t)) for t in (1, 6, 16, 96)],
    "temperature": [(f"{t} tokens", temperature_case(t)) for t in (1, 16, 96)],
    "sink_candidates": [
        (f"{r} requests t={temp}", candidates_case(r, temp))
        for temp in (0.0, 1.0)
        for r in (1, 16, 64)
    ],
    "sink_policy": [(f"{r} requests", policy_case(r)) for r in (1, 16, 64)],
    "greedy_markov_shard_pair": [
        (f"{r} rows", markov_pair_case(r)) for r in (1, 6, 16, 96)
    ],
    "greedy_shard_pair": [(f"{r} rows", shard_pair_case(r)) for r in (6, 36, 96, 384)],
    # The DSpark drafter's last step (a request a row; verification's rows go
    # through greedy_rejection, which selects its targets itself).
    "select_first_argmax": [(f"{r} rows", select_case(r)) for r in (1, 16)],
    "sample_greedy_markov": [(f"{r} rows", markov_full_case(r)) for r in (1, 16)],
    # The fused selection and verification against Triton's two launches.
    "greedy_rejection": [(f"{r} requests", rejection_case(r)) for r in (1, 16, 64)],
}


# Kernels reported but not gated, each with the owner's acceptance. The plain
# select_first_argmax runs once a decode step (the drafter's last step, 0.86 us at
# c1) and has trailed Triton's by about 0.05 us at 1-16 rows since run 20; two
# layout rewrites did not close it, and the gap is about 0.0001% of a step
# (accepted 2026-10-25, experiments/2026-10-24-tilelang-completion).
ACCEPTED = {"select_first_argmax"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--only", nargs="*", choices=sorted(CASES))
    args = parser.parse_args()
    rows = []
    for name in args.only or CASES:
        for shape, make in CASES[name]:
            rows.append(measure(name, shape, make))
    slower = [
        f"{r['kernel']} {r['shape']}"
        for r in rows
        if r["slower"] and r["kernel"] not in ACCEPTED
    ]
    accepted = [
        f"{r['kernel']} {r['shape']}"
        for r in rows
        if r["slower"] and r["kernel"] in ACCEPTED
    ]
    print(
        json.dumps(
            {
                "summary": "slower" if slower else "level or faster",
                "slower": slower,
                "accepted": accepted,
            }
        ),
        flush=True,
    )
    if args.gate and slower:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
