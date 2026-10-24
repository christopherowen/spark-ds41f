#!/usr/bin/env python3
"""The sampling family's free shapes, measured: which layout of each kernel beats Triton.

Round 2 of the ports window left the greedy reductions 17-42% slower than Triton per call,
temperature 70% slower at one token, and the sink candidates and the bf16 Gumbel sampler a
few percent slower. The greedy reductions' combine is a total order and temperature is
element-wise, so their layout is free: this sweeps the partial argmax's threads a block, a
warp per row against a thread per row for the finish and select kernels, and temperature's
threads a block. Every variant is first compared with the Triton kernel bit for bit, then
timed against it: eight fresh inputs, a CUDA graph of 100 calls each, replayed as a pair
whose order alternates. A profiler pass then reports each kernel's own device time (CUPTI)
for the production kernels and Triton's, which separates kernel time from launch time.
Each production kernel is also compiled with TileLang's IfStmtBinding replay of bound loads
off (``tl.if_stmt_binding_inline_replayable_binds``), which re-issues loads after each store
and branch. One JSON line per variant and shape.
"""
import json
import math
import statistics
import sys

import tilelang
import torch
import triton

from vllm.v1.worker.gpu.sample import gumbel, sink
from vllm.v1.worker.gpu.sample import tilelang_sampling as tls
from vllm.v1.worker.gpu.spec_decode.dspark import greedy

import sampling_tile

DEVICE = torch.device("cuda")
VOCAB = 129280
SHARD = VOCAB // 4
CALLS = 100
SAMPLES = 8


def graph(function):
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


def elapsed_us(capture) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS


def paired(make) -> dict:
    ours, theirs, differences = [], [], []
    for sample in range(SAMPLES):
        a_fn, b_fn = make(sample)
        a, b = graph(a_fn), graph(b_fn)
        pair = (a, b) if sample % 2 == 0 else (b, a)
        times = {id(c): elapsed_us(c) for c in pair}
        ours.append(times[id(a)])
        theirs.append(times[id(b)])
        differences.append(times[id(a)] - times[id(b)])
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(SAMPLES)
    return {"tilelang_us": round(statistics.median(ours), 3),
            "triton_us": round(statistics.median(theirs), 3),
            "ratio": round(statistics.median(ours) / statistics.median(theirs), 4),
            "interval_us": [round(mean - half, 3), round(mean + half, 3)]}


def kernel_times(function, calls: int = 50) -> dict:
    """Each kernel's mean device time over ``calls`` calls (CUPTI)."""
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            function()
        torch.cuda.synchronize()
    out = {}
    for event in prof.key_averages():
        if event.device_type == torch.autograd.DeviceType.CUDA and event.count >= calls:
            total = getattr(event, "device_time_total", None)
            if total is None:
                total = event.cuda_time_total
            out[event.key[:60]] = round(total / event.count, 3)
    return out


BINDS_OFF = {"tl.if_stmt_binding_inline_replayable_binds": False}
_COMPILED: dict = {}


def build(factory, *key, extra=None):
    """``factory(*key)``, or its source compiled with ``extra`` pass configs."""
    if not extra:
        return tls._kernel(factory, *key)
    slot = (id(factory), key, tuple(sorted(extra.items())))
    if slot not in _COMPILED:
        config = {getattr(k, "value", k): v for k, v in tls._PASS.items()}
        _COMPILED[slot] = tilelang.compile(
            factory.get_tir(*key), pass_configs={**config, **extra}
        )
    return _COMPILED[slot]


def generator(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


# --- Greedy reductions --------------------------------------------------------


def run_partial(base, bias, values, indices, threads, extra=None):
    vector = tls._vector_rows(base) if bias is None else tls._vector_rows(base, bias)
    key = tls.partial_argmax_key(base.dtype, None if bias is None else bias.dtype, vector)
    build(tls.partial_argmax_kernel, *key, threads, extra=extra)(
        base, base if bias is None else bias, values.view(-1), indices.view(-1)
    )


def run_finish(values, indices, out, rows, parts, out_stride, vocab_start, output, mode,
               extra=None):
    build(tls.finish_argmax_kernel, output, mode, extra=extra)(
        values.view(-1), indices.view(-1), out, parts, rows, out_stride, vocab_start
    )


def triton_pair(base, bias, vocab_start, pairs, values, indices):
    rows, vocab = base.shape
    parts = triton.cdiv(vocab, greedy._BLOCK)
    if bias is None:
        greedy._partial_argmax_logits[(rows, parts)](
            base, values, indices, base.stride(0), vocab, parts, greedy._BLOCK
        )
    else:
        greedy._partial_argmax[(rows, parts)](
            base, bias, values, indices, base.stride(0), bias.stride(0), vocab, parts,
            greedy._BLOCK,
        )
    greedy._finish_argmax_pair[(rows,)](
        values, indices, pairs, vocab_start, parts, triton.next_power_of_2(parts)
    )


def pair_inputs(rows, biased, sample):
    g = generator(sample)
    base = torch.randn(rows, 5, SHARD, generator=g, device=DEVICE).to(torch.bfloat16)[:, 2]
    bias = torch.randn(rows, SHARD, generator=g, device=DEVICE).to(torch.bfloat16) if biased else None
    shape = greedy.scratch_shape(rows, SHARD)
    return base, bias, shape


def pair_case(rows, biased, threads, mode, extra=None):
    parts = triton.cdiv(SHARD, greedy._BLOCK)

    def make(sample):
        base, bias, shape = pair_inputs(rows, biased, sample)
        ours_pairs = torch.empty(rows, 2, device=DEVICE)
        theirs_pairs = torch.empty(rows, 2, device=DEVICE)
        v1, i1 = torch.empty(shape, device=DEVICE), torch.empty(shape, dtype=torch.int32, device=DEVICE)
        v2, i2 = torch.empty(shape, device=DEVICE), torch.empty(shape, dtype=torch.int32, device=DEVICE)

        def ours():
            run_partial(base, bias, v1, i1, threads, extra)
            run_finish(v1, i1, ours_pairs.view(-1), rows, parts, 2, SHARD, "pair", mode, extra)

        def theirs():
            triton_pair(base, bias, SHARD, theirs_pairs, v2, i2)

        if sample == 0:
            ours()
            theirs()
            torch.cuda.synchronize()
            assert torch.equal(ours_pairs.view(torch.int32), theirs_pairs.view(torch.int32))
            assert torch.equal(v1.view(torch.int32), v2.view(torch.int32)) and torch.equal(i1, i2)
        return ours, theirs

    return make


def select_case(rows, mode, extra=None, tile=False):
    def make(sample):
        g = generator(sample)
        values = torch.randn(rows, 4, generator=g, device=DEVICE)
        index = torch.randint(0, VOCAB, (rows, 4), generator=g, device=DEVICE).float()
        pairs = torch.stack([values, index], dim=-1).view(rows, 8)
        o1 = torch.empty(rows, dtype=torch.int64, device=DEVICE)
        o2 = torch.empty_like(o1)
        kernel = (
            build(sampling_tile.select_tile_kernel, "int64", 4)
            if tile
            else build(tls.select_first_argmax_kernel, "int64", mode, extra=extra)
        )

        def ours():
            kernel(pairs, o1, 1)

        def theirs():
            greedy._select_first_argmax[(rows,)](pairs, o2, 1, 4, 4)

        if sample == 0:
            ours()
            theirs()
            torch.cuda.synchronize()
            assert torch.equal(o1, o2)
        return ours, theirs

    return make


def temperature_case(tokens, threads, extra=None, factory=None):
    def make(sample):
        g = generator(sample)
        logits = torch.randn(tokens, VOCAB, generator=g, device=DEVICE)
        theirs_logits = logits.clone()
        mapping = torch.arange(tokens, device=DEVICE)
        temperature = torch.full((tokens,), 0.7, device=DEVICE)
        kernel = build(
            factory or tls.temperature_kernel, "int64", tls._triton_run(logits), threads,
            extra=extra,
        )

        def ours():
            kernel(logits, mapping, temperature)

        def theirs():
            gumbel._temperature_kernel[(tokens, triton.cdiv(VOCAB, 8192))](
                theirs_logits, theirs_logits.stride(0), mapping, temperature, VOCAB,
                BLOCK_SIZE=8192,
            )

        if sample == 0:
            ours()
            theirs()
            torch.cuda.synchronize()
            assert torch.equal(logits.view(torch.int32), theirs_logits.view(torch.int32))
        return ours, theirs

    return make


# --- Device times of the production kernels ----------------------------------


def gumbel_inputs(tokens, dtype, extra=None, seed=0):
    g = generator(seed)
    logits = (torch.randn(tokens, VOCAB, generator=g, device=DEVICE) * 4).to(dtype)
    mapping = torch.arange(tokens, device=DEVICE)
    temperature = torch.ones(tokens, device=DEVICE)
    seeds = torch.randint(0, 1 << 62, (tokens,), generator=g, device=DEVICE)
    pos = torch.randint(0, 1 << 20, (tokens,), generator=g, device=DEVICE)
    blocks = triton.cdiv(VOCAB, tls.GUMBEL_BLOCK)
    argmax = torch.empty(tokens, blocks, dtype=torch.int64, device=DEVICE)
    best = torch.empty(tokens, blocks, device=DEVICE)
    kernel = build(
        tls.gumbel_sample_kernel,
        *tls.gumbel_sample_key(dtype, torch.int64, False, False, run=tls._triton_run(logits)),
        extra=extra,
    )
    argmax2, best2 = torch.empty_like(argmax), torch.empty_like(best)

    def ours():
        kernel(argmax, best, logits, mapping, seeds, pos, temperature, best.unsqueeze(1), pos)

    def theirs():
        gumbel._gumbel_sample_kernel[(tokens, blocks)](
            argmax2, argmax2.stride(0), best2, best2.stride(0), None, 0, 0, None, logits,
            logits.stride(0), mapping, seeds, pos, temperature, VOCAB,
            BLOCK_SIZE=tls.GUMBEL_BLOCK, IS_DRAFTING=False, APPLY_TEMPERATURE=False,
            USE_FP64=False, PER_TOKEN_COL=False,
        )

    if seed == 0:
        ours()
        theirs()
        torch.cuda.synchronize()
        assert torch.equal(argmax, argmax2) and torch.equal(best.view(torch.int32), best2.view(torch.int32))
    return ours, theirs


def candidates_inputs(reqs, temperature, extra=None, seed=0):
    g = generator(seed)
    width = 6
    cu = torch.arange(0, (reqs + 1) * width, width, dtype=torch.int32, device=DEVICE)
    num_sampled = torch.randint(1, width + 1, (reqs,), generator=g, device=DEVICE).to(torch.int32)
    drafts = torch.randint(0, VOCAB, (reqs * width + 1,), generator=g, device=DEVICE).to(torch.int32)
    sampled = torch.randint(0, VOCAB, (reqs, width), generator=g, device=DEVICE)
    mapping = torch.arange(reqs, device=DEVICE)
    logits = (torch.randn(reqs * width, VOCAB, generator=g, device=DEVICE) * 4).to(torch.bfloat16)
    temp = torch.full((reqs,), temperature, device=DEVICE)
    seeds = torch.randint(0, 1 << 62, (reqs,), generator=g, device=DEVICE)
    pos = torch.randint(0, 1 << 20, (reqs * width,), generator=g, device=DEVICE)
    blocks = triton.cdiv(VOCAB, sink._CANDIDATE_BLOCK_SIZE)
    values = torch.empty(reqs, blocks * 4, device=DEVICE)
    indices = torch.empty(reqs, blocks * 4, dtype=torch.int64, device=DEVICE)
    kernel = build(
        tls.sink_candidates_kernel,
        *tls.sink_candidates_key(torch.bfloat16, torch.int64, 4, True, tls._triton_run(logits)),
        extra=extra,
    )
    values2, indices2 = torch.empty_like(values), torch.empty_like(indices)

    def ours():
        kernel(values, indices, logits, cu, num_sampled, sampled, drafts, mapping, temp, seeds, pos)

    def theirs():
        sink._sink_candidates_kernel[(reqs, blocks)](
            values2, indices2, blocks * 4, logits, logits.stride(0), VOCAB, cu, num_sampled,
            sampled, sampled.stride(0), drafts, mapping, temp, seeds, pos,
            BLOCK_SIZE=sink._CANDIDATE_BLOCK_SIZE, NUM_CANDIDATES=4, USE_FP64=False,
            MASK_REJECTED_DRAFT=True,
        )

    if seed == 0:
        ours()
        theirs()
        torch.cuda.synchronize()
        assert torch.equal(values.view(torch.int32), values2.view(torch.int32))
        assert torch.equal(indices, indices2)
    return ours, theirs


def emit(row: dict) -> None:
    print(json.dumps(row), flush=True)


def round3() -> None:
    # Production kernels as built, and with bound loads not replayed, against Triton.
    fixed = {
        "gumbel bf16 16 tokens": lambda extra: lambda s: gumbel_inputs(16, torch.bfloat16, extra, s),
        "gumbel bf16 96 tokens": lambda extra: lambda s: gumbel_inputs(96, torch.bfloat16, extra, s),
        "gumbel f32 16 tokens": lambda extra: lambda s: gumbel_inputs(16, torch.float32, extra, s),
        "candidates 1 request t=0": lambda extra: lambda s: candidates_inputs(1, 0.0, extra, s),
        "candidates 16 requests t=1": lambda extra: lambda s: candidates_inputs(16, 1.0, extra, s),
        "temperature 1 token": lambda extra: temperature_case(1, 128, extra),
        "temperature 16 tokens": lambda extra: temperature_case(16, 128, extra),
        "markov pair 1 row": lambda extra: pair_case(1, True, 128, 0, extra),
        "markov pair 6 rows": lambda extra: pair_case(6, True, 128, 0, extra),
        "select 1 row": lambda extra: select_case(1, 0, extra),
    }
    for name, make in fixed.items():
        for label, extra in (("prod", None), ("binds off", BINDS_OFF)):
            row = {"fixed": name, "build": label}
            row.update(paired(make(extra)))
            emit(row)
    for biased in (True, False):
        name = "greedy_markov_shard_pair" if biased else "greedy_shard_pair"
        for rows in (1, 6, 16, 96):
            for threads in (64, 128, 256):
                for mode in (0, 32, 128):
                    row = {"kernel": name, "rows": rows, "partial_threads": threads,
                           "finish": "warp" if mode == 0 else f"thread x{mode}"}
                    row.update(paired(pair_case(rows, biased, threads, mode)))
                    emit(row)
    for rows in (1, 16, 96):
        for mode in (0, 32, 128):
            row = {"kernel": "select_first_argmax", "rows": rows,
                   "mode": "warp" if mode == 0 else f"thread x{mode}"}
            row.update(paired(select_case(rows, mode)))
            emit(row)
    for tokens in (1, 16, 96):
        for threads in (64, 128, 256):
            row = {"kernel": "temperature", "tokens": tokens, "threads": threads}
            row.update(paired(temperature_case(tokens, threads)))
            emit(row)
    # Device time of each kernel, production configuration against Triton.
    profiles = {
        "gumbel bf16 16 tokens": gumbel_inputs(16, torch.bfloat16),
        "gumbel bf16 96 tokens": gumbel_inputs(96, torch.bfloat16),
        "gumbel f32 16 tokens": gumbel_inputs(16, torch.float32),
        "candidates 1 request t=0": candidates_inputs(1, 0.0),
        "candidates 16 requests t=1": candidates_inputs(16, 1.0),
    }
    for name, (ours, theirs) in profiles.items():
        emit({"profile": name, "tilelang": kernel_times(ours), "triton": kernel_times(theirs)})
    for rows in (1, 16):
        base, bias, shape = pair_inputs(rows, True, 0)
        parts = triton.cdiv(SHARD, greedy._BLOCK)
        pairs = torch.empty(rows, 2, device=DEVICE)
        v, i = torch.empty(shape, device=DEVICE), torch.empty(shape, dtype=torch.int32, device=DEVICE)
        emit({"profile": f"greedy markov pair {rows} rows",
              "tilelang": kernel_times(lambda: (run_partial(base, bias, v, i, 128),
                                                run_finish(v, i, pairs.view(-1), rows, parts, 2,
                                                           SHARD, "pair", 0))),
              "triton": kernel_times(lambda: triton_pair(base, bias, SHARD, pairs, v, i))})


def round4a() -> None:
    """Temperature and select in Triton's tile form (sampling_tile) against the
    production kernels, each against Triton; then each one's device time."""
    for tokens in (1, 16, 96):
        for threads in (128, 256):
            for form, factory in (("prod", None), ("tile", sampling_tile.temperature_tile_kernel)):
                row = {"kernel": "temperature", "form": form, "tokens": tokens, "threads": threads}
                row.update(paired(temperature_case(tokens, threads, factory=factory)))
                emit(row)
    for rows in (1, 16, 96):
        for form in ("prod", "tile"):
            row = {"kernel": "select_first_argmax", "form": form, "rows": rows}
            row.update(paired(select_case(rows, 0, tile=form == "tile")))
            emit(row)
    for tokens in (1, 96):
        for form, factory in (("prod", None), ("tile", sampling_tile.temperature_tile_kernel)):
            ours, theirs = temperature_case(tokens, 256, factory=factory)(0)
            emit({"profile": f"temperature {form} {tokens} tokens 256 threads",
                  "tilelang": kernel_times(ours), "triton": kernel_times(theirs)})
    for form in ("prod", "tile"):
        ours, theirs = select_case(1, 0, tile=form == "tile")(0)
        emit({"profile": f"select {form} 1 row", "tilelang": kernel_times(ours),
              "triton": kernel_times(theirs)})


def main() -> None:
    rounds = {"3": round3, "4a": round4a}
    for name in sys.argv[1:] or ["4a"]:
        rounds[name]()


if __name__ == "__main__":
    main()
