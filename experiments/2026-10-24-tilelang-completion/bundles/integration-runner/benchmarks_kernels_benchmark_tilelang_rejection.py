# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired CUDA graph timings of the TileLang rejection sampler against the Triton
kernels it replaces, in production's configuration: one-hot drafts, block
verification, FP32 draws, DeepSeek V4.1's vocabulary and five draft steps.

``rejection_sample`` times the whole verification: ``rejection_sample`` itself
(TileLang's three kernels) against the four Triton kernels they replace, both with
the shared insertion; ``local_stats``, ``verify`` (against the cumulative-ratio and
rejection kernels together) and ``resample`` time one step each, from inputs the
steps before it computed once with Triton. Every graph node has its own inputs and
outputs, so no call reads logits another left in L2. Outputs start at a sentinel,
reset outside the timed replay, and the two implementations must agree bit for bit
before anything is timed, the elements they leave alone included. Both see the same
tensor addresses.
"""

import argparse
import json
import math
import statistics

import torch
import triton

from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rs
from vllm.v1.worker.gpu.spec_decode import tilelang_rejection as candidate

VOCAB = 129280
STEPS = 5
KERNELS = ("rejection_sample", "local_stats", "verify", "resample")
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def requests(rows):
    """Blocks of STEPS + 1 rows, the remainder as one shorter block (adaptive
    verification)."""
    counts = [STEPS + 1] * (rows // (STEPS + 1))
    if rows % (STEPS + 1):
        counts.append(rows % (STEPS + 1))
    return counts


def make_inputs(rows, dtype, temps, seed):
    """One graph node's verification inputs. Each row's next input is lifted by a
    margin that varies by row, so some drafts are accepted and others rejected;
    with ``temps`` "mixed" every other request is greedy."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    counts = requests(rows)
    reqs = len(counts)
    states = reqs + 9
    mapping = torch.randperm(states, generator=g, device="cuda")[:reqs]
    sizes = torch.tensor(counts, device="cuda")
    temperature = torch.ones(states, device="cuda")
    if temps == "mixed":
        temperature[mapping[::2]] = 0
    drafts = torch.randint(0, VOCAB, (rows,), generator=g, device="cuda")
    logits = torch.randn(rows, VOCAB, generator=g, device="cuda") * 4
    if rows > 1:
        r = torch.arange(rows - 1, device="cuda")
        margin = torch.tensor([0.0, 3.0, 6.0, 9.0, 14.0, 40.0], device="cuda")
        logits[r, drafts[1:]] += margin[r % 6]
    return {
        "logits": logits.to(dtype),
        "drafts": drafts.int(),
        "cu": torch.cat((sizes.new_zeros(1), sizes.cumsum(0))).int(),
        "pos": torch.randint(0, 1 << 20, (rows,), generator=g, device="cuda"),
        "idx_mapping": mapping,
        "expanded": mapping.repeat_interleave(sizes),
        "local": torch.cat([torch.arange(c, device="cuda") for c in counts]).int(),
        "temperature": temperature,
        "seeds": torch.randint(
            -(2**63), 2**63 - 1, (states,), generator=g, device="cuda"
        ),
    }


# --- rejection_sample's Triton launches for this configuration ----------------


def triton_local_stats(x, stats):
    rows, vocab = x["logits"].shape
    argmax, maximum, sumexp = stats
    unused = torch.empty_like(maximum)
    rs._compute_local_logits_stats_kernel[(rows, maximum.shape[1])](
        argmax,
        argmax.stride(0),
        maximum,
        maximum.stride(0),
        sumexp,
        sumexp.stride(0),
        unused,
        unused.stride(0),
        unused,
        unused.stride(0),
        x["logits"],
        x["logits"].stride(0),
        None,
        0,
        0,
        x["expanded"],
        x["local"],
        x["temperature"],
        vocab,
        STEPS,
        BLOCK_SIZE=candidate.STATS_BLOCK,
        HAS_DRAFT_LOGITS=False,
    )


def triton_verify(x, stats, verified):
    argmax, maximum, sumexp = stats
    cumulative, sampled, num_sampled, target_lse, draft_lse = verified
    reqs = num_sampled.shape[0]
    blocks = maximum.shape[1]
    padded = triton.next_power_of_2(blocks)
    unused = torch.empty_like(maximum)
    logits = x["logits"]
    rs._compute_cumulative_log_p_kernel[(reqs,)](
        cumulative,
        logits,
        logits.stride(0),
        maximum,
        maximum.stride(0),
        sumexp,
        sumexp.stride(0),
        x["drafts"],
        None,
        0,
        0,
        unused,
        unused.stride(0),
        unused,
        unused.stride(0),
        x["cu"],
        x["idx_mapping"],
        x["temperature"],
        blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded,
        HAS_DRAFT_LOGITS=False,
        num_warps=1,
    )
    rs._rejection_kernel[(reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_lse,
        draft_lse,
        logits,
        logits.stride(0),
        argmax,
        argmax.stride(0),
        maximum,
        maximum.stride(0),
        sumexp,
        sumexp.stride(0),
        x["drafts"],
        None,
        0,
        0,
        unused,
        unused.stride(0),
        unused,
        unused.stride(0),
        x["cu"],
        x["idx_mapping"],
        x["temperature"],
        x["seeds"],
        x["pos"],
        None,
        cumulative,
        None,
        0,
        blocks,
        target_lse,
        target_lse,
        0,
        PADDED_VOCAB_NUM_BLOCKS=padded,
        HAS_DRAFT_LOGITS=False,
        SYNTHETIC_MODE=False,
        USE_BLOCK_VERIFICATION=True,
        VP_DRAFT=False,
        num_warps=1,
    )


def triton_resample(x, verified, resampled):
    cumulative, _, num_sampled, target_lse, draft_lse = verified
    ids, values = resampled
    logits = x["logits"]
    rs._resample_kernel[(num_sampled.shape[0], ids.shape[1])](
        ids,
        ids.stride(0),
        values,
        values.stride(0),
        logits,
        logits.stride(0),
        target_lse,
        None,
        0,
        0,
        draft_lse,
        num_sampled,
        x["cu"],
        x["expanded"],
        x["drafts"],
        x["temperature"],
        x["seeds"],
        x["pos"],
        cumulative,
        None,
        0,
        None,
        0,
        0,
        logits.shape[1],
        BLOCK_SIZE=candidate.RESAMPLE_BLOCK,
        HAS_DRAFT_LOGITS=False,
        USE_FP64=False,
        USE_BLOCK_VERIFICATION=True,
        CONTEXT_WIDTH=1,
        WATERMARK=False,
    )


def stats_buffers(x):
    rows, vocab = x["logits"].shape
    blocks = triton.cdiv(vocab, candidate.STATS_BLOCK)
    return (
        torch.empty((rows, blocks), device="cuda", dtype=torch.int64),
        torch.empty((rows, blocks), device="cuda"),
        torch.empty((rows, blocks), device="cuda"),
    )


def verify_buffers(x):
    rows = x["logits"].shape[0]
    reqs = x["cu"].shape[0] - 1
    return (
        torch.empty(rows, device="cuda"),
        torch.empty((reqs, STEPS + 1), device="cuda", dtype=torch.int64),
        torch.empty(reqs, device="cuda", dtype=torch.int32),
        torch.empty(reqs, device="cuda"),
        torch.empty(reqs, device="cuda"),
    )


def resample_buffers(x):
    reqs = x["cu"].shape[0] - 1
    blocks = triton.cdiv(x["logits"].shape[1], candidate.RESAMPLE_BLOCK)
    return (
        torch.empty((reqs, blocks), device="cuda", dtype=torch.int64),
        torch.empty((reqs, blocks), device="cuda"),
    )


def triton_rejection_sample(x):
    """rejection_sample's Triton path, with its allocations: the four kernels the
    TileLang ones replace, then the insertion both paths share."""
    stats = stats_buffers(x)
    triton_local_stats(x, stats)
    verified = verify_buffers(x)
    triton_verify(x, stats, verified)
    resampled = resample_buffers(x)
    triton_resample(x, verified, resampled)
    _, sampled, num_sampled, _, _ = verified
    rs.insert_resampled(
        sampled, num_sampled, *resampled, x["cu"], x["expanded"], x["temperature"]
    )
    return sampled, num_sampled


def tilelang_rejection_sample(x):
    """rejection_sample itself, which takes the TileLang kernels here."""
    return rs.rejection_sample(
        x["logits"],
        None,
        x["drafts"],
        x["cu"],
        x["pos"],
        x["idx_mapping"],
        x["expanded"],
        x["local"],
        x["temperature"],
        x["seeds"],
        STEPS,
        use_block_verification=True,
    )


def make_case(kind, rows, dtype, temps, seed, count):
    """``reset``, ``invoke(index, ours)`` and ``output(index)`` for ``count`` graph
    nodes, each with its own inputs and outputs."""
    nodes = [make_inputs(rows, dtype, temps, seed * 1000 + i) for i in range(count)]

    if kind == "rejection_sample":
        results = [None] * count

        def invoke(index, ours):
            run = tilelang_rejection_sample if ours else triton_rejection_sample
            results[index] = run(nodes[index])

        def output(index):
            # Only the committed tokens are defined.
            sampled, num_sampled = results[index]
            columns = torch.arange(sampled.shape[1], device="cuda")
            valid = columns[None, :] < num_sampled[:, None]
            return (num_sampled, torch.where(valid, sampled, -1))

        return lambda: None, invoke, output

    # The steps before the timed one, once, with Triton.
    for x in nodes:
        x["stats"] = stats_buffers(x)
        triton_local_stats(x, x["stats"])
        if kind == "resample":
            x["verified"] = verify_buffers(x)
            triton_verify(x, x["stats"], x["verified"])
    outputs = {
        "local_stats": lambda x: stats_buffers(x),
        "verify": lambda x: verify_buffers(x),
        "resample": lambda x: resample_buffers(x),
    }[kind]
    buffers = [outputs(x) for x in nodes]

    def reset():
        for tensors in buffers:
            for t in tensors:
                t.fill_(-7)

    def invoke(index, ours):
        x, out = nodes[index], buffers[index]
        if kind == "local_stats":
            if ours:
                candidate.compute_local_stats(
                    *out,
                    x["logits"],
                    x["expanded"],
                    x["local"],
                    x["temperature"],
                    STEPS,
                )
            else:
                triton_local_stats(x, out)
        elif kind == "verify":
            if ours:
                cumulative, sampled, num_sampled, target_lse, draft_lse = out
                candidate.verify(
                    sampled,
                    num_sampled,
                    target_lse,
                    draft_lse,
                    cumulative,
                    x["logits"],
                    *x["stats"],
                    x["drafts"],
                    x["cu"],
                    x["idx_mapping"],
                    x["temperature"],
                    x["seeds"],
                    x["pos"],
                    STEPS,
                )
            else:
                triton_verify(x, x["stats"], out)
        elif ours:
            candidate.resample(
                *out,
                x["logits"],
                x["verified"][2],
                x["cu"],
                x["expanded"],
                x["drafts"],
                x["temperature"],
                x["seeds"],
                x["pos"],
            )
        else:
            triton_resample(x, x["verified"], out)

    return reset, invoke, lambda index: buffers[index]


def bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def capture(invoke, ours, count):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        invoke(0, ours)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(count):
            invoke(index, ours)
    return graph


def elapsed(graph, reset, count):
    reset()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / count


def measure(kind, rows, dtype_name, temps, seed):
    count = 8
    ours, theirs = [], []
    for sample in range(8):
        reset, invoke, output = make_case(
            kind, rows, DTYPES[dtype_name], temps, seed + sample, count
        )
        reset()
        invoke(0, False)
        expected = [value.clone() for value in output(0)]
        reset()
        invoke(0, True)
        for actual, target in zip(output(0), expected):
            if torch.equal(bits(actual), bits(target)):
                continue
            indices = (bits(actual) != bits(target)).nonzero()[:8]
            print(
                json.dumps(
                    {
                        "sample": sample,
                        "differences": [
                            [
                                index.tolist(),
                                actual[tuple(index)].item(),
                                target[tuple(index)].item(),
                            ]
                            for index in indices
                        ],
                    }
                ),
                flush=True,
            )
            raise RuntimeError(
                f"outputs differ: {kind} rows={rows} {dtype_name} {temps}"
            )
        reset()
        a = capture(invoke, True, count)
        reset()
        b = capture(invoke, False, count)
        order = (a, b) if sample % 2 == 0 else (b, a)
        times = {id(graph): elapsed(graph, reset, count) for graph in order}
        ours.append(times[id(a)])
        theirs.append(times[id(b)])
        del a, b, order, reset, invoke, output, expected
    paired = [a - b for a, b in zip(ours, theirs)]
    mean = statistics.fmean(paired)
    half = 2.365 * statistics.stdev(paired) / math.sqrt(len(paired))
    result = dict(
        kernel=kind,
        rows=rows,
        dtype=dtype_name,
        temperatures=temps,
        tilelang_us=statistics.median(ours),
        triton_us=statistics.median(theirs),
        interval_us=[mean - half, mean + half],
        slower=mean - half > 0,
        tilelang_samples_us=ours,
        triton_samples_us=theirs,
    )
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--seed", type=int, default=48291)
    parser.add_argument("--kernels", default=",".join(KERNELS))
    parser.add_argument("--rows", default="1,6,16,48,96")
    parser.add_argument("--dtypes", default="float32,bfloat16")
    parser.add_argument("--temperatures", default="sampled,mixed")
    args = parser.parse_args()
    cases = [
        (kind, rows, dtype, temps)
        for kind in args.kernels.split(",")
        for dtype in args.dtypes.split(",")
        for temps in args.temperatures.split(",")
        for rows in map(int, args.rows.split(","))
    ]
    results = [measure(*case, args.seed) for case in cases]
    failed = [
        (r["kernel"], r["rows"], r["dtype"], r["temperatures"])
        for r in results
        if r["slower"]
    ]
    print(json.dumps({"summary": "rejection sampler", "slower": failed}), flush=True)
    if args.gate and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
