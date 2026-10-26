# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired CUDA graph timings of the TileLang rejection sampler against a baseline,
in production's configuration: one-hot drafts, block verification, FP32 draws,
DeepSeek V4.1's vocabulary and five draft steps. The baseline is the Triton kernels
the TileLang ones replaced, or with ``--baseline DIR`` the release's TileLang kernels
(``DIR`` holds the release's ``tilelang_rejection.py``, loaded beside the installed
module).

``rejection_sample`` times the whole verification: ``rejection_sample`` itself
(TileLang's three kernels) against the baseline's, both with the shared insertion;
``local_stats``, ``verify`` (against the cumulative-ratio and rejection kernels
together) and ``resample`` time one step each, from inputs the steps before it
computed once: each implementation's own block statistics for ``verify``, Triton's
verification for ``resample``. Every graph node has its own inputs and outputs, so no
call reads logits another left in L2. Outputs start at a sentinel, reset outside the
timed replay, and are compared before anything is timed, the elements the kernels
leave alone included: bit for bit where the arithmetic is shared (argmaxes, maxima,
sampled tokens, resampling), and where the TileLang kernels follow Barron et al.'s
fix within the baseline's rounding: a block sum without its maximum's column plus 1
is the baseline's, the cumulative log ratios within 5e-5 (the baseline rounds them
at the magnitude of the maximum logit), and at most one request in twenty may take
another count (a draw between the two thresholds). Both see the same tensor
addresses.

With ``--dead-rows``, every request has a dead-row cut, as adaptive verification
gives it (from none of its drafts live to all, a request in four unclamped). The
TileLang kernels take the cut, which ends each block at it; the release's do not,
and its ``rejection_sample`` caps each count at the cut plus one in the insertion,
as the runner served it. The outputs differ by design, so before timing ours are
held to themselves on the same drafts with the dead ones as -1 placeholders, bit
for bit, and a greedy request's committed tokens and count to the baseline's.
"""

import argparse
import importlib.util
import json
import math
import statistics

import torch
import triton

from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rs
from vllm.v1.worker.gpu.spec_decode import tilelang_rejection as candidate
from vllm.v1.worker.gpu.spec_decode.tilelang_resampled import insert_resampled

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


def make_inputs(rows, dtype, temps, seed, dead_rows=False):
    """One graph node's verification inputs. Each row's next input is lifted by a
    margin that varies by row, so some drafts are accepted and others rejected;
    with ``temps`` "mixed" every other request is greedy. With ``dead_rows``, each
    request's cut ("cut") and the drafts with the dead ones as placeholders
    ("placeholders")."""
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
    x = {
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
    if dead_rows:
        # The draft combine's cut: a request's live drafts, or for one it does not
        # clamp its block of eight plus one. Row j is draft j - 1's input.
        live = torch.randint(0, 1 << 30, (reqs,), generator=g, device="cuda")
        cut = live % sizes
        cut[::4] = 9
        x["cut"] = cut.int()
        dead = x["local"].long() > cut.repeat_interleave(sizes)
        x["placeholders"] = torch.where(dead, -1, x["drafts"]).int()
    return x


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


def tilelang_rejection_sample(x, placeholders=False):
    """rejection_sample itself, which takes the TileLang kernels here, with the
    inputs' dead-row cut if they have one; with ``placeholders``, without the cut
    and with the dead drafts as -1 placeholders (what the cut must equal)."""
    return rs.rejection_sample(
        x["logits"],
        None,
        x["placeholders"] if placeholders else x["drafts"],
        x["cu"],
        x["pos"],
        x["idx_mapping"],
        x["expanded"],
        x["local"],
        x["temperature"],
        x["seeds"],
        STEPS,
        use_block_verification=True,
        dead_row_cut=None if placeholders else x.get("cut"),
    )


class Triton:
    """The Triton kernels the TileLang ones replaced."""

    name = "triton"

    def local_stats(self, x, out):
        triton_local_stats(x, out)

    def verify(self, x, stats, out):
        triton_verify(x, stats, out)

    def resample(self, x, verified, out):
        triton_resample(x, verified, out)

    def rejection_sample(self, x):
        return triton_rejection_sample(x)


class Release:
    """The release's TileLang kernels, from its ``tilelang_rejection.py``."""

    name = "release"

    def __init__(self, directory):
        path = f"{directory}/tilelang_rejection.py"
        spec = importlib.util.spec_from_file_location("release_rejection", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def local_stats(self, x, out):
        self.module.compute_local_stats(
            *out, x["logits"], x["expanded"], x["local"], x["temperature"], STEPS
        )

    def verify(self, x, stats, out):
        cumulative, sampled, num_sampled, target_lse, draft_lse = out
        self.module.verify(
            sampled,
            num_sampled,
            target_lse,
            draft_lse,
            cumulative,
            x["logits"],
            *stats,
            x["drafts"],
            x["cu"],
            x["idx_mapping"],
            x["temperature"],
            x["seeds"],
            x["pos"],
            STEPS,
        )

    def resample(self, x, verified, out):
        self.module.resample(
            *out,
            x["logits"],
            verified[2],
            x["cu"],
            x["expanded"],
            x["drafts"],
            x["temperature"],
            x["seeds"],
            x["pos"],
        )

    def rejection_sample(self, x):
        """rejection_sample's TileLang branch with the release's kernels; a
        dead-row cut caps the counts in the insertion, as the release's did."""
        sampled, num_sampled, ids, values = self.module.verify_and_resample(
            x["logits"],
            x["drafts"],
            x["cu"],
            x["pos"],
            x["idx_mapping"],
            x["expanded"],
            x["local"],
            x["temperature"],
            x["seeds"],
            STEPS,
        )
        insert_resampled(
            sampled,
            num_sampled,
            ids,
            values,
            x["cu"],
            x["expanded"],
            x["temperature"],
            x.get("cut"),
        )
        return sampled, num_sampled


def make_case(kind, rows, dtype, temps, seed, count, baseline, dead_rows=False):
    """``reset``, ``invoke(index, ours, placeholders=False)``, ``output(index)`` and
    the inputs for ``count`` graph nodes, each with its own inputs and outputs.
    ``placeholders``: ours without the cut, the dead drafts as placeholders."""
    nodes = [
        make_inputs(rows, dtype, temps, seed * 1000 + i, dead_rows)
        for i in range(count)
    ]

    if kind == "rejection_sample":
        results = [None] * count

        def invoke(index, ours, placeholders=False):
            if ours:
                results[index] = tilelang_rejection_sample(nodes[index], placeholders)
            else:
                results[index] = baseline.rejection_sample(nodes[index])

        def output(index):
            # Only the committed tokens are defined.
            sampled, num_sampled = results[index]
            columns = torch.arange(sampled.shape[1], device="cuda")
            valid = columns[None, :] < num_sampled[:, None]
            return (num_sampled, torch.where(valid, sampled, -1))

        return lambda: None, invoke, output, nodes

    # The steps before the timed one, once: Triton's verification for resampling;
    # each implementation's own block statistics for verification (the TileLang
    # block sums leave the maximum's column out).
    for x in nodes:
        x["stats"] = stats_buffers(x)
        triton_local_stats(x, x["stats"])
        if kind == "verify":
            x["baseline_stats"] = stats_buffers(x)
            baseline.local_stats(x, x["baseline_stats"])
            x["our_stats"] = stats_buffers(x)
            candidate.compute_local_stats(
                *x["our_stats"],
                x["logits"],
                x["expanded"],
                x["local"],
                x["temperature"],
                STEPS,
            )
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

    def invoke(index, ours, placeholders=False):
        x, out = nodes[index], buffers[index]
        drafts = x["placeholders"] if placeholders else x["drafts"]
        cut = None if placeholders else x.get("cut")
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
                baseline.local_stats(x, out)
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
                    *x["our_stats"],
                    drafts,
                    x["cu"],
                    x["idx_mapping"],
                    x["temperature"],
                    x["seeds"],
                    x["pos"],
                    STEPS,
                    cut,
                )
            else:
                baseline.verify(x, x["baseline_stats"], out)
        elif ours:
            candidate.resample(
                *out,
                x["logits"],
                x["verified"][2],
                x["cu"],
                x["expanded"],
                drafts,
                x["temperature"],
                x["seeds"],
                x["pos"],
                cut,
            )
        else:
            baseline.resample(x, x["verified"], out)

    return reset, invoke, lambda index: buffers[index], nodes


def differences(kind, actual, expected):
    """What the outputs may not differ in (see the module docstring): a list of
    ``(what, indices)``, empty when they agree."""
    found = []

    def same(what, a, b):
        if not torch.equal(bits(a), bits(b)):
            found.append((what, (bits(a) != bits(b)).nonzero()[:8].tolist()))

    def close(what, a, b, where, rtol, atol):
        bad = where & ~((a - b).abs() <= atol + rtol * b.abs())
        if bad.any():
            found.append((what, bad.nonzero()[:8].tolist()))

    def counts(a, b):
        if (a != b).sum().item() * 20 > a.numel():
            found.append(("counts", (a != b).nonzero()[:8].tolist()))
        return a == b

    if kind == "local_stats":
        same("argmax", actual[0], expected[0])
        same("maximum", actual[1], expected[1])
        written = expected[2] != -7
        same("written", actual[2] != -7, written)
        same("rows left alone", actual[2][~written], expected[2][~written])
        empty = written & (expected[2] == 0)
        same("empty blocks", actual[2][empty], expected[2][empty])
        live = written & (expected[2] > 0)
        close("block sums", actual[2] + 1, expected[2], live, 1e-5, 0)
    elif kind == "verify":
        same("sampled", actual[1], expected[1])
        same("draft lse", actual[4], expected[4])
        written = expected[0] != -7
        same("cumulative written", actual[0] != -7, written)
        close("cumulative", actual[0], expected[0], written, 0, 5e-5)
        equal = counts(actual[2], expected[2])
        close("target lse", actual[3], expected[3], equal, 2e-5, 2e-5)
    elif kind == "rejection_sample":
        equal = counts(actual[0], expected[0])
        same("tokens", actual[1][equal], expected[1][equal])
    else:
        for what, a, b in zip(("ids", "values"), actual, expected):
            same(what, a, b)
    return found


def identical(actual, expected):
    """Every output bit for bit, in ``differences``' form."""
    return [
        (f"output {i}", (bits(a) != bits(b)).nonzero()[:8].tolist())
        for i, (a, b) in enumerate(zip(actual, expected))
        if not torch.equal(bits(a), bits(b))
    ]


def greedy_differences(actual, expected, x):
    """The greedy requests' counts and committed tokens (``rejection_sample``'s
    outputs), in ``differences``' form."""
    greedy = x["temperature"][x["idx_mapping"]] == 0
    return [
        (f"greedy {what}", (a[greedy] != b[greedy]).nonzero()[:8].tolist())
        for what, a, b in zip(("counts", "tokens"), actual, expected)
        if not torch.equal(a[greedy], b[greedy])
    ]


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


def measure(kind, rows, dtype_name, temps, seed, baseline, dead_rows=False):
    count = 8
    ours, theirs = [], []
    for sample in range(8):
        reset, invoke, output, nodes = make_case(
            kind,
            rows,
            DTYPES[dtype_name],
            temps,
            seed + sample,
            count,
            baseline,
            dead_rows,
        )
        reset()
        invoke(0, False)
        expected = [value.clone() for value in output(0)]
        reset()
        invoke(0, True)
        if dead_rows:
            # Ours with the cut against ours on placeholders; a greedy request
            # against the baseline.
            actual = [value.clone() for value in output(0)]
            found = []
            if kind == "rejection_sample":
                found = greedy_differences(actual, expected, nodes[0])
            reset()
            invoke(0, True, placeholders=True)
            found += identical(actual, output(0))
        else:
            found = differences(kind, output(0), expected)
        if found:
            print(json.dumps({"sample": sample, "differences": found}), flush=True)
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
        del a, b, order, reset, invoke, output, expected, nodes
    paired = [a - b for a, b in zip(ours, theirs)]
    mean = statistics.fmean(paired)
    half = 2.365 * statistics.stdev(paired) / math.sqrt(len(paired))
    result = dict(
        kernel=kind,
        rows=rows,
        dtype=dtype_name,
        temperatures=temps,
        dead_rows=dead_rows,
        baseline=baseline.name,
        tilelang_us=statistics.median(ours),
        baseline_us=statistics.median(theirs),
        interval_us=[mean - half, mean + half],
        slower=mean - half > 0,
        tilelang_samples_us=ours,
        baseline_samples_us=theirs,
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
    parser.add_argument(
        "--baseline", metavar="DIR", help="the release's tilelang_rejection.py"
    )
    parser.add_argument(
        "--dead-rows", action="store_true", help="every request with a dead-row cut"
    )
    args = parser.parse_args()
    baseline = Release(args.baseline) if args.baseline else Triton()
    cases = [
        (kind, rows, dtype, temps)
        for kind in args.kernels.split(",")
        for dtype in args.dtypes.split(",")
        for temps in args.temperatures.split(",")
        for rows in map(int, args.rows.split(","))
    ]
    results = [measure(*case, args.seed, baseline, args.dead_rows) for case in cases]
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
