#!/usr/bin/env python3
"""The V2 runner's tiny input-preparation kernels, measured: which shape of each beats Triton.

Round 2 of the ports window (2026-10-24 14:55 CET, -next-4) left these TileLang ports slower
than the Triton kernels they replace per call: combine_sampled_and_draft_tokens with dead
rows (+27-34%) and plain at the 16K prefill (+9%), expand_idx_mapping (+14-16%),
prepare_dflash_inputs at decode (+18-21%), prepare_prefill_inputs at decode (+5-16%),
post_update at the 16K prefill (+9%), gather_lookback (+4%) and zero_kv_blocks at one
request (+2%).

Their generated CUDA re-reads the same scalars: a Python name bound to ``buf[i]`` is inlined
at every use, and TileLang's IfStmtBinding (``tl.if_stmt_binding_inline_replayable_binds``,
on by default) re-substitutes binds under each guarded statement, so loads are re-issued
after every store and branch, in a dependent chain (plain combine: 31 global loads per
thread for 14 values; Triton issues 8 per lane, all up front). The variants
(runner_variants.py) sweep:

- ``prod``: the production kernel; ``prod-binds``: the same source compiled with
  IfStmtBinding's bind inlining off;
- ``hoist``: every scalar read once into a register at the top;
- threads a block (32/64/128/256), requests a block (a thread a request against Triton's
  lanes a request; ``spread``: a block a request, Triton's grid), work a thread;
- ``i32``: request index and row offsets in int32 instead of int64;
- ``static``: row strides baked in instead of passed (post_update).

Every variant is first compared with the Triton kernel bit for bit: every buffer the
namespace holds (outputs and in-place state) after one call from identical inputs, on the
port's randomized test cases (tests/v1/worker/test_gpu_tilelang_inputs.py's generators) and
on the benchmark's inputs. Then each is timed against Triton at the benchmark's shapes (decode
c1, decode c16, 16K prefill): eight fresh inputs, a CUDA graph of 100 calls each, the pair
replayed in alternating order. Last, a profiler pass (CUPTI) reports each kernel's own device
time against Triton's, which separates kernel time from the graph's launch time.

One JSON line per variant (its check) and per variant and shape (its times). Exit 1 when a
variant fails to compile or differs from Triton.
"""

import argparse
import contextlib
import json
import math
import statistics
import sys
import traceback
from types import SimpleNamespace

import plain_loads
import runner_input_tests as rt
import tilelang
import torch
from vllm.models.deepseek_v4_1.nvidia.model_state import _gather_lookback_kernel
from vllm.models.deepseek_v4_1.tilelang import lookback
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu import input_batch as ib
from vllm.v1.worker.gpu import tilelang_inputs as tli
from vllm.v1.worker.gpu.buffer_utils import UvaBuffer
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash
from vllm.v1.worker.utils import _zero_kv_blocks_kernel

import runner_variants as rv

DEVICE = torch.device("cuda")
MAX_REQS = 16
MAX_TOKENS = 16384
DRAFTS = 5
ROWS = 1 + DRAFTS
MODEL_LEN = 1 << 20
HISTORY = 1 << 16
VOCAB = 129280
SHAPES = {"decode c1": (1, False), "decode c16": (16, False), "prefill 16K": (1, True)}
CALLS = 100
BINDS_OFF = {"tl.if_stmt_binding_inline_replayable_binds": False}
assert {getattr(k, "value", k): v for k, v in rv.PASS.items()} == {
    getattr(k, "value", k): v for k, v in tli._PASS.items()
}, "the variants compile with the production pass configs"


# --- Timing ---------------------------------------------------------------------


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
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS


def paired(variant, triton_call, build, samples: int) -> dict:
    """Per-call time of the variant and Triton on ``samples`` fresh inputs, graphs
    replayed as a pair whose order alternates; the 95% interval of the difference."""
    ours, theirs, differences = [], [], []
    for sample in range(samples):
        ns = build(sample)
        a = graph(lambda ns=ns: variant(ns))
        b = graph(lambda ns=ns: triton_call(ns))
        pair = (a, b) if sample % 2 == 0 else (b, a)
        times = {id(c): elapsed_us(c) for c in pair}
        ours.append(times[id(a)])
        theirs.append(times[id(b)])
        differences.append(times[id(a)] - times[id(b)])
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(samples)  # t(7), 95%
    return {
        "tilelang_us": round(statistics.median(ours), 3),
        "triton_us": round(statistics.median(theirs), 3),
        "ratio": round(statistics.median(ours) / statistics.median(theirs), 4),
        "difference_us": round(mean, 3),
        "interval_us": [round(mean - half, 3), round(mean + half, 3)],
    }


def device_times(function, calls: int = 100) -> dict:
    """Each kernel's mean device time a call over ``calls`` eager calls (CUPTI)."""
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            function()
        torch.cuda.synchronize()
    out = {}
    for event in prof.key_averages():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        total = getattr(event, "device_time_total", None)
        if total is None:
            total = event.cuda_time_total
        out[event.key[:60]] = round(total / calls, 3)
    return out


# --- Inputs -----------------------------------------------------------------------


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _randint(low, high, shape, gen, dtype=torch.int32):
    return torch.randint(low, high, shape, generator=gen, device=DEVICE, dtype=dtype)


def _garbage(shape, dtype, gen):
    return rt._garbage(shape, dtype, gen)


def _step(num_reqs: int, prefill: bool):
    lens = [MAX_TOKENS] if prefill else [ROWS] * num_reqs
    starts = torch.tensor([0] + lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return lens, starts.to(DEVICE)


def _cumsum(counts) -> torch.Tensor:
    cu = torch.tensor([0] + list(counts), dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return cu.to(DEVICE)


def _uva_history(gen) -> tuple:
    """The runner's token history: pinned host memory over UVA, as in serving."""
    buffer = UvaBuffer((MAX_REQS, HISTORY), torch.int32)
    history = buffer.uva()
    history.copy_(_randint(0, VOCAB, (MAX_REQS, HISTORY), gen))
    return history, buffer


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def _differences(a, b, path="") -> list[str]:
    """The fields of two namespaces built alike whose bytes differ."""
    if isinstance(a, torch.Tensor):
        return [] if torch.equal(_bits(a), _bits(b)) else [path]
    if isinstance(a, (list, tuple)):
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += _differences(x, y, f"{path}[{i}]")
        return out
    if isinstance(a, SimpleNamespace):
        out = []
        for key, value in vars(a).items():
            if key.startswith("_") or key in getattr(a, "_skip", ()):
                continue
            out += _differences(value, getattr(b, key), f"{path}.{key}")
        return out
    return []


# --- Variant plumbing ---------------------------------------------------------------


@contextlib.contextmanager
def _swapped(factory, key, kernel):
    """The production wrapper launching ``kernel`` in place of its cached kernel."""
    slot = (id(factory), key)
    previous = tli._KERNELS.get(slot)
    tli._KERNELS[slot] = kernel
    try:
        yield
    finally:
        if previous is None:
            tli._KERNELS.pop(slot, None)
        else:
            tli._KERNELS[slot] = previous


_COMPILED: dict = {}


def compiled(factory, *key):
    if (id(factory), key) not in _COMPILED:
        _COMPILED[(id(factory), key)] = factory(*key)
    return _COMPILED[(id(factory), key)]


def recompiled(factory, key, extra):
    """``factory(*key)``'s source compiled with ``extra`` pass configs."""
    slot = (id(factory), key, tuple(sorted(extra.items())))
    if slot not in _COMPILED:
        prim = factory.get_tir(*key)
        config = {**tli._PASS, **extra}
        _COMPILED[slot] = tilelang.compile(prim, pass_configs=config)
    return _COMPILED[slot]


def via_production(wrapper, factory, key_of, extra):
    """A launcher running the production wrapper with its kernel recompiled."""

    def launch(ns):
        key = key_of(ns)
        with _swapped(factory, key, recompiled(factory, key, extra)):
            wrapper(ns)

    return launch


class Kernel(SimpleNamespace):
    """name, shapes: shape -> build(seed), checks: list of build(seed), triton(ns),
    variants: name -> launcher(ns) (and ``bench_only``: variants checked on the
    benchmark's inputs alone)."""


# --- combine_sampled_and_draft_tokens -----------------------------------------------


def combine_bench(num_reqs, prefill, dead):
    def build(seed):
        gen = _gen(seed)
        lens, starts = _step(num_reqs, prefill)
        reqs = len(lens)
        logits = [1] if prefill else [ROWS] * reqs
        idx = torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:reqs]
        prefill_len = _randint(1, 1024, (MAX_REQS,), gen)
        dead_rows = None
        if dead:
            dead_rows = ib.DeadRows(
                torch.rand(MAX_REQS, DRAFTS, generator=gen, device=DEVICE),
                torch.zeros(MAX_TOKENS, dtype=torch.bool, device=DEVICE),
                _garbage((reqs,), torch.int32, gen),
                math.log(0.2),
            )
        return SimpleNamespace(
            ids=_garbage((MAX_TOKENS,), torch.int32, gen),
            idx=idx,
            last=_randint(0, VOCAB, (MAX_REQS, 1), gen, torch.int64),
            starts=starts,
            seq_lens=prefill_len[idx] + (MAX_TOKENS if prefill else 4096),
            prefill_len=prefill_len,
            drafts=_randint(0, VOCAB, (MAX_REQS, DRAFTS), gen, torch.int64),
            cu=_cumsum(logits),
            out=_garbage((sum(logits),), torch.int64, gen),
            num_new=1,
            dead=dead_rows,
        )

    return build


def combine_checks(dead):
    builds = []
    if not dead:
        for num_reqs in [n for n in rt.REQUEST_COUNTS if n]:

            def build(seed, num_reqs=num_reqs):
                gen = rt._gen(num_reqs * 5 + DRAFTS + 1 + 1000 * seed)
                starts, cu, seq_lens, prefill_len, num_logits, tokens = rt._combine_case(
                    num_reqs, DRAFTS, 1, gen
                )
                return SimpleNamespace(
                    idx=rt._idx(num_reqs, gen),
                    last=rt._randint(-(2**40), 2**40, (MAX_REQS, 1), gen, torch.int64),
                    drafts=rt._randint(-(2**40), 2**40, (MAX_REQS, DRAFTS), gen, torch.int64),
                    ids=rt._garbage((tokens + 8,), torch.int32, gen),
                    starts=starts,
                    seq_lens=seq_lens,
                    prefill_len=prefill_len,
                    cu=cu,
                    out=torch.full((num_logits,), -7, dtype=torch.int64, device=DEVICE),
                    num_new=1,
                    dead=None,
                )

            builds.append(build)
        return builds
    # Dead rows: thresholds at every running sum of log confidences (either
    # summation order) and one float either side, as test_combine_dead_rows.
    for num_reqs in (1, 3, 8, 16):

        def case(num_reqs=num_reqs):
            gen = rt._gen(num_reqs)
            starts, cu, seq_lens, prefill_len, num_logits, tokens = rt._combine_case(
                num_reqs, DRAFTS, 1, gen
            )
            idx = rt._idx(num_reqs, gen)
            last = rt._randint(0, 1 << 17, (MAX_REQS, 1), gen, torch.int64)
            drafts = rt._randint(0, 1 << 17, (MAX_REQS, DRAFTS), gen, torch.int64)
            confidence = torch.rand(MAX_REQS, DRAFTS, generator=gen, device=DEVICE) ** 0.25
            confidence[0, 2] = 1.0
            confidence[1 % MAX_REQS, 1] = 0.0
            confidence[2 % MAX_REQS, 3] = 1e-40
            confidence[3 % MAX_REQS, 4] = float("nan")
            return (starts, cu, seq_lens, prefill_len, num_logits, tokens, idx, last,
                    drafts, confidence, gen)  # fmt: skip

        *_, idx, _, _, confidence, gen = case()
        picked = rt._survivals(confidence[idx], DRAFTS)
        picked = picked[torch.isfinite(picked)]
        order = torch.randperm(picked.numel(), generator=gen, device=DEVICE)
        thresholds = [math.log(0.2), -0.0, -1e30] + picked[order][:16].tolist()
        for log_tau in thresholds:
            for direction in (0, 1, -1):
                tau = (
                    log_tau
                    if direction == 0
                    else float(
                        rt.np.nextafter(
                            rt.np.float32(log_tau), rt.np.float32(direction * rt.np.inf)
                        )
                    )
                )

                def build(seed, case=case, tau=tau):
                    (starts, cu, seq_lens, prefill_len, num_logits, tokens, idx, last,
                     drafts, confidence, _) = case()  # fmt: skip
                    padding = torch.zeros(tokens + 8, dtype=torch.bool, device=DEVICE)
                    padding[::3] = True
                    return SimpleNamespace(
                        ids=rt._garbage((tokens + 8,), torch.int32, rt._gen(1)),
                        idx=idx,
                        last=last,
                        starts=starts,
                        seq_lens=seq_lens,
                        prefill_len=prefill_len,
                        drafts=drafts,
                        cu=cu,
                        out=torch.full((num_logits,), -7, dtype=torch.int64, device=DEVICE),
                        num_new=1,
                        dead=ib.DeadRows(
                            confidence,
                            padding,
                            rt._garbage((idx.shape[0],), torch.int32, rt._gen(2)),
                            tau,
                        ),
                    )

                builds.append(build)
    return builds


def triton_combine(ns):
    dead = ns.dead
    extra = (
        (dead.confidence, dead.confidence.stride(0), dead.is_padding.view(torch.uint8),
         dead.cut, dead.log_tau)
        if dead is not None
        else (ns.ids, 0, ns.ids, ns.ids, 0.0)
    )  # fmt: skip
    ib._combine_sampled_and_draft_tokens_kernel[(ns.idx.shape[0],)](
        ns.ids, ns.idx, ns.last, ns.starts, ns.seq_lens, ns.prefill_len, ns.drafts,
        ns.drafts.stride(0), ns.cu, ns.out, *extra, NUM_NEW_SAMPLED_TOKENS=ns.num_new,
        DEAD_ROWS=dead is not None,
        BLOCK_SIZE=triton.next_power_of_2(ns.drafts.shape[1] + ns.num_new),
    )  # fmt: skip


def prod_combine(ns):
    tli.combine_sampled_and_draft_tokens(
        ns.ids, ns.idx, ns.last, ns.starts, ns.seq_lens, ns.prefill_len, ns.drafts,
        ns.cu, ns.out, ns.num_new, ns.dead,
    )  # fmt: skip


def combine_variant(factory, threads, spread, index):
    def launch(ns):
        dead = ns.dead is not None
        kernel = compiled(factory, DRAFTS, ns.num_new, dead, threads, spread, index)
        args = [ns.ids, ns.idx, ns.last.view(-1), ns.starts, ns.seq_lens,
                ns.prefill_len, ns.drafts, ns.cu, ns.out]  # fmt: skip
        if dead:
            args += [ns.dead.confidence, ns.dead.is_padding.view(torch.uint8),
                     ns.dead.cut, ns.dead.log_tau]  # fmt: skip
        kernel(*args)

    return launch


def combine_kernel(dead: bool) -> Kernel:
    def key_of(ns):
        return tli.combine_key(DRAFTS, ns.num_new, ns.dead is not None, "int64")

    variants = {
        "prod": prod_combine,
        "prod-binds": via_production(
            prod_combine, tli.combine_sampled_and_draft_tokens_kernel, key_of, BINDS_OFF
        ),
        "hoist-thread-t32": combine_variant(rv.combine_thread_kernel, 32, False, "int64"),
        "hoist-thread-t32-i32": combine_variant(rv.combine_thread_kernel, 32, False, "int32"),
        "hoist-thread-t128": combine_variant(rv.combine_thread_kernel, 128, False, "int64"),
        "hoist-thread-spread": combine_variant(rv.combine_thread_kernel, 32, True, "int64"),
        "lanes-t32": combine_variant(rv.combine_lanes_kernel, 32, False, "int64"),
        "lanes-t64": combine_variant(rv.combine_lanes_kernel, 64, False, "int64"),
        "lanes-t128": combine_variant(rv.combine_lanes_kernel, 128, False, "int64"),
        "lanes-t256": combine_variant(rv.combine_lanes_kernel, 256, False, "int64"),
        "lanes-spread": combine_variant(rv.combine_lanes_kernel, 32, True, "int64"),
        "lanes-t32-i32": combine_variant(rv.combine_lanes_kernel, 32, False, "int32"),
    }
    return Kernel(
        name="combine_sampled_and_draft_tokens" + (" (dead rows)" if dead else ""),
        shapes={s: combine_bench(n, p, dead) for s, (n, p) in SHAPES.items()},
        checks=combine_checks(dead),
        triton=triton_combine,
        variants=variants,
        bench_only=(),
    )


# --- expand_idx_mapping ---------------------------------------------------------------


def expand_bench(num_reqs, prefill):
    def build(seed):
        gen = _gen(seed)
        reqs = 1 if prefill else num_reqs
        logits = [1] if prefill else [ROWS] * reqs
        return SimpleNamespace(
            idx=torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:reqs],
            expanded=_garbage((sum(logits),), torch.int64, gen),
            local=_garbage((sum(logits),), torch.int32, gen),
            cu=_cumsum(logits),
        )

    return build


def expand_checks():
    builds = []
    for num_reqs in [n for n in rt.REQUEST_COUNTS if n]:

        def build(seed, num_reqs=num_reqs):
            # Up to Triton's block of logits a request, and past it (left unwritten).
            gen = rt._gen(num_reqs + 3 + 1000 * seed)
            logits = rt._randint(0, 11, (num_reqs,), gen).tolist()
            cu = rt._starts(logits)
            idx = rt._idx(num_reqs, gen, torch.int64)
            total = int(cu[-1])
            return SimpleNamespace(
                idx=idx,
                expanded=rt._garbage((total,), torch.int64, gen),
                local=rt._garbage((total,), torch.int32, gen),
                cu=cu,
            )

        builds.append(build)
    return builds


def triton_expand(ns):
    ib._expand_idx_mapping_kernel[(ns.idx.shape[0],)](
        ns.idx, ns.expanded, ns.local, ns.cu, BLOCK_SIZE=triton.next_power_of_2(ROWS)
    )


def prod_expand(ns):
    tli.expand_idx_mapping(ns.idx, ns.expanded, ns.local, ns.cu, ROWS)


def expand_variant(threads, spread):
    def launch(ns):
        kernel = compiled(rv.expand_kernel, rv.pow2(ROWS), threads, spread)
        kernel(ns.idx, ns.expanded, ns.local, ns.cu)

    return launch


def expand_tile_variant(threads):
    def launch(ns):
        kernel = compiled(rv.expand_tile_kernel, rv.pow2(ROWS), threads)
        kernel(ns.idx, ns.expanded, ns.local, ns.cu)

    return launch


def expand_kernel() -> Kernel:
    variants = {
        "prod": prod_expand,
        "prod-binds": via_production(
            prod_expand,
            tli.expand_idx_mapping_kernel,
            lambda ns: tli.expand_idx_mapping_key(ROWS, "int64"),
            BINDS_OFF,
        ),
    }
    for threads in (32, 64, 128, 256):
        variants[f"hoist-t{threads}"] = expand_variant(threads, False)
    variants["hoist-spread"] = expand_variant(32, True)
    for threads in (32, 128):
        variants[f"tile-t{threads}"] = expand_tile_variant(threads)
    return Kernel(
        name="expand_idx_mapping",
        shapes={s: expand_bench(n, p) for s, (n, p) in SHAPES.items()},
        checks=expand_checks(),
        triton=triton_expand,
        variants=variants,
        bench_only=(),
    )


# --- prepare_prefill_inputs -------------------------------------------------------


def prefill_bench(num_reqs, prefill):
    def build(seed):
        gen = _gen(seed)
        lens, starts = _step(num_reqs, prefill)
        history, buffer = _uva_history(gen)
        computed = _randint(0, 1024, (MAX_REQS,), gen)
        return SimpleNamespace(
            ids=_garbage((MAX_TOKENS,), torch.int32, gen),
            next_tokens=_garbage((1, MAX_REQS), torch.int32, gen),
            idx=torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[: len(lens)],
            starts=starts,
            history=history,
            _buffer=buffer,
            # Prefilling requests: the chunk is part of the prompt.
            prefill_len=computed + MAX_TOKENS + 8,
            computed=computed,
        )

    return build


def prefill_checks():
    builds = []
    for num_reqs in [n for n in rt.REQUEST_COUNTS if n]:
        for prefill in (False, True):

            def build(seed, num_reqs=num_reqs, prefill=prefill):
                gen = rt._gen(num_reqs + 31 * prefill + 1 + 1000 * seed)
                starts = rt._starts(rt._query_lens(num_reqs, prefill))
                idx = rt._idx(num_reqs, gen)
                history = rt._history(gen)
                computed = rt._randint(0, 40, (MAX_REQS,), gen)
                prefill_len = computed + rt._randint(-3, MAX_TOKENS, (MAX_REQS,), gen)
                return SimpleNamespace(
                    ids=rt._garbage((MAX_TOKENS,), torch.int32, gen),
                    next_tokens=rt._garbage((1, MAX_REQS), torch.int32, gen),
                    idx=idx,
                    starts=starts,
                    history=history,
                    prefill_len=prefill_len,
                    computed=computed,
                )

            builds.append(build)
    return builds


def triton_prefill(ns):
    lookahead = ns.next_tokens.shape[0]
    ib._prepare_prefill_inputs_kernel[(ns.idx.shape[0],)](
        ns.ids, ns.next_tokens, ns.next_tokens.stride(0), lookahead, ns.idx, ns.starts,
        ns.history, ns.history.stride(0), ns.prefill_len, ns.computed, BLOCK_SIZE=1024,
        LOOKAHEAD_BLOCK=triton.next_power_of_2(lookahead),
    )  # fmt: skip


def prod_prefill(ns):
    tli.prepare_prefill_inputs(
        ns.ids, ns.next_tokens, ns.idx, ns.starts, ns.history, ns.prefill_len,
        ns.computed,
    )  # fmt: skip


def prefill_variant(threads, per, index):
    def launch(ns):
        block = rv.pow2(ns.next_tokens.shape[0])
        kernel = compiled(rv.prefill_inputs_kernel, block, threads, per, index)
        kernel(ns.ids, ns.next_tokens, ns.idx, ns.starts, ns.history, ns.prefill_len,
               ns.computed)  # fmt: skip

    return launch


def prefill_kernel() -> Kernel:
    variants = {
        "prod": prod_prefill,
        "prod-binds": via_production(
            prod_prefill,
            tli.prepare_prefill_inputs_kernel,
            lambda ns: tli.prepare_prefill_inputs_key(ns.next_tokens.shape[0], "int64"),
            BINDS_OFF,
        ),
    }
    for threads, per in ((32, 1), (64, 1), (128, 1), (128, 4), (256, 1), (256, 4)):
        variants[f"hoist-t{threads}-p{per}"] = prefill_variant(threads, per, "int64")
    variants["hoist-t32-p1-i32"] = prefill_variant(32, 1, "int32")
    variants["hoist-t256-p4-i32"] = prefill_variant(256, 4, "int32")
    return Kernel(
        name="prepare_prefill_inputs",
        shapes={s: prefill_bench(n, p) for s, (n, p) in SHAPES.items()},
        checks=prefill_checks(),
        triton=triton_prefill,
        variants=variants,
        bench_only=(),
    )


# --- post_update ------------------------------------------------------------------


def post_bench(num_reqs, prefill):
    def build(seed):
        gen = _gen(seed)
        reqs = 1 if prefill else num_reqs
        width = 1 if prefill else ROWS
        num_sampled = _randint(1, width + 1, (reqs,), gen)
        lens = [MAX_TOKENS] if prefill else [ROWS] * reqs
        history, buffer = _uva_history(gen)
        return SimpleNamespace(
            idx=torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:reqs],
            computed=_randint(0, 1 << 16, (MAX_REQS,), gen),
            last=_garbage((MAX_REQS, 1), torch.int64, gen),
            bins=_randint(0, 5, (MAX_REQS, VOCAB), gen),
            sampled=_randint(0, VOCAB, (reqs, width), gen, torch.int64),
            num_sampled=num_sampled,
            num_rejected=width - num_sampled,
            starts=_cumsum(lens),
            history=history,
            _buffer=buffer,
            total=_randint(1024, 2048, (MAX_REQS,), gen),
        )

    return build


def post_checks():
    builds = []
    for num_reqs in [n for n in rt.REQUEST_COUNTS if n]:
        for width in (1, ROWS):

            def build(seed, num_reqs=num_reqs, width=width):
                gen = rt._gen(num_reqs * 2 + width + 1000 * seed)
                idx, state, sampled, ns_, nr, starts = rt._post_update_case(num_reqs, width, gen)
                return SimpleNamespace(
                    idx=idx,
                    computed=state.computed,
                    last=state.last,
                    bins=state.bins,
                    sampled=sampled,
                    num_sampled=ns_,
                    num_rejected=nr,
                    starts=starts,
                    history=state.history,
                    total=state.total,
                )

            builds.append(build)
    return builds


def triton_post(ns):
    ib._post_update_kernel[(ns.idx.shape[0],)](
        ns.idx, ns.computed, ns.last, ns.bins, ns.bins.stride(0), ns.sampled,
        ns.sampled.stride(0), ns.num_sampled, ns.num_rejected, ns.starts, ns.history,
        ns.history.stride(0), ns.total, None, None, None, None, None, None,
        NUM_CAPTURES=0, BOUNDARY_METADATA_WIDTH=0, num_warps=1,
    )  # fmt: skip


def prod_post(ns):
    tli.post_update(
        ns.idx, ns.computed, ns.last, ns.bins, ns.sampled, ns.num_sampled,
        ns.num_rejected, ns.starts, ns.history, ns.total,
    )  # fmt: skip


def post_variant(threads, spread, index, static):
    def launch(ns):
        strides = (ns.bins.stride(0), ns.history.stride(0)) if static else ()
        kernel = compiled(
            rv.post_update_kernel, ns.sampled.shape[1], threads, spread, index, strides
        )
        kernel(ns.idx, ns.computed, ns.last.view(-1), ns.bins, ns.sampled,
               ns.num_sampled, ns.num_rejected, ns.starts, ns.history, ns.total)  # fmt: skip

    return launch


def post_kernel() -> Kernel:
    def key_of(ns):
        return tli.post_update_key(ns.sampled.shape[1], True, True, False, "int64")

    return Kernel(
        name="post_update",
        shapes={s: post_bench(n, p) for s, (n, p) in SHAPES.items()},
        checks=post_checks(),
        triton=triton_post,
        variants={
            "prod": prod_post,
            "prod-binds": via_production(prod_post, tli.post_update_kernel, key_of, BINDS_OFF),
            "hoist-t32": post_variant(32, False, "int64", False),
            "hoist-t32-i32": post_variant(32, False, "int32", False),
            "hoist-spread": post_variant(32, True, "int64", False),
            # Strides baked in: one kernel per (bins, history) stride pair.
            "hoist-t32-static": post_variant(32, False, "int64", True),
        },
        bench_only=(),
    )


# --- prepare_dflash_inputs --------------------------------------------------------


def _dflash_outputs(gen):
    return [
        _garbage((MAX_TOKENS,), torch.int32, gen),
        _garbage((MAX_TOKENS,), torch.int64, gen),
        _garbage((MAX_REQS + 1,), torch.int32, gen),
        _garbage((MAX_REQS,), torch.int32, gen),
        _garbage((MAX_TOKENS,), torch.int64, gen),
        _garbage((MAX_TOKENS,), torch.int64, gen),
        _garbage((MAX_TOKENS,), torch.int64, gen),
        _garbage((MAX_REQS * DRAFTS,), torch.int64, gen),
        _garbage((MAX_REQS * DRAFTS,), torch.int64, gen),
        _garbage((MAX_REQS * DRAFTS,), torch.int32, gen),
        _garbage((MAX_REQS,), torch.float32, gen),
        _garbage((MAX_REQS,), torch.int64, gen),
    ]


def dflash_bench(num_reqs, prefill):
    def build(seed):
        gen = _gen(seed)
        lens, starts = _step(num_reqs, prefill)
        reqs = len(lens)
        positions = torch.zeros(MAX_TOKENS, dtype=torch.int64, device=DEVICE)
        for r, n in enumerate(lens):
            first = int(_randint(0, MODEL_LEN - n, (), gen))
            s = int(starts[r])
            positions[s : s + n] = torch.arange(first, first + n, device=DEVICE)
        num_sampled = _randint(1, ROWS + 1, (reqs,), gen)
        num_rejected = ROWS - num_sampled if not prefill else torch.zeros_like(num_sampled)
        return SimpleNamespace(
            outs=_dflash_outputs(gen),
            positions=positions,
            starts=starts,
            idx=torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:reqs],
            last=_randint(0, VOCAB, (MAX_REQS, 1), gen, torch.int64),
            next_tokens=_randint(0, VOCAB, (1, MAX_REQS), gen),
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            temperature=torch.rand(MAX_REQS, generator=gen, device=DEVICE),
            seeds=_randint(0, 1 << 30, (MAX_REQS,), gen, torch.int64),
            table=_randint(0, 1 << 20, (MAX_REQS, MODEL_LEN // 256), gen),
            scalars=(123, 256, ROWS, DRAFTS, MAX_REQS, MAX_REQS * ROWS, MODEL_LEN, 0),
            reqs=reqs,
            rows=max(lens) + ROWS,
        )

    return build


def dflash_checks():
    builds = []
    for num_reqs in [n for n in rt.REQUEST_COUNTS if n]:
        for prefill in (False, True):

            def build(seed, num_reqs=num_reqs, prefill=prefill):
                gen = rt._gen(num_reqs + 50 * prefill + 1 + 1000 * seed)
                batch, inputs, outputs, max_tokens = rt._dflash_case(num_reqs, gen, prefill, 1)
                nqpr = 1 + DRAFTS
                return SimpleNamespace(
                    outs=[getattr(outputs, k) for k in vars(outputs)],
                    positions=batch.positions,
                    starts=batch.query_start_loc,
                    idx=batch.idx_mapping,
                    last=inputs.last_sampled,
                    next_tokens=inputs.next_prefill_tokens,
                    num_sampled=inputs.num_sampled,
                    num_rejected=inputs.num_rejected,
                    temperature=inputs.temperature,
                    seeds=inputs.seeds,
                    table=inputs.block_table,
                    scalars=(123, 16, nqpr, DRAFTS, MAX_REQS, max_tokens, 1 << 20, 0),
                    reqs=num_reqs,
                    rows=int(batch.num_scheduled_tokens.max()) + nqpr,
                )

            builds.append(build)
    return builds


def _dflash_inputs(ns):
    return (ns.positions, ns.starts, ns.idx, ns.last, ns.next_tokens, ns.num_sampled,
            ns.num_rejected, ns.temperature, ns.seeds, ns.table)  # fmt: skip


def triton_dflash(ns):
    dflash._prepare_dflash_inputs_kernel[
        (ns.reqs, triton.cdiv(ns.rows, dflash._PREPARE_BLOCK_SIZE))
    ](
        *ns.outs, *_dflash_inputs(ns), ns.table.stride(0), *ns.scalars,
        SAMPLE_FROM_ANCHOR=False, PAD_SLOT_ID=PAD_SLOT_ID, CP_SIZE=1, CP_INTERLEAVE=1,
        BLOCK_SIZE=dflash._PREPARE_BLOCK_SIZE,
    )  # fmt: skip


def prod_dflash(ns):
    tli.prepare_dflash_inputs(
        *ns.outs, *_dflash_inputs(ns), *ns.scalars, 1, 1, False, ns.reqs, ns.rows
    )


def dflash_variant(threads, per, index):
    def launch(ns):
        kernel = compiled(rv.dflash_kernel, threads, per, index)
        token, block_size, nqpr, steps, max_reqs, max_tokens, model_len, _ = ns.scalars
        kernel(
            *ns.outs, ns.positions, ns.starts, ns.idx, ns.last.view(-1),
            ns.next_tokens.view(-1), ns.num_sampled, ns.num_rejected, ns.temperature,
            ns.seeds, ns.table, token, block_size, nqpr, steps, max_reqs, max_tokens,
            model_len, ns.reqs, -(-ns.rows // (threads * per)),
        )  # fmt: skip

    return launch


def dflash_kernel() -> Kernel:
    variants = {
        "prod": prod_dflash,
        "prod-binds": via_production(
            prod_dflash,
            tli.prepare_dflash_inputs_kernel,
            lambda ns: tli.prepare_dflash_inputs_key(False, 1, 1, "int64"),
            BINDS_OFF,
        ),
    }
    for threads, per in ((64, 1), (128, 1), (256, 1), (64, 2), (128, 2)):
        variants[f"hoist-t{threads}-p{per}"] = dflash_variant(threads, per, "int64")
    variants["hoist-t256-p1-i32"] = dflash_variant(256, 1, "int32")
    return Kernel(
        name="prepare_dflash_inputs",
        shapes={s: dflash_bench(n, p) for s, (n, p) in SHAPES.items()},
        checks=dflash_checks(),
        triton=triton_dflash,
        variants=variants,
        bench_only=(),
    )


# --- gather_lookback --------------------------------------------------------------

DEPTH = 3


def lookback_bench(num_reqs, prefill):
    def build(seed):
        gen = _gen(seed)
        reqs = 1 if prefill else num_reqs
        history, buffer = _uva_history(gen)
        return SimpleNamespace(
            window=_garbage((MAX_REQS, DEPTH), torch.int32, gen),
            idx=torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:reqs],
            computed=_randint(DEPTH, HISTORY, (MAX_REQS,), gen),
            history=history,
            _buffer=buffer,
        )

    return build


def lookback_checks():
    builds = []
    for num_reqs in rt.REQUEST_COUNTS:

        def build(seed, num_reqs=num_reqs):
            gen = rt._gen(num_reqs * 9 + DEPTH + 1000 * seed)
            return SimpleNamespace(
                idx=rt._idx(num_reqs, gen),
                computed=rt._randint(0, 12, (MAX_REQS,), gen),  # windows before history
                history=rt._history(gen, 64),
                window=rt._garbage((MAX_REQS, DEPTH), torch.int32, gen),
            )

        builds.append(build)
    return builds


def triton_lookback(ns):
    _gather_lookback_kernel[(ns.window.shape[0],)](
        ns.window, ns.idx, ns.computed, ns.history, ns.history.stride(0),
        ns.idx.shape[0], DEPTH=DEPTH, BLOCK_DEPTH=triton.next_power_of_2(DEPTH),
    )  # fmt: skip


def prod_lookback(ns):
    lookback.gather_lookback(ns.window, ns.idx, ns.computed, ns.history)


def lookback_binds(ns):
    key = lookback.gather_lookback_key(DEPTH, "int64")
    recompiled(lookback.gather_lookback_kernel, key, BINDS_OFF)(
        ns.window, ns.idx, ns.computed, ns.history
    )


def lookback_variant(threads, spread, index):
    def launch(ns):
        kernel = compiled(rv.lookback_kernel, DEPTH, threads, spread, index)
        kernel(ns.window, ns.idx, ns.computed, ns.history)

    return launch


def lookback_kernel() -> Kernel:
    variants = {"prod": prod_lookback, "prod-binds": lookback_binds}
    for threads in (32, 64, 128):
        variants[f"hoist-t{threads}"] = lookback_variant(threads, False, "int64")
    variants["hoist-spread"] = lookback_variant(32, True, "int64")
    variants["hoist-t128-i32"] = lookback_variant(128, False, "int32")
    return Kernel(
        name="gather_lookback",
        shapes={s: lookback_bench(n, p) for s, (n, p) in SHAPES.items()},
        checks=lookback_checks(),
        triton=triton_lookback,
        variants=variants,
        bench_only=(),
    )


# --- zero_kv_blocks ---------------------------------------------------------------


def _zero_case(layouts, num_blocks, ids):
    """Segments laid out as (block stride, page, element offset) over garbage."""

    def build(seed):
        gen = rt._gen(7 + seed)
        storage = [
            rt._garbage((num_blocks * stride + offset + 8,), torch.int32, gen)
            for stride, _, offset in layouts
        ]
        addrs = [
            s.data_ptr() + offset * 4 for s, (_, _, offset) in zip(storage, layouts, strict=True)
        ]
        return SimpleNamespace(
            storage=storage,
            addrs=torch.tensor(addrs, dtype=torch.uint64, device=DEVICE),
            strides=torch.tensor([s for s, _, _ in layouts], device=DEVICE),
            pages=torch.tensor([p for _, p, _ in layouts], device=DEVICE),
            ids=torch.tensor(ids, dtype=torch.int64, device=DEVICE),
            max_page=max(p for _, p, _ in layouts),
            _skip=("addrs",),  # each copy's own addresses
        )

    return build


def zero_bench(num_reqs, prefill):
    # The pages of a step's new blocks over 43 layers' caches: a block a request in
    # decode, a 16384-token chunk's 64 blocks in prefill.
    pages = [33792, 4224] * 21 + [33792]
    count = 64 if prefill else num_reqs
    ids = torch.randperm(72, generator=torch.Generator().manual_seed(count))[:count]
    return _zero_case([(p, p, 0) for p in pages], 72, ids.tolist())


def zero_checks():
    layouts = [(33792, 33792, 0), (1100, 1027, 0), (12, 4, 3), (2048, 2048, 1), (5, 5, 0)]
    builds = []
    for num_ids in (1, 5, 32):
        ids = torch.randperm(40, generator=torch.Generator().manual_seed(num_ids))[:num_ids]
        builds.append(_zero_case(layouts, 40, ids.tolist()))
    return builds


def triton_zero(ns):
    blk = min(1 << (ns.max_page - 1).bit_length(), 1024)
    _zero_kv_blocks_kernel[(ns.ids.shape[0], ns.addrs.shape[0], math.ceil(ns.max_page / blk))](
        ns.addrs, ns.strides, ns.pages, ns.ids, BLOCK_SIZE=blk
    )


def prod_zero(ns):
    tli.zero_kv_blocks(ns.addrs.view(torch.int64), ns.strides, ns.pages, ns.ids, ns.max_page)


def zero_variant(threads, per):
    def launch(ns):
        kernel = compiled(rv.zero_kv_kernel, threads, per)
        kernel(ns.addrs.view(torch.int64), ns.strides, ns.pages, ns.ids,
               -(-ns.max_page // (threads * per)))  # fmt: skip

    return launch


def zero_kernel() -> Kernel:
    variants = {
        "prod": prod_zero,
        "prod-binds": via_production(
            prod_zero, tli.zero_kv_blocks_kernel, lambda ns: (), BINDS_OFF
        ),
    }
    for threads, per in ((128, 4), (256, 4), (512, 4), (128, 8), (256, 8)):
        variants[f"hoist-t{threads}-p{per}"] = zero_variant(threads, per)
    return Kernel(
        name="zero_kv_blocks",
        shapes={s: zero_bench(n, p) for s, (n, p) in SHAPES.items()},
        checks=zero_checks(),
        triton=triton_zero,
        variants=variants,
        bench_only=(),
    )


# --- Driver -----------------------------------------------------------------------


_PLAIN_INPUTS = plain_loads.Plain(tli, tli._PASS)
_PLAIN_LOOKBACK = plain_loads.Plain(
    lookback,
    {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    },
    "_bound",
    lookback.gather_lookback_kernel,
)


def plain_variant(prod):
    """The production launch with every kernel's read-only buffers loaded by
    ordinary loads (plain_loads)."""

    def launch(ns):
        with _PLAIN_INPUTS.active(), _PLAIN_LOOKBACK.active():
            prod(ns)

    return launch


def kernels() -> list[Kernel]:
    found = [
        combine_kernel(dead=True),
        combine_kernel(dead=False),
        expand_kernel(),
        dflash_kernel(),
        prefill_kernel(),
        post_kernel(),
        lookback_kernel(),
        zero_kernel(),
    ]
    for kernel in found:
        kernel.variants["prod-plain"] = plain_variant(kernel.variants["prod"])
    return found


def check(kernel: Kernel, name: str, launch) -> dict:
    """The variant against Triton on every check case and the benchmark's inputs
    (two seeds a shape): every buffer of the namespace, byte for byte."""
    builds = [] if name in kernel.bench_only else list(kernel.checks)
    builds += list(kernel.shapes.values())
    cases = 0
    for build in builds:
        for seed in (0, 1):
            ref, ours = build(seed), build(seed)
            kernel.triton(ref)
            launch(ours)
            torch.cuda.synchronize()
            cases += 1
            differ = _differences(ref, ours)
            if differ:
                return {"correct": False, "cases": cases, "differs": differ[:8]}
            del ref, ours
    return {"correct": True, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels", nargs="*", help="substrings of kernel names")
    parser.add_argument("--variants", nargs="*", help="substrings of variant names")
    parser.add_argument("--shapes", nargs="*", choices=sorted(SHAPES), default=None)
    parser.add_argument("--mode", choices=("both", "graph", "cupti"), default="both")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    shapes = args.shapes or list(SHAPES)
    failed = False
    for kernel in kernels():
        if args.kernels and not any(k in kernel.name for k in args.kernels):
            continue
        for name, launch in kernel.variants.items():
            if args.variants and not any(v in name for v in args.variants):
                continue
            record = {"kernel": kernel.name, "variant": name}
            try:
                record.update(check(kernel, name, launch))
            except Exception as error:  # a variant that fails to compile or launch
                record.update(correct=False, error=f"{type(error).__name__}: {error}"[:2000])
                traceback.print_exc(file=sys.stderr)
            print(json.dumps(record), flush=True)
            if not record["correct"]:
                failed = True
                continue
            for shape in shapes:
                build = kernel.shapes[shape]
                line = {"kernel": kernel.name, "variant": name, "shape": shape}
                if args.mode in ("both", "graph"):
                    line.update(paired(launch, kernel.triton, build, args.samples))
                if args.mode in ("both", "cupti"):
                    ns = build(0)
                    line["device_us"] = device_times(lambda f=launch, ns=ns: f(ns))
                    line["triton_device_us"] = device_times(lambda f=kernel.triton, ns=ns: f(ns))
                print(json.dumps(line), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
