# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired per-call timing of the V2 runner's TileLang input-preparation kernels
against the Triton kernels they replace, at serving shapes: a decode step of one
request and of sixteen (1 + 5 drafted rows each), and a 16384-token prefill step.

Each kernel and shape is measured on eight fresh inputs; on each, CUDA graphs of 100
calls of both run back to back, alternating which goes first, and the case's 95%
interval comes from the eight per-input differences. The gate (``--gate``) is that
no TileLang kernel is slower at any of the serving shapes (the interval's lower
bound above zero).

The block tables follow the deployment's shape: seventeen KV cache groups whose
rows hold about 93K block ids at a 1M-token context (``--widths`` replaces the
group widths, in kernel blocks; a width of 1 is a per-request ring). The token
history is the runner's pinned host memory, as in serving.
"""

import argparse
import json
import math
import statistics
from collections.abc import Callable

import torch

from vllm.models.deepseek_v4_1.nvidia.model_state import _gather_lookback_kernel
from vllm.models.deepseek_v4_1.tilelang import dspark_context, lookback
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu import input_batch as ib
from vllm.v1.worker.gpu import tilelang_inputs as tli
from vllm.v1.worker.gpu.block_table import (
    _compute_slot_mappings_kernel,
    _gather_block_tables_kernel,
)
from vllm.v1.worker.gpu.buffer_utils import UvaBuffer, _apply_write_kernel
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash
from vllm.v1.worker.utils import _zero_kv_blocks_kernel

DEVICE = torch.device("cuda")
MAX_REQS = 16
MAX_TOKENS = 16384
DRAFTS = 5
ROWS = 1 + DRAFTS
MODEL_LEN = 1 << 20
HISTORY = 1 << 16  # token history columns (the runner's are max_model_len)
VOCAB = 129280
# Kernel blocks a row at a 1M-token context: 64- and 256-token blocks of the
# attention caches, the compressed caches, the indexer and two rings.
WIDTHS = (16384,) * 4 + (4096,) * 6 + (1024,) * 3 + (64, 32, 1, 1)
SHAPES = {"decode c1": (1, False), "decode c16": (16, False), "prefill 16K": (1, True)}


def graph(function: Callable[[], None]) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(side)
    torch.accelerator.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for _ in range(100):
            function()
    return capture


def elapsed(capture: torch.cuda.CUDAGraph) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 10  # ms / 100 calls to microseconds


def _randint(low, high, shape, gen, dtype=torch.int32):
    return torch.randint(low, high, shape, generator=gen, device=DEVICE, dtype=dtype)


def _ptrs(tensors) -> torch.Tensor:
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.uint64).to(DEVICE)


def _step(num_reqs: int, prefill: bool) -> tuple[list[int], torch.Tensor]:
    """Query lengths and starts: decode rows, or one 16384-token prefill chunk."""
    lens = [MAX_TOKENS] if prefill else [ROWS] * num_reqs
    starts = torch.tensor([0] + lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return lens, starts.to(DEVICE)


def _history(gen) -> torch.Tensor:
    """The runner's token history: pinned host memory, read and written over UVA."""
    buffer = UvaBuffer((MAX_REQS, HISTORY), torch.int32)
    history = buffer.uva()
    history.uva_buffer = buffer  # the pinned host tensor lives as long as the view
    history.copy_(_randint(0, VOCAB, (MAX_REQS, HISTORY), gen))
    return history


def _tables(gen, widths):
    tables = [_randint(1, 1 << 20, (MAX_REQS, w), gen) for w in widths]
    counts = torch.stack(
        [_randint(max(w // 2, 1), w + 1, (MAX_REQS,), gen) for w in widths]
    )
    strides = torch.tensor(widths, dtype=torch.int64, device=DEVICE)
    return tables, counts, strides


# --- Cases: (Triton call, TileLang call) on one fresh input --------------------
# A case that hands the kernels a pointer table also returns the tensors it
# points at: nothing else holds them once the case returns.


def case_gather_block_tables(gen, num_reqs, prefill, widths):
    tables, counts, strides = _tables(gen, widths)
    out = [torch.empty_like(t) for t in tables]
    src, dst = _ptrs(tables), _ptrs(out)
    idx = torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:num_reqs]
    rows, groups = num_reqs, len(widths)

    def triton_call():
        _gather_block_tables_kernel[(groups, rows)](
            idx, src, dst, strides, counts, counts.stride(0), num_reqs, BLOCK_SIZE=1024
        )

    src64, dst64 = src.view(torch.int64), dst.view(torch.int64)
    chunks = tli.gather_chunks(widths, DEVICE)

    def tilelang_call():
        tli.gather_block_tables(idx, src64, dst64, strides, counts, chunks, rows)

    return triton_call, tilelang_call, tables, out


def case_compute_slot_mappings(gen, num_reqs, prefill, widths):
    tables, counts, strides = _tables(gen, widths)
    groups = len(widths)
    kernel_blocks = [max(MODEL_LEN // w, 8) if w > 1 else 8 for w in widths]
    kernel_block_sizes = torch.tensor(kernel_blocks, dtype=torch.int32, device=DEVICE)
    block_sizes = kernel_block_sizes.clone()
    enabled = torch.ones(groups, dtype=torch.bool, device=DEVICE)
    circular = torch.tensor([w == 1 for w in widths], device=DEVICE)
    group_cp = torch.ones(groups, dtype=torch.int32, device=DEVICE)
    idx = torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:num_reqs]
    lens, starts = _step(num_reqs, prefill)
    positions = torch.zeros(MAX_TOKENS, dtype=torch.int64, device=DEVICE)
    for r, n in enumerate(lens):
        first = int(_randint(0, MODEL_LEN - n, (), gen))
        s = int(starts[r])
        positions[s : s + n] = torch.arange(first, first + n, device=DEVICE)
    out = torch.empty(groups, MAX_TOKENS, dtype=torch.int64, device=DEVICE)
    ptrs = _ptrs(tables)

    def triton_call():
        _compute_slot_mappings_kernel[(groups, num_reqs + 1)](
            MAX_TOKENS, idx, starts, positions, ptrs, strides, counts,
            counts.stride(0), block_sizes, kernel_block_sizes, enabled, circular,
            group_cp, out, out.stride(0), 0, CP_SIZE=1, CP_INTERLEAVE=1,
            PAD_ID=PAD_SLOT_ID, TRITON_BLOCK_SIZE=1024,
        )  # fmt: skip

    ptrs64 = ptrs.view(torch.int64)

    def tilelang_call():
        tli.compute_slot_mappings(
            idx, starts, positions, ptrs64, strides, counts, block_sizes,
            kernel_block_sizes, enabled, circular, group_cp, out, 0, 1, 1,
        )  # fmt: skip

    return triton_call, tilelang_call, tables


def case_apply_write(gen, num_reqs, prefill, widths):
    """A new request's prompt into the token history (a prefill step), or one new
    block id a request into a block table (a decode step)."""
    if prefill:
        output = _history(gen)
        lengths = [MAX_TOKENS]
    else:
        output = torch.zeros(MAX_REQS, widths[0], dtype=torch.int32, device=DEVICE)
        lengths = [1] * num_reqs
    rows = list(range(len(lengths)))
    host_cu = torch.tensor(lengths).cumsum(0).tolist()
    indices = torch.tensor(rows, dtype=torch.int32, device=DEVICE)
    starts = _randint(0, 64, (len(lengths),), gen)
    contents = _randint(0, VOCAB, (host_cu[-1],), gen)
    cu = torch.tensor(host_cu, dtype=torch.int32, device=DEVICE)
    flat = output.view(-1)

    def triton_call():
        _apply_write_kernel[(len(lengths),)](
            output, output.stride(0), indices, starts, contents, cu, None,
            BLOCK_SIZE=1024, MULTI_GROUP=False,
        )  # fmt: skip

    def tilelang_call():
        tli.apply_staged_writes(
            flat, output.stride(0), indices, starts, contents, cu, host_cu
        )

    return triton_call, tilelang_call


def case_zero_kv_blocks(gen, num_reqs, prefill, widths):
    """The pages of a step's new blocks over 43 layers' caches: a block a request
    in decode, a 16384-token chunk's 64 blocks in prefill."""
    pages = [33792, 4224] * 21 + [33792]
    num_blocks = 72
    storage = [
        torch.empty(num_blocks * p, dtype=torch.int32, device=DEVICE) for p in pages
    ]
    addrs = _ptrs(storage)
    strides = torch.tensor(pages, dtype=torch.int64, device=DEVICE)
    sizes = strides.clone()
    count = 64 if prefill else num_reqs
    ids = torch.randperm(num_blocks, generator=gen, device=DEVICE)[:count]
    blk = 1024
    chunks = math.ceil(max(pages) / blk)

    def triton_call():
        _zero_kv_blocks_kernel[(count, len(pages), chunks)](
            addrs, strides, sizes, ids, BLOCK_SIZE=blk
        )

    addrs64 = addrs.view(torch.int64)

    def tilelang_call():
        tli.zero_kv_blocks(addrs64, strides, sizes, ids, max(pages))

    return triton_call, tilelang_call, storage


def case_prepare_prefill_inputs(gen, num_reqs, prefill, widths):
    lens, starts = _step(num_reqs, prefill)
    reqs = len(lens)
    idx = torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:reqs]
    history = _history(gen)
    computed = _randint(0, 1024, (MAX_REQS,), gen)
    # Prefilling requests: the chunk is part of the prompt.
    prefill_len = computed + MAX_TOKENS + 8
    ids = torch.empty(MAX_TOKENS, dtype=torch.int32, device=DEVICE)
    next_tokens = torch.empty(1, MAX_REQS, dtype=torch.int32, device=DEVICE)

    def triton_call():
        ib._prepare_prefill_inputs_kernel[(reqs,)](
            ids, next_tokens, next_tokens.stride(0), 1, idx, starts, history,
            history.stride(0), prefill_len, computed, BLOCK_SIZE=1024,
            LOOKAHEAD_BLOCK=1,
        )  # fmt: skip

    def tilelang_call():
        tli.prepare_prefill_inputs(
            ids, next_tokens, idx, starts, history, prefill_len, computed
        )

    return triton_call, tilelang_call


def case_prepare_pos_seq_lens(gen, num_reqs, prefill, widths):
    lens, starts = _step(num_reqs, prefill)
    reqs = len(lens)
    idx = torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:reqs]
    computed = _randint(0, MODEL_LEN // 2, (MAX_REQS,), gen)
    pos = torch.empty(MAX_TOKENS, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.empty(MAX_REQS, dtype=torch.int32, device=DEVICE)

    def triton_call():
        ib._prepare_pos_seq_lens_kernel[(reqs + 1,)](
            pos, seq_lens, idx, starts, computed, MAX_REQS, BLOCK_SIZE=1024
        )

    def tilelang_call():
        tli.prepare_pos_seq_lens(idx, starts, computed, pos, seq_lens)

    return triton_call, tilelang_call


def _combine(gen, num_reqs, prefill, dead):
    lens, starts = _step(num_reqs, prefill)
    reqs = len(lens)
    logits = [1] if prefill else [ROWS] * reqs
    cu = torch.tensor([0] + logits, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu = cu.to(DEVICE)
    idx = torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:reqs]
    prefill_len = _randint(1, 1024, (MAX_REQS,), gen)
    seq_lens = prefill_len[idx] + (MAX_TOKENS if prefill else 4096)
    last = _randint(0, VOCAB, (MAX_REQS, 1), gen, torch.int64)
    drafts = _randint(0, VOCAB, (MAX_REQS, DRAFTS), gen, torch.int64)
    ids = torch.empty(MAX_TOKENS, dtype=torch.int32, device=DEVICE)
    out = torch.empty(sum(logits), dtype=torch.int64, device=DEVICE)
    dead_rows = None
    if dead:
        dead_rows = ib.DeadRows(
            torch.rand(MAX_REQS, DRAFTS, generator=gen, device=DEVICE),
            torch.zeros(MAX_TOKENS, dtype=torch.bool, device=DEVICE),
            torch.empty(reqs, dtype=torch.int32, device=DEVICE),
            math.log(0.2),
        )
    extra = (
        (
            dead_rows.confidence,
            dead_rows.confidence.stride(0),
            dead_rows.is_padding.view(torch.uint8),
            dead_rows.cut,
            dead_rows.log_tau,
        )
        if dead_rows is not None
        else (ids, 0, ids, ids, 0.0)
    )

    def triton_call():
        ib._combine_sampled_and_draft_tokens_kernel[(reqs,)](
            ids, idx, last, starts, seq_lens, prefill_len, drafts, drafts.stride(0),
            cu, out, *extra, NUM_NEW_SAMPLED_TOKENS=1, DEAD_ROWS=dead,
            BLOCK_SIZE=triton.next_power_of_2(ROWS),
        )  # fmt: skip

    def tilelang_call():
        tli.combine_sampled_and_draft_tokens(
            ids, idx, last, starts, seq_lens, prefill_len, drafts, cu, out, 1,
            dead_rows,
        )  # fmt: skip

    return triton_call, tilelang_call


def case_combine(gen, num_reqs, prefill, widths):
    return _combine(gen, num_reqs, prefill, dead=False)


def case_combine_dead_rows(gen, num_reqs, prefill, widths):
    return _combine(gen, num_reqs, prefill, dead=True)


def case_expand_idx_mapping(gen, num_reqs, prefill, widths):
    reqs = 1 if prefill else num_reqs
    logits = [1] if prefill else [ROWS] * reqs
    cu = torch.tensor([0] + logits, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu = cu.to(DEVICE)
    idx = torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:reqs]
    expanded = torch.empty(sum(logits), dtype=torch.int64, device=DEVICE)
    local = torch.empty(sum(logits), dtype=torch.int32, device=DEVICE)

    def triton_call():
        ib._expand_idx_mapping_kernel[(reqs,)](
            idx, expanded, local, cu, BLOCK_SIZE=triton.next_power_of_2(ROWS)
        )

    def tilelang_call():
        tli.expand_idx_mapping(idx, expanded, local, cu, ROWS)

    return triton_call, tilelang_call


def case_adaptive_offsets(gen, num_reqs, prefill, widths):
    """Adaptive verification's offsets: before the port, five torch launches
    (two zeroes, two adds and cumsums, the tail fill)."""
    reqs = 1 if prefill else num_reqs
    capacities = _randint(0, ROWS, (MAX_REQS,), gen)[:reqs]
    non_draft = _randint(1, 2, (MAX_REQS,), gen)[:reqs]
    cu = torch.empty(MAX_REQS + 1, dtype=torch.int32, device=DEVICE)
    starts = torch.empty(MAX_REQS + 1, dtype=torch.int32, device=DEVICE)
    tokens = reqs * ROWS

    def torch_call():
        cu[:1].zero_()
        torch.cumsum(capacities + 1, dim=0, out=cu[1 : reqs + 1])
        starts[:1].zero_()
        torch.cumsum(capacities + non_draft, dim=0, out=starts[1 : reqs + 1])
        starts[reqs + 1 :].fill_(tokens)

    def tilelang_call():
        tli.adaptive_offsets(capacities, non_draft, cu, starts, 1, tokens, True)

    return torch_call, tilelang_call


def case_dspark_context(gen, num_reqs, prefill, widths):
    """DSpark's context-graph inputs (DSparkContextCudaGraphs.run): the three
    auxiliary hidden states into the graph's buffer and its tail scrubbed; before
    the port, three strided copies and three fills. The prefill shape stands for
    a batch shrinking into a larger graph (90 rows in a 96-row graph)."""
    width = 5120
    num_tokens = 90 if prefill else num_reqs * ROWS
    # The context graphs' capacities: powers of two up to 96 rows.
    capacity = 96 if prefill else min(1 << (num_tokens - 1).bit_length(), 96)
    wide = [
        torch.randn(MAX_TOKENS, width + 64, generator=gen, device=DEVICE).to(
            torch.bfloat16
        )
        for _ in range(3)
    ]
    sources = [buffer[:, :width] for buffer in wide]
    aux = torch.empty(capacity, 3 * width, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.empty(MAX_TOKENS, dtype=torch.int64, device=DEVICE)
    slots = torch.empty(2, MAX_TOKENS, dtype=torch.int64, device=DEVICE)

    def torch_call():
        for i, source in enumerate(sources):
            aux[:num_tokens, i * width : (i + 1) * width].copy_(source[:num_tokens])
        aux[num_tokens:capacity].zero_()
        positions[num_tokens:capacity].zero_()
        slots[:, num_tokens:capacity].fill_(PAD_SLOT_ID)

    def tilelang_call():
        dspark_context.context_inputs(
            aux, sources, positions, slots, num_tokens, capacity
        )

    return torch_call, tilelang_call, wide


def case_post_update(gen, num_reqs, prefill, widths):
    """The commit of a verified step (up to 1 + 5 tokens a request), or of a
    prefill chunk's one sampled token. Repeated calls keep committing: both
    kernels see the same growing state."""
    reqs = 1 if prefill else num_reqs
    width = 1 if prefill else ROWS
    idx = torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:reqs]
    computed = _randint(0, 1 << 16, (MAX_REQS,), gen)
    last = torch.empty(MAX_REQS, 1, dtype=torch.int64, device=DEVICE)
    bins = torch.zeros(MAX_REQS, VOCAB, dtype=torch.int32, device=DEVICE)
    sampled = _randint(0, VOCAB, (reqs, width), gen, torch.int64)
    num_sampled = _randint(1, width + 1, (reqs,), gen)
    num_rejected = width - num_sampled
    lens = [MAX_TOKENS] if prefill else [ROWS] * reqs
    starts = torch.tensor([0] + lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    starts = starts.to(DEVICE)
    history = _history(gen)
    total = _randint(1024, 2048, (MAX_REQS,), gen)

    def triton_call():
        ib._post_update_kernel[(reqs,)](
            idx, computed, last, bins, bins.stride(0), sampled, sampled.stride(0),
            num_sampled, num_rejected, starts, history, history.stride(0), total,
            None, None, None, None, None, None, NUM_CAPTURES=0,
            BOUNDARY_METADATA_WIDTH=0, num_warps=1,
        )  # fmt: skip

    def tilelang_call():
        tli.post_update(
            idx, computed, last, bins, sampled, num_sampled, num_rejected, starts,
            history, total,
        )  # fmt: skip

    return triton_call, tilelang_call


def case_prepare_dflash_inputs(gen, num_reqs, prefill, widths):
    """One draft group's inputs (the drafter prepares one a draft KV cache group)."""
    lens, starts = _step(num_reqs, prefill)
    reqs = len(lens)
    positions = torch.zeros(MAX_TOKENS, dtype=torch.int64, device=DEVICE)
    for r, n in enumerate(lens):
        first = int(_randint(0, MODEL_LEN - n, (), gen))
        s = int(starts[r])
        positions[s : s + n] = torch.arange(first, first + n, device=DEVICE)
    idx = torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:reqs]
    num_sampled = _randint(1, ROWS + 1, (reqs,), gen)
    num_rejected = ROWS - num_sampled if not prefill else torch.zeros_like(num_sampled)
    last = _randint(0, VOCAB, (MAX_REQS, 1), gen, torch.int64)
    next_tokens = _randint(0, VOCAB, (1, MAX_REQS), gen)
    temperature = torch.rand(MAX_REQS, generator=gen, device=DEVICE)
    seeds = _randint(0, 1 << 30, (MAX_REQS,), gen, torch.int64)
    table = _randint(0, 1 << 20, (MAX_REQS, MODEL_LEN // 256), gen)
    max_tokens = MAX_REQS * ROWS
    outs = [
        torch.empty(MAX_TOKENS, dtype=torch.int32, device=DEVICE),
        torch.empty(MAX_TOKENS, dtype=torch.int64, device=DEVICE),
        torch.empty(MAX_REQS + 1, dtype=torch.int32, device=DEVICE),
        torch.empty(MAX_REQS, dtype=torch.int32, device=DEVICE),
        torch.empty(MAX_TOKENS, dtype=torch.int64, device=DEVICE),
        torch.empty(MAX_TOKENS, dtype=torch.int64, device=DEVICE),
        torch.empty(MAX_TOKENS, dtype=torch.int64, device=DEVICE),
        torch.empty(MAX_REQS * DRAFTS, dtype=torch.int64, device=DEVICE),
        torch.empty(MAX_REQS * DRAFTS, dtype=torch.int64, device=DEVICE),
        torch.empty(MAX_REQS * DRAFTS, dtype=torch.int32, device=DEVICE),
        torch.empty(MAX_REQS, dtype=torch.float32, device=DEVICE),
        torch.empty(MAX_REQS, dtype=torch.int64, device=DEVICE),
    ]
    rows = max(lens) + ROWS
    inputs = (positions, starts, idx, last, next_tokens, num_sampled, num_rejected,
              temperature, seeds, table)  # fmt: skip
    scalars = (123, 256, ROWS, DRAFTS, MAX_REQS, max_tokens, MODEL_LEN, 0)

    def triton_call():
        dflash._prepare_dflash_inputs_kernel[
            (reqs, triton.cdiv(rows, dflash._PREPARE_BLOCK_SIZE))
        ](
            *outs, *inputs, table.stride(0), *scalars, SAMPLE_FROM_ANCHOR=False,
            PAD_SLOT_ID=PAD_SLOT_ID, CP_SIZE=1, CP_INTERLEAVE=1,
            BLOCK_SIZE=dflash._PREPARE_BLOCK_SIZE,
        )  # fmt: skip

    def tilelang_call():
        tli.prepare_dflash_inputs(*outs, *inputs, *scalars, 1, 1, False, reqs, rows)

    return triton_call, tilelang_call


def case_gather_lookback(gen, num_reqs, prefill, widths):
    """The Engram window of every row (the batch's, then -1) at depth 3."""
    reqs = 1 if prefill else num_reqs
    idx = torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:reqs]
    computed = _randint(3, HISTORY, (MAX_REQS,), gen)
    history = _history(gen)
    window = torch.empty(MAX_REQS, 3, dtype=torch.int32, device=DEVICE)

    def triton_call():
        _gather_lookback_kernel[(MAX_REQS,)](
            window, idx, computed, history, history.stride(0), reqs, DEPTH=3,
            BLOCK_DEPTH=4,
        )  # fmt: skip

    def tilelang_call():
        lookback.gather_lookback(window, idx, computed, history)

    return triton_call, tilelang_call


CASES = {
    "gather_block_tables": case_gather_block_tables,
    "compute_slot_mappings": case_compute_slot_mappings,
    "apply_write": case_apply_write,
    "zero_kv_blocks": case_zero_kv_blocks,
    "prepare_prefill_inputs": case_prepare_prefill_inputs,
    "prepare_pos_seq_lens": case_prepare_pos_seq_lens,
    "combine_sampled_and_draft_tokens": case_combine,
    "combine_sampled_and_draft_tokens (dead rows)": case_combine_dead_rows,
    "expand_idx_mapping": case_expand_idx_mapping,
    "adaptive_offsets": case_adaptive_offsets,
    "dspark_context": case_dspark_context,
    "post_update": case_post_update,
    "prepare_dflash_inputs": case_prepare_dflash_inputs,
    "gather_lookback": case_gather_lookback,
}


def measure(case, num_reqs: int, prefill: bool, widths, samples: int = 8) -> dict:
    ours_us, triton_us, differences = [], [], []
    for sample in range(samples):
        gen = torch.Generator(device=DEVICE).manual_seed(sample)
        # The tensors behind the case's pointer tables live as long as its calls:
        # graph capture empties the allocator's cache, so a kernel reaching a
        # dropped one through its pointer reads unmapped memory.
        triton_call, tilelang_call, *pointed_at = case(gen, num_reqs, prefill, widths)
        ours = graph(tilelang_call)
        theirs = graph(triton_call)
        pair = (ours, theirs) if sample % 2 == 0 else (theirs, ours)
        times = {id(capture): elapsed(capture) for capture in pair}
        ours_us.append(times[id(ours)])
        triton_us.append(times[id(theirs)])
        differences.append(times[id(ours)] - times[id(theirs)])
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(samples)  # t(7), 95%
    return {
        "tilelang_us": statistics.median(ours_us),
        "triton_us": statistics.median(triton_us),
        "difference_us": mean,
        "interval_us": [mean - half, mean + half],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument(
        "--kernels", nargs="*", choices=sorted(CASES), help="default: every kernel"
    )
    parser.add_argument(
        "--widths",
        type=lambda text: tuple(int(w) for w in text.split(",")),
        default=WIDTHS,
        help="block-table widths of the KV cache groups, in kernel blocks",
    )
    args = parser.parse_args()
    names = args.kernels or list(CASES)
    results = {
        name: {
            shape: measure(CASES[name], num_reqs, prefill, args.widths)
            for shape, (num_reqs, prefill) in SHAPES.items()
        }
        for name in names
    }
    print(json.dumps(results, indent=1))
    if args.gate:
        slower = [
            f"{name} at {shape}"
            for name, shapes in results.items()
            for shape, result in shapes.items()
            if result["interval_us"][0] > 0
        ]
        if slower:
            raise SystemExit(f"TileLang slower: {', '.join(slower)}")


if __name__ == "__main__":
    main()
