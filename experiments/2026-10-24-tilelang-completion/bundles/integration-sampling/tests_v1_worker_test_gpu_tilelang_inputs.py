# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V2 runner's TileLang input-preparation kernels against the Triton kernels
they replace, bit for bit: every buffer a launch may write is compared whole
(starting from identical garbage), on randomized inputs and edge cases (no
requests, the most requests, request counts at and between CUDA graph capture
sizes, padded rows, PAD and -1 ids, a 16384-token step). The Triton kernels stay
importable as the reference; the runner launches the TileLang kernels through the
call sites exercised here. Last, the kernels the call sites launch are exactly
calls the runner declares for preparation.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang runner kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from vllm.models.deepseek_v4_1.nvidia.model_state import (  # noqa: E402
    _gather_lookback_kernel,
)
from vllm.models.deepseek_v4_1.tilelang import lookback  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # noqa: E402
from vllm.v1.worker.gpu import input_batch as ib  # noqa: E402
from vllm.v1.worker.gpu import tilelang_inputs as tli  # noqa: E402
from vllm.v1.worker.gpu.block_table import (  # noqa: E402
    BlockTables,
    _compute_slot_mappings_kernel,
    _gather_block_tables_kernel,
)
from vllm.v1.worker.gpu.buffer_utils import (  # noqa: E402
    StagedWriteTensor,
    _apply_write_kernel,
)
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash  # noqa: E402
from vllm.v1.worker.utils import KVBlockZeroer, _zero_kv_blocks_kernel  # noqa: E402

DEVICE = torch.device("cuda")
MAX_REQS = 16
MAX_TOKENS = 16384  # max_num_batched_tokens of the TP3 prefill
DRAFTS = 5  # num_speculative_steps: 1 + 5 rows a decoding request
# Request counts at and between the deployment's capture sizes (1-16 requests).
REQUEST_COUNTS = (0, 1, 2, 3, 4, 5, 7, 8, 11, 15, 16)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _randint(low, high, shape, gen, dtype=torch.int32):
    return torch.randint(low, high, shape, generator=gen, device=DEVICE, dtype=dtype)


def _garbage(shape, dtype, gen):
    """Identical random contents for both kernels' outputs: a write either kernel
    makes that the other does not shows."""
    if dtype.is_floating_point:
        return torch.randn(shape, generator=gen, device=DEVICE).to(dtype)
    info = torch.iinfo(dtype)
    low, high = max(info.min, -(2**31)), min(info.max, 2**31 - 1)
    return _randint(low, high, shape, gen, dtype)


def _query_lens(num_reqs: int, prefill: bool, first: int = 0) -> list[int]:
    """Decode rows (1 + drafts) a request; with ``prefill``, the middle request
    is a chunk filling the step, from row ``first``, to 16384 tokens."""
    lens = [1 + DRAFTS] * num_reqs
    if prefill and num_reqs:
        lens[num_reqs // 2] = MAX_TOKENS - first - (num_reqs - 1) * (1 + DRAFTS)
    return lens


def _starts(lens: list[int], first: int = 0) -> torch.Tensor:
    return torch.tensor(
        np.cumsum([first] + lens), dtype=torch.int32, device=DEVICE
    ).contiguous()


def _idx(num_reqs: int, gen: torch.Generator, dtype=torch.int64) -> torch.Tensor:
    return torch.randperm(MAX_REQS, generator=gen, device=DEVICE)[:num_reqs].to(dtype)


# --- Block tables -------------------------------------------------------------


def _block_tables(gen, *, cp_size=1, cp_rank=0, cp_interleave=1, group_cp=None):
    """A geometry like the deployment's: wide and narrow tables, widths that are
    and are not multiples of four, rings and disabled groups."""
    block_sizes = [256, 128, 256, 64, 8, 256, 16]
    kernel_block_sizes = [64, 128, 256, 64, 8, 128, 16]
    max_blocks = [4096, 33, 1024, 7, 1, 64, 3]
    tables = BlockTables(
        block_sizes=block_sizes,
        max_num_reqs=MAX_REQS,
        max_num_batched_tokens=MAX_TOKENS,
        max_num_blocks_per_group=max_blocks,
        device=DEVICE,
        kernel_block_sizes=kernel_block_sizes,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_interleave=cp_interleave,
        slot_mapping_enabled=[True, True, True, False, True, True, True],
        group_cp_sizes=group_cp,
        slot_mapping_circular=[False, False, False, False, True, False, False],
    )
    for g, table in enumerate(tables.block_tables):
        width = table.gpu.shape[1]
        # Block 0 (the null block) and real ids alike.
        table.gpu.copy_(_randint(0, 1 << 20, table.gpu.shape, gen))
        tables.num_blocks.np[g] = torch.randint(
            0, width + 1, (MAX_REQS,), generator=torch.Generator().manual_seed(g)
        ).numpy()
    tables.num_blocks.copy_to_uva()
    return tables


def _ptrs(tensors) -> torch.Tensor:
    return torch.tensor(
        [t.data_ptr() for t in tensors], dtype=torch.uint64, device=DEVICE
    )


@pytest.mark.parametrize("num_reqs", REQUEST_COUNTS)
@pytest.mark.parametrize("padded", [0, 3])
@pytest.mark.parametrize("idx_dtype", [torch.int64, torch.int32])
def test_gather_block_tables(num_reqs, padded, idx_dtype):
    """Groups whose width is a multiple of four take 16-byte moves, the others a
    column at a time; rows past ``rows`` keep their contents."""
    gen = _gen(num_reqs * 7 + padded)
    tables = _block_tables(gen)
    rows = min(num_reqs + padded, MAX_REQS)
    idx = _idx(num_reqs, gen, idx_dtype)

    def outputs():
        return [
            _garbage(table.gpu.shape, torch.int32, _gen(table.gpu.shape[1]))
            for table in tables.block_tables
        ]

    ref, ours = outputs(), outputs()
    if rows:
        _gather_block_tables_kernel[(tables.num_kv_cache_groups, rows)](
            idx,
            tables.block_table_ptrs,
            _ptrs(ref),
            tables.block_table_strides,
            tables.num_blocks.gpu,
            tables.num_blocks.gpu.stride(0),
            num_reqs,
            BLOCK_SIZE=1024,
        )
    tables.gather_block_tables(idx, rows, out=tuple(ours), out_ptrs=_ptrs(ours))
    torch.accelerator.synchronize()
    for r, o in zip(ref, ours):
        assert torch.equal(r, o)


def test_gather_block_tables_into_the_input_tables():
    """The runner's own call: into ``input_block_tables`` through their pointer
    table, padded rows zeroed."""
    gen = _gen(3)
    tables = _block_tables(gen)
    idx = _idx(5, gen)
    ref = [t.clone() for t in tables.input_block_tables]
    _gather_block_tables_kernel[(tables.num_kv_cache_groups, 8)](
        idx,
        tables.block_table_ptrs,
        _ptrs(ref),
        tables.block_table_strides,
        tables.num_blocks.gpu,
        tables.num_blocks.gpu.stride(0),
        5,
        BLOCK_SIZE=1024,
    )
    tables.gather_block_tables(idx, num_reqs_padded=8)
    torch.accelerator.synchronize()
    for r, o in zip(ref, tables.input_block_tables):
        assert torch.equal(r, o)


def _slot_case(gen, num_reqs, prefill, first):
    lens = _query_lens(num_reqs, prefill, first)
    starts = _starts(lens, first)
    total = int(starts[-1])
    assert total <= MAX_TOKENS
    # Positions past each request's blocks and at block edges, and of a request
    # continuing its sequence.
    positions = _randint(0, 1 << 20, (MAX_TOKENS,), gen, torch.int64)
    if total > first:
        positions[first:total] = torch.arange(
            total - first, device=DEVICE, dtype=torch.int64
        ) + _randint(0, 1 << 18, (), gen, torch.int64)
    return starts, positions


@pytest.mark.parametrize("num_reqs", REQUEST_COUNTS)
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize(
    "cp", [(1, 0, 1, None), (2, 1, 16, [2, 1, 2, 2, 1, 2, 2]), (4, 3, 128, None)]
)
def test_compute_slot_mappings(num_reqs, prefill, cp):
    cp_size, cp_rank, cp_interleave, group_cp = cp
    gen = _gen(num_reqs * 3 + prefill)
    tables = _block_tables(
        gen,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_interleave=cp_interleave,
        group_cp=group_cp,
    )
    idx = _idx(num_reqs, gen)
    for first in (0, 5):  # a step whose rows start past row 0 leaves them alone
        if first and num_reqs == 0:
            continue
        starts, positions = _slot_case(gen, num_reqs, prefill, first)
        garbage = _garbage(tables.slot_mappings.shape, torch.int64, gen)
        ref = garbage.clone()
        _compute_slot_mappings_kernel[(tables.num_kv_cache_groups, num_reqs + 1)](
            ref.shape[1],
            idx,
            starts,
            positions,
            tables.block_table_ptrs,
            tables.block_table_strides,
            tables.num_blocks.gpu,
            tables.num_blocks.gpu.stride(0),
            tables.block_sizes_tensor,
            tables.kernel_block_sizes_tensor,
            tables.slot_mapping_enabled,
            tables.slot_mapping_circular,
            tables.group_cp_sizes,
            ref,
            ref.stride(0),
            cp_rank,
            CP_SIZE=cp_size,
            CP_INTERLEAVE=cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,
        )
        tables.slot_mappings.copy_(garbage)
        ours = tables.compute_slot_mappings(idx, starts, positions, MAX_TOKENS)
        torch.accelerator.synchronize()
        assert torch.equal(ref, ours)


# --- Staged writes ------------------------------------------------------------


def _stage(tensor: StagedWriteTensor, gen, num_writes: int, long: bool) -> None:
    """Random writes, several to a row and overlapping one another. Triton
    applies each write in its own program, in no order, so where two writes
    overlap neither kernel defines which lands: an element staged again takes
    the value it was first staged with, and the result is defined."""
    rows = tensor.num_rows
    cols = tensor.gpu.shape[1] if tensor.gpu.dim() == 2 else 1
    staged: dict[tuple[int, int], int | float] = {}
    for w in range(num_writes):
        row = int(_randint(0, rows, (), gen))
        if cols == 1:
            value = int(_randint(-100, 100, (), gen))
            tensor.stage_write_elem(row, staged.setdefault((row, 0), value))
            continue
        start = int(_randint(0, cols // 2, (), gen))
        length = cols - start if long and w == 0 else int(_randint(0, 9, (), gen))
        length = min(length, cols - start)
        values = _garbage((length,), tensor.dtype, gen).tolist()
        values = [staged.setdefault((row, start + j), v) for j, v in enumerate(values)]
        tensor.stage_write(row, start, values)


def _reference_write(tensor: StagedWriteTensor, out: torch.Tensor) -> None:
    n = len(tensor._staged_write_indices)
    if n == 0:
        return

    def device(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device=DEVICE)

    _apply_write_kernel[(n,)](
        out,
        out.stride(0),
        device(tensor._staged_write_indices),
        device(tensor._staged_write_starts),
        device(tensor._staged_write_contents, tensor.dtype),
        device(tensor._staged_write_cu_lens),
        None,
        BLOCK_SIZE=1024,
        MULTI_GROUP=False,
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
@pytest.mark.parametrize("shape", [(MAX_REQS,), (MAX_REQS, 5000)])
@pytest.mark.parametrize("uva", [False, True])
@pytest.mark.parametrize("num_writes", [1, 3, 17])
def test_apply_staged_writes(dtype, shape, uva, num_writes):
    gen = _gen(num_writes + len(shape))
    tensor = StagedWriteTensor(shape, dtype, DEVICE, uva_instead_of_gpu=uva)
    tensor.gpu.copy_(_garbage(shape, dtype, gen))
    _stage(tensor, gen, num_writes, long=True)
    ref = tensor.gpu.clone()
    _reference_write(tensor, ref)
    tensor.apply_write()
    torch.accelerator.synchronize()
    assert torch.equal(ref.view(torch.uint8), tensor.gpu.view(torch.uint8))


def test_apply_fused_staged_writes():
    """``FusedStagedWriter``: int32 writes of several tables through int64
    pointers, each to its group's table and stride."""
    gen = _gen(11)
    tables = _block_tables(gen)
    for t in tables.block_tables:
        _stage(t, gen, 9, long=True)
    group_ids, indices, starts, contents, cu_lens = [], [], [], [], []
    for g, t in enumerate(tables.block_tables):
        group_ids += [g] * len(t._staged_write_indices)
        indices += t._staged_write_indices
        starts += t._staged_write_starts
        base = len(contents)
        contents += t._staged_write_contents
        cu_lens += [base + c for c in t._staged_write_cu_lens]
    ref = [t.gpu.clone() for t in tables.block_tables]

    def device(values):
        return torch.tensor(values, dtype=torch.int32, device=DEVICE)

    _apply_write_kernel[(len(indices),)](
        _ptrs(ref),
        tables.block_table_strides,
        device(indices),
        device(starts),
        device(contents),
        device(cu_lens),
        device(group_ids),
        BLOCK_SIZE=1024,
        MULTI_GROUP=True,
    )
    tables.apply_staged_writes()
    torch.accelerator.synchronize()
    for r, t in zip(ref, tables.block_tables):
        assert torch.equal(r, t.gpu)


# --- KV block zeroing ---------------------------------------------------------


@pytest.mark.parametrize("num_ids", [1, 5, 32])
def test_zero_kv_blocks(num_ids):
    """Segments of 16-byte aligned and misaligned addresses, block strides that
    are and are not multiples of four, and pages within and past a chunk."""
    gen = _gen(num_ids)
    num_blocks = 40
    layouts = [  # (block stride, page, element offset)
        (33792, 33792, 0),
        (1100, 1027, 0),
        (12, 4, 3),
        (2048, 2048, 1),
        (5, 5, 0),
    ]
    storages = [
        _garbage((num_blocks * stride + offset + 8,), torch.int32, gen)
        for stride, _, offset in layouts
    ]
    addrs = [s.data_ptr() + offset * 4 for s, (_, _, offset) in zip(storages, layouts)]
    strides = torch.tensor([s for s, _, _ in layouts], device=DEVICE)
    pages = torch.tensor([p for _, p, _ in layouts], device=DEVICE)
    max_page = max(p for _, p, _ in layouts)
    blk = min(1 << (max_page - 1).bit_length(), 1024)
    ids = torch.randperm(num_blocks, generator=gen, device=DEVICE)[:num_ids].tolist()
    ref = [s.clone() for s in storages]
    ref_addrs = [
        r.data_ptr() + (a - s.data_ptr()) for r, s, a in zip(ref, storages, addrs)
    ]
    _zero_kv_blocks_kernel[(num_ids, len(layouts), math.ceil(max_page / blk))](
        torch.tensor(ref_addrs, dtype=torch.uint64, device=DEVICE),
        strides,
        pages,
        torch.tensor(ids, dtype=torch.int64, device=DEVICE),
        BLOCK_SIZE=blk,
    )
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = DEVICE
    zeroer._meta = (
        torch.tensor(addrs, dtype=torch.uint64, device=DEVICE),
        strides,
        pages,
        math.ceil(max_page / blk),
        blk,
        len(layouts),
    )
    zeroer.zero_block_ids(ids)
    torch.accelerator.synchronize()
    for r, s in zip(ref, storages):
        assert torch.equal(r, s)


# --- Step inputs --------------------------------------------------------------


def _history(gen, model_len=MAX_TOKENS + 64):
    return _randint(0, 1 << 17, (MAX_REQS, model_len), gen)


@pytest.mark.parametrize("num_reqs", REQUEST_COUNTS)
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("lookahead", [1, 3])
def test_prepare_prefill_inputs(num_reqs, prefill, lookahead):
    gen = _gen(num_reqs + 31 * prefill + lookahead)
    lens = _query_lens(num_reqs, prefill)
    starts = _starts(lens)
    idx = _idx(num_reqs, gen)
    history = _history(gen)
    computed = _randint(0, 40, (MAX_REQS,), gen)
    # Prefilling requests (computed < prefill length, some ending inside the
    # lookahead), and decoding ones the kernel skips.
    prefill_len = computed + _randint(-3, MAX_TOKENS, (MAX_REQS,), gen)
    ids = _garbage((MAX_TOKENS,), torch.int32, gen)
    next_tokens = _garbage((lookahead, MAX_REQS), torch.int32, gen)
    ref_ids, ref_next = ids.clone(), next_tokens.clone()
    if num_reqs:
        ib._prepare_prefill_inputs_kernel[(num_reqs,)](
            ref_ids,
            ref_next,
            ref_next.stride(0),
            lookahead,
            idx,
            starts,
            history,
            history.stride(0),
            prefill_len,
            computed,
            BLOCK_SIZE=1024,
            LOOKAHEAD_BLOCK=triton.next_power_of_2(lookahead),
        )
    ib.prepare_prefill_inputs(
        ids, next_tokens, idx, starts, history, prefill_len, computed
    )
    torch.accelerator.synchronize()
    assert torch.equal(ref_ids, ids)
    assert torch.equal(ref_next, next_tokens)


@pytest.mark.parametrize("num_reqs", REQUEST_COUNTS)
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("idx_dtype", [torch.int64, torch.int32])
def test_prepare_pos_seq_lens(num_reqs, prefill, idx_dtype):
    gen = _gen(num_reqs + 17 * prefill)
    starts = _starts(_query_lens(num_reqs, prefill))
    idx = _idx(num_reqs, gen, idx_dtype)
    computed = _randint(0, 1 << 20, (MAX_REQS,), gen)
    pos = _garbage((MAX_TOKENS,), torch.int64, gen)
    seq_lens = _garbage((MAX_REQS,), torch.int32, gen)
    ref_pos, ref_seq = pos.clone(), seq_lens.clone()
    ib._prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
        ref_pos, ref_seq, idx, starts, computed, MAX_REQS, BLOCK_SIZE=1024
    )
    ib.prepare_pos_seq_lens(idx, starts, computed, pos, seq_lens)
    torch.accelerator.synchronize()
    assert torch.equal(ref_pos, pos)
    assert torch.equal(ref_seq, seq_lens)


def _combine_case(num_reqs, drafts, num_new, gen):
    """Decoding requests with up to ``drafts`` drafts each (fewer under adaptive
    verification), a chunked prefill, and a prompt tail."""
    num_drafts = _randint(0, drafts + 1, (num_reqs,), gen).tolist()
    logits = [d + num_new for d in num_drafts]
    lens = [max(n, 1) + int(_randint(0, 3, (), gen)) for n in logits]
    starts = _starts(lens)
    cu = _starts(logits)
    seq_lens = _randint(1, 4096, (num_reqs,), gen)
    prefill_len = _randint(0, 4096, (MAX_REQS,), gen)
    return starts, cu, seq_lens, prefill_len, sum(logits), int(starts[-1])


def _combine_ref(args, num_logits, drafts, num_new, dead_rows):
    (ids, idx, last, starts, seq_lens, prefill_len, draft_tokens, cu) = args
    logits = torch.full((num_logits,), -7, dtype=torch.int64, device=DEVICE)
    if idx.shape[0]:
        ib._combine_sampled_and_draft_tokens_kernel[(idx.shape[0],)](
            ids,
            idx,
            last,
            starts,
            seq_lens,
            prefill_len,
            draft_tokens,
            draft_tokens.stride(0),
            cu,
            logits,
            *(
                (
                    dead_rows.confidence,
                    dead_rows.confidence.stride(0),
                    dead_rows.is_padding.view(torch.uint8),
                    dead_rows.cut,
                    dead_rows.log_tau,
                )
                if dead_rows is not None
                else (ids, 0, ids, ids, 0.0)
            ),
            NUM_NEW_SAMPLED_TOKENS=num_new,
            DEAD_ROWS=dead_rows is not None,
            BLOCK_SIZE=triton.next_power_of_2(drafts + num_new),
        )
    return logits


def _combine_ours(args, num_logits, num_new, dead_rows):
    (ids, idx, last, starts, seq_lens, prefill_len, draft_tokens, cu) = args
    logits = torch.full((num_logits,), -7, dtype=torch.int64, device=DEVICE)
    if idx.shape[0]:
        tli.combine_sampled_and_draft_tokens(
            ids, idx, last, starts, seq_lens, prefill_len, draft_tokens, cu, logits,
            num_new, dead_rows,
        )  # fmt: skip
    return logits


@pytest.mark.parametrize("num_reqs", REQUEST_COUNTS)
@pytest.mark.parametrize("drafts,num_new", [(DRAFTS, 1), (DRAFTS, 0), (3, 1), (0, 1)])
def test_combine_sampled_and_draft_tokens(num_reqs, drafts, num_new):
    gen = _gen(num_reqs * 5 + drafts + num_new)
    starts, cu, seq_lens, prefill_len, num_logits, tokens = _combine_case(
        num_reqs, drafts, num_new, gen
    )
    idx = _idx(num_reqs, gen)
    last = _randint(-(2**40), 2**40, (MAX_REQS, 1), gen, torch.int64)
    draft_tokens = _randint(-(2**40), 2**40, (MAX_REQS, drafts), gen, torch.int64)
    ids = _garbage((tokens + 8,), torch.int32, gen)
    ref_ids = ids.clone()
    common = (idx, last, starts, seq_lens, prefill_len, draft_tokens, cu)
    ref = _combine_ref((ref_ids, *common), num_logits, drafts, num_new, None)
    ours = _combine_ours((ids, *common), num_logits, num_new, None)
    torch.accelerator.synchronize()
    assert torch.equal(ref_ids, ids)
    assert torch.equal(ref, ours)
    # The runner's wrapper: the same, through the switched call site.
    if num_reqs:
        wrapped_ids = ref_ids.clone()
        wrapped = ib.combine_sampled_and_draft_tokens(
            wrapped_ids, idx, last, starts, seq_lens, prefill_len, draft_tokens, cu,
            num_logits, num_new,
        )  # fmt: skip
        torch.accelerator.synchronize()
        assert torch.equal(ref_ids, wrapped_ids)
        assert torch.equal(ref, wrapped)


def _survivals(confidence: torch.Tensor, drafts: int) -> torch.Tensor:
    """Every running sum of the drafts' log confidences, in the warp's
    Kogge-Stone order and in sequence: thresholds at which an order that differs
    in one rounding moves a cut."""
    block = triton.next_power_of_2(drafts + 1)
    logs = torch.zeros(confidence.shape[0], block, device=DEVICE)
    logs[:, :drafts] = torch.log(confidence[:, :drafts])
    scan = logs.clone()
    stride = 1
    while stride < block:
        shifted = torch.zeros_like(scan)
        shifted[:, stride:] = scan[:, :-stride]
        scan = shifted + scan
        stride *= 2
    sequential = logs.clone()
    for j in range(1, block):
        sequential[:, j] = sequential[:, j - 1] + logs[:, j]
    return torch.cat((scan.flatten(), sequential.flatten()))


@pytest.mark.parametrize("num_reqs", [1, 3, 8, 16])
def test_combine_dead_rows(num_reqs):
    """Dead rows: identical cuts and padding at thresholds equal to every running
    sum of log confidences (either summation order) and between them, with
    confidences of 1, 0, denormals and NaN."""
    gen = _gen(num_reqs)
    drafts, num_new = DRAFTS, 1
    starts, cu, seq_lens, prefill_len, num_logits, tokens = _combine_case(
        num_reqs, drafts, num_new, gen
    )
    idx = _idx(num_reqs, gen)
    last = _randint(0, 1 << 17, (MAX_REQS, 1), gen, torch.int64)
    draft_tokens = _randint(0, 1 << 17, (MAX_REQS, drafts), gen, torch.int64)
    confidence = torch.rand(MAX_REQS, drafts, generator=gen, device=DEVICE) ** 0.25
    confidence[0, 2] = 1.0
    confidence[1 % MAX_REQS, 1] = 0.0
    confidence[2 % MAX_REQS, 3] = 1e-40  # denormal
    confidence[3 % MAX_REQS, 4] = float("nan")
    thresholds = [math.log(0.2), -0.0, -1e30]
    picked = _survivals(confidence[idx], drafts)
    picked = picked[torch.isfinite(picked)]
    thresholds += picked[torch.randperm(picked.numel(), generator=gen, device=DEVICE)][
        :48
    ].tolist()
    for log_tau in thresholds:
        for direction in (0, 1, -1):
            tau = (
                log_tau
                if direction == 0
                else float(
                    np.nextafter(np.float32(log_tau), np.float32(direction * np.inf))
                )
            )
            results = []
            for kind in ("ref", "ours"):
                ids = _garbage((tokens + 8,), torch.int32, _gen(1))
                padding = torch.zeros(tokens + 8, dtype=torch.bool, device=DEVICE)
                padding[::3] = True
                cut = _garbage((num_reqs,), torch.int32, _gen(2))
                dead = ib.DeadRows(confidence, padding, cut, tau)
                args = (ids, idx, last, starts, seq_lens, prefill_len, draft_tokens, cu)
                if kind == "ref":
                    logits = _combine_ref(args, num_logits, drafts, num_new, dead)
                else:
                    logits = _combine_ours(args, num_logits, num_new, dead)
                results.append((ids, padding, cut, logits))
            torch.accelerator.synchronize()
            for r, o in zip(*results):
                assert torch.equal(r, o), tau


@pytest.mark.parametrize("num_reqs", REQUEST_COUNTS)
@pytest.mark.parametrize("idx_dtype", [torch.int64, torch.int32])
def test_expand_idx_mapping(num_reqs, idx_dtype):
    gen = _gen(num_reqs + 3)
    block_len = DRAFTS + 1
    # Up to Triton's block of logits a request, and past it (left unwritten).
    logits = _randint(0, 11, (num_reqs,), gen).tolist()
    cu = _starts(logits)
    idx = _idx(num_reqs, gen, idx_dtype)
    total = int(cu[-1])
    ref_idx = _garbage((total,), idx_dtype, gen)
    ref_pos = _garbage((total,), torch.int32, gen)
    ours_idx, ours_pos = ref_idx.clone(), ref_pos.clone()
    if num_reqs:
        ib._expand_idx_mapping_kernel[(num_reqs,)](
            idx, ref_idx, ref_pos, cu, BLOCK_SIZE=triton.next_power_of_2(block_len)
        )
    tli.expand_idx_mapping(idx, ours_idx, ours_pos, cu, block_len)
    torch.accelerator.synchronize()
    assert torch.equal(ref_idx, ours_idx)
    assert torch.equal(ref_pos, ours_pos)


@pytest.mark.parametrize("num_logits", [0, 1, 6, 96, 1000])
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
def test_logits_rows(num_logits, index_dtype):
    """The logits rows' input ids and positions against torch's indexing,
    repeated rows and a negative index (which wraps) included."""
    gen = _gen(num_logits + 5)
    input_ids = _randint(0, 1 << 17, (MAX_TOKENS,), gen)
    positions = _randint(0, 1 << 20, (MAX_TOKENS,), gen, torch.int64)
    indices = _randint(0, MAX_TOKENS, (num_logits,), gen, index_dtype)
    if num_logits > 2:
        indices[1] = indices[0]
        indices[2] = -1
    ids, pos = tli.logits_rows(input_ids, positions, indices)
    torch.accelerator.synchronize()
    assert torch.equal(ids, input_ids[indices])
    assert torch.equal(pos, positions[indices])


@pytest.mark.parametrize(
    "live,rows", [(0, 0), (0, 8), (6, 8), (96, 96), (90, 128), (16000, MAX_TOKENS)]
)
def test_padding_rows(live, rows):
    """The routers' padding rows, torch's two fills (the target) and comparison
    (the drafter): rows past ``rows`` keep their contents."""
    gen = _gen(live + rows)
    start = torch.randint(0, 2, (MAX_TOKENS + 5,), generator=gen, device=DEVICE)
    expected, actual = start.bool(), start.bool()
    expected[:live].fill_(False)
    expected[live:rows].fill_(True)
    tli.padding_rows(actual, live, rows)
    torch.accelerator.synchronize()
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("num_reqs", REQUEST_COUNTS)
@pytest.mark.parametrize("width", [1, 3, DRAFTS])
@pytest.mark.parametrize("dtype,gather", [(torch.int64, True), (torch.float32, False)])
def test_scatter_rows(num_reqs, width, dtype, gather):
    """The runner's commit of a step's drafts (int64, read back) and adaptive
    verification's confidences (float32): ``state[idx] = source`` against torch's
    indexing, a filtered row's -1 wrapping to the last row as torch's does, with
    sources that are column views of wider rows."""
    gen = _gen(num_reqs * 7 + width)
    idx = _idx(num_reqs, gen)
    if 2 < num_reqs < MAX_REQS:
        # -1 wraps to the last row, which no other request then takes.
        idx = torch.randperm(MAX_REQS - 1, generator=gen, device=DEVICE)[:num_reqs]
        idx[1] = -1
    wide = _garbage((MAX_REQS, DRAFTS + 2), dtype, gen)
    source = wide[:num_reqs, :width]
    state = _garbage((MAX_REQS, DRAFTS), dtype, gen)
    expected = state.clone()
    expected[idx, :width] = source
    gathered = expected[idx, :width]
    out = tli.scatter_rows(idx, source, state, gather=gather)
    torch.accelerator.synchronize()
    bits = torch.int32 if dtype == torch.float32 else dtype
    assert torch.equal(state.view(bits), expected.view(bits))
    if gather:
        assert torch.equal(out, gathered)
    else:
        assert out is None


def _adaptive_offsets_ref(capacities, non_draft, cu, starts, bonus, tokens, drafted):
    """reallocate_drafts' torch chain before the port."""
    num_reqs = capacities.shape[0]
    if not drafted:
        capacities.zero_()
    cu[:1].zero_()
    torch.cumsum(capacities + bonus, dim=0, out=cu[1 : num_reqs + 1])
    starts[:1].zero_()
    torch.cumsum(capacities + non_draft, dim=0, out=starts[1 : num_reqs + 1])
    starts[num_reqs + 1 :].fill_(tokens)


# The batch's request counts, and counts at and across the scan's 256-request passes.
@pytest.mark.parametrize("num_reqs", REQUEST_COUNTS[1:] + (255, 256, 257, 600))
@pytest.mark.parametrize("drafted", [True, False])
def test_adaptive_offsets(num_reqs, drafted):
    """reallocate_drafts' offsets: both prefix sums, query_start_loc's tail and, on a
    step without drafts, the zeroed capacities, in buffers longer than the batch."""
    gen = _gen(num_reqs + 17)
    size = num_reqs + 7
    capacities = _randint(0, DRAFTS + 1, (size,), gen)
    non_draft = _randint(0, MAX_TOKENS, (size,), gen)
    cu = _garbage((size + 1,), torch.int32, gen)
    starts = _garbage((size + 1,), torch.int32, gen)
    bonus, tokens = 1, int(_randint(0, 1 << 20, (1,), gen))
    results = []
    for kind in ("ref", "ours"):
        buffers = [capacities.clone(), cu.clone(), starts.clone()]
        args = (
            buffers[0][:num_reqs],
            non_draft[:num_reqs],
            buffers[1],
            buffers[2],
            bonus,
            tokens,
            drafted,
        )
        if kind == "ref":
            _adaptive_offsets_ref(*args)
        else:
            tli.adaptive_offsets(*args)
        results.append(buffers)
    torch.accelerator.synchronize()
    for r, o in zip(*results):
        assert torch.equal(r, o)


def _post_update_case(num_reqs, width, gen, vocab=64):
    idx = _idx(num_reqs, gen)
    if num_reqs > 2:
        idx[1] = -1  # filtered row
    state = SimpleNamespace(
        computed=_randint(0, 100, (MAX_REQS,), gen),
        last=_garbage((MAX_REQS, 1), torch.int64, gen),
        bins=_randint(0, 5, (MAX_REQS, vocab), gen),
        history=_garbage((MAX_REQS, 256), torch.int32, gen),
        total=_randint(0, 100, (MAX_REQS,), gen),
    )
    # Repeated tokens within a row: the counts take each occurrence.
    sampled = _randint(0, 6, (num_reqs, width), gen, torch.int64)
    num_sampled = _randint(0, width + 1, (num_reqs,), gen)
    num_rejected = _randint(0, width + 1, (num_reqs,), gen)
    starts = _starts(_randint(1, 9, (num_reqs,), gen).tolist())
    return idx, state, sampled, num_sampled, num_rejected, starts


def _post_update_args(idx, state, sampled, ns, nr, starts, bins, query):
    return (
        idx,
        state.computed,
        state.last,
        state.bins if bins else None,
        sampled,
        ns,
        nr,
        starts if query else None,
        state.history,
        state.total,
    )


def _post_update_ref(args, boundary_state=None, capture=None):
    (idx, computed, last, bins, sampled, ns, nr, starts, history, total) = args
    if not idx.shape[0]:
        return
    ib._post_update_kernel[(idx.shape[0],)](
        idx,
        computed,
        last,
        bins,
        bins.stride(0) if bins is not None else 0,
        sampled,
        sampled.stride(0),
        ns,
        nr,
        starts,
        history,
        history.stride(0),
        total,
        boundary_state.metadata if boundary_state is not None else None,
        boundary_state.stop_tokens if boundary_state is not None else None,
        boundary_state.seen if boundary_state is not None else None,
        capture[0] if capture is not None else None,
        capture[1] if capture is not None else None,
        capture[2] if capture is not None else None,
        NUM_CAPTURES=4 if capture is not None else 0,
        BOUNDARY_METADATA_WIDTH=(
            boundary_state.metadata.shape[1] if boundary_state is not None else 0
        ),
        num_warps=1,
    )


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(**{k: _clone(v) for k, v in vars(value).items()})
    return value


@pytest.mark.parametrize("num_reqs", REQUEST_COUNTS)
@pytest.mark.parametrize("width", [1, DRAFTS + 1])
@pytest.mark.parametrize("bins,query", [(True, True), (False, False)])
def test_post_update(num_reqs, width, bins, query):
    gen = _gen(num_reqs * 2 + width)
    idx, state, sampled, ns, nr, starts = _post_update_case(num_reqs, width, gen)
    ref, ours = _clone(state), _clone(state)
    ref_ns, ref_nr = ns.clone(), nr.clone()
    _post_update_ref(
        _post_update_args(idx, ref, sampled, ref_ns, ref_nr, starts, bins, query)
    )
    ib.post_update(*_post_update_args(idx, ours, sampled, ns, nr, starts, bins, query))
    torch.accelerator.synchronize()
    for name in vars(state):
        assert torch.equal(getattr(ref, name), getattr(ours, name)), name
    assert torch.equal(ref_ns, ns) and torch.equal(ref_nr, nr)


@pytest.mark.parametrize("num_reqs", [1, 5, 16])
@pytest.mark.parametrize("query", [True, False])
def test_post_update_boundary_captures(num_reqs, query):
    """Request-boundary checkpoints: stops at the first stop token or length
    bound, the captures, and the trimmed commit."""
    gen = _gen(num_reqs + 100 * query)
    width = DRAFTS + 1
    idx, state, sampled, ns, nr, starts = _post_update_case(num_reqs, width, gen)
    metadata = torch.stack(
        [
            _randint(0, 2, (MAX_REQS,), gen),  # enabled
            state.computed + _randint(-2, 9, (MAX_REQS,), gen),  # prompt
            _randint(0, 3, (MAX_REQS,), gen) * (state.computed + 2),  # instruction
            state.total + _randint(-1, 4, (MAX_REQS,), gen),  # min length
            state.total + _randint(0, 8, (MAX_REQS,), gen),  # max length
            _randint(0, 140, (MAX_REQS,), gen),  # stops (past the capacity too)
            _randint(0, 2, (MAX_REQS,), gen) * (state.computed + 3),  # prefill tail
        ],
        dim=1,
    ).contiguous()
    boundary = SimpleNamespace(
        metadata=metadata,
        stop_tokens=_randint(0, 6, (MAX_REQS, 128), gen),
        seen=_randint(0, 2, (MAX_REQS, 4), gen),
    )
    capture = _garbage((3, num_reqs, 4), torch.int32, gen)
    ref, ours = _clone(state), _clone(state)
    ref_boundary, ours_boundary = _clone(boundary), _clone(boundary)
    ref_capture, ours_capture = capture.clone(), capture.clone()
    ref_ns, ref_nr = ns.clone(), nr.clone()
    _post_update_ref(
        _post_update_args(idx, ref, sampled, ref_ns, ref_nr, starts, True, query),
        ref_boundary,
        ref_capture,
    )
    ib.post_update(
        *_post_update_args(idx, ours, sampled, ns, nr, starts, True, query),
        ours_boundary,
        ours_capture,
    )
    torch.accelerator.synchronize()
    for name in vars(state):
        assert torch.equal(getattr(ref, name), getattr(ours, name)), name
    assert torch.equal(ref_boundary.seen, ours_boundary.seen)
    assert torch.equal(ref_capture, ours_capture)
    assert torch.equal(ref_ns, ns) and torch.equal(ref_nr, nr)


# --- DFlash drafting ----------------------------------------------------------


def _dflash_case(num_reqs, gen, prefill, cp_size):
    nqpr = 1 + DRAFTS
    lens = _query_lens(num_reqs, prefill)
    lens = [min(n, 2000) for n in lens]
    starts = _starts(lens)
    total = int(starts[-1])
    positions = torch.zeros(MAX_TOKENS, dtype=torch.int64, device=DEVICE)
    for r, (s, n) in enumerate(zip(starts.tolist(), lens)):
        positions[s : s + n] = torch.arange(n, device=DEVICE) + 300 * r + 1
    width = 64
    block_table = _randint(0, 1 << 16, (MAX_REQS, width), gen)
    block_table[:, ::5] = 0  # null blocks
    max_tokens = MAX_REQS * nqpr
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_scheduled_tokens=np.array(lens),
        positions=positions[: max(total, 1)],
        query_start_loc=starts,
        idx_mapping=_idx(num_reqs, gen),
    )
    inputs = SimpleNamespace(
        num_sampled=_randint(0, 3, (num_reqs,), gen),
        # Some requests reject every row (sink rejection of draft zero).
        num_rejected=torch.minimum(
            _randint(0, nqpr + 1, (num_reqs,), gen),
            torch.tensor(lens, dtype=torch.int32, device=DEVICE),
        ),
        last_sampled=_randint(-(2**40), 2**40, (MAX_REQS, 1), gen, torch.int64),
        next_prefill_tokens=_randint(0, 1 << 17, (1, MAX_REQS), gen),
        temperature=torch.rand(MAX_REQS, generator=gen, device=DEVICE),
        seeds=_randint(-(2**40), 2**40, (MAX_REQS,), gen, torch.int64),
        block_table=block_table,
    )
    outputs = SimpleNamespace(
        input_ids=_garbage((MAX_TOKENS,), torch.int32, gen),
        positions=_garbage((MAX_TOKENS,), torch.int64, gen),
        query_start_loc=_garbage((MAX_REQS + 1,), torch.int32, gen),
        seq_lens=_garbage((MAX_REQS,), torch.int32, gen),
        query_slots=_garbage((MAX_TOKENS,), torch.int64, gen),
        context_positions=_garbage((MAX_TOKENS,), torch.int64, gen),
        context_slots=_garbage((MAX_TOKENS,), torch.int64, gen),
        sample_indices=_garbage((MAX_REQS * DRAFTS,), torch.int64, gen),
        sample_pos=_garbage((MAX_REQS * DRAFTS,), torch.int64, gen),
        sample_map=_garbage((MAX_REQS * DRAFTS,), torch.int32, gen),
        temperature=_garbage((MAX_REQS,), torch.float32, gen),
        seeds=_garbage((MAX_REQS,), torch.int64, gen),
    )
    return batch, inputs, outputs, max_tokens


@pytest.mark.parametrize("num_reqs", [n for n in REQUEST_COUNTS if n])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("cp", [(1, 0, 1), (2, 1, 4)])
def test_prepare_dflash_inputs(num_reqs, prefill, cp):
    cp_size, cp_rank, cp_interleave = cp
    gen = _gen(num_reqs + 50 * prefill + cp_size)
    batch, inputs, outputs, max_tokens = _dflash_case(num_reqs, gen, prefill, cp_size)
    nqpr, block_size, model_len = 1 + DRAFTS, 16, 1 << 20
    ref = _clone(outputs)
    num_blocks = triton.cdiv(
        int(batch.num_scheduled_tokens.max()) + nqpr, dflash._PREPARE_BLOCK_SIZE
    )
    dflash._prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
        ref.input_ids,
        ref.positions,
        ref.query_start_loc,
        ref.seq_lens,
        ref.query_slots,
        ref.context_positions,
        ref.context_slots,
        ref.sample_indices,
        ref.sample_pos,
        ref.sample_map,
        ref.temperature,
        ref.seeds,
        batch.positions,
        batch.query_start_loc,
        batch.idx_mapping,
        inputs.last_sampled,
        inputs.next_prefill_tokens,
        inputs.num_sampled,
        inputs.num_rejected,
        inputs.temperature,
        inputs.seeds,
        inputs.block_table,
        inputs.block_table.stride(0),
        123,
        block_size,
        nqpr,
        DRAFTS,
        MAX_REQS,
        max_tokens,
        model_len,
        cp_rank,
        SAMPLE_FROM_ANCHOR=False,
        PAD_SLOT_ID=PAD_SLOT_ID,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        BLOCK_SIZE=dflash._PREPARE_BLOCK_SIZE,
    )
    buffers = SimpleNamespace(
        input_ids=outputs.input_ids,
        positions=outputs.positions,
        query_start_loc=outputs.query_start_loc,
        seq_lens=outputs.seq_lens,
    )
    dflash.prepare_dflash_inputs(
        buffers,
        outputs.query_slots,
        outputs.context_positions,
        outputs.context_slots,
        outputs.sample_indices,
        outputs.sample_pos,
        outputs.sample_map,
        outputs.temperature,
        outputs.seeds,
        batch,
        inputs.num_sampled,
        inputs.num_rejected,
        inputs.last_sampled,
        inputs.next_prefill_tokens,
        inputs.temperature,
        inputs.seeds,
        inputs.block_table,
        block_size,
        cp_rank,
        cp_size,
        cp_interleave,
        123,
        nqpr,
        DRAFTS,
        MAX_REQS,
        max_tokens,
        model_len,
        False,
    )
    torch.accelerator.synchronize()
    for name in vars(outputs):
        assert torch.equal(
            getattr(ref, name).view(torch.uint8),
            getattr(outputs, name).view(torch.uint8),
        ), name


# --- Engram lookback (DeepSeek V4.1) -----------------------------------------


@pytest.mark.parametrize("num_reqs", REQUEST_COUNTS)
@pytest.mark.parametrize("depth", [2, 3, 8])
def test_gather_lookback(num_reqs, depth):
    gen = _gen(num_reqs * 9 + depth)
    idx = _idx(num_reqs, gen)
    computed = _randint(0, 12, (MAX_REQS,), gen)  # windows before the history
    history = _history(gen, 64)
    window = _garbage((MAX_REQS, depth), torch.int32, gen)
    ref = window.clone()
    _gather_lookback_kernel[(MAX_REQS,)](
        ref,
        idx,
        computed,
        history,
        history.stride(0),
        num_reqs,
        DEPTH=depth,
        BLOCK_DEPTH=triton.next_power_of_2(depth),
    )
    lookback.gather_lookback(window, idx, computed, history)
    torch.accelerator.synchronize()
    assert torch.equal(ref, window)


# --- Preparation --------------------------------------------------------------


def test_step_glue_declarations_cover_their_launches(monkeypatch):
    """The step's glue kernels with the runner's dtypes (int64 request indices,
    int64 drafts, float32 confidences, int64 logits indices and a dummy batch's
    int32 ones) launch exactly the calls the runner declares for them."""
    launched = set()
    original = tli._kernel

    def record(factory, *key):
        launched.add((id(factory), key))
        return original(factory, *key)

    monkeypatch.setattr(tli, "_kernel", record)
    gen = _gen(1)
    idx = _idx(4, gen)
    drafts = _randint(0, 1 << 17, (MAX_REQS, DRAFTS), gen, torch.int64)
    state = torch.zeros_like(drafts)
    tli.scatter_rows(idx, drafts[:4], state, gather=True)
    confidences = torch.rand(MAX_REQS, DRAFTS, generator=gen, device=DEVICE)
    tli.scatter_rows(idx, confidences[:4], torch.zeros_like(confidences))
    capacities = _randint(0, DRAFTS + 1, (4,), gen)
    starts = torch.zeros(MAX_REQS + 1, dtype=torch.int32, device=DEVICE)
    tli.adaptive_offsets(
        capacities, _randint(1, 2, (4,), gen), starts.clone(), starts, 1, 24, True
    )
    tli.padding_rows(torch.zeros(32, dtype=torch.bool, device=DEVICE), 24, 32)
    ids = _randint(0, 1 << 17, (64,), gen)
    positions = _randint(0, 1 << 20, (64,), gen, torch.int64)
    for index_dtype in (torch.int64, torch.int32):
        tli.logits_rows(ids, positions, _randint(0, 64, (24,), gen, index_dtype))
    torch.accelerator.synchronize()
    declared = (
        tli.scatter_rows_calls(torch.int64, torch.int64, gather=True)
        + tli.scatter_rows_calls(torch.int64, torch.float32, gather=False)
        + tli.adaptive_offsets_calls()
        + tli.padding_rows_calls()
        + tli.logits_rows_calls(torch.int64)
        + tli.logits_rows_calls(torch.int32)
    )
    assert launched == {(id(function), args) for function, args, _ in declared}


def test_declared_calls_cover_the_call_sites(monkeypatch):
    """Every kernel the runner's call sites launch, with the runner's buffers and
    dtypes, is a call its declarations list (after the kernel cache freezes, an
    undeclared one raises KernelNotPreparedError)."""
    from vllm.v1.worker.gpu import input_batch

    launched = set()
    original = tli._kernel

    def record(factory, *key):
        launched.add((id(factory), key))
        return original(factory, *key)

    monkeypatch.setattr(tli, "_kernel", record)
    gen = _gen(0)

    # Block tables and their staged writes (several groups: the fused writer).
    tables = _block_tables(gen)
    for t in tables.block_tables:
        _stage(t, gen, 2, long=False)
    tables.apply_staged_writes()
    idx = _idx(4, gen)
    starts = _starts([1 + DRAFTS] * 4)
    positions = torch.arange(MAX_TOKENS, device=DEVICE, dtype=torch.int64)
    tables.gather_block_tables(idx, 8)
    tables.compute_slot_mappings(idx, starts, positions, MAX_TOKENS)
    for dtype in (torch.int32, torch.int64, torch.float32):
        tensor = StagedWriteTensor((MAX_REQS, 8), dtype, DEVICE)
        _stage(tensor, gen, 2, long=False)
        tensor.apply_write()

    # The step's inputs, as the runner prepares and commits a decode step.
    history = _history(gen, 256)
    computed = _randint(0, 100, (MAX_REQS,), gen)
    next_tokens = torch.zeros((1, MAX_REQS), dtype=torch.int32, device=DEVICE)
    ids = torch.zeros(MAX_TOKENS, dtype=torch.int32, device=DEVICE)
    input_batch.prepare_prefill_inputs(
        ids, next_tokens, idx, starts, history, computed + 50, computed
    )
    seq_lens = torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE)
    input_batch.prepare_pos_seq_lens(idx, starts, computed, positions, seq_lens)
    last = torch.zeros((MAX_REQS, 1), dtype=torch.int64, device=DEVICE)
    drafts = torch.zeros((MAX_REQS, DRAFTS), dtype=torch.int64, device=DEVICE)
    cu = starts.clone()
    padding = torch.zeros(MAX_TOKENS, dtype=torch.bool, device=DEVICE)
    dead = input_batch.DeadRows(
        torch.ones(MAX_REQS, DRAFTS, device=DEVICE),
        padding,
        torch.zeros(4, dtype=torch.int32, device=DEVICE),
        math.log(0.2),
    )
    for dead_rows in (None, dead):
        input_batch.combine_sampled_and_draft_tokens(
            ids, idx, last, starts, seq_lens[:4], computed, drafts, cu,
            4 * (1 + DRAFTS), 1, dead_rows,
        )  # fmt: skip
    input_batch.expand_idx_mapping(idx, 4 * (1 + DRAFTS), cu, 1 + DRAFTS)
    bins = torch.zeros((MAX_REQS, 64), dtype=torch.int32, device=DEVICE)
    total = _randint(0, 100, (MAX_REQS,), gen)
    for width in (1, 1 + DRAFTS):
        input_batch.post_update(
            idx, computed, last, bins,
            torch.zeros((4, width), dtype=torch.int64, device=DEVICE),
            torch.ones(4, dtype=torch.int32, device=DEVICE),
            torch.zeros(4, dtype=torch.int32, device=DEVICE),
            starts, history, total,
        )  # fmt: skip

    # KV block zeroing.
    storage = torch.ones(64, dtype=torch.int32, device=DEVICE)
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = DEVICE
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=DEVICE),
        torch.tensor([16], device=DEVICE),
        torch.tensor([16], device=DEVICE),
        1,
        16,
        1,
    )
    zeroer.zero_block_ids([1])

    # DFlash drafting.
    batch, inputs, outputs, max_tokens = _dflash_case(4, gen, False, 1)
    buffers = SimpleNamespace(
        input_ids=outputs.input_ids,
        positions=outputs.positions,
        query_start_loc=outputs.query_start_loc,
        seq_lens=outputs.seq_lens,
    )
    dflash.prepare_dflash_inputs(
        buffers, outputs.query_slots, outputs.context_positions,
        outputs.context_slots, outputs.sample_indices, outputs.sample_pos,
        outputs.sample_map, outputs.temperature, outputs.seeds, batch,
        inputs.num_sampled, inputs.num_rejected, inputs.last_sampled,
        inputs.next_prefill_tokens, inputs.temperature, inputs.seeds,
        inputs.block_table, 16, 0, 1, 1, 123, 1 + DRAFTS, DRAFTS, MAX_REQS,
        max_tokens, 1 << 20, False,
    )  # fmt: skip
    torch.accelerator.synchronize()

    declared = (
        tli.block_table_calls(cp_size=1, cp_interleave=1)
        + tli.staged_write_calls()
        + tli.zero_kv_blocks_calls()
        + tli.input_batch_calls(
            num_lookahead=1,
            num_speculative_steps=DRAFTS,
            num_new_sampled_tokens=1,
            dead_rows=True,
            sampled_widths=(1, 1 + DRAFTS),
            commits=((True, True, False),),
        )
        + tli.dflash_calls(False, [(1, 1)])
    )
    declared_keys = {(id(function), args) for function, args, kwargs in declared}
    assert all(not kwargs for _, _, kwargs in declared)
    assert launched == declared_keys

    # The Engram lookback: the model state declares the key the gather uses.
    window = torch.zeros((MAX_REQS, 3), dtype=torch.int32, device=DEVICE)
    used = []
    bound = lookback._bound
    monkeypatch.setattr(
        lookback, "_bound", lambda *key: used.append(key) or bound(*key)
    )
    lookback.gather_lookback(window, idx, computed, history)
    assert used == [lookback.gather_lookback_key(3, tli.RUNNER_IDX_DTYPE)]
