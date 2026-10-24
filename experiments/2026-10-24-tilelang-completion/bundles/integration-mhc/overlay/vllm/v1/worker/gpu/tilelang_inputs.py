# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang kernels of the V2 model runner's input preparation.

Integer and index bookkeeping launched every step, each the contract of the Triton
kernel it replaces, element for element, including which elements a launch writes
and which it leaves alone (the Triton kernels stay importable as the tests'
reference):

- ``gather_block_tables``: ``block_table._gather_block_tables_kernel``
- ``compute_slot_mappings``: ``block_table._compute_slot_mappings_kernel``
- ``apply_staged_writes``: ``buffer_utils._apply_write_kernel``
- ``zero_kv_blocks``: ``worker.utils._zero_kv_blocks_kernel``
- ``prepare_prefill_inputs``: ``input_batch._prepare_prefill_inputs_kernel``
- ``prepare_pos_seq_lens``: ``input_batch._prepare_pos_seq_lens_kernel``
- ``combine_sampled_and_draft_tokens``:
  ``input_batch._combine_sampled_and_draft_tokens_kernel``
- ``expand_idx_mapping``: ``input_batch._expand_idx_mapping_kernel``
- ``post_update``: ``input_batch._post_update_kernel``
- ``prepare_dflash_inputs``: ``dflash.speculator._prepare_dflash_inputs_kernel``

Integer arithmetic follows Triton's: C division and remainder (``truncdiv``,
``truncmod``), int32 products where Triton multiplies int32 values, and stores that
truncate to the destination type. Every kernel has a device tensor argument, so it
launches on the caller's stream (graph capture included).

Preparation: every launch is ``_kernel(factory, *key)`` with ``key`` from the same
key function the owner's declaration uses (``*_calls``), so the declared call and the
launched call cannot drift apart. Owners declare their calls through
``prepare.declare`` (DeepSeek V4.1's TileLang preparation, the runner's
``JitWarmupRegistry`` owner ``TILELANG``) where the Triton kernels registered their
warmups. After the kernel cache is frozen, a call that was not declared raises
``KernelNotPreparedError`` at its first launch.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
from collections.abc import Callable, Sequence
from typing import Any

import torch

from vllm.tilelang_utils import T, tilelang
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.core.boundary_checkpoint import (
    INSTRUCTION_CHECKPOINT_SLOT,
    MAX_BOUNDARY_STOP_TOKENS,
    NUM_BOUNDARY_CHECKPOINT_SLOTS,
    PREFILL_TAIL_CHECKPOINT_SLOT,
    PROMPT_CHECKPOINT_SLOT,
    RESPONSE_CHECKPOINT_SLOT,
)

_PASS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    # Every access is guarded by the kernel's own masks, as in Triton; buffers
    # reached through pointer tables have no meaningful extent to guard against.
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    # A name bound to a load stays one load: with replay on (TileLang's default),
    # IfStmtBinding re-substitutes binds under each guarded statement, and the
    # combine kernel re-read its request's scalars after every store (47 loads where
    # Triton's has 9).
    "tl.if_stmt_binding_inline_replayable_binds": False,
}
_THREADS = 256
_PER_THREAD = 4
_CHUNK = _THREADS * _PER_THREAD  # elements a block covers in one pass
_REQUESTS = 32  # requests a block handles, a thread each
_SPAN = 1 << 40  # extent of a buffer reached through a pointer table, elements
_PAD = PAD_SLOT_ID

_DTYPES = {
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float32: "float32",
}

_KERNELS: dict[tuple[int, tuple[Any, ...]], Any] = {}


def _kernel(factory: Any, *key: Any) -> Any:
    """``factory(*key)``, resolved once: before the kernel cache freezes it
    compiles on a miss; after, an undeclared key raises by name."""
    kernel = _KERNELS.get((id(factory), key))
    if kernel is None:
        kernel = _KERNELS[(id(factory), key)] = factory(*key)
    return kernel


def _next_power_of_2(value: int) -> int:
    return 1 << max(value - 1, 0).bit_length()


def _ceil(value: int, unit: int) -> int:
    return -(-value // unit)


def _as(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """An input in the kernel's element type: a conversion is exact where Triton's
    load converts (widening) or its store truncates (narrowing)."""
    return tensor if tensor.dtype == dtype else tensor.to(dtype)


def _local_slot(position, block_id, block_size, cp_rank, cp_size, cp_interleave):
    """``cp_utils.cp_local_slot``: the rank-local KV slot of ``position`` in block
    ``block_id`` (int64), or PAD where another context-parallel rank owns it."""
    offset = T.truncmod(position, block_size * cp_size)
    if cp_size == 1:
        return block_id * block_size + offset
    mine = T.truncmod(T.truncdiv(offset, cp_interleave), cp_size) == cp_rank
    rounds = T.truncdiv(offset, cp_interleave * cp_size)
    local = rounds * cp_interleave + T.truncmod(offset, cp_interleave)
    return T.if_then_else(mine, block_id * block_size + local, T.int64(_PAD))


# --- Block tables -------------------------------------------------------------


@tilelang.jit(pass_configs=_PASS)
def gather_block_tables_kernel(idx_dtype: str):
    """Each batch row of every KV cache group's block table: the request's row
    copied up to its block count and zeroed to the row's end; rows from the batch's
    requests up to ``rows`` (the padded count) are zeroed. A block covers 1024
    columns of one row of one group: ``chunk_groups`` and ``chunk_firsts`` list
    each group's 1024-column chunks (``gather_chunks``), so no block is launched
    past a narrow group's end. Groups whose width is a multiple of four and whose
    tables are 16-byte aligned move four columns a thread in 16-byte loads and
    stores; the others a column at a time, block-strided."""
    num_reqs = T.dynamic("num_reqs")
    groups = T.dynamic("groups")
    counts_reqs = T.dynamic("counts_reqs")
    counts_stride = T.dynamic("counts_stride")
    chunk_count = T.dynamic("chunk_count")

    @T.prim_func
    def gather_block_tables(
        idx_mapping: T.Tensor((num_reqs,), idx_dtype),
        src_ptrs: T.Tensor((groups,), T.ptr),
        dst_ptrs: T.Tensor((groups,), T.ptr),
        widths: T.Tensor((groups,), T.int64),
        num_blocks: T.StridedTensor((groups, counts_reqs), (counts_stride, 1), T.int32),
        chunk_groups: T.Tensor((chunk_count,), T.int32),
        chunk_firsts: T.Tensor((chunk_count,), T.int64),
        rows: T.int32,
    ):
        with T.Kernel(chunk_count, rows, threads=_THREADS) as (c, row):
            tx = T.get_thread_binding()
            g = chunk_groups[c]
            first = chunk_firsts[c]
            src = T.make_tensor(src_ptrs[g], (_SPAN,), T.int32)
            dst = T.make_tensor(dst_ptrs[g], (_SPAN,), T.int32)
            width = widths[g]
            live = row < num_reqs
            req = T.if_then_else(live, T.cast(idx_mapping[row], T.int64), T.int64(0))
            count = T.if_then_else(
                live, T.cast(num_blocks[g, req], T.int64), T.int64(0)
            )
            src_row = req * width
            dst_row = T.cast(row, T.int64) * width
            if (
                T.truncmod(width, 4) == 0
                and T.truncmod(src_ptrs[g], 16) == 0
                and T.truncmod(dst_ptrs[g], 16) == 0
            ):
                col = first + tx * _PER_THREAD
                if col < width:
                    if col + _PER_THREAD <= count or col >= count:
                        # Live columns, or zeros: a predicated-off load reads
                        # zero bits and touches no memory.
                        piece = T.ldg128(
                            src[src_row + col], pred=col + _PER_THREAD <= count
                        )
                        T.stg128(dst[dst_row + col], piece)
                    else:
                        for k in T.unroll(_PER_THREAD):
                            dst[dst_row + col + k] = T.if_then_else(
                                col + k < count, src[src_row + col + k], 0
                            )
            else:
                for k in T.unroll(_PER_THREAD):
                    col = first + k * _THREADS + tx
                    if col < width:
                        dst[dst_row + col] = T.if_then_else(
                            col < count, src[src_row + col], 0
                        )

    return gather_block_tables


@tilelang.jit(pass_configs=_PASS)
def compute_slot_mappings_kernel(cp_size: int, cp_interleave: int, idx_dtype: str):
    """Each group's slot for every token of the step, and PAD from the step's
    token count to the end of the row; columns before the step's first token are
    left alone. A block covers 1024 columns of one group's row, four a thread; a
    token finds its request by binary search of the query starts (the same small
    array for the whole block), so a long prefill chunk spreads over the row's
    blocks instead of one program."""
    num_reqs = T.dynamic("num_reqs")
    starts_len = T.dynamic("starts_len")
    positions_len = T.dynamic("positions_len")
    groups = T.dynamic("groups")
    counts_reqs = T.dynamic("counts_reqs")
    counts_stride = T.dynamic("counts_stride")
    out_rows = T.dynamic("out_rows")
    cols = T.dynamic("cols")
    out_stride = T.dynamic("out_stride")

    @T.prim_func
    def compute_slot_mappings(
        idx_mapping: T.Tensor((num_reqs,), idx_dtype),
        query_start_loc: T.Tensor((starts_len,), T.int32),
        positions: T.Tensor((positions_len,), T.int64),
        table_ptrs: T.Tensor((groups,), T.ptr),
        table_strides: T.Tensor((groups,), T.int64),
        num_blocks: T.StridedTensor((groups, counts_reqs), (counts_stride, 1), T.int32),
        block_sizes: T.Tensor((groups,), T.int32),
        kernel_block_sizes: T.Tensor((groups,), T.int32),
        mapping_enabled: T.Tensor((groups,), T.bool),
        mapping_circular: T.Tensor((groups,), T.bool),
        group_cp_sizes: T.Tensor((groups,), T.int32),
        slot_mappings: T.StridedTensor((out_rows, cols), (out_stride, 1), T.int64),
        cp_rank: T.int32,
    ):
        with T.Kernel(T.ceildiv(cols, _CHUNK), groups, threads=_THREADS) as (cx, g):
            tx = T.get_thread_binding()
            lo = T.alloc_var(T.int32)
            hi = T.alloc_var(T.int32)
            first_token = query_start_loc[0]
            # From the step's token count, not the padded count: the gap can hold
            # stale slots of an earlier chunk (CUDA graphs read it).
            end_token = query_start_loc[num_reqs]
            table = T.make_tensor(table_ptrs[g], (_SPAN,), T.int32)
            stride = table_strides[g]
            kernel_block = kernel_block_sizes[g]
            enabled = mapping_enabled[g]
            paged = enabled and not mapping_circular[g]
            for k in T.unroll(_PER_THREAD):
                t = cx * _CHUNK + k * _THREADS + tx
                if t < cols:
                    if t >= end_token:
                        slot_mappings[g, t] = T.int64(_PAD)
                    elif t >= first_token:
                        # The request whose rows hold t: the first whose end is
                        # past it.
                        lo = 0
                        hi = num_reqs
                        while lo < hi:
                            mid = (lo + hi) >> 1
                            if query_start_loc[mid + 1] <= t:
                                lo = mid + 1
                            else:
                                hi = mid
                        req = T.cast(idx_mapping[lo], T.int64)
                        count = num_blocks[g, req]
                        row = req * stride
                        position = positions[t]
                        if cp_size == 1:
                            index = T.if_then_else(
                                paged, T.truncdiv(position, kernel_block), T.int64(0)
                            )
                            valid = index < count
                            number = T.if_then_else(valid, table[row + index], 0)
                            slot = T.cast(number * kernel_block, T.int64) + (
                                T.truncmod(position, kernel_block)
                            )
                        else:
                            kv_block = block_sizes[g]
                            single = group_cp_sizes[g] == 1
                            virtual = kv_block * cp_size
                            offset = T.truncmod(position, virtual)
                            mine = (
                                T.truncmod(T.truncdiv(offset, cp_interleave), cp_size)
                                == cp_rank
                            )
                            local = T.if_then_else(
                                single,
                                position,
                                T.truncdiv(position, virtual) * kv_block
                                + T.truncdiv(offset, cp_interleave * cp_size)
                                * cp_interleave
                                + T.truncmod(offset, cp_interleave),
                            )
                            is_local = single or mine
                            index = T.if_then_else(
                                paged, T.truncdiv(local, kernel_block), T.int64(0)
                            )
                            valid = index < count
                            number = T.if_then_else(
                                is_local and valid, table[row + index], 0
                            )
                            slot = T.if_then_else(
                                is_local,
                                T.cast(number * kernel_block, T.int64)
                                + T.truncmod(local, kernel_block),
                                T.int64(_PAD),
                            )
                        slot_mappings[g, t] = T.if_then_else(
                            enabled and valid, slot, T.int64(_PAD)
                        )

    return compute_slot_mappings


def gather_block_tables_key(idx_dtype: str) -> tuple:
    return (idx_dtype,)


def compute_slot_mappings_key(
    cp_size: int, cp_interleave: int, idx_dtype: str
) -> tuple:
    # Without context parallelism the interleave is dead code: one kernel.
    return (cp_size, cp_interleave if cp_size > 1 else 1, idx_dtype)


def gather_chunks(
    widths: Sequence[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """The group and first column of each 1024-column chunk of the groups' rows,
    for ``gather_block_tables``."""
    groups = [g for g, width in enumerate(widths) for _ in range(0, width, _CHUNK)]
    firsts = [first for width in widths for first in range(0, width, _CHUNK)]
    return (
        torch.tensor(groups, dtype=torch.int32, device=device),
        torch.tensor(firsts, dtype=torch.int64, device=device),
    )


def gather_block_tables(
    idx_mapping: torch.Tensor,
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    widths: torch.Tensor,
    num_blocks: torch.Tensor,
    chunks: tuple[torch.Tensor, torch.Tensor],
    rows: int,
) -> None:
    """``BlockTables.gather_block_tables``' launch: pointer tables as int64, and
    ``chunks`` from ``gather_chunks`` of the groups' widths."""
    chunk_groups, chunk_firsts = chunks
    if rows == 0 or chunk_groups.shape[0] == 0:
        return
    key = gather_block_tables_key(_DTYPES[idx_mapping.dtype])
    _kernel(gather_block_tables_kernel, *key)(
        idx_mapping, src_ptrs, dst_ptrs, widths, num_blocks, chunk_groups,
        chunk_firsts, rows,
    )  # fmt: skip


def compute_slot_mappings(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    table_ptrs: torch.Tensor,
    table_strides: torch.Tensor,
    num_blocks: torch.Tensor,
    block_sizes: torch.Tensor,
    kernel_block_sizes: torch.Tensor,
    mapping_enabled: torch.Tensor,
    mapping_circular: torch.Tensor,
    group_cp_sizes: torch.Tensor,
    slot_mappings: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    cp_interleave: int,
) -> None:
    """``BlockTables.compute_slot_mappings``' launch: pointer table as int64."""
    if table_ptrs.shape[0] == 0 or slot_mappings.shape[1] == 0:
        return
    key = compute_slot_mappings_key(cp_size, cp_interleave, _DTYPES[idx_mapping.dtype])
    _kernel(compute_slot_mappings_kernel, *key)(
        idx_mapping,
        _as(query_start_loc, torch.int32),
        _as(positions, torch.int64),
        table_ptrs,
        table_strides,
        num_blocks,
        block_sizes,
        kernel_block_sizes,
        mapping_enabled,
        mapping_circular,
        group_cp_sizes,
        slot_mappings,
        cp_rank,
    )


# --- Staged writes ------------------------------------------------------------


@tilelang.jit(pass_configs=_PASS)
def apply_write_kernel(dtype: str, multi_group: bool):
    """Each staged write's contents into its row from its start column. A block
    covers 1024 elements of one write; ``chunks`` blocks cover the longest. One
    tensor's writes (``multi_group=False``), or int32 writes to the tensors of a
    pointer table, each write to its group's (``FusedStagedWriter``)."""
    writes = T.dynamic("writes")
    starts_len = T.dynamic("starts_len")
    contents_len = T.dynamic("contents_len")
    cu_len = T.dynamic("cu_len")

    if multi_group:
        groups = T.dynamic("groups")
        group_len = T.dynamic("group_len")

        @T.prim_func
        def apply_write(
            output_ptrs: T.Tensor((groups,), T.ptr),
            output_strides: T.Tensor((groups,), T.int64),
            indices: T.Tensor((writes,), T.int32),
            starts: T.Tensor((starts_len,), T.int32),
            contents: T.Tensor((contents_len,), T.int32),
            cu_lens: T.Tensor((cu_len,), T.int32),
            group_ids: T.Tensor((group_len,), T.int32),
            chunks: T.int32,
        ):
            with T.Kernel(writes, chunks, threads=_THREADS) as (w, cx):
                tx = T.get_thread_binding()
                begin = T.if_then_else(w > 0, cu_lens[w - 1], 0)
                length = cu_lens[w] - begin
                group = group_ids[w]
                output = T.make_tensor(output_ptrs[group], (_SPAN,), T.int32)
                base = T.cast(indices[w], T.int64) * output_strides[group] + starts[w]
                for k in T.unroll(_PER_THREAD):
                    j = cx * _CHUNK + k * _THREADS + tx
                    if j < length:
                        output[base + j] = contents[begin + j]

        return apply_write

    out_len = T.dynamic("out_len")

    @T.prim_func
    def apply_write(
        output: T.Tensor((out_len,), dtype),
        output_stride: T.int64,
        indices: T.Tensor((writes,), T.int32),
        starts: T.Tensor((starts_len,), T.int32),
        contents: T.Tensor((contents_len,), dtype),
        cu_lens: T.Tensor((cu_len,), T.int32),
        chunks: T.int32,
    ):
        with T.Kernel(writes, chunks, threads=_THREADS) as (w, cx):
            tx = T.get_thread_binding()
            begin = T.if_then_else(w > 0, cu_lens[w - 1], 0)
            length = cu_lens[w] - begin
            base = T.cast(indices[w], T.int64) * output_stride + starts[w]
            for k in T.unroll(_PER_THREAD):
                j = cx * _CHUNK + k * _THREADS + tx
                if j < length:
                    output[base + j] = contents[begin + j]

    return apply_write


def _longest(cu_lens: Sequence[int]) -> int:
    longest, previous = 0, 0
    for end in cu_lens:
        longest = max(longest, end - previous)
        previous = end
    return longest


def apply_staged_writes(
    output: torch.Tensor,
    output_stride: int,
    indices: torch.Tensor,
    starts: torch.Tensor,
    contents: torch.Tensor,
    cu_lens: torch.Tensor,
    host_cu_lens: Sequence[int],
) -> None:
    """``StagedWriteTensor.apply_write``'s launch; ``output`` is the tensor's flat
    view and ``host_cu_lens`` the host copy of ``cu_lens``."""
    chunks = _ceil(_longest(host_cu_lens), _CHUNK)
    if indices.shape[0] == 0 or chunks == 0:
        return
    _kernel(apply_write_kernel, _DTYPES[output.dtype], False)(
        output, output_stride, indices, starts, contents, cu_lens, chunks
    )


def apply_fused_staged_writes(
    output_ptrs: torch.Tensor,
    output_strides: torch.Tensor,
    indices: torch.Tensor,
    starts: torch.Tensor,
    contents: torch.Tensor,
    cu_lens: torch.Tensor,
    group_ids: torch.Tensor,
    host_cu_lens: Sequence[int],
) -> None:
    """``FusedStagedWriter.apply``'s launch: int32 writes through an int64 pointer
    table."""
    chunks = _ceil(_longest(host_cu_lens), _CHUNK)
    if indices.shape[0] == 0 or chunks == 0:
        return
    _kernel(apply_write_kernel, "int32", True)(
        output_ptrs, output_strides, indices, starts, contents, cu_lens, group_ids,
        chunks,
    )  # fmt: skip


# --- KV block zeroing ---------------------------------------------------------


_ZERO_THREADS = 128
_ZERO_PER_THREAD = 8
_ZERO_CHUNK = _ZERO_THREADS * _ZERO_PER_THREAD  # Triton's BLOCK_SIZE cap


@tilelang.jit(pass_configs=_PASS)
def zero_kv_blocks_kernel():
    """Zero every segment's page of each block: a block covers 1024 int32s of one
    segment's page of one KV block, in Triton's form. Only the page size is read
    before the early exit; the block id, stride and address are read together
    after it, and every store is an int32 predicated on the page (``_STORE_IF``),
    each thread's eight 128 apart."""
    segments = T.dynamic("segments")
    stride_len = T.dynamic("stride_len")
    page_len = T.dynamic("page_len")
    num_ids = T.dynamic("num_ids")

    @T.prim_func
    def zero_kv_blocks(
        seg_addrs: T.Tensor((segments,), T.ptr),
        seg_block_strides: T.Tensor((stride_len,), T.int64),
        seg_page_sizes: T.Tensor((page_len,), T.int64),
        block_ids: T.Tensor((num_ids,), T.int64),
        chunks: T.int32,
    ):
        # Blocks on x (up to 2^31 - 1), as Triton's grid has them.
        with T.Kernel(num_ids, segments, chunks, threads=_ZERO_THREADS) as (b, s, cx):
            T.import_source(_STORE_IF)
            tx = T.get_thread_binding()
            page = T.alloc_var(T.int64, init=seg_page_sizes[s])
            first = T.cast(cx, T.int64) * _ZERO_CHUNK
            if first < page:
                stride = T.alloc_var(T.int64, init=seg_block_strides[s])
                block_id = T.alloc_var(T.int64, init=block_ids[b])
                segment = T.make_tensor(seg_addrs[s], (_SPAN,), T.int32)
                base = block_id * stride
                for k in T.unroll(_ZERO_PER_THREAD):
                    col = first + k * _ZERO_THREADS + tx
                    T.call_extern(
                        "handle", "vllm_store_if_b32",
                        T.address_of(segment[base + col]), 0, col < page,
                    )  # fmt: skip

    return zero_kv_blocks


def zero_kv_blocks(
    seg_addrs: torch.Tensor,
    seg_block_strides: torch.Tensor,
    seg_page_sizes: torch.Tensor,
    block_ids: torch.Tensor,
    max_page_size: int,
) -> None:
    """``KVBlockZeroer.zero_block_ids``' launch: segment addresses as int64."""
    chunks = _ceil(max_page_size, _ZERO_CHUNK)
    if chunks == 0 or seg_addrs.shape[0] == 0 or block_ids.shape[0] == 0:
        return
    _kernel(zero_kv_blocks_kernel)(
        seg_addrs, seg_block_strides, seg_page_sizes, block_ids, chunks
    )


# --- Step inputs --------------------------------------------------------------


@tilelang.jit(pass_configs=_PASS)
def prepare_prefill_inputs_kernel(lookahead_block: int, idx_dtype: str):
    """A block a request: a prefilling request's chunk of prompt tokens into the
    input ids, and its next ``num_lookahead`` prefill tokens (0 past the prompt)."""
    ids_len = T.dynamic("ids_len")
    lookahead = T.dynamic("lookahead")
    lookahead_cols = T.dynamic("lookahead_cols")
    lookahead_stride = T.dynamic("lookahead_stride")
    num_reqs = T.dynamic("num_reqs")
    starts_len = T.dynamic("starts_len")
    history_rows = T.dynamic("history_rows")
    history_cols = T.dynamic("history_cols")
    history_stride = T.dynamic("history_stride")
    prefill_rows = T.dynamic("prefill_rows")
    counts_len = T.dynamic("counts_len")

    @T.prim_func
    def prepare_prefill_inputs(
        input_ids: T.Tensor((ids_len,), T.int32),
        next_prefill_tokens: T.StridedTensor(
            (lookahead, lookahead_cols), (lookahead_stride, 1), T.int32
        ),
        idx_mapping: T.Tensor((num_reqs,), idx_dtype),
        query_start_loc: T.Tensor((starts_len,), T.int32),
        all_token_ids: T.StridedTensor(
            (history_rows, history_cols), (history_stride, 1), T.int32
        ),
        prefill_len: T.Tensor((prefill_rows,), T.int32),
        num_computed_tokens: T.Tensor((counts_len,), T.int32),
    ):
        with T.Kernel(num_reqs, threads=_THREADS) as b:
            tx = T.get_thread_binding()
            req = T.cast(idx_mapping[b], T.int64)
            prefill = prefill_len[req]
            computed = num_computed_tokens[req]
            if computed < prefill:
                start = query_start_loc[b]
                length = query_start_loc[b + 1] - start
                for it in T.serial(T.ceildiv(length, _CHUNK)):
                    for k in T.unroll(_PER_THREAD):
                        j = it * _CHUNK + k * _THREADS + tx
                        if j < length:
                            input_ids[start + j] = all_token_ids[req, computed + j]
                if tx < lookahead_block and tx < lookahead:
                    position = computed + length + tx
                    next_prefill_tokens[tx, req] = T.if_then_else(
                        position < prefill, all_token_ids[req, position], 0
                    )

    return prepare_prefill_inputs


@tilelang.jit(pass_configs=_PASS)
def prepare_pos_seq_lens_kernel(idx_dtype: str):
    """A block a request: its sequence length and its tokens' positions; the last
    block zeroes the sequence lengths past the batch (full CUDA graphs read them)."""
    positions_len = T.dynamic("positions_len")
    max_reqs = T.dynamic("max_reqs")
    num_reqs = T.dynamic("num_reqs")
    starts_len = T.dynamic("starts_len")
    counts_len = T.dynamic("counts_len")

    @T.prim_func
    def prepare_pos_seq_lens(
        positions: T.Tensor((positions_len,), T.int64),
        seq_lens: T.Tensor((max_reqs,), T.int32),
        idx_mapping: T.Tensor((num_reqs,), idx_dtype),
        query_start_loc: T.Tensor((starts_len,), T.int32),
        num_computed_tokens: T.Tensor((counts_len,), T.int32),
    ):
        with T.Kernel(num_reqs + 1, threads=_THREADS) as b:
            tx = T.get_thread_binding()
            if b == num_reqs:
                for it in T.serial(T.ceildiv(max_reqs - num_reqs, _THREADS)):
                    i = num_reqs + it * _THREADS + tx
                    if i < max_reqs:
                        seq_lens[i] = 0
            else:
                computed = num_computed_tokens[idx_mapping[b]]
                start = query_start_loc[b]
                length = query_start_loc[b + 1] - start
                if tx == 0:
                    seq_lens[b] = computed + length
                for it in T.serial(T.ceildiv(length, _CHUNK)):
                    for k in T.unroll(_PER_THREAD):
                        j = it * _CHUNK + k * _THREADS + tx
                        if j < length:
                            positions[start + j] = T.cast(computed + j, T.int64)

    return prepare_pos_seq_lens


# The dead-row survival takes logf of the drafts' confidences under nvcc's default
# (non-flush-to-zero) flags: the same libdevice logf, instruction for instruction,
# as Triton's PTX for this kernel (sm_121a, Triton 3.5), subnormal inputs scaled
# by 2^23 before the reduction alike. Triton's polynomial FMAs carry .ftz, nvcc's
# do not; no operand or result of those FMAs is subnormal for any float32 input,
# so the bits agree. -ftz=true would flush subnormal confidences to zero (-inf
# where Triton gives a finite log).
@tilelang.jit(pass_configs=_PASS)
def combine_sampled_and_draft_tokens_kernel(
    drafts: int, num_new: int, dead_rows: bool, idx_dtype: str
):
    """A request's logits indices, and for a decoding request its last sampled
    token and its drafts into the input ids; ``drafts`` is the draft tokens' width
    (Triton's BLOCK_SIZE is the next power of two of ``drafts + num_new``). Every
    scalar the request needs is read up front, so the loads issue together; its
    drafts and confidences are predicated loads of the drafts it has (``_LOAD_IF``).
    With dead rows the drafts' survival is the
    running sum of log confidences in Triton's warp-scan order (Kogge-Stone,
    strides 1, 2, 4, ...), the cut counts the drafts whose survival is at least
    ``log_tau``, and rows of drafts past it become padding. Without dead rows a
    thread takes a request; with them a block does (its first thread), which is
    how the scan's chain of logs and sums ran fastest."""
    block = _next_power_of_2(drafts + num_new)
    if dead_rows and block > 32:
        raise ValueError("dead rows scan at most one warp of drafts")
    ids_len = T.dynamic("ids_len")
    num_reqs = T.dynamic("num_reqs")
    sampled_len = T.dynamic("sampled_len")
    starts_len = T.dynamic("starts_len")
    seq_len = T.dynamic("seq_len")
    prefill_rows = T.dynamic("prefill_rows")
    draft_rows = T.dynamic("draft_rows")
    draft_stride = T.dynamic("draft_stride")
    cu_len = T.dynamic("cu_len")
    logits_len = T.dynamic("logits_len")
    conf_rows = T.dynamic("conf_rows")
    conf_cols = T.dynamic("conf_cols")
    conf_stride = T.dynamic("conf_stride")
    padding_len = T.dynamic("padding_len")
    cut_len = T.dynamic("cut_len")
    scan_strides = tuple(1 << i for i in range(max(block - 1, 0).bit_length()))

    @T.macro
    def request(
        b, input_ids, idx_mapping, last_sampled, query_start_loc, seq_lens, prefill_len,
        draft_tokens, cu_num_logits, logits_indices, confidence, is_padding, dead_cut,
        log_tau,
    ):  # fmt: skip
        req = T.alloc_var(T.int64, init=T.cast(idx_mapping[b], T.int64))
        first = T.alloc_var(T.int32, init=cu_num_logits[b])
        end = T.alloc_var(T.int32, init=cu_num_logits[b + 1])
        query_end = T.alloc_var(T.int32, init=query_start_loc[b + 1])
        seq = T.alloc_var(T.int32, init=seq_lens[b])
        prefill = T.alloc_var(T.int32, init=prefill_len[req])
        last = T.alloc_var(T.int64, init=last_sampled[req])
        num_logits = end - first
        num_drafts = num_logits - num_new
        # Only the drafts there are: a prefill chunk's request waits on none.
        tokens = T.alloc_local((drafts,), T.int64)
        for j in T.unroll(drafts):
            tokens[j] = T.call_extern(
                "int64", "vllm_load_if_b64", T.address_of(draft_tokens[req, j]),
                T.int64(0), j < num_drafts,
            )  # fmt: skip
        if dead_rows:
            confs = T.alloc_local((drafts,), T.float32)
            for j in T.unroll(drafts):
                confs[j] = T.call_extern(
                    "float32", "vllm_load_if_f32", T.address_of(confidence[req, j]),
                    T.float32(1.0), j < num_drafts,
                )  # fmt: skip
        for j in T.unroll(block):
            if j < num_logits:
                logits_indices[first + j] = T.cast(query_end - num_logits + j, T.int64)
        if seq > prefill:
            # Keep prompt-tail slots; only generated-token slots take it.
            if num_new > 0 and seq - num_logits >= prefill:
                input_ids[query_end - num_logits] = T.cast(last, T.int32)
            for j in T.unroll(drafts):
                if j < num_drafts:
                    input_ids[query_end - num_drafts + j] = T.cast(tokens[j], T.int32)
        if dead_rows:
            # Requests without drafts are never clamped.
            cut = T.alloc_var(T.int32, init=block + 1)
            if seq > prefill and num_drafts > 0:
                survival = T.alloc_local((block,), T.float32)
                for j in tuple(range(block)):
                    if j < drafts:
                        survival[j] = T.log(
                            T.if_then_else(j < num_drafts, confs[j], T.float32(1.0))
                        )
                    else:
                        survival[j] = T.log(T.float32(1.0))
                for stride in scan_strides:
                    # Descending, so each lane adds its neighbour's sum of the
                    # previous stage, as the warp's shuffles do.
                    for j in tuple(range(block - 1, stride - 1, -1)):
                        survival[j] = survival[j - stride] + survival[j]
                cut = 0
                for j in tuple(range(block)):
                    if survival[j] >= log_tau and j < num_drafts:
                        cut = cut + 1
                for j in T.unroll(block):
                    if j < num_drafts and j >= cut:
                        is_padding[query_end - num_drafts + j] = T.uint8(1)
            dead_cut[b] = cut

    if not dead_rows:

        @T.prim_func
        def combine_sampled_and_draft_tokens(
            input_ids: T.Tensor((ids_len,), T.int32),
            idx_mapping: T.Tensor((num_reqs,), idx_dtype),
            last_sampled: T.Tensor((sampled_len,), T.int64),
            query_start_loc: T.Tensor((starts_len,), T.int32),
            seq_lens: T.Tensor((seq_len,), T.int32),
            prefill_len: T.Tensor((prefill_rows,), T.int32),
            draft_tokens: T.StridedTensor(
                (draft_rows, drafts), (draft_stride, 1), T.int64
            ),
            cu_num_logits: T.Tensor((cu_len,), T.int32),
            logits_indices: T.Tensor((logits_len,), T.int64),
        ):
            with T.Kernel(T.ceildiv(num_reqs, _REQUESTS), threads=_REQUESTS) as bx:
                T.import_source(_LOAD_IF)
                b = bx * _REQUESTS + T.get_thread_binding()
                if b < num_reqs:
                    request(
                        b, input_ids, idx_mapping, last_sampled, query_start_loc,
                        seq_lens, prefill_len, draft_tokens, cu_num_logits,
                        logits_indices, None, None, None, None,
                    )  # fmt: skip

        return combine_sampled_and_draft_tokens

    @T.prim_func
    def combine_sampled_and_draft_tokens(
        input_ids: T.Tensor((ids_len,), T.int32),
        idx_mapping: T.Tensor((num_reqs,), idx_dtype),
        last_sampled: T.Tensor((sampled_len,), T.int64),
        query_start_loc: T.Tensor((starts_len,), T.int32),
        seq_lens: T.Tensor((seq_len,), T.int32),
        prefill_len: T.Tensor((prefill_rows,), T.int32),
        draft_tokens: T.StridedTensor((draft_rows, drafts), (draft_stride, 1), T.int64),
        cu_num_logits: T.Tensor((cu_len,), T.int32),
        logits_indices: T.Tensor((logits_len,), T.int64),
        confidence: T.StridedTensor(
            (conf_rows, conf_cols), (conf_stride, 1), T.float32
        ),
        is_padding: T.Tensor((padding_len,), T.uint8),
        dead_cut: T.Tensor((cut_len,), T.int32),
        log_tau: T.float32,
    ):
        with T.Kernel(num_reqs, threads=32) as b:
            T.import_source(_LOAD_IF)
            if T.get_thread_binding() == 0:
                request(
                    b, input_ids, idx_mapping, last_sampled, query_start_loc, seq_lens,
                    prefill_len, draft_tokens, cu_num_logits, logits_indices,
                    confidence, is_padding, dead_cut, log_tau,
                )  # fmt: skip

    return combine_sampled_and_draft_tokens


# Triton stores under a mask with predicated instructions. A guarded store written as
# a branch lets ptxas sink the loads that feed it below the branch, behind the loads
# the guard waits for: two round trips to memory where Triton has one. Kernels whose
# loads must issue together store through these helpers instead of branching.
_STORE_IF = r"""
__device__ __forceinline__ void vllm_store_if_b64(void* p, long long v, bool ok) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n"
               " @q st.global.b64 [%0], %1;\n}"
               :: "l"(p), "l"(v), "r"((int)ok) : "memory");
}
__device__ __forceinline__ void vllm_store_if_b32(void* p, int v, bool ok) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n"
               " @q st.global.b32 [%0], %1;\n}"
               :: "l"(p), "r"(v), "r"((int)ok) : "memory");
}
__device__ __forceinline__ void vllm_store_if_f32(void* p, float v, bool ok) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n"
               " @q st.global.f32 [%0], %1;\n}"
               :: "l"(p), "f"(v), "r"((int)ok) : "memory");
}
"""

# Triton's masked loads, likewise: a lane whose mask is off keeps ``other`` and never
# waits on memory.
_LOAD_IF = r"""
__device__ __forceinline__ long long vllm_load_if_b64(const void* p, long long other,
                                                      bool ok) {
  long long v = other;
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n"
               " @q ld.global.b64 %0, [%1];\n}"
               : "+l"(v) : "l"(p), "r"((int)ok));
  return v;
}
__device__ __forceinline__ float vllm_load_if_f32(const void* p, float other, bool ok) {
  float v = other;
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n"
               " @q ld.global.f32 %0, [%1];\n}"
               : "+f"(v) : "l"(p), "r"((int)ok));
  return v;
}
"""

_STORE_BITS = {"int32": 32, "int64": 64}


@tilelang.jit(pass_configs=_PASS)
def expand_idx_mapping_kernel(block: int, idx_dtype: str):
    """A thread an expanded entry: request ``r``'s ``j``-th logit (``j <
    min(num_logits, block)``) takes the request's state index and ``j``. In
    Triton's form: the request's starts and state index are read together, as
    ordinary loads (the inputs are not ``__restrict__``, which would route them
    through the read-only cache path), and the two guarded stores are predicated
    (``_STORE_IF``)."""
    num_reqs = T.dynamic("num_reqs")
    expanded_len = T.dynamic("expanded_len")
    local_len = T.dynamic("local_len")
    cu_len = T.dynamic("cu_len")
    store_state = f"vllm_store_if_b{_STORE_BITS[idx_dtype]}"

    @T.prim_func
    def expand_idx_mapping(
        idx_mapping: T.Tensor((num_reqs,), idx_dtype),
        expanded_idx_mapping: T.Tensor((expanded_len,), idx_dtype),
        expanded_local_pos: T.Tensor((local_len,), T.int32),
        cu_num_logits: T.Tensor((cu_len,), T.int32),
    ):
        with T.Kernel(T.ceildiv(num_reqs * block, 32), threads=32) as bx:
            T.annotate_restrict_buffers(idx_mapping, cu_num_logits)
            T.import_source(_STORE_IF)
            element = bx * 32 + T.get_thread_binding()
            r = element // block
            j = element % block
            if r < num_reqs:
                start = T.alloc_var(T.int32, init=cu_num_logits[r])
                end = T.alloc_var(T.int32, init=cu_num_logits[r + 1])
                state = T.alloc_var(idx_dtype, init=idx_mapping[r])
                ok = j < end - start
                T.call_extern(
                    "handle",
                    store_state,
                    T.address_of(expanded_idx_mapping[start + j]),
                    state,
                    ok,
                )
                T.call_extern(
                    "handle",
                    "vllm_store_if_b32",
                    T.address_of(expanded_local_pos[start + j]),
                    j,
                    ok,
                )

    return expand_idx_mapping


@tilelang.jit(pass_configs=_PASS)
def post_update_kernel(
    width: int,
    bin_counts: bool,
    has_query: bool,
    boundary: bool,
    idx_dtype: str,
    last_dtype: str,
):
    """A thread a request: commit its ``num_sampled`` tokens (``width`` is the
    sampled tokens' column count): the last one, the token history and its length,
    the output-token counts, and the computed-token count less the rejected rows.
    The counts are read for every distinct token before any is written, then each
    gets its multiplicity: Triton's increments one at a time, without their chain of
    dependent loads. With ``boundary`` (request-boundary checkpoints), the commit
    first stops at a request's first stop token or length bound and records the
    checkpoint captures (``boundary_checkpoint.prepare_boundary_capture``)."""
    num_reqs = T.dynamic("num_reqs")
    computed_len = T.dynamic("computed_len")
    last_len = T.dynamic("last_len")
    bin_rows = T.dynamic("bin_rows")
    bin_cols = T.dynamic("bin_cols")
    bin_stride = T.dynamic("bin_stride")
    sampled_rows = T.dynamic("sampled_rows")
    sampled_stride = T.dynamic("sampled_stride")
    num_sampled_len = T.dynamic("num_sampled_len")
    num_rejected_len = T.dynamic("num_rejected_len")
    starts_len = T.dynamic("starts_len")
    history_rows = T.dynamic("history_rows")
    history_cols = T.dynamic("history_cols")
    history_stride = T.dynamic("history_stride")
    total_len_len = T.dynamic("total_len_len")
    slots = NUM_BOUNDARY_CHECKPOINT_SLOTS

    @T.macro
    def commit(
        r, req, count, rejected, computed, total, query_len, num_computed_tokens,
        last_sampled, output_bin_counts, sampled_tokens, all_token_ids, total_len,
    ):  # fmt: skip
        tokens = T.alloc_local((width,), T.int64)
        for i in tuple(range(width)):
            tokens[i] = T.if_then_else(i < count, sampled_tokens[r, i], T.int64(0))
        if count > 0:
            last_sampled[req] = T.cast(sampled_tokens[r, count - 1], last_dtype)
            total_len[req] = total + count
        for i in tuple(range(width)):
            if i < count:
                all_token_ids[req, total + i] = T.cast(tokens[i], T.int32)
        if bin_counts:
            firsts = []
            for i in tuple(range(width)):
                first = i < count
                for j in tuple(range(i)):
                    first = first and tokens[j] != tokens[i]
                firsts.append(first)
            counts = T.alloc_local((width,), T.int32)
            for i in tuple(range(width)):
                counts[i] = T.if_then_else(
                    firsts[i], output_bin_counts[req, tokens[i]], 0
                )
            for i in tuple(range(width)):
                if firsts[i]:
                    occurrences = T.cast(i < count, T.int32)
                    for j in tuple(range(i + 1, width)):
                        occurrences = occurrences + T.cast(
                            j < count and tokens[j] == tokens[i], T.int32
                        )
                    output_bin_counts[req, tokens[i]] = counts[i] + occurrences
        delta = query_len - rejected
        if delta != 0:
            num_computed_tokens[req] = computed + delta

    if not boundary:

        @T.prim_func
        def post_update(
            idx_mapping: T.Tensor((num_reqs,), idx_dtype),
            num_computed_tokens: T.Tensor((computed_len,), T.int32),
            last_sampled: T.Tensor((last_len,), last_dtype),
            output_bin_counts: T.StridedTensor(
                (bin_rows, bin_cols), (bin_stride, 1), T.int32
            ),
            sampled_tokens: T.StridedTensor(
                (sampled_rows, width), (sampled_stride, 1), T.int64
            ),
            num_sampled: T.Tensor((num_sampled_len,), T.int32),
            num_rejected: T.Tensor((num_rejected_len,), T.int32),
            query_start_loc: T.Tensor((starts_len,), T.int32),
            all_token_ids: T.StridedTensor(
                (history_rows, history_cols), (history_stride, 1), T.int32
            ),
            total_len: T.Tensor((total_len_len,), T.int32),
        ):
            with T.Kernel(T.ceildiv(num_reqs, _REQUESTS), threads=_REQUESTS) as bx:
                r = bx * _REQUESTS + T.get_thread_binding()
                if r < num_reqs:
                    req = T.cast(idx_mapping[r], T.int64)
                    # Rows with negative indices are filtered.
                    if req >= 0:
                        count = num_sampled[r]
                        rejected = num_rejected[r]
                        computed = num_computed_tokens[req]
                        total = total_len[req]
                        if has_query:
                            query_len = query_start_loc[r + 1] - query_start_loc[r]
                        else:
                            query_len = 0
                        commit(
                            r, req, count, rejected, computed, total, query_len,
                            num_computed_tokens, last_sampled, output_bin_counts,
                            sampled_tokens, all_token_ids, total_len,
                        )  # fmt: skip

        return post_update

    metadata_rows = T.dynamic("metadata_rows")
    metadata_cols = T.dynamic("metadata_cols")
    metadata_stride = T.dynamic("metadata_stride")
    stop_rows = T.dynamic("stop_rows")
    stop_cols = T.dynamic("stop_cols")
    stop_stride = T.dynamic("stop_stride")
    seen_rows = T.dynamic("seen_rows")
    seen_stride = T.dynamic("seen_stride")
    capture_rows_len = T.dynamic("capture_rows_len")
    capture_stride = T.dynamic("capture_stride")

    @T.prim_func
    def post_update(
        idx_mapping: T.Tensor((num_reqs,), idx_dtype),
        num_computed_tokens: T.Tensor((computed_len,), T.int32),
        last_sampled: T.Tensor((last_len,), last_dtype),
        output_bin_counts: T.StridedTensor(
            (bin_rows, bin_cols), (bin_stride, 1), T.int32
        ),
        sampled_tokens: T.StridedTensor(
            (sampled_rows, width), (sampled_stride, 1), T.int64
        ),
        num_sampled: T.Tensor((num_sampled_len,), T.int32),
        num_rejected: T.Tensor((num_rejected_len,), T.int32),
        query_start_loc: T.Tensor((starts_len,), T.int32),
        all_token_ids: T.StridedTensor(
            (history_rows, history_cols), (history_stride, 1), T.int32
        ),
        total_len: T.Tensor((total_len_len,), T.int32),
        metadata: T.StridedTensor(
            (metadata_rows, metadata_cols), (metadata_stride, 1), T.int32
        ),
        stop_tokens: T.StridedTensor((stop_rows, stop_cols), (stop_stride, 1), T.int32),
        seen: T.StridedTensor((seen_rows, slots), (seen_stride, 1), T.int32),
        capture_tokens: T.StridedTensor(
            (capture_rows_len, slots), (capture_stride, 1), T.int32
        ),
        capture_bias: T.StridedTensor(
            (capture_rows_len, slots), (capture_stride, 1), T.int32
        ),
        capture_rows: T.StridedTensor(
            (capture_rows_len, slots), (capture_stride, 1), T.int32
        ),
    ):
        with T.Kernel(T.ceildiv(num_reqs, _REQUESTS), threads=_REQUESTS) as bx:
            r = bx * _REQUESTS + T.get_thread_binding()
            if r < num_reqs:
                req = T.cast(idx_mapping[r], T.int64)
                if req < 0:
                    for kind in tuple(range(slots)):
                        capture_tokens[r, kind] = 0
                else:
                    computed = num_computed_tokens[req]
                    total = total_len[req]
                    if has_query:
                        query_start = query_start_loc[r]
                        query_len = query_start_loc[r + 1] - query_start
                    else:
                        query_start = 0
                        query_len = 0
                    count = T.alloc_var(T.int32, init=num_sampled[r])
                    rejected = T.alloc_var(T.int32, init=num_rejected[r])
                    prompt = T.alloc_var(T.int32, init=0)
                    response = T.alloc_var(T.int32, init=0)
                    instruction = T.alloc_var(T.int32, init=0)
                    tail = T.alloc_var(T.int32, init=0)
                    if metadata[req, 0] != 0:
                        prompt_len = metadata[req, 1]
                        instruction_len = metadata[req, 2]
                        min_len = metadata[req, 3]
                        max_len = metadata[req, 4]
                        num_stops = T.min(metadata[req, 5], MAX_BOUNDARY_STOP_TOKENS)
                        tail_len = metadata[req, 6]
                        stopped = T.alloc_var(T.int32, init=0)
                        valid = T.alloc_var(T.int32, init=count)
                        for i in T.serial(count):
                            token = sampled_tokens[r, i]
                            length = total + i + 1
                            hit = T.alloc_var(T.int32, init=0)
                            for s in T.serial(num_stops):
                                if T.cast(stop_tokens[req, s], T.int64) == token:
                                    hit = 1
                            if (
                                stopped == 0
                                and length >= min_len
                                and (hit != 0 or length >= max_len)
                            ):
                                valid = i + 1
                                stopped = 1
                        rejected = rejected + count - valid
                        count = valid
                        after = computed + query_len - rejected
                        if (
                            computed < prompt_len
                            and after == prompt_len
                            and seen[req, PROMPT_CHECKPOINT_SLOT] == 0
                        ):
                            prompt = prompt_len
                            seen[req, PROMPT_CHECKPOINT_SLOT] = 1
                        if (
                            instruction_len > 0
                            and computed < instruction_len
                            and after == instruction_len
                            and seen[req, INSTRUCTION_CHECKPOINT_SLOT] == 0
                        ):
                            instruction = instruction_len
                            seen[req, INSTRUCTION_CHECKPOINT_SLOT] = 1
                        if (
                            tail_len > 0
                            and computed < tail_len
                            and after == tail_len
                            and seen[req, PREFILL_TAIL_CHECKPOINT_SLOT] == 0
                        ):
                            tail = tail_len
                            seen[req, PREFILL_TAIL_CHECKPOINT_SLOT] = 1
                        if stopped != 0 and seen[req, RESPONSE_CHECKPOINT_SLOT] == 0:
                            response = after
                            seen[req, RESPONSE_CHECKPOINT_SLOT] = 1
                    capture_tokens[r, PROMPT_CHECKPOINT_SLOT] = prompt
                    capture_tokens[r, RESPONSE_CHECKPOINT_SLOT] = response
                    capture_tokens[r, INSTRUCTION_CHECKPOINT_SLOT] = instruction
                    capture_tokens[r, PREFILL_TAIL_CHECKPOINT_SLOT] = tail
                    capture_bias[r, PROMPT_CHECKPOINT_SLOT] = 0
                    capture_bias[r, RESPONSE_CHECKPOINT_SLOT] = T.max(count - 1, 0)
                    capture_bias[r, INSTRUCTION_CHECKPOINT_SLOT] = 0
                    capture_bias[r, PREFILL_TAIL_CHECKPOINT_SLOT] = 0
                    capture_rows[r, PREFILL_TAIL_CHECKPOINT_SLOT] = (
                        query_start + tail - computed - 1
                    )
                    capture_rows[r, PROMPT_CHECKPOINT_SLOT] = (
                        query_start + prompt - computed - 1
                    )
                    capture_rows[r, RESPONSE_CHECKPOINT_SLOT] = (
                        query_start + response - computed - 1
                    )
                    capture_rows[r, INSTRUCTION_CHECKPOINT_SLOT] = (
                        query_start + instruction - computed - 1
                    )
                    num_sampled[r] = count
                    num_rejected[r] = rejected
                    commit(
                        r, req, count, rejected, computed, total, query_len,
                        num_computed_tokens, last_sampled, output_bin_counts,
                        sampled_tokens, all_token_ids, total_len,
                    )  # fmt: skip

    return post_update


def prepare_prefill_inputs_key(num_lookahead: int, idx_dtype: str) -> tuple:
    block = _next_power_of_2(num_lookahead)
    if block > _THREADS:
        raise ValueError(f"at most {_THREADS} prefill lookahead tokens")
    return (block, idx_dtype)


def prepare_pos_seq_lens_key(idx_dtype: str) -> tuple:
    return (idx_dtype,)


def combine_key(
    num_speculative_steps: int,
    num_new_sampled_tokens: int,
    dead_rows: bool,
    idx_dtype: str,
) -> tuple:
    # The kernel's Triton BLOCK_SIZE covers the last sampled token and every draft.
    return (num_speculative_steps, num_new_sampled_tokens, dead_rows, idx_dtype)


def expand_idx_mapping_key(max_expand_len: int, idx_dtype: str) -> tuple:
    return (_next_power_of_2(max_expand_len), idx_dtype)


def post_update_key(
    width: int,
    bin_counts: bool,
    has_query: bool,
    boundary: bool,
    idx_dtype: str,
    last_dtype: str = "int64",
) -> tuple:
    return (width, bin_counts, has_query, boundary, idx_dtype, last_dtype)


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    if idx_mapping.shape[0] == 0:
        return
    key = prepare_prefill_inputs_key(
        next_prefill_tokens.shape[0], _DTYPES[idx_mapping.dtype]
    )
    _kernel(prepare_prefill_inputs_kernel, *key)(
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        _as(query_start_loc, torch.int32),
        _as(all_token_ids, torch.int32),
        _as(prefill_len, torch.int32),
        _as(num_computed_tokens, torch.int32),
    )


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    key = prepare_pos_seq_lens_key(_DTYPES[idx_mapping.dtype])
    _kernel(prepare_pos_seq_lens_kernel, *key)(
        pos,
        seq_lens,
        idx_mapping,
        _as(query_start_loc, torch.int32),
        _as(num_computed_tokens, torch.int32),
    )


def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    logits_indices: torch.Tensor,
    num_new_sampled_tokens: int,
    dead_rows: Any = None,
) -> None:
    """``dead_rows``: ``input_batch.DeadRows`` or None."""
    if idx_mapping.shape[0] == 0:
        return
    key = combine_key(
        draft_tokens.shape[-1], num_new_sampled_tokens, dead_rows is not None,
        _DTYPES[idx_mapping.dtype],
    )  # fmt: skip
    args = [
        input_ids,
        idx_mapping,
        _as(last_sampled_tokens, torch.int64).view(-1),
        _as(query_start_loc, torch.int32),
        _as(seq_lens, torch.int32),
        _as(prefill_len, torch.int32),
        _as(draft_tokens, torch.int64),
        _as(cu_num_logits, torch.int32),
        logits_indices,
    ]
    if dead_rows is not None:
        args += [
            _as(dead_rows.confidence, torch.float32),
            dead_rows.is_padding.view(torch.uint8),
            dead_rows.cut,
            dead_rows.log_tau,
        ]
    _kernel(combine_sampled_and_draft_tokens_kernel, *key)(*args)


def expand_idx_mapping(
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> None:
    if idx_mapping.shape[0] == 0:
        return
    key = expand_idx_mapping_key(max_expand_len, _DTYPES[idx_mapping.dtype])
    _kernel(expand_idx_mapping_kernel, *key)(
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        _as(cu_num_logits, torch.int32),
    )


def post_update(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor | None,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    boundary_state: Any = None,
    boundary_capture: torch.Tensor | None = None,
) -> None:
    """``input_batch.post_update``'s launch. Absent optional inputs are replaced by
    a tensor of the same kind that the kernel never reads."""
    if idx_mapping.shape[0] == 0:
        return
    if sampled_tokens.dim() == 1:
        sampled_tokens = sampled_tokens.view(-1, 1)
    sampled_tokens = _as(sampled_tokens, torch.int64)
    boundary = boundary_capture is not None
    key = post_update_key(
        sampled_tokens.shape[1], output_bin_counts is not None,
        query_start_loc is not None, boundary, _DTYPES[idx_mapping.dtype],
        _DTYPES[last_sampled_tokens.dtype],
    )  # fmt: skip
    args = [
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens.view(-1),
        output_bin_counts if output_bin_counts is not None else num_sampled.view(-1, 1),
        sampled_tokens,
        num_sampled,
        num_rejected,
        _as(query_start_loc, torch.int32)
        if query_start_loc is not None
        else num_sampled,
        all_token_ids,
        total_len,
    ]
    if boundary:
        assert boundary_state is not None and boundary_capture is not None
        args += [
            boundary_state.metadata,
            boundary_state.stop_tokens,
            boundary_state.seen,
            boundary_capture[0],
            boundary_capture[1],
            boundary_capture[2],
        ]
    _kernel(post_update_kernel, *key)(*args)


# --- DFlash drafting ----------------------------------------------------------

_DFLASH_THREADS = 256  # Triton's BLOCK_SIZE: a thread a context or query row
# Blocks a step (in the last request's row) that pad the query slots past the batch.
_DFLASH_PAD_BLOCKS = 8


@tilelang.jit(pass_configs=_PASS)
def prepare_dflash_inputs_kernel(
    sample_from_anchor: bool, cp_size: int, cp_interleave: int, idx_dtype: str
):
    """``num_blocks`` blocks of 256 rows a request: the accepted context's
    positions and slots (rejected rows PAD), the 1 + N query rows' ids, positions
    and slots, and their sampling rows; the first block of each request its query
    start, sequence length and sampling state, and the first block of the last
    request pads every per-request buffer to the captured sizes."""
    ids_len = T.dynamic("ids_len")
    query_positions_len = T.dynamic("query_positions_len")
    query_starts_len = T.dynamic("query_starts_len")
    seq_lens_len = T.dynamic("seq_lens_len")
    query_slots_len = T.dynamic("query_slots_len")
    context_positions_len = T.dynamic("context_positions_len")
    context_slots_len = T.dynamic("context_slots_len")
    sample_indices_len = T.dynamic("sample_indices_len")
    sample_pos_len = T.dynamic("sample_pos_len")
    sample_map_len = T.dynamic("sample_map_len")
    temperature_out_len = T.dynamic("temperature_out_len")
    seeds_out_len = T.dynamic("seeds_out_len")
    target_positions_len = T.dynamic("target_positions_len")
    target_starts_len = T.dynamic("target_starts_len")
    idx_len = T.dynamic("idx_len")
    last_len = T.dynamic("last_len")
    next_len = T.dynamic("next_len")
    num_sampled_len = T.dynamic("num_sampled_len")
    num_rejected_len = T.dynamic("num_rejected_len")
    temperature_len = T.dynamic("temperature_len")
    seeds_len = T.dynamic("seeds_len")
    table_rows = T.dynamic("table_rows")
    table_cols = T.dynamic("table_cols")
    table_stride = T.dynamic("table_stride")
    sample_offset = 0 if sample_from_anchor else 1

    @T.prim_func
    def prepare_dflash_inputs(
        out_input_ids: T.Tensor((ids_len,), T.int32),
        out_query_positions: T.Tensor((query_positions_len,), T.int64),
        out_query_start_loc: T.Tensor((query_starts_len,), T.int32),
        out_seq_lens: T.Tensor((seq_lens_len,), T.int32),
        out_query_slot_mapping: T.Tensor((query_slots_len,), T.int64),
        out_context_positions: T.Tensor((context_positions_len,), T.int64),
        out_context_slot_mapping: T.Tensor((context_slots_len,), T.int64),
        out_sample_indices: T.Tensor((sample_indices_len,), T.int64),
        out_sample_pos: T.Tensor((sample_pos_len,), T.int64),
        out_sample_idx_mapping: T.Tensor((sample_map_len,), T.int32),
        out_temperature: T.Tensor((temperature_out_len,), T.float32),
        out_seeds: T.Tensor((seeds_out_len,), T.int64),
        target_positions: T.Tensor((target_positions_len,), T.int64),
        target_query_start_loc: T.Tensor((target_starts_len,), T.int32),
        idx_mapping: T.Tensor((idx_len,), idx_dtype),
        last_sampled: T.Tensor((last_len,), T.int64),
        next_prefill_tokens: T.Tensor((next_len,), T.int32),
        num_sampled: T.Tensor((num_sampled_len,), T.int32),
        num_rejected: T.Tensor((num_rejected_len,), T.int32),
        temperature: T.Tensor((temperature_len,), T.float32),
        seeds: T.Tensor((seeds_len,), T.int64),
        block_table: T.StridedTensor(
            (table_rows, table_cols), (table_stride, 1), T.int32
        ),
        parallel_drafting_token_id: T.int32,
        block_size: T.int32,
        num_query_per_req: T.int32,
        num_speculative_steps: T.int32,
        max_num_reqs: T.int32,
        max_num_tokens: T.int32,
        max_model_len: T.int32,
        cp_rank: T.int32,
        num_reqs: T.int32,
        num_blocks: T.int32,
    ):
        with T.Kernel(
            num_blocks + _DFLASH_PAD_BLOCKS, num_reqs, threads=_DFLASH_THREADS
        ) as (bk, r):
            T.import_source(_STORE_IF)
            tx = T.get_thread_binding()
            if bk >= num_blocks:
                # The last request's extra blocks pad the query slots past the
                # batch, a slice each: a captured graph replayed at a larger size
                # writes no KV for them. (One block writing all of them was the
                # kernel's longest path at decode.)
                if r == num_reqs - 1:
                    query_end = num_reqs * num_query_per_req
                    chunk = T.ceildiv(max_num_tokens - query_end, _DFLASH_PAD_BLOCKS)
                    first = query_end + (bk - num_blocks) * chunk
                    for it in T.serial(T.ceildiv(chunk, _DFLASH_THREADS)):
                        i = first + it * _DFLASH_THREADS + tx
                        if (i < first + chunk) and (i < max_num_tokens):
                            out_query_slot_mapping[i] = T.int64(_PAD)
            else:
                # Triton's form: every load the thread needs is issued up front, at
                # an index that exists whatever the masks say; the masks are
                # selects and the stores predicated (_STORE_IF), so no load waits
                # behind a branch for another.
                req = T.alloc_var(T.int64, init=T.cast(idx_mapping[r], T.int64))
                context_start = T.alloc_var(T.int32, init=target_query_start_loc[r])
                context_end = T.alloc_var(T.int32, init=target_query_start_loc[r + 1])
                rejected = T.alloc_var(T.int32, init=num_rejected[r])
                sampled = T.alloc_var(T.int32, init=num_sampled[r])
                j = bk * _DFLASH_THREADS + tx
                rows = context_end - context_start
                valid_end = context_end - rejected
                num_valid = valid_end - context_start
                anchor = T.alloc_var(
                    T.int64,
                    init=target_positions[T.max(valid_end - 1, context_start)],
                )
                first = T.alloc_var(T.int64, init=target_positions[context_start])
                row_position = T.alloc_var(
                    T.int64,
                    init=target_positions[context_start + T.min(j, T.max(rows - 1, 0))],
                )
                last_token = T.alloc_var(T.int64, init=last_sampled[req])
                next_token = T.alloc_var(T.int32, init=next_prefill_tokens[req])
                temp = T.alloc_var(T.float32, init=temperature[req])
                seed = T.alloc_var(T.int64, init=seeds[req])
                # No valid row: every row was rejected, the first one included (sink
                # rejection replaced draft zero). The anchor takes draft zero's
                # position; the context before it is in the cache already.
                last_valid = T.if_then_else(num_valid > 0, anchor, first - 1)
                span = block_size * cp_size
                last_column = T.cast(table_stride - 1, T.int64)
                in_context = j < rows
                live = j < num_valid
                position = T.if_then_else(live, row_position, T.int64(0))
                offset = j - num_valid
                is_query = (j >= num_valid) & (j < num_valid + num_query_per_req)
                query_position = last_valid + 1 + offset
                context_block = T.alloc_var(
                    T.int64,
                    init=T.cast(
                        block_table[r, T.min(T.truncdiv(position, span), last_column)],
                        T.int64,
                    ),
                )
                query_block = T.alloc_var(
                    T.int64,
                    init=T.cast(
                        block_table[
                            r,
                            T.min(
                                T.truncdiv(T.max(query_position, T.int64(0)), span),
                                last_column,
                            ),
                        ],
                        T.int64,
                    ),
                )
                # Block 0 is the null block: evicted sliding-window positions and
                # rejected rows write no draft KV.
                context_slot = T.if_then_else(
                    live and context_block != 0,
                    _local_slot(
                        position, context_block, block_size, cp_rank, cp_size,
                        cp_interleave,
                    ),
                    T.int64(_PAD),
                )  # fmt: skip
                T.call_extern(
                    "handle", "vllm_store_if_b64",
                    T.address_of(out_context_positions[context_start + j]), position,
                    in_context,
                )  # fmt: skip
                T.call_extern(
                    "handle", "vllm_store_if_b64",
                    T.address_of(out_context_slot_mapping[context_start + j]),
                    context_slot, in_context,
                )  # fmt: skip
                query = r * num_query_per_req + offset
                input_id = T.if_then_else(
                    offset == 0,
                    T.if_then_else(
                        sampled > 0, T.cast(last_token, T.int32), next_token
                    ),
                    parallel_drafting_token_id,
                )
                query_slot = T.if_then_else(
                    query_block != 0,
                    _local_slot(
                        query_position, query_block, block_size, cp_rank, cp_size,
                        cp_interleave,
                    ),
                    T.int64(_PAD),
                )  # fmt: skip
                T.call_extern(
                    "handle", "vllm_store_if_b32",
                    T.address_of(out_input_ids[query]), input_id, is_query,
                )  # fmt: skip
                T.call_extern(
                    "handle", "vllm_store_if_b64",
                    T.address_of(out_query_positions[query]),
                    T.min(query_position, T.cast(max_model_len - 1, T.int64)), is_query,
                )  # fmt: skip
                T.call_extern(
                    "handle", "vllm_store_if_b64",
                    T.address_of(out_query_slot_mapping[query]), query_slot, is_query,
                )  # fmt: skip
                is_sample = is_query and offset >= sample_offset
                sample = r * num_speculative_steps + offset - sample_offset
                T.call_extern(
                    "handle", "vllm_store_if_b64",
                    T.address_of(out_sample_indices[sample]), T.cast(query, T.int64),
                    is_sample,
                )  # fmt: skip
                T.call_extern(
                    "handle", "vllm_store_if_b64",
                    T.address_of(out_sample_pos[sample]),
                    query_position + 1 - sample_offset, is_sample,
                )  # fmt: skip
                T.call_extern(
                    "handle", "vllm_store_if_b32",
                    T.address_of(out_sample_idx_mapping[sample]), T.cast(req, T.int32),
                    is_sample,
                )  # fmt: skip
                if bk == 0:
                    lead = tx == 0
                    T.call_extern(
                        "handle", "vllm_store_if_b32",
                        T.address_of(out_query_start_loc[r]), r * num_query_per_req,
                        lead,
                    )  # fmt: skip
                    # The absolute length the draft attention reads up to.
                    T.call_extern(
                        "handle", "vllm_store_if_b32",
                        T.address_of(out_seq_lens[r]),
                        T.cast(
                            T.min(
                                last_valid + 1 + num_query_per_req,
                                T.cast(max_model_len, T.int64),
                            ),
                            T.int32,
                        ),
                        lead,
                    )  # fmt: skip
                    T.call_extern(
                        "handle", "vllm_store_if_f32",
                        T.address_of(out_temperature[req]), temp, lead,
                    )  # fmt: skip
                    T.call_extern(
                        "handle", "vllm_store_if_b64",
                        T.address_of(out_seeds[req]), seed, lead,
                    )  # fmt: skip
                    if r == num_reqs - 1:
                        # Pad the per-request buffers to max_num_reqs for graph replay.
                        last_query_end = num_reqs * num_query_per_req
                        for it in T.serial(
                            T.ceildiv(max_num_reqs + 1 - num_reqs, _DFLASH_THREADS)
                        ):
                            i = num_reqs + it * _DFLASH_THREADS + tx
                            if i < max_num_reqs + 1:
                                out_query_start_loc[i] = last_query_end
                            if i < max_num_reqs:
                                out_seq_lens[i] = 0
                        # Padded sample rows read query row 0 and scatter nowhere.
                        pad_start = num_reqs * num_speculative_steps
                        pad_end = max_num_reqs * num_speculative_steps
                        for it in T.serial(
                            T.ceildiv(pad_end - pad_start, _DFLASH_THREADS)
                        ):
                            i = pad_start + it * _DFLASH_THREADS + tx
                            if i < pad_end:
                                out_sample_indices[i] = T.int64(0)
                                out_sample_pos[i] = T.int64(0)
                                out_sample_idx_mapping[i] = -1
                        # The query slots past the batch: the extra blocks above.

    return prepare_dflash_inputs


def prepare_dflash_inputs_key(
    sample_from_anchor: bool, cp_size: int, cp_interleave: int, idx_dtype: str
) -> tuple:
    return (sample_from_anchor, cp_size, cp_interleave if cp_size > 1 else 1, idx_dtype)


def prepare_dflash_inputs(
    out_input_ids: torch.Tensor,
    out_query_positions: torch.Tensor,
    out_query_start_loc: torch.Tensor,
    out_seq_lens: torch.Tensor,
    out_query_slot_mapping: torch.Tensor,
    out_context_positions: torch.Tensor,
    out_context_slot_mapping: torch.Tensor,
    out_sample_indices: torch.Tensor,
    out_sample_pos: torch.Tensor,
    out_sample_idx_mapping: torch.Tensor,
    out_temperature: torch.Tensor,
    out_seeds: torch.Tensor,
    target_positions: torch.Tensor,
    target_query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    block_table: torch.Tensor,
    parallel_drafting_token_id: int,
    block_size: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    cp_rank: int,
    cp_size: int,
    cp_interleave: int,
    sample_from_anchor: bool,
    num_reqs: int,
    max_rows: int,
) -> None:
    """``dflash.speculator.prepare_dflash_inputs``' launch for ``num_reqs``
    requests of at most ``max_rows`` context and query rows each."""
    num_blocks = _ceil(max_rows, _DFLASH_THREADS)
    if num_reqs == 0 or num_blocks == 0:
        return
    key = prepare_dflash_inputs_key(
        sample_from_anchor, cp_size, cp_interleave, _DTYPES[idx_mapping.dtype]
    )
    _kernel(prepare_dflash_inputs_kernel, *key)(
        out_input_ids,
        out_query_positions,
        out_query_start_loc,
        out_seq_lens,
        out_query_slot_mapping,
        out_context_positions,
        out_context_slot_mapping,
        out_sample_indices,
        out_sample_pos,
        out_sample_idx_mapping,
        out_temperature,
        out_seeds,
        _as(target_positions, torch.int64),
        _as(target_query_start_loc, torch.int32),
        idx_mapping,
        _as(last_sampled, torch.int64).view(-1),
        _as(next_prefill_tokens, torch.int32).view(-1),
        _as(num_sampled, torch.int32),
        _as(num_rejected, torch.int32),
        _as(temperature, torch.float32),
        _as(seeds, torch.int64),
        _as(block_table, torch.int32),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        cp_rank,
        num_reqs,
        num_blocks,
    )


# --- Preparation --------------------------------------------------------------

# The runner's batch-to-request-state index is ``np.intp``.
RUNNER_IDX_DTYPE = "int64"


def declare(calls: Callable[[], list]) -> None:
    """Declare an owner's calls with the runner's JIT registry: its TileLang owner
    compiles them with the deployment's other TileLang calls before memory
    profiling and the kernel cache freezes. Like a Triton dispatcher's
    ``register_warmup``, a no-op where no registry collects (the first launch then
    compiles, the cache not being frozen)."""
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    if JitWarmupRegistry.active() is None:
        return
    from vllm.models.deepseek_v4_1.tilelang import prepare

    prepare.declare(calls)


def _call(factory: Any, *key: Any) -> tuple:
    from vllm.models.deepseek_v4_1.tilelang import prepare

    return prepare.call(factory, *key)


def block_table_calls(cp_size: int, cp_interleave: int) -> list:
    """``BlockTables``' gather and slot mappings for the runner's index."""
    return [
        _call(gather_block_tables_kernel, *gather_block_tables_key(RUNNER_IDX_DTYPE)),
        _call(
            compute_slot_mappings_kernel,
            *compute_slot_mappings_key(cp_size, cp_interleave, RUNNER_IDX_DTYPE),
        ),
    ]


def staged_write_calls(
    dtypes: Sequence[torch.dtype] = tuple(_DTYPES), fused: bool = True
) -> list:
    """``StagedWriteTensor``'s write for each of ``dtypes`` (every element type a
    staged tensor may have, by default) and ``FusedStagedWriter``'s: the key holds
    no shape or stride, so tensors of one element type share a kernel."""
    calls = [_call(apply_write_kernel, _DTYPES[dtype], False) for dtype in dtypes]
    if fused:
        calls.append(_call(apply_write_kernel, "int32", True))
    return calls


def zero_kv_blocks_calls() -> list:
    return [_call(zero_kv_blocks_kernel)]


def input_batch_calls(
    *,
    num_lookahead: int,
    num_speculative_steps: int,
    num_new_sampled_tokens: int,
    dead_rows: bool,
    sampled_widths: Sequence[int],
    commits: Sequence[tuple[bool, bool, bool]],
) -> list:
    """The step-input kernels for the runner's buffers: prefill inputs, positions
    and sequence lengths, the draft combine (with its dead-row variant when
    adaptive verification marks dead rows), the logits index expansion when
    drafting, and the commit of ``sampled_widths``-column sampled tokens for each
    ``(bin_counts, has_query, boundary)`` of ``commits``."""
    calls = [
        _call(
            prepare_prefill_inputs_kernel,
            *prepare_prefill_inputs_key(num_lookahead, RUNNER_IDX_DTYPE),
        ),
        _call(prepare_pos_seq_lens_kernel, *prepare_pos_seq_lens_key(RUNNER_IDX_DTYPE)),
    ]
    for dead in (False, True) if dead_rows else (False,):
        calls.append(
            _call(
                combine_sampled_and_draft_tokens_kernel,
                *combine_key(
                    num_speculative_steps,
                    num_new_sampled_tokens,
                    dead,
                    RUNNER_IDX_DTYPE,
                ),
            )
        )
    if num_speculative_steps > 0:
        calls.append(
            _call(
                expand_idx_mapping_kernel,
                *expand_idx_mapping_key(
                    num_speculative_steps + num_new_sampled_tokens, RUNNER_IDX_DTYPE
                ),
            )
        )
    for width in dict.fromkeys(sampled_widths):
        for bin_counts, has_query, boundary in dict.fromkeys(commits):
            calls.append(
                _call(
                    post_update_kernel,
                    *post_update_key(
                        width, bin_counts, has_query, boundary, RUNNER_IDX_DTYPE
                    ),
                )
            )
    return calls


def dflash_calls(
    sample_from_anchor: bool, cp_layouts: Sequence[tuple[int, int]]
) -> list:
    """The DFlash input preparation for each draft group's ``(cp_size,
    cp_interleave)``."""
    return [
        _call(
            prepare_dflash_inputs_kernel,
            *prepare_dflash_inputs_key(
                sample_from_anchor, cp_size, cp_interleave, RUNNER_IDX_DTYPE
            ),
        )
        for cp_size, cp_interleave in dict.fromkeys(cp_layouts)
    ]
