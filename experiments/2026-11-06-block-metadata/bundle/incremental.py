# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental GPU gather only; not installed in the serving runtime.

Each CTA owns a separate destination-cache record. An overwrite generation comes
from the producer; append ranges come from counts. No context-sized host scan,
additional gather launch, or persistent copy of the page IDs is needed.
"""
from functools import lru_cache

import torch

from vllm.tilelang_utils import T, tilelang
from vllm.v1.worker.gpu import tilelang_inputs as inputs
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor


@lru_cache(None)
@tilelang.jit(pass_configs=inputs._PASS)
def incremental_kernel(idx_dtype: str, mode: str):
    assert mode in ("row", "chunk", "span")
    num_reqs = T.dynamic("num_reqs")
    groups = T.dynamic("groups")
    capacity = T.dynamic("capacity")
    counts_stride = T.dynamic("counts_stride")
    chunks = T.dynamic("chunks")
    slots = T.dynamic("slots")

    @T.prim_func
    def incremental_gather(
        idx: T.Tensor((num_reqs,), idx_dtype),
        src_ptrs: T.Tensor((groups,), T.ptr),
        dst_ptrs: T.Tensor((groups,), T.ptr),
        widths: T.Tensor((groups,), T.int64),
        counts: T.StridedTensor((groups, capacity), (counts_stride, 1), T.int32),
        epochs: T.Tensor((capacity,), T.int64),
        chunk_groups: T.Tensor((chunks,), T.int32),
        chunk_firsts: T.Tensor((chunks,), T.int64),
        state: T.Tensor((chunks, slots, 3), T.int64),
        rows: T.int32,
    ):
        with T.Kernel(chunks, rows, threads=256) as (c, row):
            tx = T.get_thread_binding()
            group = chunk_groups[c]
            first = chunk_firsts[c]
            width = widths[group]
            live = row < num_reqs
            req = T.if_then_else(live, T.cast(idx[row], T.int64), T.int64(0))
            owner = T.if_then_else(live, req, T.int64(-1))
            epoch = T.if_then_else(live, epochs[req], T.int64(-1))
            count = T.if_then_else(live, T.cast(counts[group, req], T.int64), T.int64(0))
            old_owner = state[c, row, 0]
            old_epoch = state[c, row, 1]
            old_count = state[c, row, 2]
            reset = owner != old_owner or epoch != old_epoch or count < old_count
            changed = reset or count != old_count
            if changed:
                begin = T.if_then_else(reset, T.int64(0), old_count)
                end = T.if_then_else(reset, width, count)
                if mode == "row":
                    begin = T.int64(0)
                    end = width
                elif mode == "chunk":
                    begin = T.truncdiv(begin, 1024) * 1024
                    end = T.min(T.truncdiv(end + 1023, 1024) * 1024, width)
                if first < end and first + 1024 > begin:
                    src = T.make_tensor(src_ptrs[group], (1 << 40,), T.int32)
                    dst = T.make_tensor(dst_ptrs[group], (1 << 40,), T.int32)
                    src_row = req * width
                    dst_row = T.cast(row, T.int64) * width
                    if (
                        T.truncmod(width, 4) == 0
                        and T.truncmod(src_ptrs[group], 16) == 0
                        and T.truncmod(dst_ptrs[group], 16) == 0
                    ):
                        col = first + tx * 4
                        if col < width and col < end and col + 4 > begin:
                            if col >= begin and col + 4 <= end and (
                                col + 4 <= count or col >= count
                            ):
                                value = T.ldg128(src[src_row + col], pred=col + 4 <= count)
                                T.stg128(dst[dst_row + col], value)
                            else:
                                for k in T.unroll(4):
                                    if col + k >= begin and col + k < end:
                                        dst[dst_row + col + k] = T.if_then_else(
                                            col + k < count, src[src_row + col + k], 0
                                        )
                    else:
                        for k in T.unroll(4):
                            col = first + k * 256 + tx
                            if col < width and col >= begin and col < end:
                                dst[dst_row + col] = T.if_then_else(
                                    col < count, src[src_row + col], 0
                                )
                # Other warps must finish reading this CTA's old state before
                # thread zero publishes the next state. No cross-CTA sharing.
                T.sync_threads()
                if tx == 0:
                    state[c, row, 0] = owner
                    state[c, row, 1] = epoch
                    state[c, row, 2] = count

    return incremental_gather


class IncrementalBlockTables(BlockTables):
    """Harness-only adapter for the real producer/staging/gather entry points.

    Serving integration still needs prepared-kernel declarations and all direct
    destination writers audited. This class is deliberately not a runtime patch.
    """

    def __init__(self, *args, mode="span", **kwargs):
        if mode not in ("row", "chunk", "span"):
            raise ValueError(mode)
        self.mode = mode
        super().__init__(*args, **kwargs)
        self.epochs = UvaBackedTensor((self.max_num_reqs,), dtype=torch.int64)
        self._epochs_dirty = False

    def init_block_table_layout_tensors(self):
        super().init_block_table_layout_tensors()
        self.gather_state = torch.full(
            (self._gather_chunks[0].shape[0], self.max_num_reqs, 3),
            -2, dtype=torch.int64, device=self.device,
        )

    def invalidate_gather(self):
        self.gather_state.fill_(-2)

    def append_block_ids(self, req_index, new_block_ids, overwrite):
        if overwrite and int(self.epochs.np[req_index]) == (1 << 63) - 1:
            raise OverflowError("block-table overwrite generation exhausted")
        super().append_block_ids(req_index, new_block_ids, overwrite)
        if overwrite:
            self.epochs.np[req_index] += 1
            self._epochs_dirty = True

    def apply_staged_writes(self):
        super().apply_staged_writes()
        if self._epochs_dirty:
            self.epochs.copy_to_uva()
            self._epochs_dirty = False

    def get_dummy_block_tables(self, num_reqs):
        self.invalidate_gather()
        return super().get_dummy_block_tables(num_reqs)

    def gather_block_tables(self, idx_mapping, num_reqs_padded, out=None, out_ptrs=None):
        if out is not None:
            # An explicitly supplied destination may alias the default tables.
            self.invalidate_gather()
            return super().gather_block_tables(idx_mapping, num_reqs_padded, out, out_ptrs)
        chunks = self._gather_chunks
        if num_reqs_padded and chunks[0].shape[0]:
            incremental_kernel(inputs._DTYPES[idx_mapping.dtype], self.mode)(
                idx_mapping, self._block_table_ptrs_i64, self._input_block_table_ptrs_i64,
                self.block_table_strides, self.num_blocks.gpu, self.epochs.gpu,
                *chunks, self.gather_state, num_reqs_padded,
            )
        return tuple(bt[:num_reqs_padded] for bt in self.input_block_tables)
