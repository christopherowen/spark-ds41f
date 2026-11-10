# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact wide coarse merges with direct, ordered writes to admitted outputs."""
import tilelang
import tilelang.language as T

from . import pdl
from .indexer import _row_extent, _key16, _value16, _select_macros, _final_writer


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def wide_indexer_topk_merge(width: int, topk: int, group: int, lists: int,
                            span: int, fan_in: int, candidates: bool,
                            threads: int = 512):
    assert group == 1 and not candidates, 'wide merge serves coarse entries'
    rows = T.dynamic('rows')
    groups = tilelang.cdiv(lists, fan_in)
    final = groups == 1
    n = min(lists, fan_in) * topk
    n_pad = tilelang.cdiv(n, threads) * threads
    per = n_pad // threads
    assert n_pad * 4 <= 64 * 1024, 'wide merge keys exceed the admitted tile'
    steps = threads.bit_length() - 1
    threshold, _ = _select_macros(threads, topk)
    write_final, write_empty = _final_writer(threads, topk, 1)

    @T.macro
    def block_scan(values, tx):
        carry = T.alloc_var(T.int32)
        for s in T.serial(steps):
            off = 1 << s
            carry = T.if_then_else(tx >= off, values[T.max(tx-off, 0)], 0)
            T.sync_threads()
            values[tx] += carry
            T.sync_threads()

    @T.macro
    def body(WorkValues, WorkEntries, Lengths, Active, Indices, Values):
        with T.Kernel(groups, rows, threads=threads) as (g, row):
            pdl.chain()
            tx = T.get_thread_binding()
            keys = T.alloc_shared([n_pad], T.int32)
            hist = T.alloc_shared([257], T.int32)
            scan = T.alloc_shared([threads], T.int32)
            params = T.alloc_shared([4], T.int32)
            extent = _row_extent(Lengths, Active, row, width)
            live = T.min(T.ceildiv(extent, span), lists)
            base = g * fan_in * topk
            group_live = T.min(T.max(live-g*fan_in, 0), fan_in)
            if group_live <= 0:
                if final:
                    write_empty(Indices, Values, None, row)
            elif group_live == 1:
                if final:
                    write_final(Indices, Values, None, None, row, extent,
                        lambda slot: WorkEntries[row, base+slot] >= 0,
                        lambda slot: WorkEntries[row, base+slot],
                        lambda slot: WorkValues[row, base+slot], params)
                else:
                    for i in T.serial(T.ceildiv(topk, threads)):
                        slot = i*threads+tx
                        if slot < topk:
                            Values[row, g*topk+slot] = WorkValues[row, base+slot]
                            Indices[row, g*topk+slot] = WorkEntries[row, base+slot]
            else:
                for i in T.serial(per):
                    j = i*threads+tx
                    c = T.min(base+j, lists*topk-1)
                    keys[j] = T.if_then_else(
                        (j < n) & (g*fan_in+j//topk < live),
                        T.if_then_else(WorkEntries[row, c] >= 0,
                                       _key16(WorkValues[row, c]), 0), 0)
                T.sync_threads()
                threshold(keys, n_pad, tx, hist, scan, params)
                key = T.alloc_var(T.int32)
                n_gt = T.alloc_var(T.int32)
                n_eq = T.alloc_var(T.int32)
                sel_eq = T.alloc_var(T.int32)
                out_pos = T.alloc_var(T.int32)
                taken = T.alloc_var(T.int32)
                thr = T.alloc_var(T.int32)
                need = T.alloc_var(T.int32)
                thr = params[0]
                need = params[1]
                n_gt = 0
                n_eq = 0
                for v in T.serial(per):
                    key = keys[tx*per+v]
                    if key > thr:
                        n_gt += 1
                    elif key == thr:
                        n_eq += 1
                scan[tx] = n_eq
                T.sync_threads()
                block_scan(scan, tx)
                sel_eq = T.max(T.min(need-(scan[tx]-n_eq), n_eq), 0)
                T.sync_threads()
                scan[tx] = n_gt+sel_eq
                T.sync_threads()
                block_scan(scan, tx)
                out_pos = scan[tx]-n_gt-sel_eq
                taken = 0
                # Emit each winner once, retaining input entry order and ties.
                # The output arrays replace two topk-sized shared staging arrays.
                for v in T.serial(per):
                    key = keys[tx*per+v]
                    if (key > thr) | ((key == thr) & (taken < sel_eq)):
                        if key == thr:
                            taken += 1
                        Values[row, g*topk+out_pos] = _value16(key)
                        Indices[row, g*topk+out_pos] = WorkEntries[row, base+tx*per+v]
                        out_pos += 1
                T.sync_threads()
                total = scan[threads-1]
                for i in T.serial(T.ceildiv(topk, threads)):
                    slot = i*threads+tx
                    if (slot < topk) & (slot >= total):
                        Values[row, g*topk+slot] = -T.infinity(T.bfloat16)
                        Indices[row, g*topk+slot] = -1

    if final:
        @T.prim_func
        def merge(WorkValues: T.Tensor([rows, lists*topk], T.bfloat16),
                  WorkEntries: T.Tensor([rows, lists*topk], T.int32),
                  Lengths: T.Tensor([rows], T.int32),
                  Active: T.Tensor([1], T.int32),
                  Indices: T.Tensor([rows, topk], T.int32),
                  Values: T.Tensor([rows, topk], T.bfloat16)):
            body(WorkValues, WorkEntries, Lengths, Active, Indices, Values)
        return merge

    @T.prim_func
    def merge(WorkValues: T.Tensor([rows, lists*topk], T.bfloat16),
              WorkEntries: T.Tensor([rows, lists*topk], T.int32),
              Lengths: T.Tensor([rows], T.int32),
              Active: T.Tensor([1], T.int32),
              OutValues: T.Tensor([rows, groups*topk], T.bfloat16),
              OutEntries: T.Tensor([rows, groups*topk], T.int32)):
        body(WorkValues, WorkEntries, Lengths, Active, OutEntries, OutValues)
    return merge
