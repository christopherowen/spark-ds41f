# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page-owned MXFP4 block means and native coarse/fine scoring."""
import math

import tilelang
import tilelang.language as T
import torch

from . import prepare
from .block_shortlist import BlockMeanShortlistPlan
from .indexer import (
    CANDIDATE_BLOCK, CANDIDATE_BLOCKS, IndexerTopK, IndexerV41Plan,
    _mxfp4_byte, indexer_score_v41,
)
from .shortlist_indexer import decode, remap_blocks
from .fused_candidates import FusedCoarse

STATE_BYTES = 80


@tilelang.jit
def summarize_compact(page: int, slot_dtype: str = 'int64'):
    rows, pool_pages, stride = T.dynamic('rows, pool_pages, stride')
    small = page // 8
    assert page in (128, 256)

    @T.prim_func
    def update(Pool: T.StridedTensor([pool_pages, page*STATE_BYTES], [stride, 1], T.uint8),
               Half: T.StridedTensor([pool_pages, page*STATE_BYTES//2], [stride//2, 1], T.bfloat16),
               Slots: T.Tensor([rows], slot_dtype)):
        with T.Kernel(rows, threads=128) as row:
            tx = T.get_thread_binding()
            slot = Slots[row]
            vals = T.alloc_local([32], T.bfloat16)
            sums = T.alloc_local([32], T.float32)
            amax = T.alloc_var(T.float32)
            if (slot >= 0) & (slot % page == page-1) & (slot // page < pool_pages):
                physical = slot // page
                for group in T.serial(T.ceildiv(small*4, 128)):
                    g = group*128+tx
                    block, part = g//4, g%4
                    if g < small*4:
                        T.fill(sums, 0.)
                        for k in T.serial(8):
                            for i in T.serial(32):
                                d = part*32+i
                                token = block*8+k
                                sums[i] += decode(Pool[physical, token*64+d//2],
                                    Pool[physical, page*64+token*4+part], d)*.125
                        amax = 6.0 * 2.0**-126
                        for i in T.serial(32):
                            vals[i] = T.cast(sums[i], T.bfloat16)
                            amax = T.max(amax, T.abs(T.cast(vals[i], T.float32)))
                        bits = T.reinterpret(T.fmul(amax, T.float32(1.0/6.0)), T.uint32)
                        sf = T.cast(T.shift_right(bits, 23) & 0xFF, T.int32) + T.cast(
                            (bits & 0x7FFFFF) != 0, T.int32)
                        scale = T.reinterpret(T.cast(T.shift_left(254-sf, 23), T.uint32), T.float32)
                        for i in T.serial(16):
                            Pool[physical, page*68+block*64+part*16+i] = _mxfp4_byte(vals, 2*i, scale)
                        Pool[physical, page*68+small*64+block*4+part] = T.cast(sf, T.uint8)
    return update


class CompactShortlistPlan(BlockMeanShortlistPlan):
    def __init__(self, width, page, *, candidate_blocks=False, budget=None, row_cap=256):
        self.width, self.page, self.source, self.row_cap = width, page, candidate_blocks, row_cap
        self.budget = budget or (512 if candidate_blocks else 64)
        self.block_budget = self.budget*8
        self.capacity = (self.budget+8)*64
        self.score = indexer_score_v41(width//8, page//8, block_P=page//8,
                                      summary_page=page, summary_budget=self.block_budget)
        self.coarse = FusedCoarse(width//8, page, self.block_budget)
        self.final = IndexerV41Plan(self.capacity, page, candidates=True)
        if self.source:
            self.select_blocks = IndexerTopK(self.capacity, CANDIDATE_BLOCKS, CANDIDATE_BLOCK)
            self.remap = remap_blocks(self.capacity)

    @staticmethod
    def calls(width, page, *, candidate_blocks=False, budget=None, row_cap=256):
        b = budget or (512 if candidate_blocks else 64)
        coarse = FusedCoarse.calls(width//8,page,b*8)
        capacity=(b+8)*64
        calls=[prepare.call(indexer_score_v41,width//8,page//8,block_P=page//8,
                            summary_page=page,summary_budget=b*8),*coarse,
               *IndexerV41Plan.calls(capacity,page,candidates=True)]
        if candidate_blocks:
            calls += [*IndexerTopK.calls(capacity,CANDIDATE_BLOCKS,CANDIDATE_BLOCK),
                      prepare.call(remap_blocks,capacity)]
        return calls

    def _specs(self,rows):
        result=super()._specs(rows)
        result.pop('selected')
        result.pop('values')
        return result

    def run(self, q, scales, weights, pool, table, lengths, active, candidates=None, candidate_lengths=None,
            *, out=None, candidate_out=None, candidate_out_lengths=None, scratch=None):
        if candidates is not None or candidate_lengths is not None:
            raise ValueError('compact summaries cannot consume reindex candidates')
        if out is None or scratch is None:
            raise ValueError('outputs and scratch must be admitted')
        if self.source and (candidate_out is None or candidate_out_lengths is None):
            raise ValueError('source handoff outputs are required')
        rows = len(q)
        specs = self._specs(rows)
        if len(scratch) != len(specs):
            raise ValueError('workspace differs from admission')
        full = dict(zip(specs, scratch))
        means = pool[:, self.page*68:self.page*68+(self.page//8)*68]
        def buffers(name, plan, b):
            return [b[f'{name}_{i}'] for i in range(len(plan.scratch_specs(1)))]
        for start in range(0, rows, self.row_cap):
            end = min(start+self.row_cap, rows)
            r = end-start
            current = self._specs(r)
            b = {name:t.view(-1)[:math.prod(current[name][0])].view(current[name][0])
                 for name, t in full.items()}
            qr, sr, wr, tr, lr = q[start:end], scales[start:end], weights[start:end], table[start:end], lengths[start:end]
            self.score(qr.view(torch.int8), sr.view(torch.uint32).reshape(r, 32), wr,
                       means.view(torch.int8), means.view(torch.uint32), tr, lr, active, b['scores'], b['counts'])
            self.coarse.run(b['scores'],b['counts'],lr,active,b['candidates'],b['candidate_lengths'],
                            scratch=buffers('coarse',self.coarse,b))
            _, scores = self.final.run(qr, sr, wr, pool, tr, lr, active, b['candidates'], b['candidate_lengths'],
                                      out=out[start:end], scratch=buffers('final', self.final, b))
            if self.source:
                self.select_blocks.run(scores, b['candidate_lengths'], active, b['compact'], b['block_values'],
                                       counts=candidate_out_lengths[start:end], scratch=buffers('blocks', self.select_blocks, b))
                self.remap(b['compact'], b['candidates'], candidate_out[start:end])
        return out, None
