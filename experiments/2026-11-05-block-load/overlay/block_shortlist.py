# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eight-key means, ordered shortlist assembly and native fine scoring."""
import math

import tilelang
import tilelang.language as T
import torch

from . import prepare
from .indexer import IndexerV41Plan, IndexerTopK, CANDIDATE_BLOCKS, CANDIDATE_BLOCK
from .paged_shortlist import PageShortlistPlan
from .shortlist_indexer import decode, remap_blocks

STATE_BYTES = 100


@tilelang.jit
def summarize_blocks(page: int, slot_dtype: str = 'int64'):
    rows,pool_pages,stride = T.dynamic('rows, pool_pages, stride')
    blocks = page//8

    @T.prim_func
    def update(Pool: T.StridedTensor([pool_pages,page*STATE_BYTES],[stride,1],T.uint8),
               Half: T.StridedTensor([pool_pages,page*STATE_BYTES//2],[stride//2,1],T.bfloat16),
               Slots: T.Tensor([rows],slot_dtype)):
        with T.Kernel(rows,threads=128) as row:
            means = T.alloc_fragment([blocks,128],T.float32)
            slot = Slots[row]
            if (slot >= 0) & (slot%page == page-1) & (slot//page < pool_pages):
                physical = slot//page
                T.fill(means,0.)
                for k in T.serial(8):
                    for block,d in T.Parallel(blocks,128):
                        token = block*8+k
                        means[block,d] += decode(Pool[physical,token*64+d//2],
                            Pool[physical,page*64+token*4+d//32],d)*.125
                for block,d in T.Parallel(blocks,128):
                    Half[physical,page*68//2+block*128+d] = T.cast(means[block,d],T.bfloat16)
    return update


@tilelang.jit
def block_scores(width: int, page: int, budget: int, block_c: int = 64):
    rows,pool_pages,stride = T.dynamic('rows, pool_pages, stride')
    n = width//8
    ctas = min(tilelang.cdiv(n,block_c),128)

    @T.prim_func
    def score(Q: T.Tensor([rows,32,64],T.uint8), Scale: T.Tensor([rows,32,4],T.uint8),
              Weight: T.Tensor([rows,32],T.bfloat16),
              Pool: T.StridedTensor([pool_pages,page*STATE_BYTES//2],[stride,1],T.bfloat16),
              Table: T.Tensor([rows,width//page],T.int32), Length: T.Tensor([rows],T.int32),
              Active: T.Tensor([1],T.int32), Score: T.Tensor([rows,n],T.bfloat16),
              Count: T.Tensor([rows],T.int32)):
        with T.Kernel(ctas,rows,threads=128) as (cta,row):
            a = T.alloc_shared([block_c,128],T.bfloat16)
            b = T.alloc_shared([32,128],T.bfloat16)
            acc = T.alloc_fragment([block_c,32],T.float32)
            terms = T.alloc_shared([block_c,32],T.float32)
            total = T.alloc_fragment([block_c],T.float32)
            extent = T.min(T.max(Length[row],0),T.max(T.min(Active[0],width),0))
            full = extent//page*(page//8)
            count = T.if_then_else(full <= budget,0,full)
            if (cta == 0) & (T.get_thread_binding() == 0):
                Count[row] = count
            if cta*block_c < count:
                for h,d in T.Parallel(32,128):
                    b[h,d] = T.cast(decode(Q[row,h,d//2],Scale[row,h,d//32],d),T.bfloat16)
                for iteration in T.serial(T.ceildiv(T.max(T.ceildiv(count,block_c)-cta,0),ctas)):
                    start = (cta+iteration*ctas)*block_c
                    for c,d in T.Parallel(block_c,128):
                        a[c,d] = T.if_then_else(start+c < count,
                            Pool[Table[row,(start+c)//(page//8)],
                                 page*68//2+((start+c)%(page//8))*128+d],
                            T.cast(0,T.bfloat16))
                    T.gemm(a,b,acc,transpose_B=True,clear_accum=True)
                    for c,h in T.Parallel(block_c,32):
                        dot = T.max(T.cast(T.cast(acc[c,h],T.bfloat16),T.float32),0.)
                        terms[c,h] = T.cast(T.cast(T.fmul(dot,T.cast(Weight[row,h],T.float32)),T.bfloat16),T.float32)
                    T.sync_threads()
                    T.fill(total,0.)
                    for h in T.serial(32):
                        for c in T.Parallel(block_c):total[c] += terms[c,h]
                    for c in T.Parallel(block_c):
                        if start+c < count:Score[row,start+c] = T.cast(total[c],T.bfloat16)
    return score


@tilelang.jit
def block_candidates(width: int, page: int, budget: int):
    rows = T.dynamic('rows')
    capacity = (budget//8+8)*8

    @T.prim_func
    def assemble(Selected: T.Tensor([rows,budget],T.int32),
                 Length: T.Tensor([rows],T.int32), Active: T.Tensor([1],T.int32),
                 Candidates: T.Tensor([rows,capacity],T.int32),
                 CandidateLength: T.Tensor([rows],T.int32)):
        with T.Kernel(rows,threads=128) as row:
            search = T.alloc_local([2],T.int32)
            extent = T.min(T.max(Length[row],0),T.max(T.min(Active[0],width),0))
            visible = T.ceildiv(extent,8)
            full = extent//page*(page//8)
            partial = visible-full
            if full <= budget:
                for i in T.Parallel(capacity):
                    Candidates[row,i] = T.if_then_else(i < visible,i,-1)
                if T.get_thread_binding() == 0:CandidateLength[row] = extent
            else:
                search[0] = 0
                search[1] = budget
                for step in T.serial(budget.bit_length()):
                    if search[0] < search[1]:
                        middle = (search[0]+search[1])//2
                        if Selected[row,middle] < 0:search[1] = middle
                        else:search[0] = middle+1
                used = search[0]
                last = T.if_then_else(used > 0,Selected[row,T.max(used-1,0)],-1)
                extra = T.cast((extent > 0) & (partial == 0) & (last != visible-1),T.int32)
                total = used+partial+extra
                for i in T.Parallel(capacity):
                    Candidates[row,i] = -1
                    if i < used:Candidates[row,i] = Selected[row,i]
                    elif i < used+partial:Candidates[row,i] = full+i-used
                    elif i < total:Candidates[row,i] = visible-1
                if T.get_thread_binding() == 0:
                    CandidateLength[row] = total*8-T.if_then_else(extent > 0,(8-extent%8)%8,0)
    return assemble


class BlockMeanShortlistPlan(PageShortlistPlan):
    def __init__(self,width,page,*,candidate_blocks=False,budget=None,row_cap=256):
        self.width,self.page,self.source,self.row_cap = width,page,candidate_blocks,row_cap
        self.budget = budget or (512 if candidate_blocks else 64)
        self.block_budget = self.budget*8
        self.capacity = (self.budget+8)*64
        self.score = block_scores(width,page,self.block_budget)
        self.coarse = IndexerTopK(width//8,self.block_budget)
        self.assemble = block_candidates(width,page,self.block_budget)
        self.final = IndexerV41Plan(self.capacity,page,candidates=True)
        if self.source:
            self.select_blocks = IndexerTopK(self.capacity,CANDIDATE_BLOCKS,CANDIDATE_BLOCK)
            self.remap = remap_blocks(self.capacity)

    @staticmethod
    def calls(width,page,*,candidate_blocks=False,budget=None,row_cap=256):
        b = budget or (512 if candidate_blocks else 64)
        capacity = (b+8)*64
        calls = [prepare.call(block_scores,width,page,b*8),*IndexerTopK.calls(width//8,b*8),
                 prepare.call(block_candidates,width,page,b*8),
                 *IndexerV41Plan.calls(capacity,page,candidates=True)]
        if candidate_blocks:
            calls += [*IndexerTopK.calls(capacity,CANDIDATE_BLOCKS,CANDIDATE_BLOCK),
                      prepare.call(remap_blocks,capacity)]
        return calls

    def _specs(self,rows):
        r = min(rows,self.row_cap)
        specs = {'scores':((r,self.width//8),torch.bfloat16),'counts':((r,),torch.int32),
                 'selected':((r,self.block_budget),torch.int32),'values':((r,self.block_budget),torch.bfloat16),
                 'candidates':((r,self.capacity//8),torch.int32),'candidate_lengths':((r,),torch.int32)}
        for name,plan in [('coarse',self.coarse),('final',self.final)]:
            for i,spec in enumerate(plan.scratch_specs(r)):specs[f'{name}_{i}'] = spec
        if self.source:
            specs.update(compact=((r,CANDIDATE_BLOCKS),torch.int32),block_values=((r,CANDIDATE_BLOCKS),torch.bfloat16))
            for i,spec in enumerate(self.select_blocks.scratch_specs(r)):specs[f'blocks_{i}'] = spec
        return specs

    def run(self,q,scales,weights,pool,table,lengths,active,candidates=None,candidate_lengths=None,
            *,out=None,candidate_out=None,candidate_out_lengths=None,scratch=None):
        if candidates is not None or candidate_lengths is not None:
            raise ValueError('block summaries cannot consume reindex candidates')
        if out is None or scratch is None:raise ValueError('outputs and scratch must be admitted')
        if self.source and (candidate_out is None or candidate_out_lengths is None):
            raise ValueError('source handoff outputs are required')
        rows = len(q)
        specs = self._specs(rows)
        if len(scratch) != len(specs):raise ValueError('workspace differs from admission')
        full = dict(zip(specs,scratch))
        def buffers(name,plan,b):return [b[f'{name}_{i}'] for i in range(len(plan.scratch_specs(1)))]
        for start in range(0,rows,self.row_cap):
            end = min(start+self.row_cap,rows)
            r = end-start
            current = self._specs(r)
            b = {name:t.view(-1)[:math.prod(current[name][0])].view(current[name][0]) for name,t in full.items()}
            qr,sr,wr,tr,lr = q[start:end],scales[start:end],weights[start:end],table[start:end],lengths[start:end]
            self.score(qr.view(torch.uint8),sr.view(torch.uint8).reshape(r,32,4),wr,
                       pool.view(torch.bfloat16),tr,lr,active,b['scores'],b['counts'])
            self.coarse.run(b['scores'],b['counts'],active,b['selected'],b['values'],scratch=buffers('coarse',self.coarse,b))
            self.assemble(b['selected'],lr,active,b['candidates'],b['candidate_lengths'])
            _,scores = self.final.run(qr,sr,wr,pool,tr,lr,active,b['candidates'],b['candidate_lengths'],
                                     out=out[start:end],scratch=buffers('final',self.final,b))
            if self.source:
                self.select_blocks.run(scores,b['candidate_lengths'],active,b['compact'],b['block_values'],
                                       counts=candidate_out_lengths[start:end],scratch=buffers('blocks',self.select_blocks,b))
                self.remap(b['compact'],b['candidates'],candidate_out[start:end])
        return out,None
