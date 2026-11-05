# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single shared-memory store per mean element; identical score arithmetic."""
import tilelang
import tilelang.language as T
from vllm.models.deepseek_v4_1.tilelang.shortlist_indexer import decode
from vllm.models.deepseek_v4_1.tilelang.block_shortlist import STATE_BYTES

@tilelang.jit
def prepared_scores(width: int, page: int, budget: int, block_c: int = 64):
    rows,pool_pages,stride = T.dynamic('rows, pool_pages, stride')
    n = width//8
    ctas = min(tilelang.cdiv(n,block_c),128)

    @T.prim_func
    def score(Q: T.Tensor([rows,32,128],T.bfloat16), Scale: T.Tensor([rows,32,4],T.uint8),
              Weight: T.Tensor([rows,32],T.bfloat16),
              Pool: T.StridedTensor([pool_pages,page//8,128],[stride,128,1],T.bfloat16),
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
                T.copy(Q[row,0:32,0:128],b)
                for iteration in T.serial(T.ceildiv(T.max(T.ceildiv(count,block_c)-cta,0),ctas)):
                    start = (cta+iteration*ctas)*block_c
                    for part in T.serial(block_c//(page//8)):
                        first = start+part*(page//8)
                        if first < count:
                            physical = Table[row,first//(page//8)]
                            T.copy(Pool[physical,0:page//8,0:128],
                                   a[part*(page//8):(part+1)*(page//8),0:128])
                        else:
                            for c,d in T.Parallel(page//8,128):
                                a[part*(page//8)+c,d] = 0
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
def prepare_queries(width:int,page:int,budget:int):
    rows = T.dynamic('rows')
    @T.prim_func
    def unpack(Q:T.Tensor([rows,32,64],T.uint8),Scale:T.Tensor([rows,32,4],T.uint8),
               Length:T.Tensor([rows],T.int32),Active:T.Tensor([1],T.int32),
               Decoded:T.Tensor([rows,32,128],T.bfloat16)):
        with T.Kernel(rows,threads=128) as row:
            extent = T.min(T.max(Length[row],0),T.max(T.min(Active[0],width),0))
            if extent//page*(page//8) > budget:
                for h,d in T.Parallel(32,128):
                    Decoded[row,h,d] = T.cast(decode(Q[row,h,d//2],Scale[row,h,d//32],d),T.bfloat16)
    return unpack


import math
import torch
from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang.block_shortlist import BlockMeanShortlistPlan,block_candidates
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerTopK,IndexerV41Plan,CANDIDATE_BLOCKS,CANDIDATE_BLOCK
from vllm.models.deepseek_v4_1.tilelang.shortlist_indexer import remap_blocks


class QueryReusePlan(BlockMeanShortlistPlan):
    def __init__(self,width,page,*,candidate_blocks=False,budget=None,row_cap=256):
        self.width,self.page,self.source,self.row_cap = width,page,candidate_blocks,row_cap
        self.budget = budget or (512 if candidate_blocks else 64)
        self.block_budget = self.budget*8
        self.capacity = (self.budget+8)*64
        self.coarse = IndexerTopK(width//8,self.block_budget)
        self.assemble = block_candidates(width,page,self.block_budget)
        self.final = IndexerV41Plan(self.capacity,page,candidates=True)
        if self.source:
            self.select_blocks = IndexerTopK(self.capacity,CANDIDATE_BLOCKS,CANDIDATE_BLOCK)
            self.remap = remap_blocks(self.capacity)
        self.unpack = prepare_queries(width,page,self.block_budget)
        self.prepared_score = prepared_scores(width,page,self.block_budget)

    def _specs(self,rows):
        specs = super()._specs(rows)
        specs['prepared_q'] = ((min(rows,self.row_cap),32,128),torch.bfloat16)
        return specs

    @staticmethod
    def calls(width,page,*,candidate_blocks=False,budget=None,row_cap=256):
        b = budget or (512 if candidate_blocks else 64)
        parent = BlockMeanShortlistPlan.calls(width,page,candidate_blocks=candidate_blocks,budget=budget,row_cap=row_cap)
        return [prepare.call(prepare_queries,width,page,b*8),
                prepare.call(prepared_scores,width,page,b*8),*parent[1:]]

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
            self.unpack(qr.view(torch.uint8),sr.view(torch.uint8).reshape(r,32,4),lr,active,b['prepared_q'])
            means = pool.view(torch.bfloat16)[:,self.page*68//2:self.page*100//2].view(len(pool),self.page//8,128)
            self.prepared_score(b['prepared_q'],sr.view(torch.uint8).reshape(r,32,4),wr,
                                means,tr,lr,active,b['scores'],b['counts'])
            self.coarse.run(b['scores'],b['counts'],active,b['selected'],b['values'],scratch=buffers('coarse',self.coarse,b))
            self.assemble(b['selected'],lr,active,b['candidates'],b['candidate_lengths'])
            _,scores = self.final.run(qr,sr,wr,pool,tr,lr,active,b['candidates'],b['candidate_lengths'],
                                     out=out[start:end],scratch=buffers('final',self.final,b))
            if self.source:
                self.select_blocks.run(scores,b['candidate_lengths'],active,b['compact'],b['block_values'],
                                       counts=candidate_out_lengths[start:end],scratch=buffers('blocks',self.select_blocks,b))
                self.remap(b['compact'],b['candidates'],candidate_out[start:end])
        return out,None
