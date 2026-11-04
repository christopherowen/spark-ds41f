# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query-only semantic shortlist. Validation/reference work lives elsewhere.

Metadata is an explicit immutable snapshot for one cache owner. Full512-key
chunks contain eight balanced buckets of native eight-key blocks. Queries use
only complete chunks; every visible key in the partial chunk remains eligible.
This module does not claim a persistent serving-cache update protocol.
"""
import tilelang
import tilelang.language as T
import torch

from .indexer import IndexerTopK, IndexerV41Plan, INDEX_TOPK, CANDIDATE_BLOCKS

CHUNK = 512
BUCKET = 64
BLOCK = 8
TILE = 256


def decode(byte, scale, dimension):
    code = (T.cast(byte, T.int32) >> ((dimension % 2)*4)) & 15
    magnitude = code & 7
    value = T.if_then_else(magnitude < 2, T.cast(magnitude, T.float32)*.5,
                          (1.+T.cast(magnitude & 1, T.float32)*.5)*T.cast(1 << T.max(magnitude//2-1, 0), T.float32))
    value = value*T.reinterpret(T.cast(scale, T.int32) << 23, T.float32)
    return T.if_then_else((code & 8) != 0, -value, value)


@tilelang.jit
def centroid_scores(width: int, block_c: int = 64):
    rows = T.dynamic('rows')
    buckets = width//BUCKET
    ctas = min(tilelang.cdiv(buckets, block_c), 128)

    @T.prim_func
    def score(Q: T.Tensor([rows,32,64],T.uint8), Scale: T.Tensor([rows,32,4],T.uint8),
              Weight: T.Tensor([rows,32],T.bfloat16), Centers: T.Tensor([buckets,128],T.bfloat16),
              Length: T.Tensor([rows],T.int32), Active: T.Tensor([1],T.int32),
              Score: T.Tensor([rows,buckets],T.bfloat16), Count: T.Tensor([rows],T.int32)):
        with T.Kernel(ctas, rows, threads=128) as (cta,row):
            a = T.alloc_shared([block_c,128],T.bfloat16)
            b = T.alloc_shared([32,128],T.bfloat16)
            acc = T.alloc_fragment([block_c,32],T.float32)
            terms = T.alloc_shared([block_c,32],T.float32)
            total = T.alloc_fragment([block_c],T.float32)
            extent = T.min(T.max(Length[row],0),T.max(T.min(Active[0],width),0))
            count = extent//CHUNK*(CHUNK//BUCKET)
            if cta == 0:
                if T.get_thread_binding() == 0:
                    Count[row] = count
            if cta*block_c < count:
                for h,d in T.Parallel(32,128):
                    b[h,d] = T.cast(decode(Q[row,h,d//2],Scale[row,h,d//32],d),T.bfloat16)
                for iteration in T.serial(T.ceildiv(T.max(T.ceildiv(count,block_c)-cta,0),ctas)):
                    start = (cta+iteration*ctas)*block_c
                    for c,d in T.Parallel(block_c,128):
                        a[c,d] = T.if_then_else(start+c < count,Centers[T.min(start+c,buckets-1),d],T.cast(0,T.bfloat16))
                    T.gemm(a,b,acc,transpose_B=True,clear_accum=True)
                    for c,h in T.Parallel(block_c,32):
                        dot = T.max(T.cast(T.cast(acc[c,h],T.bfloat16),T.float32),0.)
                        terms[c,h] = T.cast(T.cast(T.fmul(dot,T.cast(Weight[row,h],T.float32)),T.bfloat16),T.float32)
                    T.sync_threads()
                    T.fill(total,0.)
                    for h in T.serial(32):
                        for c in T.Parallel(block_c):
                            total[c] += terms[c,h]
                    for c in T.Parallel(block_c):
                        if start+c < count:
                            Score[row,start+c] = T.cast(total[c],T.bfloat16)
    return score


@tilelang.jit
def shortlist_flags(width: int, budget: int):
    rows = T.dynamic('rows')
    blocks, tiles = width//BLOCK, tilelang.cdiv(width//BLOCK,TILE)
    ctas = min(tiles,128)
    search_steps = budget.bit_length()

    @T.prim_func
    def flags(Selected: T.Tensor([rows,budget],T.int32), Membership: T.Tensor([blocks],T.int32),
              Length: T.Tensor([rows],T.int32), Active: T.Tensor([1],T.int32),
              Local: T.Tensor([rows,blocks],T.int32), Counts: T.Tensor([rows,tiles],T.int32)):
        with T.Kernel(ctas,rows,threads=128) as (cta,row):
            keep = T.alloc_fragment([TILE],T.int32)
            prefix = T.alloc_fragment([TILE],T.int32)
            low = T.alloc_fragment([TILE],T.int32)
            high = T.alloc_fragment([TILE],T.int32)
            extent = T.min(T.max(Length[row],0),T.max(T.min(Active[0],width),0))
            visible = T.ceildiv(extent,BLOCK)
            complete = extent//CHUNK*(CHUNK//BLOCK)
            for iteration in T.serial(T.ceildiv(T.max(T.ceildiv(visible,TILE)-cta,0),ctas)):
                tile = cta+iteration*ctas
                for i in T.Parallel(TILE):
                    low[i] = 0
                    high[i] = budget
                # Native top-k returns sorted bucket ids with trailing -1.
                # Treat padding as +infinity to keep the binary search ordered.
                for step in T.serial(search_steps):
                    for i in T.Parallel(TILE):
                        block = tile*TILE+i
                        if (block < complete) & (low[i] < high[i]):
                            middle = (low[i]+high[i])//2
                            selected = Selected[row,T.min(middle,budget-1)]
                            selected = T.if_then_else(selected < 0,2147483647,selected)
                            if selected < Membership[block]:
                                low[i] = middle+1
                            else:
                                high[i] = middle
                for i in T.Parallel(TILE):
                    block = tile*TILE+i
                    keep[i] = 0
                    if block < visible:
                        chosen = (block >= complete) | (block == visible-1)
                        if block < complete:
                            chosen = chosen | ((low[i] < budget) &
                                (Selected[row,T.min(low[i],budget-1)] == Membership[block]))
                        keep[i] = T.cast(chosen,T.int32)
                T.cumsum(keep,prefix,dim=0)
                for i in T.Parallel(TILE):
                    block = tile*TILE+i
                    if block < visible:
                        Local[row,block] = T.if_then_else(keep[i] > 0,prefix[i],0)
                    if i == TILE-1:
                        Counts[row,tile] = prefix[i]
    return flags


@tilelang.jit
def shortlist_prefix(width: int, budget: int):
    rows = T.dynamic('rows')
    tiles = tilelang.cdiv(width//BLOCK,TILE)
    scan_size = 1 << (tiles-1).bit_length()
    capacity = (budget+8)*BUCKET//BLOCK

    @T.prim_func
    def prefix(Counts: T.Tensor([rows,tiles],T.int32), Length: T.Tensor([rows],T.int32),
               Active: T.Tensor([1],T.int32), Base: T.Tensor([rows,tiles],T.int32),
               Candidates: T.Tensor([rows,capacity],T.int32), CandidateLength: T.Tensor([rows],T.int32)):
        with T.Kernel(rows,threads=128) as row:
            value = T.alloc_fragment([scan_size],T.int32)
            cumulative = T.alloc_fragment([scan_size],T.int32)
            extent = T.min(T.max(Length[row],0),T.max(T.min(Active[0],width),0))
            live_tiles = T.ceildiv(T.ceildiv(extent,BLOCK),TILE)
            for i in T.Parallel(scan_size):
                value[i] = T.if_then_else(i < live_tiles,Counts[row,T.min(i,tiles-1)],0)
            T.cumsum(value,cumulative,dim=0)
            for i in T.Parallel(scan_size):
                if i < live_tiles:
                    Base[row,i] = cumulative[i]-value[i]
                if i == scan_size-1:
                    CandidateLength[row] = cumulative[i]*BLOCK-T.if_then_else(extent > 0,(BLOCK-extent%BLOCK)%BLOCK,0)
            for i in T.Parallel(capacity):
                Candidates[row,i] = -1
    return prefix


@tilelang.jit
def shortlist_scatter(width: int, budget: int):
    rows = T.dynamic('rows')
    blocks, tiles = width//BLOCK, tilelang.cdiv(width//BLOCK,TILE)
    ctas, capacity = min(tiles,128), (budget+8)*BUCKET//BLOCK

    @T.prim_func
    def scatter(Local: T.Tensor([rows,blocks],T.int32), Base: T.Tensor([rows,tiles],T.int32),
                Length: T.Tensor([rows],T.int32), Active: T.Tensor([1],T.int32),
                Candidates: T.Tensor([rows,capacity],T.int32)):
        with T.Kernel(ctas,rows,threads=128) as (cta,row):
            extent = T.min(T.max(Length[row],0),T.max(T.min(Active[0],width),0))
            visible = T.ceildiv(extent,BLOCK)
            for iteration in T.serial(T.ceildiv(T.max(T.ceildiv(visible,TILE)-cta,0),ctas)):
                tile = cta+iteration*ctas
                for i in T.Parallel(TILE):
                    block = tile*TILE+i
                    if block < visible:
                        position = Local[row,block]
                        if position > 0:
                            Candidates[row,Base[row,tile]+position-1] = block
    return scatter


@tilelang.jit
def remap_blocks(capacity: int):
    rows = T.dynamic('rows')

    @T.prim_func
    def remap(Selected: T.Tensor([rows,CANDIDATE_BLOCKS],T.int32),
              Candidates: T.Tensor([rows,capacity//BLOCK],T.int32),
              Original: T.Tensor([rows,CANDIDATE_BLOCKS],T.int32)):
        with T.Kernel(tilelang.cdiv(CANDIDATE_BLOCKS,128),rows,threads=128) as (tile,row):
            for i in T.Parallel(128):
                slot = tile*128+i
                entry = Selected[row,slot]
                Original[row,slot] = T.if_then_else(entry >= 0,Candidates[row,T.max(entry,0)],-1)
    return remap


class ShortlistPlan:
    """No reference, proof, comparison, host copies or metadata rebuild in run."""
    def __init__(self,width,page,budget,centers,membership,rows=1,source=False):
        assert width % CHUNK == 0 and budget > 0 and budget & (budget-1) == 0
        self.width,self.page,self.budget,self.source = width,page,budget,source
        self.centers,self.membership = centers,membership
        self.capacity = (budget+8)*BUCKET
        self.score = centroid_scores(width)
        self.coarse = IndexerTopK(width//BUCKET,budget)
        self.flags = shortlist_flags(width,budget)
        self.prefix = shortlist_prefix(width,budget)
        self.scatter = shortlist_scatter(width,budget)
        self.final = IndexerV41Plan(self.capacity,page,candidates=True)
        def empty(shape,dtype=torch.int32):
            return torch.empty(shape,dtype=dtype,device='cuda')
        def scratch(plan):
            return [empty(shape,dtype) for shape,dtype in plan.scratch_specs(rows)]
        self.scores = empty((rows,width//BUCKET),torch.bfloat16)
        self.counts = empty((rows,))
        self.selected = empty((rows,budget))
        self.values = empty((rows,budget),torch.bfloat16)
        self.coarse_scratch = scratch(self.coarse)
        self.local = empty((rows,width//BLOCK))
        self.tile_counts = empty((rows,tilelang.cdiv(width//BLOCK,TILE)))
        self.bases = empty(self.tile_counts.shape)
        self.candidates = empty((rows,self.capacity//BLOCK))
        self.candidate_lengths = empty((rows,))
        self.out = empty((rows,INDEX_TOPK))
        self.final_scratch = scratch(self.final)
        if source:
            self.block_select = IndexerTopK(self.capacity,CANDIDATE_BLOCKS,BLOCK)
            self.block_scratch = scratch(self.block_select)
            self.compact_blocks = empty((rows,CANDIDATE_BLOCKS))
            self.block_values = empty((rows,CANDIDATE_BLOCKS),torch.bfloat16)
            self.blocks = empty((rows,CANDIDATE_BLOCKS))
            self.block_lengths = empty((rows,))
            self.remap = remap_blocks(self.capacity)

    def run(self,q,scales,weights,pool,table,lengths,active):
        rows = q.shape[0]
        self.score(q.view(torch.uint8),scales.view(torch.uint8).reshape(rows,32,4),weights,
                   self.centers,lengths,active,self.scores,self.counts)
        self.coarse.run(self.scores,self.counts,active,self.selected,self.values,scratch=self.coarse_scratch)
        self.flags(self.selected,self.membership,lengths,active,self.local,self.tile_counts)
        self.prefix(self.tile_counts,lengths,active,self.bases,self.candidates,self.candidate_lengths)
        self.scatter(self.local,self.bases,lengths,active,self.candidates)
        _,scores = self.final.run(q,scales,weights,pool,table,lengths,active,self.candidates,
                                 self.candidate_lengths,out=self.out,scratch=self.final_scratch)
        if self.source:
            self.block_select.run(scores,self.candidate_lengths,active,self.compact_blocks,self.block_values,
                                  counts=self.block_lengths,scratch=self.block_scratch)
            self.remap(self.compact_blocks,self.candidates,self.blocks)
        return self.out
