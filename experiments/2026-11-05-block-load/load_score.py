# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single shared-memory store per mean element; identical score arithmetic."""
import tilelang
import tilelang.language as T
from vllm.models.deepseek_v4_1.tilelang.shortlist_indexer import decode

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

