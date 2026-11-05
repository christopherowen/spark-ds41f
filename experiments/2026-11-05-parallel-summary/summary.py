# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coalesced exact eight-key means with cooperative scale reduction."""
import tilelang
import tilelang.language as T
from vllm.models.deepseek_v4_1.tilelang.shortlist_indexer import decode
from vllm.models.deepseek_v4_1.tilelang.indexer import _e2m1_code

@tilelang.jit
def summarize_parallel(page:int,slot_dtype:str='int64'):
    rows,pages,stride=T.dynamic('rows,pages,stride')
    small=page//8;groups=small*4
    assert page in (128,256)
    @T.prim_func
    def update(Pool:T.StridedTensor([pages,page*80],[stride,1],T.uint8),
               Half:T.StridedTensor([pages,page*40],[stride//2,1],T.bfloat16),
               Slots:T.Tensor([rows],slot_dtype)):
        with T.Kernel(rows,threads=128) as row:
            slot=Slots[row]
            sums=T.alloc_fragment([groups,32],T.float32)
            terms=T.alloc_fragment([groups,32],T.float32)
            maximum=T.alloc_fragment([groups],T.float32)
            means=T.alloc_shared([groups,32],T.bfloat16)
            inverse=T.alloc_shared([groups],T.float32)
            if (slot>=0)&(slot%page==page-1)&(slot//page<pages):
                physical=slot//page
                T.fill(sums,0.)
                for k in T.serial(8):
                    for g,i in T.Parallel(groups,32):
                        d=g%4*32+i;token=g//4*8+k
                        sums[g,i]+=decode(Pool[physical,token*64+d//2],
                            Pool[physical,page*64+token*4+g%4],d)*.125
                for g,i in T.Parallel(groups,32):
                    means[g,i]=T.cast(sums[g,i],T.bfloat16)
                    terms[g,i]=T.max(T.abs(T.cast(T.cast(sums[g,i],T.bfloat16),T.float32)),6.0*2.0**-126)
                T.reduce_max(terms,maximum,dim=1)
                for g in T.Parallel(groups):
                    bits=T.reinterpret(T.fmul(maximum[g],T.float32(1.0/6.0)),T.uint32)
                    sf=T.cast(T.shift_right(bits,23)&0xFF,T.int32)+T.cast((bits&0x7FFFFF)!=0,T.int32)
                    inverse[g]=T.reinterpret(T.cast(T.shift_left(254-sf,23),T.uint32),T.float32)
                    Pool[physical,page*68+small*64+g]=T.cast(sf,T.uint8)
                for g,i in T.Parallel(groups,16):
                    lo=_e2m1_code(T.fmul(T.cast(means[g,2*i],T.float32),inverse[g]))
                    hi=_e2m1_code(T.fmul(T.cast(means[g,2*i+1],T.float32),inverse[g]))
                    Pool[physical,page*68+g//4*64+g%4*16+i]=T.cast(lo|(hi<<4),T.uint8)
    return update
