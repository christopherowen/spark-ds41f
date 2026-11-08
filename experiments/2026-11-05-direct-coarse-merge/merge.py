# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact source coarse selection without selecting every entry of a segment."""
import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import prepare,pdl
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    IndexerTopK,_make,_row_extent,_key16,_value16,_select_macros,_final_writer,
)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC:True})
def raw_pair(width:int):
    rows=T.dynamic('rows')
    topk,threads=4096,512
    span=2*topk
    groups=tilelang.cdiv(width,span)
    final=groups==1
    threshold,compact=_select_macros(threads,topk)
    write_final,write_empty=_final_writer(threads,topk,1)

    @T.macro
    def body(Scores,Lengths,Active,Indices,Values):
        with T.Kernel(groups,rows,threads=threads) as (g,row):
            pdl.chain()
            tx=T.get_thread_binding()
            keys=T.alloc_shared([span],T.int32)
            hist=T.alloc_shared([257],T.int32)
            scan=T.alloc_shared([threads],T.int32)
            params=T.alloc_shared([4],T.int32)
            selected=T.alloc_shared([topk],T.int32)
            selected_keys=T.alloc_shared([topk],T.int32)
            extent=_row_extent(Lengths,Active,row,width)
            base=g*span
            if base < extent:
                for i in T.serial(span//threads):
                    j=i*threads+tx
                    keys[j]=T.if_then_else(base+j < extent,
                        _key16(Scores[row,T.min(base+j,width-1)]),0)
                T.sync_threads()
                threshold(keys,span,tx,hist,scan,params)
                compact(keys,span//threads,tx,scan,params,selected,selected_keys)
                total=scan[threads-1]
                if final:
                    write_final(Indices,Values,None,None,row,extent,
                        lambda slot:slot < total,lambda slot:base+selected[slot],
                        lambda slot:_value16(selected_keys[slot]),params)
                else:
                    for i in T.serial(topk//threads):
                        slot=i*threads+tx
                        if slot < total:
                            Indices[row,g*topk+slot]=base+selected[slot]
                            Values[row,g*topk+slot]=_value16(selected_keys[slot])
                        else:
                            Indices[row,g*topk+slot]=-1
                            Values[row,g*topk+slot]=-T.infinity(T.bfloat16)
            elif final:
                write_empty(Indices,Values,None,row)

    @T.prim_func
    def run(Scores:T.Tensor([rows,width],T.bfloat16),Lengths:T.Tensor([rows],T.int32),
            Active:T.Tensor([1],T.int32),Indices:T.Tensor([rows,groups*topk],T.int32),
            Values:T.Tensor([rows,groups*topk],T.bfloat16)):
        body(Scores,Lengths,Active,Indices,Values)
    return run


class DirectCoarseTopK:
    """Raw pairs replace the all-entry segment pass and first merge exactly."""
    def __init__(self,width):
        self.width=width
        self.topk=4096
        self.groups=tilelang.cdiv(width,8192)
        segments,first,levels=IndexerTopK.routes(width,self.topk)
        assert segments>=2 and self.groups==levels[0][2]
        self.raw=raw_pair(width)
        self.levels=[(_make(kernel),lists,out) for kernel,lists,out in levels[1:]]

    @staticmethod
    def calls(width):
        _,_,levels=IndexerTopK.routes(width,4096)
        return [prepare.call(raw_pair,width),*(kernel for kernel,_,_ in levels[1:])]

    def scratch_specs(self,rows):
        if self.groups==1:return ()
        widest=(self.groups*4096,self.levels[0][2]*4096)
        return tuple(((rows*cols,),dtype) for cols in widest for dtype in (torch.bfloat16,torch.int32))

    def run(self,scores,lengths,active,indices,values,*,scratch=None):
        rows=len(scores)
        if self.groups==1:
            self.raw(scores,lengths,active,indices,values)
            return indices
        specs=self.scratch_specs(rows)
        if scratch is None:scratch=[torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in specs]
        assert len(scratch)==4
        buffers=((scratch[0],scratch[1]),(scratch[2],scratch[3]))
        def views(buffer,lists):
            return tuple(t[:rows*lists*4096].view(rows,lists*4096) for t in buffer)
        first=views(buffers[0],self.groups)
        self.raw(scores,lengths,active,first[1],first[0])
        for level,(kernel,lists,outputs) in enumerate(self.levels):
            work=views(buffers[level%2],lists)
            if outputs==1:kernel(*work,lengths,active,indices,values)
            else:kernel(*work,lengths,active,*views(buffers[(level+1)%2],outputs))
        return indices
