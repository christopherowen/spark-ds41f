# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact coarse winners directly form the fine candidate list."""
import tilelang
import tilelang.language as T
import torch
from vllm.models.deepseek_v4_1.tilelang import pdl,prepare
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    IndexerTopK,_make,_row_extent,_key16,_select_macros,
)
from vllm.models.deepseek_v4_1.tilelang.coarse_merge import raw_pair


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC:True})
def finish(width:int,page:int,budget:int,lists:int=1,span:int=4096,raw:bool=False):
    rows=T.dynamic('rows');threads=512
    columns=width if raw else lists*budget
    n=tilelang.cdiv(columns,threads)*threads
    capacity=budget+64
    assert n<=16384
    threshold,compact=_select_macros(threads,budget)

    @T.macro
    def write(Candidates,TokenCount,row,extent,valid,entry,params):
        tx=T.get_thread_binding()
        if tx==0:params[0]=0
        T.sync_threads()
        for i in T.serial(T.ceildiv(capacity,threads)):
            slot=i*threads+tx
            if slot<capacity:
                Candidates[row,slot]=-1
                if slot<budget:
                    if valid(slot):
                        Candidates[row,slot]=entry(slot)
                        if slot==budget-1:
                            params[0]=budget
                        elif not valid(slot+1):
                            params[0]=slot+1
        T.sync_threads()
        used=params[0]
        visible=T.ceildiv(extent,8)
        full=extent//page*(page//8)
        partial=visible-full
        last=T.if_then_else(used>0,Candidates[row,T.max(used-1,0)],-1)
        extra=T.cast((extent>0)&(partial==0)&(last!=visible-1),T.int32)
        for i in T.Parallel(page//8):
            if i<partial:Candidates[row,used+i]=full+i
        if tx==0:
            if extra:Candidates[row,used+partial]=visible-1
            TokenCount[row]=(used+partial+extra)*8-T.if_then_else(extent>0,(8-extent%8)%8,0)

    @T.macro
    def body(WorkValues,WorkEntries,Counts,Lengths,Active,Candidates,TokenCount):
        with T.Kernel(rows,threads=threads) as row:
            pdl.chain()
            tx=T.get_thread_binding()
            keys=T.alloc_shared([n],T.int32)
            hist=T.alloc_shared([257],T.int32)
            scan=T.alloc_shared([threads],T.int32)
            params=T.alloc_shared([4],T.int32)
            selected=T.alloc_shared([budget],T.int32)
            selected_keys=T.alloc_shared([budget],T.int32)
            extent=_row_extent(Lengths,Active,row,width*8)
            full=extent//page*(page//8)
            coarse=_row_extent(Counts,Active,row,width)
            if full<=budget:
                for i in T.Parallel(capacity):
                    Candidates[row,i]=T.if_then_else(i<T.ceildiv(extent,8),i,-1)
                if tx==0:TokenCount[row]=extent
            elif raw:
                for i in T.serial(n//threads):
                    j=i*threads+tx
                    keys[j]=T.if_then_else(j<coarse,_key16(WorkValues[row,T.min(j,width-1)]),0)
                T.sync_threads()
                threshold(keys,n,tx,hist,scan,params)
                compact(keys,n//threads,tx,scan,params,selected,selected_keys)
                total=scan[threads-1]
                write(Candidates,TokenCount,row,extent,lambda slot:slot<total,
                      lambda slot:selected[slot],params)
            else:
                live=T.min(T.ceildiv(coarse,span),lists)
                if live==1:
                    write(Candidates,TokenCount,row,extent,lambda slot:WorkEntries[row,slot]>=0,
                          lambda slot:WorkEntries[row,slot],params)
                elif live>1:
                    for i in T.serial(n//threads):
                        j=i*threads+tx
                        c=T.min(j,columns-1)
                        keys[j]=T.if_then_else((j<columns)&(j//budget<live),
                            T.if_then_else(WorkEntries[row,c]>=0,_key16(WorkValues[row,c]),0),0)
                    T.sync_threads()
                    threshold(keys,n,tx,hist,scan,params)
                    compact(keys,n//threads,tx,scan,params,selected,selected_keys)
                    total=scan[threads-1]
                    write(Candidates,TokenCount,row,extent,lambda slot:slot<total,
                          lambda slot:WorkEntries[row,selected[slot]],params)
                else:
                    write(Candidates,TokenCount,row,extent,lambda slot:False,lambda slot:0,params)

    if raw:
        @T.prim_func
        def run(Scores:T.Tensor([rows,width],T.bfloat16),Counts:T.Tensor([rows],T.int32),
                Lengths:T.Tensor([rows],T.int32),Active:T.Tensor([1],T.int32),
                Candidates:T.Tensor([rows,capacity],T.int32),TokenCount:T.Tensor([rows],T.int32)):
            body(Scores,None,Counts,Lengths,Active,Candidates,TokenCount)
        return run

    @T.prim_func
    def run(WorkValues:T.Tensor([rows,columns],T.bfloat16),WorkEntries:T.Tensor([rows,columns],T.int32),
            Counts:T.Tensor([rows],T.int32),Lengths:T.Tensor([rows],T.int32),Active:T.Tensor([1],T.int32),
            Candidates:T.Tensor([rows,capacity],T.int32),TokenCount:T.Tensor([rows],T.int32)):
        body(WorkValues,WorkEntries,Counts,Lengths,Active,Candidates,TokenCount)
    return run


class FusedCoarse:
    def __init__(self,width,page,budget):
        self.width,self.page,self.budget=width,page,budget
        segments,first,levels=IndexerTopK.routes(width,budget)
        self.pairs=budget==4096 and segments>1
        self.raw=segments==1 or (self.pairs and len(levels)==1)
        self.middle=[]
        if self.raw:
            self.last=finish(width,page,budget,raw=True)
        else:
            self.first=raw_pair(width) if self.pairs else _make(first)
            self.initial_lists=levels[0][2] if self.pairs else segments
            self.middle=[(_make(kernel),lists,out) for kernel,lists,out in levels[1 if self.pairs else 0:-1]]
            _,args,_=levels[-1][0]
            self.last=finish(width,page,budget,lists=args[3],span=args[4])

    def scratch_specs(self,rows):
        if self.raw:return ()
        counts=[self.initial_lists]
        if self.middle:counts.append(self.middle[0][2])
        return tuple(((rows*count*self.budget,),dtype) for count in counts for dtype in (torch.bfloat16,torch.int32))

    def run(self,scores,counts,lengths,active,candidates,token_count,*,scratch):
        if self.raw:return self.last(scores,counts,lengths,active,candidates,token_count)
        rows=len(scores)
        buffers=[scratch[i:i+2] for i in range(0,len(scratch),2)]
        def views(buffer,lists):return tuple(t[:rows*lists*self.budget].view(rows,lists*self.budget) for t in buffer)
        initial=views(buffers[0],self.initial_lists)
        if self.pairs:self.first(scores,counts,active,initial[1],initial[0])
        else:self.first(scores,counts,active,*initial)
        for level,(kernel,lists,out) in enumerate(self.middle):
            kernel(*views(buffers[level%2],lists),counts,active,*views(buffers[(level+1)%2],out))
        last_lists=self.middle[-1][2] if self.middle else self.initial_lists
        return self.last(*views(buffers[len(self.middle)%2],last_lists),counts,lengths,active,candidates,token_count)
