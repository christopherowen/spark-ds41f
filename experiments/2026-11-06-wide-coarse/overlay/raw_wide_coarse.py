# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact top8192 from raw16384-entry tiles; eliminate two select-all stages."""
import tilelang
import tilelang.language as T
import torch
from . import prepare,pdl
from .indexer import IndexerTopK,_make,_row_extent,_key16,_value16,_select_macros,_final_writer

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC:True})
def raw_wide_pair(width:int):
    rows=T.dynamic('rows')
    topk,threads=8192,512
    span=2*topk;per=span//threads;steps=threads.bit_length()-1
    groups=tilelang.cdiv(width,span);final=groups==1
    threshold,_=_select_macros(threads,topk)
    _,write_empty=_final_writer(threads,topk,1)
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
    def body(Scores,Lengths,Active,Indices,Values):
        with T.Kernel(groups,rows,threads=threads) as (g,row):
            pdl.chain()
            tx=T.get_thread_binding()
            keys=T.alloc_shared([span],T.int32)
            hist=T.alloc_shared([257],T.int32)
            scan=T.alloc_shared([threads],T.int32)
            params=T.alloc_shared([4],T.int32)
            extent=_row_extent(Lengths,Active,row,width)
            base=g*span
            if base<extent:
                for i in T.serial(per):
                    j=i*threads+tx
                    keys[j]=T.if_then_else(base+j<extent,
                        _key16(Scores[row,T.min(base+j,width-1)]),0)
                T.sync_threads()
                threshold(keys,span,tx,hist,scan,params)
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
                        Indices[row, g*topk+out_pos] = base+tx*per+v
                        out_pos += 1
                T.sync_threads()
                total = scan[threads-1]
                for i in T.serial(T.ceildiv(topk, threads)):
                    slot = i*threads+tx
                    if (slot < topk) & (slot >= total):
                        Values[row, g*topk+slot] = -T.infinity(T.bfloat16)
                        Indices[row, g*topk+slot] = -1

            elif final:
                write_empty(Indices,Values,None,row)

    @T.prim_func
    def run(Scores:T.Tensor([rows,width],T.bfloat16),Lengths:T.Tensor([rows],T.int32),
            Active:T.Tensor([1],T.int32),Indices:T.Tensor([rows,groups*topk],T.int32),
            Values:T.Tensor([rows,groups*topk],T.bfloat16)):
        body(Scores,Lengths,Active,Indices,Values)
    return run

class RawWideCoarseTopK:
    """Raw tiles replace the all-entry segment pass and first two merges exactly."""
    def __init__(self,width):
        self.width=width
        self.topk=8192
        self.groups=tilelang.cdiv(width,16384)
        segments,first,levels=IndexerTopK.routes(width,self.topk)
        assert segments>=3 and self.groups==levels[1][2]
        self.raw=raw_wide_pair(width)
        self.levels=[(_make(kernel),lists,out) for kernel,lists,out in levels[2:]]

    @staticmethod
    def calls(width):
        _,_,levels=IndexerTopK.routes(width,8192)
        return [prepare.call(raw_wide_pair,width),*(kernel for kernel,_,_ in levels[2:])]

    def scratch_specs(self,rows):
        if self.groups==1:return ()
        widest=(self.groups*8192,self.levels[0][2]*8192)
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
            return tuple(t[:rows*lists*8192].view(rows,lists*8192) for t in buffer)
        first=views(buffers[0],self.groups)
        self.raw(scores,lengths,active,first[1],first[0])
        for level,(kernel,lists,outputs) in enumerate(self.levels):
            work=views(buffers[level%2],lists)
            if outputs==1:kernel(*work,lengths,active,indices,values)
            else:kernel(*work,lengths,active,*views(buffers[(level+1)%2],outputs))
        return indices
