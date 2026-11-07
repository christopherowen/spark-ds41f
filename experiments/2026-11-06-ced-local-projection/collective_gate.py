"""Real TP3 CED byte and projection-factory gate; synthetic inputs, no timing claim."""
import argparse
import datetime as dt
import json
import resource
import os
import types

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--rank',type=int,required=True)
parser.add_argument('--master',required=True)
parser.add_argument('--port',type=int,required=True)
args=parser.parse_args()
assert args.rank in (0,1,2)

import torch
import tilelang
from vllm.distributed import get_tp_group
from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment
from vllm.models.deepseek_v4_1 import ced
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
from vllm.models.deepseek_v4_1.sp_prefill import SPRows
from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang.linear import TileLangSplitLinearMethod

WORLD=3
HIDDEN=5120
SEED=20261106
records=[]

def equal_bytes(actual,expected,label):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype,label
    assert torch.equal(actual.contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8)),label

def tensor(rows,tail,dtype,generator,raw):
    shape=(rows,*tail)
    if raw:
        bits=torch.int16 if dtype==torch.bfloat16 else torch.int32
        low,high=(-32768,32768) if bits==torch.int16 else (-2**31,2**31-1)
        return torch.randint(low,high,shape,dtype=bits,device='cuda',generator=generator).view(dtype)
    return torch.randn(shape,device='cuda',generator=generator,dtype=torch.float32).to(dtype)

def row_gate(tokens,raw):
    sp=SPRows(tokens,WORLD,args.rank)
    generator=torch.Generator(device='cuda').manual_seed(SEED+tokens+int(raw))
    full=[tensor(tokens,tail,dtype,generator,raw) for tail,dtype in
          [((HIDDEN,),torch.bfloat16),((4,HIDDEN),torch.bfloat16),
           ((4,),torch.float32),((4,4),torch.float32),((4,),torch.float32)]]
    indices=torch.linspace(0,tokens-1,128,device='cuda').long()
    indices[0]=-1;indices[1]=tokens;indices[2]=tokens+7;indices[3]=indices[4]
    local=[sp.local_slice(value) for value in full]
    actual=ced.gather_sp_rows(sp,indices,*local)
    valid=(indices>=0)&(indices<tokens)
    for index,(source,result) in enumerate(zip(full,actual)):
        expected=source.new_zeros((indices.numel(),*source.shape[1:]))
        expected[valid]=source[indices[valid]]
        equal_bytes(result,expected,('compact-carry',tokens,raw,index,args.rank))
    empty=ced.gather_sp_rows(sp,indices[:0],*local)
    assert all(value.shape[0]==0 for value in empty)
    records.append({'test':'compact-carry','tokens':tokens,'raw_bits':raw,'tensors':5,'passed':True})
    print(json.dumps({'rank':args.rank,**records[-1]}),flush=True)

def projection_gate(tokens,nonfinite):
    sp=SPRows(tokens,WORLD,args.rank)
    generator=torch.Generator(device='cuda').manual_seed(SEED+tokens)
    weight=torch.randn((512,HIDDEN),device='cuda',generator=generator,dtype=torch.float32).to(torch.bfloat16)
    hidden=torch.randn((tokens,HIDDEN),device='cuda',generator=generator,dtype=torch.float32).to(torch.bfloat16)
    if nonfinite:
        hidden[0,:]=0;hidden[1,:]=-0.0
        hidden[2,0]=float('inf');hidden[3,1]=-float('inf');hidden[-1,2]=float('nan')
    method=TileLangSplitLinearMethod(1)
    layer=types.SimpleNamespace(weight=weight,quant_method=method,tp_size=1,out_dtype=torch.bfloat16)
    compressor=DeepseekCompressor.__new__(DeepseekCompressor)
    torch.nn.Module.__init__(compressor)
    compressor.compress_ratio=1;compressor.fused_wkv_wgate=layer
    native_call=DeepseekCompressor._sp_projection_call(compressor)
    baseline_call=method._kernels(layer)['gemm']
    tilelang.prepare([native_call,baseline_call],num_workers=1)
    native=prepare.launch(native_call);baseline=prepare.launch(baseline_call)
    expected=torch.empty((tokens,512),dtype=torch.bfloat16,device='cuda')
    baseline(hidden,weight,expected)
    local=sp.local_slice(hidden)
    projected=torch.empty((sp.local_rows,512),dtype=torch.bfloat16,device='cuda')
    native(local,weight,projected)
    actual=sp.gather(projected)
    equal_bytes(actual,expected,('factory-projection',tokens,nonfinite,args.rank))
    attention=types.SimpleNamespace(compressor=compressor,hidden_size=HIDDEN)
    compact=DeepseekV4Attention.ced_sp_saves_bytes(attention,sp,128,4)
    if tokens in (385,390):assert compact is False
    if tokens in (391,1025,16384):assert compact is True
    records.append({'test':'factory-projection','tokens':tokens,'nonfinite':nonfinite,
                    'split':method._split(layer),'compact_guard':compact,'passed':True})
    print(json.dumps({'rank':args.rank,**records[-1]}),flush=True)

os.environ['RANK']=str(args.rank)
config=VllmConfig(parallel_config=ParallelConfig(
    tensor_parallel_size=WORLD, distributed_executor_backend='external_launcher'))
with set_current_vllm_config(config):
    torch.cuda.set_device(0)
    try:
        init_distributed_environment(world_size=WORLD,rank=args.rank,local_rank=0,
            distributed_init_method=f'tcp://{args.master}:{args.port}',backend='nccl',timeout=dt.timedelta(seconds=120))
        initialize_model_parallel(tensor_model_parallel_size=WORLD,backend='nccl')
        group=get_tp_group();comm=group.device_communicator.pynccl_comm
        assert group.world_size==WORLD and group.rank_in_group==args.rank
        assert comm is not None and not comm.disabled,'native grouped PyNCCL path is required'
        print(json.dumps({'rank':args.rank,'world':WORLD,'device_communicator':type(group.device_communicator).__name__,
                          'pynccl_enabled':not comm.disabled,'scope':'actual TP3 synthetic correctness; no serving/timing claim'}),flush=True)
        for tokens in (385,390,391,1025,16384):
            for raw in (False,True):row_gate(tokens,raw)
            for nonfinite in (False,True):projection_gate(tokens,nonfinite)
            torch.distributed.barrier()
        torch.cuda.synchronize()
        print(json.dumps({'rank':args.rank,'passed':len(records),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                          'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                          'process_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
