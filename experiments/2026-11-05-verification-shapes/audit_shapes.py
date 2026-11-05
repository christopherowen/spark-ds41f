"""Run the source's shape/dispatch logic offline, without importing GPU packages."""
from __future__ import annotations
import argparse, ast, dataclasses, enum, hashlib, json, pathlib, types
from collections import Counter, defaultdict
from itertools import groupby, product

p=argparse.ArgumentParser(); p.add_argument('source',type=pathlib.Path); a=p.parse_args()
ns=dict(globals(),_DENSE_VARLEN_DECODE_MAX_REQS=2,round_up=lambda x,y:-(-x//y)*y)
provenance={}
def extract(relative,names,methods=None):
    path=a.source/relative; body=path.read_text(); provenance[relative]=hashlib.sha256(body.encode()).hexdigest()
    tree=ast.parse(body); selected=[]
    for node in tree.body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:
            if methods and isinstance(node,ast.ClassDef) and node.name == 'CudaGraphManager': node.body=[m for m in node.body if isinstance(m,ast.FunctionDef) and m.name in methods]
            selected.append(node)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
ns['dataclass']=dataclasses.dataclass
extract('vllm/config/compilation.py',{'CUDAGraphMode'})
extract('vllm/v1/worker/gpu/cudagraph_utils.py',{'dense_varlen_decode_shapes','BatchExecutionDescriptor','_is_compatible','CudaGraphManager'},{'_init_candidates','dispatch','_resolve_effective_loras','_maybe_ubatch_twin'})
extract('vllm/models/deepseek_v4_1/nvidia/dspark.py',{'_context_capture_sizes'})
S=types.SimpleNamespace; mode=ns['CUDAGraphMode']; cls=ns['CudaGraphManager']
results=[]
for reqs in (8,16):
 for width in (6,17):
    cap=reqs*6+width-6
    ladder=[1,2,3,4,6,8,*range(12,33,4),*range(40,reqs*6+1,8)]
    ladder=sorted(set([n for n in ladder if n<reqs*6]+[reqs*6,*range(reqs*6//8*8+8,cap,8),cap]))
    cfg=S(num_speculative_tokens=5,num_verification_tokens=width-1,uniform_decode_query_len=6,scheduler_config=S(max_num_seqs=reqs),compilation_config=S(cudagraph_capture_sizes=ladder,max_cudagraph_capture_size=cap),cache_config=S(use_kda_recoverssm=False),speculative_config=S(uses_acceptance_length_adaptation=lambda:False,uses_batch_size_dynamic_speculative_decoding=lambda:False))
    m=cls.__new__(cls)
    m.__dict__.update(vllm_config=cfg,compilation_config=cfg.compilation_config,max_num_reqs=reqs,decode_query_len=width,cudagraph_mode=mode.FULL_AND_PIECEWISE,specialize_full_decode=False,varlen_decode=True,lora_capture_cases=[0],ubatch_runner=None,full_capture_request_sizes=None,single_request_prefill_tokens=0,use_breakable_cg=False,_capture_descs={},_candidates={},_lora_dispatch_map={},_graphs_captured=True)
    m._init_candidates(); ctx=ns['_context_capture_sizes'](cfg,cap)
    rows=[]
    for nr in range(1,reqs+1):
      for nt in range(nr,min(nr*width,cap)+1):
        d=m.dispatch(nr,nt,None,0,max_query_len=min(width,nt-nr+1))
        assert d.cg_mode==mode.FULL,(nr,nt,d)
        context=next(n for n in ctx if n>=nt)
        rows.append(dict(reqs=nr,live=nt,target=d.num_tokens,target_reqs=d.num_reqs,context=context))
        if nr<=2: assert d.num_tokens==nt
    full=[next(r for r in rows if r['reqs']==nr and r['live']==nr*6) for nr in range(1,reqs+1)]
    results.append(dict(request_capacity=reqs,verification_width=width,row_capacity=cap,ladder=ladder,full_graph_count=len(m._capture_descs[mode.FULL]),context_graph_count=len(ctx),piecewise_graph_count=len(m._capture_descs[mode.PIECEWISE]),context_sizes=ctx,shape_count=len(rows),padding_histogram=dict(Counter(r['target']-r['live'] for r in rows)),native_full_depth=full,all_shapes=rows))
print(json.dumps(dict(provenance=provenance,scope='CPU-only dispatch enumeration; not GPU capture, latency, or memory measurements',results=results),indent=2))
