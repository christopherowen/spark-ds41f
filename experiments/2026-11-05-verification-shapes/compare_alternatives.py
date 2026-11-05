"""Compare capture designs using the actual source dispatch, without a GPU.

Counts use an unweighted feasible-shape census, never serving-frequency weights.
The tile-work count is a structural bound, not a kernel latency prediction.
"""
from __future__ import annotations
import argparse,ast,contextlib,hashlib,io,json,pathlib,runpy,subprocess,sys,types
from collections import Counter
p=argparse.ArgumentParser(); p.add_argument('--base-source',type=pathlib.Path,required=True); p.add_argument('--source',type=pathlib.Path,required=True); a=p.parse_args(); exp=pathlib.Path(__file__).resolve().parent

provenance = {}
for label, source, expected in (
    ("base", a.base_source, "5f926df6d3126266a9c88c82c06544060008c33c"),
    ("selected", a.source, "8dea43f62a149a3aef2be4b24fa1c68e9b33407d"),
):
    record = {"logical_revision": expected, "checkout_head": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(), "files": {}}
    for name in ("vllm/config/compilation.py", "vllm/v1/worker/gpu/cudagraph_utils.py", "vllm/models/deepseek_v4_1/nvidia/dspark.py"):
        data = (source / name).read_bytes()
        assert data == subprocess.check_output(["git", "-C", str(source), "show", expected + ":" + name])
        record["files"][name] = hashlib.sha256(data).hexdigest()
    provenance[label] = record

def namespace(source):
 old=sys.argv; sys.argv=[str(exp/'audit_shapes.py'),str(source)]
 try:
  with contextlib.redirect_stdout(io.StringIO()): data=runpy.run_path(str(exp/'audit_shapes.py'),run_name='__main__')
 finally: sys.argv=old
 return data['ns']

# Use the source's real wide/decode row-tile selector and table.
path=a.source/'vllm/models/deepseek_v4_1/tilelang/gemm.py'; tree=ast.parse(path.read_text()); selected=[]
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'DECODE_ROWS','DECODE_TILE_ROWS'} for t in node.targets): selected.append(node)
 elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=='WIDE_DECODE': selected.append(node)
 elif isinstance(node,ast.FunctionDef) and node.name=='decode_tile_rows': selected.append(node)
route={}; exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),route)
def mma_rows(rows,key):
 tile=route['decode_tile_rows'](rows,route['WIDE_DECODE'][key]); return -(-rows//tile)*tile

def run(strategy,reqs):
 source=a.base_source if strategy in ('base','native_rows_only') else a.source
 ns=namespace(source); ns['_DENSE_VARLEN_DECODE_MAX_REQS']={'dense_c3':3,'dense_c4':4,'dense_all':reqs}.get(strategy,2)
 width=17; cap=reqs*6+11; ladder=sorted(set([1,2,3,4,6,8,*range(12,33,4),*range(40,cap,8),cap]))
 if strategy in ('native_rows_only','native_rows_plus_exact'): ladder=sorted(set(ladder+[n*6 for n in range(1,reqs+1)]))
 if strategy=='all_row_ladder': ladder=list(range(1,cap+1))
 S=types.SimpleNamespace; mode=ns['CUDAGraphMode']; cls=ns['CudaGraphManager']
 cfg=S(num_speculative_tokens=5,num_verification_tokens=16,uniform_decode_query_len=6,scheduler_config=S(max_num_seqs=reqs),compilation_config=S(cudagraph_capture_sizes=ladder,max_cudagraph_capture_size=cap),cache_config=S(use_kda_recoverssm=False),speculative_config=S(uses_acceptance_length_adaptation=lambda:False,uses_batch_size_dynamic_speculative_decoding=lambda:False))
 m=cls.__new__(cls); m.__dict__.update(vllm_config=cfg,compilation_config=cfg.compilation_config,max_num_reqs=reqs,decode_query_len=width,cudagraph_mode=mode.FULL_AND_PIECEWISE,specialize_full_decode=False,varlen_decode=True,lora_capture_cases=[0],ubatch_runner=None,full_capture_request_sizes=None,single_request_prefill_tokens=0,use_breakable_cg=False,_capture_descs={},_candidates={},_lora_dispatch_map={},_graphs_captured=True)
 m._init_candidates(); ctx=ns['_context_capture_sizes'](cfg,cap); shapes=[]
 for nr in range(1,reqs+1):
  for nt in range(nr,min(nr*width,cap)+1):
   d=m.dispatch(nr,nt,None,0,max_query_len=min(width,nt-nr+1)); assert d.cg_mode==mode.FULL
   c=next(n for n in ctx if n>=nt); shapes.append(dict(requests=nr,live=nt,target=d.num_tokens,target_requests=d.num_reqs,context=c))
 native=[next(r for r in shapes if r['requests']==nr and r['live']==nr*6) for nr in range(1,reqs+1)]
 return dict(strategy=strategy,request_capacity=reqs,target_graphs=len(m._capture_descs[mode.FULL]),context_graphs=len(ctx),feasible_shapes=len(shapes),exact_rows=sum(r['live']==r['target'] for r in shapes),exact_rows_and_requests=sum(r['live']==r['target'] and r['requests']==r['target_requests'] for r in shapes),sum_padded_target_rows=sum(r['target']-r['live'] for r in shapes),sum_padded_requests=sum(r['target_requests']-r['requests'] for r in shapes),sum_padded_context_rows=sum(r['context']-r['live'] for r in shapes),worst_target_padding=max(r['target']-r['live'] for r in shapes),native_full_depth=native,shapes=shapes)

results=[]; bounds=[]
for reqs in (8,16):
 group=[run(s,reqs) for s in ('base','selected','native_rows_only','native_rows_plus_exact','dense_c3','dense_c4','all_row_ladder','dense_all')]; base=group[0]; picked=group[1]
 for key in ((448,5120),(608,5120),(5120,576),(5120,768)):
  diffs=[mma_rows(o['target'],key)-mma_rows(n['target'],key) for o,n in zip(base['shapes'],picked['shapes'])]
  bounds.append(dict(request_capacity=reqs,projection=list(key),changed_mma_row_tiles=sum(d!=0 for d in diffs),sum_saved_mma_rows=sum(diffs),shape_count=len(diffs)))
 for r in group: r.pop('shapes'); results.append(r)
print(json.dumps(dict(scope='CPU structural enumeration only; no measured latency, throughput, memory, or optimality claim',gemm_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),base_head='5f926df6d3126266a9c88c82c06544060008c33c',source_head='8dea43f62a149a3aef2be4b24fa1c68e9b33407d',provenance=provenance,results=results,selected_mma_bounds=bounds),indent=2))
