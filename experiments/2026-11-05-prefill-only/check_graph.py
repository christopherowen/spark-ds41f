"""Out-of-band graph replay of the actual phase-dispatch method on captured inputs."""
import argparse,ast,hashlib,json,os,types
from pathlib import Path
import torch
from benchmark_runtime import graph_of
from vllm.models.deepseek_v4_1.tilelang.compact_shortlist import CompactShortlistPlan,summarize_compact
from vllm.models.deepseek_v4_1.tilelang.paged_shortlist import PageShortlistPlan
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan,CANDIDATE_BLOCKS


def method(path,workspace):
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DeepseekV41TileLangAttention')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_index_chunk')
    scope={'os':os,'PageShortlistPlan':PageShortlistPlan,'current_workspace_manager':lambda:workspace,'retain_cuda_graph_capture_resource':lambda *args:None}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
    return scope['_index_chunk']


class Workspace:
    def __init__(self,plans,rows):
        self.buffers={}
        for plan in plans:
            specs=plan.scratch_specs(rows);key=tuple(specs)
            self.buffers[key]=[torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in specs]
    def get_simultaneous(self,*specs):return self.buffers[tuple(specs)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--replay-receipt',type=Path,required=True);args=p.parse_args()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();replay=json.loads(args.replay_receipt.read_text());assert replay['passed']
    os.environ['VLLM_DS41_PAGE_NATIVE_DECODE']='1';os.environ['VLLM_DS41_PAGE_NATIVE_SOURCE']='1'
    report={'passed':False,'performance_measured':False,'model_quality_tested':False,'scope':'captured prefill operands test graph dispatch correctness; not decode-query or request performance evidence','cases':[]}
    paths=sorted(args.inputs.glob('*-rank-0-*.pt'));assert len(paths)==12
    torch.cuda.reset_peak_memory_stats()
    for path in paths:
        assert sha(path)==replay['files'][path.name]['sha256'];data=torch.load(path,map_location='cpu',weights_only=True)
        page,width,source=data['page_size'],data['width'],bool(data['candidate_blocks'])
        raw=data['pool'].cuda();pool=torch.zeros(len(raw),page*80+128,dtype=torch.uint8,device='cuda')[:,:page*80];pool[:,:page*68].copy_(raw)
        complete=[i for i,n in enumerate(data['layout']['written_prefixes']) if n==page]
        slots=torch.tensor(complete,dtype=torch.int64,device='cuda')*page+page-1;summarize_compact(page)(pool,pool.view(torch.bfloat16),slots)
        for rows in (1,8,64):
            q,s,w=[data[k][:rows].cuda() for k in ('q','scales','weights')]
            table=torch.zeros(rows,width//page,dtype=torch.int32,device='cuda');n=min(table.shape[1],data['table'].shape[1]);table[:,:n].copy_(data['table'][:rows,:n])
            lengths,active=data['lengths'][:rows].cuda(),data['active'].cuda();saved_lengths=lengths.clone();saved_active=active.clone()
            native=IndexerV41Plan(width,page,candidate_blocks=source)
            prefill=native if source else CompactShortlistPlan(width,page,row_cap=256)
            workspace=Workspace([native,prefill],rows);dispatch=method(Path('/tmp/phase_attention.py'),workspace)
            plans=types.SimpleNamespace(index={'decode':prefill,'prefill':prefill},native_index={} if source else {'decode':native,'prefill':native})
            layer=types.SimpleNamespace(_tilelang=lambda:plans,indexer=types.SimpleNamespace(k_cache=types.SimpleNamespace(kv_cache=pool)),_active=active)
            for mode in ('decode','prefill'):
                out=torch.empty(rows,512,dtype=torch.int32,device='cuda');blocks=torch.empty(rows,CANDIDATE_BLOCKS,dtype=torch.int32,device='cuda');count=torch.empty(rows,dtype=torch.int32,device='cuda')
                def call():
                    dispatch(layer,mode,rows,output_indices=out,q_mxfp4=q,q_scales=s,query_weights=w,page_table=table,cache_lengths=lengths,candidate_output=blocks if source else None,candidate_output_lengths=count if source else None)
                graph=graph_of(call)
                teacher=native if mode=='decode' or source else prefill
                expected=torch.empty_like(out);expected_blocks=torch.empty_like(blocks);expected_count=torch.empty_like(count)
                scratch=[torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in teacher.scratch_specs(rows)]
                options={'candidate_out':expected_blocks,'candidate_out_lengths':expected_count} if source else {}
                for cap in ('original','empty','restored'):
                    if cap=='empty':lengths.zero_();active.zero_()
                    else:lengths.copy_(saved_lengths);active.copy_(saved_active)
                    teacher.run(q,s,w,pool,table,lengths,active,out=expected,scratch=scratch,**options)
                    graph.replay();torch.cuda.synchronize()
                    torch.testing.assert_close(out,expected,atol=0,rtol=0)
                    if source:
                        torch.testing.assert_close(blocks,expected_blocks,atol=0,rtol=0);torch.testing.assert_close(count,expected_count,atol=0,rtol=0)
                report['cases'].append({'file':path.name,'sha256':sha(path),'rows':rows,'source':source,'mode':mode,'native_selected':mode=='decode' or source,'exact_expected_outputs':True,'empty_and_restored_replays':True})
                del graph,call
        print('CASE '+json.dumps({'file':path.name,'completed_routes':6}),flush=True)
    assert len(report['cases'])==72;report['passed']=True
    report['memory']={'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved()}
    print('RESULT '+json.dumps(report),flush=True)

if __name__=='__main__':main()
