"""Isolated coarse timings on frozen prefill operands; no reference work in timers."""
import argparse,hashlib,json,statistics
from pathlib import Path
import torch
from benchmark_runtime import graph_of,sample
from vllm.models.deepseek_v4_1.tilelang.compact_shortlist import CompactShortlistPlan,summarize_compact
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerTopK
from vllm.models.deepseek_v4_1.tilelang.raw_wide_coarse import RawWideCoarseTopK

p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--replay-receipt',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(1);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
replay=json.loads(args.replay_receipt.read_text());assert replay['passed']
paths=sorted(args.inputs.glob('*-rank-0-source-*.pt'));assert paths
report={'passed':False,'request_performance_measured':False,'model_quality_tested':False,'source_budget':1024,'scope':'exact coarse selection only on captured prefill operands; not actual decode-query, whole attention or request speed','cases':[]}
for path in paths:
    assert sha(path)==replay['files'][path.name]['sha256']
    data=torch.load(path,map_location='cpu',weights_only=True);page,width=data['page_size'],data['width'];assert data['candidate_blocks']
    pool=torch.zeros(len(data['pool']),page*80+128,dtype=torch.uint8,device='cuda')[:,:page*80];pool[:,:page*68].copy_(data['pool'])
    complete=[i for i,n in enumerate(data['layout']['written_prefixes']) if n==page]
    slots=torch.tensor(complete,dtype=torch.int64,device='cuda')*page+page-1;summarize_compact(page)(pool,pool.view(torch.bfloat16),slots)
    means=pool[:,page*68:page*68+(page//8)*68]
    for rows in (1,8,64):
        q,scales,weights=[data[k][:rows].cuda() for k in ('q','scales','weights')]
        table=torch.zeros(rows,width//page,dtype=torch.int32,device='cuda');n=min(table.shape[1],data['table'].shape[1]);table[:,:n].copy_(data['table'][:rows,:n])
        lengths,active=data['lengths'][:rows].cuda(),data['active'].cuda()
        score=CompactShortlistPlan(width,page,candidate_blocks=True,budget=1024).score
        scores=torch.empty(rows,width//8,dtype=torch.bfloat16,device='cuda');counts=torch.empty(rows,dtype=torch.int32,device='cuda')
        score(q.view(torch.int8),scales.view(torch.uint32).reshape(rows,32),weights,means.view(torch.int8),means.view(torch.uint32),table,lengths,active,scores,counts)
        graphs={};buffers={}
        for name,plan in [('generic',IndexerTopK(width//8,8192)),('raw',RawWideCoarseTopK(width//8))]:
            scratch=[torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in plan.scratch_specs(rows)]
            indices=torch.empty(rows,8192,dtype=torch.int32,device='cuda');values=torch.empty(rows,8192,dtype=torch.bfloat16,device='cuda')
            def call(p=plan,s=scratch,o=indices,v=values):p.run(scores,counts,active,o,v,scratch=s)
            graphs[name]=graph_of(call);buffers[name]=(scratch,indices,values)
        torch.testing.assert_close(buffers['generic'][1],buffers['raw'][1],atol=0,rtol=0)
        torch.testing.assert_close(buffers['generic'][2].view(torch.int16),buffers['raw'][2].view(torch.int16),atol=0,rtol=0)
        before=sample(graphs['generic']);candidate=sample(graphs['raw']);after=sample(graphs['generic'])
        old=statistics.median(before['samples_ms']+after['samples_ms'])
        row={'file':path.name,'sha256':sha(path),'rows':rows,'page':page,'width':width,'coarse_visible_range':[int(counts.min()),int(counts.max())],'generic_before':before,'raw':candidate,'generic_after':after,'generic_over_raw':old/candidate['median_ms'],'outputs_exact':True,'coarse_scratch_bytes':{name:sum(x.numel()*x.element_size() for x in b[0]) for name,b in buffers.items()}}
        report['cases'].append(row);print('CASE '+json.dumps(row),flush=True)
        del graphs,buffers,call,plan,scratch,indices,values,q,scales,weights,table,lengths,active,scores,counts,score
        torch.cuda.empty_cache()
    del pool,means,slots,data;torch.cuda.empty_cache()
report['passed']=True;print('RESULT '+json.dumps(report),flush=True)
