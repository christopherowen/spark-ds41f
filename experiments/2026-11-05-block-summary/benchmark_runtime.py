"""Private captured operands; performance only, with no reference work in timers."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch

from vllm.models.deepseek_v4_1.tilelang.block_shortlist import BlockMeanShortlistPlan,summarize_blocks
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan,CANDIDATE_BLOCKS


def graph_of(call):
    for _ in range(3):call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call()
    for _ in range(5):graph.replay()
    torch.cuda.synchronize()
    return graph


def sample(graph,repeats=10,samples=12):
    values = []
    for _ in range(samples):
        begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(repeats):graph.replay()
        end.record();end.synchronize()
        values.append(begin.elapsed_time(end)/repeats)
    return {'samples_ms':values,'median_ms':statistics.median(values),'replays_per_sample':repeats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--replay-receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stages',action='store_true',help='separate diagnostic stage timing after whole-plan timing')
    args = parser.parse_args()
    assert not args.out.exists()
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    replay = json.loads(args.replay_receipt.read_text());assert replay['passed']
    result = {'passed':False,'actual_captures':True,'request_performance_measured':False,
              'scope':'captured prefill queries at row counts 1/8/64; original physical IDs; native68 versus block8 layout100; CUDA graph kernel timings, no final-answer inference',
              'replay_sha256':sha(args.replay_receipt),'cases':[]}
    paths = sorted(args.inputs.glob('validation-a-rank-0-*.pt'))
    assert len(paths) == 4
    torch.cuda.reset_peak_memory_stats()
    for path in paths:
        assert sha(path) == replay['files'][path.name]['sha256']
        data = torch.load(path,map_location='cpu',weights_only=True)
        page,width = data['page_size'],data['width']
        native_pool = data['pool'].cuda()
        pool = torch.empty(len(native_pool),page*100+128,dtype=torch.uint8,device='cuda')[:,:page*100]
        pool[:,:page*68].copy_(native_pool)
        complete = [i for i,n in enumerate(data['layout']['written_prefixes']) if n == page]
        slots = torch.tensor(complete,dtype=torch.int64,device='cuda')*page+page-1
        update = summarize_blocks(page)
        update(pool,pool.view(torch.bfloat16),slots)
        single = slots[len(slots)//2:len(slots)//2+1]
        for rows in (1,8,64):
            q,scales,weights = [data[k][:rows].cuda() for k in ('q','scales','weights')]
            table = torch.zeros(rows,width//page,dtype=torch.int32,device='cuda')
            copied = min(table.shape[1],data['table'].shape[1]);table[:,:copied].copy_(data['table'][:rows,:copied])
            lengths,active = data['lengths'][:rows].cuda(),data['active'].cuda()
            source = bool(data['candidate_blocks'])
            plans = {'native':IndexerV41Plan(width,page,candidate_blocks=source),
                     'block8':BlockMeanShortlistPlan(width,page,candidate_blocks=source)}
            graphs = {}
            buffers = {}
            for name,plan in plans.items():
                scratch = [torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in plan.scratch_specs(rows)]
                out = torch.empty(rows,512,dtype=torch.int32,device='cuda')
                blocks = torch.empty(rows,CANDIDATE_BLOCKS,dtype=torch.int32,device='cuda')
                counts = torch.empty(rows,dtype=torch.int32,device='cuda')
                options = {'candidate_out':blocks,'candidate_out_lengths':counts} if source else {}
                storage = native_pool if name == 'native' else pool
                buffers[name] = (scratch,out,blocks,counts)
                def call(p=plan,s=scratch,o=out,opt=options,storage=storage):
                    p.run(q,scales,weights,storage,table,lengths,active,out=o,scratch=s,**opt)
                graphs[name] = graph_of(call)
            before = sample(graphs['native'])
            candidate = sample(graphs['block8'])
            after = sample(graphs['native'])
            native_ms = statistics.median(before['samples_ms']+after['samples_ms'])
            row = {'file':path.name,'sha256':sha(path),'page':page,'width':width,'rows':rows,
                   'layer':data['capture_context']['layer_id'],'source':source,
                   'visible_range':[int(lengths.min()),int(lengths.max())],
                   'native_before':before,'block8':candidate,'native_after':after,
                   'native_over_block8':native_ms/candidate['median_ms']}
            if args.stages:
                plan = plans['block8']
                scratch = buffers['block8'][0]
                b = dict(zip(plan._specs(rows),scratch))
                def stage_scratch(name,p):return [b[f'{name}_{i}'] for i in range(len(p.scratch_specs(rows)))]
                calls = {
                    'mean_scores':lambda:plan.score(q.view(torch.uint8),scales.view(torch.uint8).reshape(rows,32,4),weights,pool.view(torch.bfloat16),table,lengths,active,b['scores'],b['counts']),
                    'coarse_topk':lambda:plan.coarse.run(b['scores'],b['counts'],active,b['selected'],b['values'],scratch=stage_scratch('coarse',plan.coarse)),
                    'assemble':lambda:plan.assemble(b['selected'],lengths,active,b['candidates'],b['candidate_lengths']),
                    'fine_score_and_top512':lambda:plan.final.run(q,scales,weights,pool,table,lengths,active,b['candidates'],b['candidate_lengths'],out=buffers['block8'][1],scratch=stage_scratch('final',plan.final))}
                if source:
                    calls.update(source_top2048=lambda:plan.select_blocks.run(b['final_0'],b['candidate_lengths'],active,b['compact'],b['block_values'],counts=buffers['block8'][3],scratch=stage_scratch('blocks',plan.select_blocks)),
                                 source_remap=lambda:plan.remap(b['compact'],b['candidates'],buffers['block8'][2]))
                row['diagnostic_stages'] = {}
                for name,call_stage in calls.items():
                    stage_graph = graph_of(call_stage)
                    row['diagnostic_stages'][name] = sample(stage_graph)
                    del stage_graph
                del call_stage,calls,stage_scratch,b,plan
            result['cases'].append(row)
            args.out.write_text(json.dumps(result,indent=2)+'\n')
            print('CASE '+json.dumps(row),flush=True)
            del call,graphs,buffers,plans,scratch,out,blocks,counts,q,scales,weights,table,lengths,active
            torch.cuda.empty_cache()
        update_graph = graph_of(lambda:update(pool,pool.view(torch.bfloat16),single))
        all_graph = graph_of(lambda:update(pool,pool.view(torch.bfloat16),slots))
        construction = {'single_complete_page':sample(update_graph),'all_complete_pages':sample(all_graph),
                        'complete_pages':len(slots),'page':page,'file':path.name,
                        'scope':'page-completion maintenance, measured separately from queries; not amortized away'}
        result.setdefault('construction',[]).append(construction)
        print('CONSTRUCTION '+json.dumps(construction),flush=True)
        del update_graph,all_graph,pool,native_pool,slots,single,data
        torch.cuda.empty_cache()
    result.update(passed=True,memory_observations={'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                                   'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT '+json.dumps(result),flush=True)


if __name__ == '__main__':main()
