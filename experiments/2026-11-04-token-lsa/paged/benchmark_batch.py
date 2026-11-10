"""Separate complete-query batching benchmark; no reference/validation work."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import torch
import torch.nn.functional as F

from snapshot_layout import restore_layout
from vllm.models.deepseek_v4_1.tilelang.direct_shortlist import DirectPageShortlistPlan
from vllm.models.deepseek_v4_1.tilelang.paged_shortlist import summarize_pages,STATE_BYTES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--row-cap',type=int,required=True)
    options = parser.parse_args()
    result = {'passed':False,'row_cap':options.row_cap,'cases':[],
              'scope':'immutable synthetic repeated-query component; construction separate; not request speed'}
    try:
        path = Path('/snapshots/rank-0-dense-0.pt')
        for extent in (32768,131072):
            data = torch.load(path,map_location='cpu',weights_only=True)
            page,width = int(data['page_size']),int(data['width'])
            old_pages = (int(data['lengths'][0])+page-1)//page
            logical = torch.arange(extent//page)
            compact = data['pool'][data['table'][0,:old_pages].long()]
            data['pool'] = F.pad(compact[logical%old_pages].clone(),(0,page*(STATE_BYTES-68)))
            data['table'] = F.pad(logical.int()[None],(0,width//page-len(logical)))
            data['lengths'] = torch.tensor([extent],dtype=torch.int32)
            data['active'] = torch.tensor([extent],dtype=torch.int32)
            data['layout'] = dict(data['layout'],physical_ids=logical.long(),written_prefixes=[page]*len(logical))
            args,span = restore_layout(data)
            slots = args[4][0,:extent//page].long()*page+page-1
            update = summarize_pages(page)
            torch.cuda.synchronize();began = time.perf_counter()
            update(args[3],args[3].view(torch.bfloat16),slots)
            torch.cuda.synchronize()
            construction_ms = (time.perf_counter()-began)*1000
            rows = 256
            args = [t.repeat_interleave(rows,dim=0) if i in (0,1,2,4,5) else t for i,t in enumerate(args)]
            plan = DirectPageShortlistPlan(width,page,budget=1536,row_cap=options.row_cap)
            scratch = [torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in plan.scratch_specs(rows)]
            out = torch.empty((rows,512),device='cuda',dtype=torch.int32)
            def run():
                plan.run(*args,out=out,scratch=scratch)
            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for _ in range(10):
                graph.replay()
            samples = []
            for _ in range(20):
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):
                    graph.replay()
                end.record();end.synchronize()
                samples.append(start.elapsed_time(end)/10)
            eager = []
            for _ in range(10):
                torch.cuda.synchronize();began = time.perf_counter()
                run()
                torch.cuda.synchronize()
                eager.append((time.perf_counter()-began)*1000)
            case = {'snapshot':path.name,'snapshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'synthetic_repeated_pages_and_queries':True,'rows':rows,'extent':extent,
                    'width':width,'page':page,'budget':1536,'row_cap':options.row_cap,
                    'query_graph_gpu_ms_samples':samples,'query_graph_gpu_ms_median':statistics.median(samples),
                    'query_eager_wall_ms_samples':eager,'query_eager_wall_ms_median':statistics.median(eager),
                    'construction_wall_ms':construction_ms,'construction_required':True,
                    'native_allocator_stride_with_appended_metadata':True,'pool_span_bytes':span,
                    'workspace_bytes':sum(t.numel()*t.element_size() for t in scratch)}
            result['cases'].append(case)
            print(json.dumps(case),flush=True)
            del graph,run,plan,args,scratch,out,data
            torch.cuda.empty_cache()
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
