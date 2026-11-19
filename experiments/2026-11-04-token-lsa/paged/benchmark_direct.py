"""Separate compaction performance process. No validation/reference execution."""
import argparse
import hashlib
import json
import statistics
import traceback

import torch

from vllm.models.deepseek_v4_1.tilelang.direct_shortlist import direct_candidates
from vllm.models.deepseek_v4_1.tilelang.paged_shortlist import STATE_BYTES,page_flags
from vllm.models.deepseek_v4_1.tilelang.shortlist_indexer import shortlist_prefix,shortlist_scatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['scan','direct'],required=True)
    mode = parser.parse_args().mode
    result = {'passed':False,'mode':mode,'cases':[],
              'scope':'synthetic compaction component only; no scorer, index maintenance or request speed claim'}
    try:
        width,rows = 1048576,8
        for page,budget in ((128,64),(256,512)):
            for extent in (32768,131072,524288):
                generator = torch.Generator(device='cuda').manual_seed(page+extent)
                pages,blocks = width//page,page//8
                pool = torch.empty((rows*pages+1,page*STATE_BYTES+128),dtype=torch.uint8,device='cuda')[:,:page*STATE_BYTES]
                table = (torch.randperm(rows*pages,device='cuda',generator=generator)+1).reshape(rows,pages).int()
                membership = torch.rand((len(pool),blocks),device='cuda',generator=generator).argsort(dim=1).argsort(dim=1)//8
                pool[:,page*72:page*72+blocks] = membership.to(torch.uint8)
                n = min(budget,extent//64)
                selected = torch.full((rows,budget),-1,device='cuda',dtype=torch.int32)
                selected[:,:n] = torch.stack([torch.randperm(extent//64,device='cuda',generator=generator)[:n].sort().values.int() for _ in range(rows)])
                lengths = torch.full((rows,),extent,device='cuda',dtype=torch.int32)
                active = torch.tensor([width],device='cuda',dtype=torch.int32)
                candidates = torch.empty((rows,(budget+8)*8),device='cuda',dtype=torch.int32)
                candidate_lengths = torch.empty_like(lengths)
                if mode == 'scan':
                    local = torch.empty((rows,width//8),device='cuda',dtype=torch.int32)
                    tiles = (width//8+255)//256
                    counts = torch.empty((rows,tiles),device='cuda',dtype=torch.int32)
                    bases = torch.empty_like(counts)
                    flags,prefix,scatter = page_flags(width,page,budget),shortlist_prefix(width,budget),shortlist_scatter(width,budget)
                    def run():
                        flags(selected,pool,table,lengths,active,local,counts)
                        prefix(counts,lengths,active,bases,candidates,candidate_lengths)
                        scatter(local,bases,lengths,active,candidates)
                else:
                    compact = direct_candidates(width,page,budget)
                    def run():
                        compact(selected,pool,table,lengths,active,candidates,candidate_lengths)
                for _ in range(3):
                    run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                for _ in range(10):
                    graph.replay()
                samples = []
                for _ in range(20):
                    begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(50):
                        graph.replay()
                    end.record();end.synchronize()
                    samples.append(begin.elapsed_time(end)/50)
                c = {'width':width,'extent':extent,'page':page,'budget':budget,'rows':rows,
                     'samples_ms':samples,'median_ms':statistics.median(samples),
                     'synthetic_layout':True,'full_selected_budget':n == budget,
                     'query_launches':3 if mode == 'scan' else 1,
                     'input_sha256':hashlib.sha256(table.cpu().numpy().tobytes()+
                         membership.cpu().numpy().tobytes()+selected.cpu().numpy().tobytes()).hexdigest()}
                result['cases'].append(c)
                print(json.dumps(c),flush=True)
                del graph,run,pool,table,membership,selected,candidates
                if mode == 'scan':
                    del local,counts,bases
                torch.cuda.empty_cache()
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
