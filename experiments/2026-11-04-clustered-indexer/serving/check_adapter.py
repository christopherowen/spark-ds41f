"""Serving-adapter gates: mixed owners, reused workspace and row-cap boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import torch
from check import inputs
from snapshot_layout import restore_layout
from vllm.models.deepseek_v4_1.tilelang.clustered_serving import ServingClusteredIndexerPlan
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan, CANDIDATE_BLOCKS


def check(args,width,page,source,label,cuda_graph=True):
    rows=args[0].shape[0]
    native=IndexerV41Plan(width,page,candidate_blocks=source)
    plan=ServingClusteredIndexerPlan(width,page,candidate_blocks=source)
    scratch=[torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in plan.scratch_specs(rows)]
    out=torch.empty((rows,512),dtype=torch.int32,device='cuda')
    blocks=torch.empty((rows,CANDIDATE_BLOCKS),dtype=torch.int32,device='cuda')
    counts=torch.empty((rows,),dtype=torch.int32,device='cuda')
    expected_blocks=torch.empty_like(blocks);expected_counts=torch.empty_like(counts)
    def reference():
        return native.run(*args,**({'candidate_out':expected_blocks,'candidate_out_lengths':expected_counts} if source else {}))[0]
    def run():
        plan.run(*args,out=out,scratch=scratch,
                 **({'candidate_out':blocks,'candidate_out_lengths':counts} if source else {}))
    def equal():
        expected=reference();run()
        torch.testing.assert_close(out,expected,atol=0,rtol=0)
        if source:
            torch.testing.assert_close(blocks,expected_blocks,atol=0,rtol=0)
            torch.testing.assert_close(counts,expected_counts,atol=0,rtol=0)
        return None
    equal()
    graph=None
    if cuda_graph:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run()
    lengths,active=args[5].clone(),args[6].clone()
    for kind in ['full','half','empty','restore']:
        args[5].copy_(lengths//2 if kind=='half' else lengths)
        args[6].copy_(torch.zeros_like(active) if kind=='empty' else active)
        expected=reference()
        if graph is None:run()
        else:graph.replay()
        torch.testing.assert_close(out,expected,atol=0,rtol=0)
        if source:
            torch.testing.assert_close(blocks,expected_blocks,atol=0,rtol=0)
            torch.testing.assert_close(counts,expected_counts,atol=0,rtol=0)
    return {'label':label,'width':width,'page_size':page,'rows':rows,'source':source,
            'indices_identical':True,'source_handoff_identical':True if source else None,
            'reused_workspace_visibility_identical':True,'execution':'cuda-graph' if cuda_graph else 'eager',
            'workspace_bytes':sum(x.numel()*x.element_size() for x in scratch),
            'output_sha256':hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');parser.add_argument('--real-only',action='store_true')
    parser.add_argument('--no-graph',action='store_true')
    parser.add_argument('--freeze-declared',action='store_true')
    options=parser.parse_args();result={'passed':False,'cases':[]}
    try:
        if options.freeze_declared:
            import tilelang
            from tilelang.cache import freeze
            calls=[]
            shapes=[(16384,page,False) for page in [128,256]]+[(65536,page,True) for page in [128,256]]
            for path in sorted(Path('/snapshots').glob('rank-0-*.pt')):
                data=torch.load(path,map_location='cpu',weights_only=True)
                shapes.append((data['width'],data['page_size'],data['candidate_blocks']))
            for width,page,source in set(shapes):
                calls.extend(ServingClusteredIndexerPlan.calls(width,page,candidate_blocks=source))
                calls.extend(IndexerV41Plan.calls(width,page,candidate_blocks=source))
            tilelang.prepare(calls)
            freeze('serving adapter declaration gate')
            result['declared_calls_frozen']=True
        if not options.real_only:
            for page in [128,256]:
                for rows in ([9] if options.quick else [1,8,9,17]):
                    for source in [False,True]:
                        width=65536 if source else 16384
                        extents=[width-1-(r*7)%512 for r in range(rows)]
                        extents[0]=0
                        if rows>1:extents[1]=37
                        args=inputs(width,extents,'random',page,unwritten_cache=True)
                        result['cases'].append(check(args,width,page,source,'mixed-owner random',not options.no_graph))
                        print(json.dumps(result['cases'][-1]),flush=True)
        if not options.quick:
            for path in sorted(Path('/snapshots').glob('rank-0-*.pt')):
                data=torch.load(path,map_location='cpu',weights_only=True)
                args,_=restore_layout(data)
                result['cases'].append(check(args,data['width'],data['page_size'],data['candidate_blocks'],path.name,not options.no_graph))
                print(json.dumps(result['cases'][-1]),flush=True)
        if not result['cases']:raise RuntimeError('no adapter cases executed')
        result['passed']=True
    except Exception:
        result['error']=traceback.format_exc();raise
    finally:print('RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':main()
