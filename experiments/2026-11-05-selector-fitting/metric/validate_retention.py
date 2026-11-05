"""Native-key retention on a separate document; no request or answer claim."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def recall(expected,actual):
    # Both outputs are ascending native IDs. This document has enough visible
    # keys/blocks to fill each list; no padding may be interpreted as a match.
    assert bool((expected >= 0).all()) and bool((actual >= 0).all())
    assert bool((expected[:,1:] > expected[:,:-1]).all()) and bool((actual[:,1:] > actual[:,:-1]).all())
    positions = torch.searchsorted(actual.contiguous(),expected.contiguous())
    safe = positions.clamp(max=actual.shape[1]-1)
    result = ((positions < actual.shape[1]) & (actual.gather(1,safe) == expected)).float().mean(1)
    return {'mean':float(result.mean()),'minimum':float(result.min()),
            'per_query':result.cpu().tolist(),'queries':len(result),'independent_documents':1}


def one(data,metric):
    from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan,CANDIDATE_BLOCKS
    from vllm.models.deepseek_v4_1.tilelang.direct_shortlist import DirectPageShortlistPlan
    from vllm.models.deepseek_v4_1.tilelang.paged_shortlist import summarize_pages,STATE_BYTES
    from vllm.models.deepseek_v4_1.tilelang.metric_summary import summarize_pages_metric
    page,width,rows = data['page_size'],data['width'],len(data['q'])
    source = bool(data['candidate_blocks'])
    pool = torch.zeros(len(data['pool']),page*STATE_BYTES+128,dtype=torch.uint8,device='cuda')[:,:page*STATE_BYTES]
    pool[:,:page*68].copy_(data['pool'].cuda())
    args = [data[k].cuda() for k in ('q','scales','weights')]
    # Captures retain the model's full-capacity table. This compact proxy uses
    # the captured live plan width; page scoring requires its exact table ABI.
    # Native geometry replay above separately validates the original table.
    assert width%page == 0 and data['table'].shape[1] >= width//page
    assert int(data['active'][0]) <= width and int(data['lengths'].max()) <= width
    table = data['table'][:,:width//page].contiguous().cuda()
    args += [pool,table,data['lengths'].cuda(),data['active'].cuda()]
    native = IndexerV41Plan(width,page,candidate_blocks=source)
    options = {}
    if source:
        options = {'candidate_out':torch.empty(rows,CANDIDATE_BLOCKS,dtype=torch.int32,device='cuda'),
                   'candidate_out_lengths':torch.empty(rows,dtype=torch.int32,device='cuda')}
    expected,_ = native.run(*args,**options)
    expected = expected.clone()
    expected_blocks = options['candidate_out'].clone() if source else None
    completed = [i*page+page-1 for i,n in enumerate(data['layout']['written_prefixes']) if n == page]
    slots = torch.tensor(completed,dtype=torch.int64,device='cuda')
    plan = DirectPageShortlistPlan(width,page,candidate_blocks=source,budget=512 if source else 64,row_cap=256)
    scratch = [torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in plan.scratch_specs(rows)]
    output = torch.empty(rows,512,dtype=torch.int32,device='cuda')
    blocks = torch.empty(rows,CANDIDATE_BLOCKS,dtype=torch.int32,device='cuda')
    counts = torch.empty(rows,dtype=torch.int32,device='cuda')
    report = {}
    for name in ('original','projected'):
        if name == 'original':summarize_pages(page)(pool,pool.view(torch.bfloat16),slots)
        else:summarize_pages_metric(page)(pool,pool.view(torch.bfloat16),slots,metric)
        plan.run(*args,out=output,candidate_out=blocks,candidate_out_lengths=counts,scratch=scratch)
        report[name] = {'native_top512_key_recall':recall(expected,output)}
        if source:report[name]['native_source_block_recall'] = recall(expected_blocks,blocks)
    return {'layer':data['capture_context']['layer_id'],'case_id':data['case_id'],'rows':rows,
            'page_size':page,'planned_width':width,'min_visible_keys':int(data['lengths'].min()),
            'max_visible_keys':int(data['lengths'].max()),'dense_budget':64,'source_budget':512,
            'scope':'compact-address untimed quality proxy; native geometry separately replayed; final-answer quality untested',
            'variants':report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--replay-receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    fit = json.loads((args.fit/'fit.json').read_text())
    assert fit['dimension'] == 32 and fit['validation_case'] == 'validation-a'
    assert digest(args.fit/'metrics.pt') == fit['weights_sha256']
    replay = json.loads(args.replay_receipt.read_text());assert replay['passed'] and replay['actual_captures']
    maps = torch.load(args.fit/'metrics.pt',weights_only=True,map_location='cpu')
    paths = sorted(args.inputs.glob('validation-a-rank-0-*.pt'));assert len(paths) == 4
    result = {'complete':False,'weights_sha256':fit['weights_sha256'],'replay_sha256':digest(args.replay_receipt),
              'scope':'one document; adjacent query rows and replicated ranks do not increase independent quality N',
              'model_answer_quality_tested':False,'performance_measured':False,'cases':[]}
    for path in paths:
        assert digest(path) == replay['files'][path.name]['sha256']
        data = torch.load(path,weights_only=True,map_location='cpu')
        assert data['rank'] == 0 and data['case_id'] == 'validation-a' and len(data['q']) == 64
        metric = maps[str(data['capture_context']['layer_id'])]
        assert metric.shape == (128,32) and bool(metric.isfinite().all())
        result['cases'].append(one(data,metric.bfloat16().cuda()))
        args.out.write_text(json.dumps(result,indent=2)+'\n')
    result['complete'] = True
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT '+json.dumps(result),flush=True)


if __name__ == '__main__':main()
