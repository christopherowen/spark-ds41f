"""Native replay and finer-summary selection; never a request-speed result."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from metadata import decode
from validate_retention import recall
from score import summary_scores
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan, IndexerTopK, CANDIDATE_BLOCKS
from vllm.models.deepseek_v4_1.tilelang.paged_shortlist import summarize_pages, STATE_BYTES
from vllm.models.deepseek_v4_1.tilelang.direct_shortlist import DirectPageShortlistPlan


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def one(data):
    page,width,rows = data['page_size'],data['width'],len(data['q'])
    source = bool(data['candidate_blocks'])
    assert rows == 64 and width%page == 0
    assert int(data['active'][0]) <= width and int(data['lengths'].max()) <= width
    pool = torch.zeros(len(data['pool']),page*STATE_BYTES+128,dtype=torch.uint8,device='cuda')[:,:page*STATE_BYTES]
    pool[:,:page*68].copy_(data['pool'])
    copied = min(data['table'].shape[1],width//page)
    assert copied*page >= int(data['lengths'].max())
    table = torch.zeros(rows,width//page,dtype=torch.int32,device='cuda')
    table[:,:copied].copy_(data['table'][:,:copied])
    q,scales,weights = [data[k].cuda() for k in ('q','scales','weights')]
    lengths,active = data['lengths'].cuda(),data['active'].cuda()
    args = [q,scales,weights,pool,table,lengths,active]
    complete = torch.tensor([i for i,n in enumerate(data['layout']['written_prefixes']) if n == page],dtype=torch.int64,device='cuda')
    summarize_pages(page)(pool,pool.view(torch.bfloat16),complete*page+page-1)
    native = IndexerV41Plan(width,page,candidate_blocks=source)
    options = {}
    if source:
        options = {'candidate_out':torch.empty(rows,CANDIDATE_BLOCKS,dtype=torch.int32,device='cuda'),
                   'candidate_out_lengths':torch.empty(rows,dtype=torch.int32,device='cuda')}
    expected,native_scores = native.run(*args,**options)
    expected = expected.clone()
    expected_blocks = options['candidate_out'].clone() if source else None
    plan = DirectPageShortlistPlan(width,page,candidate_blocks=source,budget=512 if source else 64,row_cap=256)
    scratch = [torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in plan.scratch_specs(rows)]
    b = dict(zip(plan._specs(rows),scratch))
    b['scores'].zero_()
    out = torch.empty(rows,512,dtype=torch.int32,device='cuda')
    blocks = torch.empty(rows,CANDIDATE_BLOCKS,dtype=torch.int32,device='cuda')
    counts = torch.empty(rows,dtype=torch.int32,device='cuda')
    plan.run(*args,out=out,candidate_out=blocks,candidate_out_lengths=counts,scratch=scratch)
    variants = {'cluster64':{'native_top512_key_recall':recall(expected,out)}}
    if source:variants['cluster64']['native_source_block_recall'] = recall(expected_blocks,blocks)
    old_lengths = b['candidate_lengths'].clone()

    # The generalized scorer must reproduce every old visible BF16 score bit.
    centers64 = pool[:,page*68:page*72].contiguous().view(torch.bfloat16).reshape(len(pool),page//64,128)
    control = torch.zeros_like(b['scores'])
    control_counts = torch.empty_like(b['counts'])
    summary_scores(width,page,64)(q,scales,weights,centers64,table,lengths,active,control,control_counts)
    torch.testing.assert_close(control_counts,b['counts'],atol=0,rtol=0)
    for row,n in enumerate(control_counts.cpu().tolist()):
        torch.testing.assert_close(control[row,:n].view(torch.int16),b['scores'][row,:n].view(torch.int16),atol=0,rtol=0)

    # Means of native 8-key blocks need neither fitted maps nor clustering.
    keys = decode(pool[:,:page*64].reshape(-1,page,64),pool[:,page*64:page*68].reshape(-1,page,4))
    assert bool(keys.isfinite().all())
    keys = keys.reshape(len(pool),page//8,8,128)
    means = torch.zeros(len(pool),page//8,128,dtype=torch.float32,device='cuda')
    for token in range(8):means.add_(keys[:,:,token,:]*.125)
    means = means.bfloat16()
    coarse = torch.zeros(rows,width//8,dtype=torch.bfloat16,device='cuda')
    coarse_counts = torch.empty(rows,dtype=torch.int32,device='cuda')
    summary_scores(width,page,8)(q,scales,weights,means,table,lengths,active,coarse,coarse_counts)
    budget = plan.budget*8
    selector = IndexerTopK(width//8,budget)
    selected = torch.empty(rows,budget,dtype=torch.int32,device='cuda')
    values = torch.empty(rows,budget,dtype=torch.bfloat16,device='cuda')
    selector.run(coarse,coarse_counts,active,selected,values)
    candidate = b['candidates']
    candidate.fill_(-1)
    for row,extent in enumerate(data['lengths'].tolist()):
        ids = selected[row][selected[row] >= 0]
        full = extent//page*(page//8)
        visible = (extent+7)//8
        tail = torch.arange(full,visible,dtype=torch.int32,device='cuda')
        last = torch.tensor([visible-1],dtype=torch.int32,device='cuda')
        ids = torch.cat((ids,tail,last)).unique(sorted=True)
        assert len(ids) <= candidate.shape[1] and bool((ids >= 0).all())
        candidate[row,:len(ids)] = ids
        b['candidate_lengths'][row] = len(ids)*8-(8-extent%8)%8
    delta = b['candidate_lengths']-old_lengths
    assert bool((delta.abs() <= 8).all()), 'keep fine work equal within the forced final block'

    def buffers(name,subplan):return [b[f'{name}_{i}'] for i in range(len(subplan.scratch_specs(rows)))]
    _,fine_scores = plan.final.run(q,scales,weights,pool,table,lengths,active,candidate,b['candidate_lengths'],out=out,
                                  scratch=buffers('final',plan.final))
    # Prove the new candidate assembly retains native arithmetic at every key.
    for row,n in enumerate(b['candidate_lengths'].cpu().tolist()):
        positions = torch.arange(n,device='cuda')
        positions = candidate[row,positions//8].long()*8+positions%8
        assert bool((positions < lengths[row]).all())
        torch.testing.assert_close(fine_scores[row,:n].view(torch.int16),native_scores[row,positions].view(torch.int16),atol=0,rtol=0)
    if source:
        plan.select_blocks.run(fine_scores,b['candidate_lengths'],active,b['compact'],b['block_values'],counts=counts,
                               scratch=buffers('blocks',plan.select_blocks))
        plan.remap(b['compact'],candidate,blocks)
    variants['block8'] = {'native_top512_key_recall':recall(expected,out)}
    if source:variants['block8']['native_source_block_recall'] = recall(expected_blocks,blocks)
    return {'case_id':data['case_id'],'layer':data['capture_context']['layer_id'],'source_interface':source,
            'rows':rows,'page_size':page,'planned_width':width,'min_visible_keys':int(lengths.min()),
            'max_visible_keys':int(lengths.max()),'retained_native_blocks':budget,
            'old_coarse_bf16_bits_equal':True,'all_retained_native_score_bits_equal':True,
            'fine_work_difference_max_keys':int(delta.abs().max()),'variants':variants}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--replay-receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    replay = json.loads(args.replay_receipt.read_text());assert replay['passed'] and replay['actual_captures']
    paths = sorted(args.inputs.glob('*-rank-0-*.pt'));assert len(paths) == 12
    torch.cuda.reset_peak_memory_stats()
    result = {'complete':False,'replay_sha256':digest(args.replay_receipt),'model_answer_quality_tested':False,
              'performance_measured':False,'scope':'untimed prefill proxy; three documents; reused diagnostic data; no serving change',
              'prospective_summary_bytes_per_key':32,'prospective_native_plus_summary_bytes_per_key':100,'files':{}}
    try:
        for path in paths:
            assert digest(path) == replay['files'][path.name]['sha256']
            data = torch.load(path,map_location='cpu',weights_only=True)
            assert data['rank'] == 0 and data['mode'] == 'prefill'
            result['files'][path.name] = one(data)
            args.out.write_text(json.dumps(result,indent=2)+'\n')
            print('CASE '+json.dumps({'file':path.name,'complete':True}),flush=True)
        result['complete'] = True
    except BaseException as error:
        result['error'] = {'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        result['memory_observations'] = {'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                         'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                                         'scope':'standalone tensor allocator, not model memory qualification'}
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        print('RESULT '+json.dumps(result),flush=True)


if __name__ == '__main__':main()
