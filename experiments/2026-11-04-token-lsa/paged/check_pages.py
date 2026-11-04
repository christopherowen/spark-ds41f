"""Separate untimed cache lifecycle and shortlist validation process."""
import json
from pathlib import Path
import traceback

import torch

from metadata import decode
from vllm.models.deepseek_v4_1.tilelang.paged_shortlist import (
    PageShortlistPlan, summarize_pages, STATE_BYTES)
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    IndexerV41Plan, CANDIDATE_BLOCKS, page_bytes)


def summaries(pool,page,physical):
    membership = pool[physical,page*72:page*72+page//8].long()
    assert bool((membership < page//64).all())
    for label in range(page//64):
        assert int((membership == label).sum()) == 8
    packed = pool[physical,:page*64].reshape(page,64)
    scales = pool[physical,page*64:page*68].reshape(page,4)
    points = decode(packed,scales).float().reshape(page//8,8,128).mean(1)
    expected = torch.stack([points[membership == label].mean(0) for label in range(page//64)]).bfloat16()
    actual = pool[physical,page*68:page*72].contiguous().view(torch.bfloat16).reshape(page//64,128)
    torch.testing.assert_close(actual.float(),expected.float(),atol=.02,rtol=.02)
    return membership.cpu().tolist()


def case(page,source,rows,extent):
    width = 32768 if source else 8192
    pages = width//page
    g = torch.Generator(device='cuda').manual_seed(page+rows+int(source))
    pool = torch.empty((rows*pages+1,page*STATE_BYTES+128),dtype=torch.uint8,device='cuda')[:,:page*STATE_BYTES]
    pool.fill_(0xa5)
    table = (torch.randperm(rows*pages,device='cuda',generator=g)+1).reshape(rows,pages).int()
    # Different physical caches per owner; no owner deduplication assumption.
    pool[:, :page*64].random_(0,256,generator=g)
    pool[:,page*64:page*68].random_(122,130,generator=g)
    lengths = torch.tensor([max(extent-(i%16)*page//2,0) for i in range(rows)],device='cuda',dtype=torch.int32)
    active = torch.tensor([width],device='cuda',dtype=torch.int32)
    q = torch.randint(0,256,(rows,32,64),device='cuda',dtype=torch.uint8,generator=g)
    scales = torch.randint(122,130,(rows,32,4),device='cuda',dtype=torch.uint8,generator=g)
    weights = (torch.randn((rows,32),device='cuda',generator=g)/64).bfloat16()
    update = summarize_pages(page)
    completed = torch.cat([table[r,:int(lengths[r])//page].long() for r in range(rows)])
    slots = completed*page+page-1
    records = pool[:,:page*68].clone()
    if len(slots):
        update(pool,pool.view(torch.bfloat16),slots)
    torch.testing.assert_close(pool[:,:page*68],records,atol=0,rtol=0)
    for physical in completed.tolist():
        summaries(pool,page,physical)
    old = pool.clone()
    update(pool,pool.view(torch.bfloat16),torch.tensor([-1,0,page,2*page-2],device='cuda',dtype=torch.int64))
    torch.testing.assert_close(pool,old,atol=0,rtol=0)
    native = IndexerV41Plan(width,page)
    plan = PageShortlistPlan(width,page,candidate_blocks=source)
    scratch = [torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in plan.scratch_specs(rows)]
    out = torch.empty((rows,512),device='cuda',dtype=torch.int32)
    blocks = torch.empty((rows,CANDIDATE_BLOCKS),device='cuda',dtype=torch.int32)
    counts = torch.empty((rows,),device='cuda',dtype=torch.int32)
    args = [q,scales,weights,pool,table,lengths,active]
    def run():
        plan.run(*args,out=out,candidate_out=blocks,candidate_out_lengths=counts,scratch=scratch)
    def check():
        expected,native_scores = native.run(*args)
        run()
        b = dict(zip(plan._specs(rows),scratch))
        for r in range(rows):
            # The workspace is reused in bounded chunks. Inspect the last chunk
            # only here; every output row is checked independently below.
            visible = min(int(lengths[r]),int(active[0]))
            live = out[r][out[r] >= 0]
            assert len(live) == min(512,visible) and len(live) == len(live.unique())
            assert bool((live < visible).all())
            if visible <= plan.budget*64:
                torch.testing.assert_close(out[r],expected[r],atol=0,rtol=0)
            if source:
                ids = blocks[r][blocks[r] >= 0]
                assert bool((ids[1:] > ids[:-1]).all())
                assert int(counts[r]) == len(ids)*8-((8-visible%8)%8 if visible else 0)
                if visible:
                    assert int(ids[-1]) == (visible-1)//8
        start = ((rows-1)//plan.row_cap)*plan.row_cap
        for r in range(rows-start):
            n = int(b['candidate_lengths'].view(-1)[r])
            ids = b['candidates'].view(-1,plan.capacity//8)[r,:(n+7)//8]
            assert bool((ids[1:] > ids[:-1]).all())
            positions = torch.arange(n,device='cuda')
            positions = ids[positions//8].long()*8+positions%8
            torch.testing.assert_close(b['final_0'].view(-1,plan.capacity)[r,:n].view(torch.int16),
                                      native_scores[start+r,positions].view(torch.int16),atol=0,rtol=0)
        return out.clone()
    result = check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.testing.assert_close(out,result,atol=0,rtol=0)
    original = lengths.clone()
    for limit in (0,37,page-1,page,page+1,extent):
        lengths.copy_(torch.minimum(original,torch.full_like(original,limit)))
        expected = check()
        graph.replay()
        torch.testing.assert_close(out,expected,atol=0,rtol=0)
    lengths.copy_(original)
    # Simulate a physical page recycled to a different owner, completing it
    # again before the new owner can query its full-page summaries.
    if len(completed):
        physical = int(completed[0])
        pool[physical,:page*64].zero_()
        pool[physical,page*64:page*68].fill_(127)
        update(pool,pool.view(torch.bfloat16),slots)
        membership = summaries(pool,page,physical)
        assert membership == sorted(membership), 'zero-score tie partition must be stable'
        expected = check()
        graph.replay()
        torch.testing.assert_close(out,expected,atol=0,rtol=0)
    return {'page':page,'source':source,'rows':rows,'extent':extent,
            'balanced_centroids':True,'native_record_bytes_unchanged':True,
            'partial_page_update_is_noop':True,'candidate_score_bits':True,
            'mixed_owners_graph_and_recycle':True,
            'reserved_bytes_per_key':STATE_BYTES,'token_gate':'not run'}


def main():
    result = {'passed':False,'cases':[]}
    try:
        for page in (128,256):
            for source,rows,extent in [(False,1,4097),(False,9,4097),(True,2,20001),(False,2,37)]:
                c = case(page,source,rows,extent)
                result['cases'].append(c)
                print(json.dumps(c),flush=True)
            torch.cuda.empty_cache()
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
