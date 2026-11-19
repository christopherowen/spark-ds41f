"""Out-of-band equivalence and scaling screen for ordered direct compaction."""
import json
import statistics
import traceback

import torch

import check_pages
from vllm.models.deepseek_v4_1.tilelang.direct_shortlist import (
    DirectPageShortlistPlan,direct_candidates)
from vllm.models.deepseek_v4_1.tilelang.paged_shortlist import page_flags,STATE_BYTES
from vllm.models.deepseek_v4_1.tilelang.shortlist_indexer import shortlist_prefix,shortlist_scatter


def case(width,page,budget,rows):
    blocks,buckets = page//8,page//64
    pages = width//page
    g = torch.Generator(device='cuda').manual_seed(width+page+budget+rows)
    pool = torch.empty((rows*pages+1,page*STATE_BYTES+128),dtype=torch.uint8,device='cuda')[:,:page*STATE_BYTES]
    table = (torch.randperm(rows*pages,device='cuda',generator=g)+1).reshape(rows,pages).int()
    member = torch.rand((len(pool),blocks),device='cuda',generator=g).argsort(dim=1).argsort(dim=1)//8
    pool[:,page*72:page*72+blocks] = member.to(torch.uint8)
    selected = torch.empty((rows,budget),device='cuda',dtype=torch.int32)
    lengths = torch.empty((rows,),device='cuda',dtype=torch.int32)
    active = torch.tensor([width],device='cuda',dtype=torch.int32)
    capacity = (budget+8)*8
    local = torch.empty((rows,width//8),device='cuda',dtype=torch.int32)
    tiles = (width//8+255)//256
    counts = torch.empty((rows,tiles),device='cuda',dtype=torch.int32)
    bases = torch.empty_like(counts)
    expected = torch.empty((rows,capacity),device='cuda',dtype=torch.int32)
    actual = torch.empty_like(expected)
    expected_lengths = torch.empty_like(lengths)
    actual_lengths = torch.empty_like(lengths)
    flags,prefix,scatter = page_flags(width,page,budget),shortlist_prefix(width,budget),shortlist_scatter(width,budget)
    compact = direct_candidates(width,page,budget)
    def old():
        flags(selected,pool,table,lengths,active,local,counts)
        prefix(counts,lengths,active,bases,expected,expected_lengths)
        scatter(local,bases,lengths,active,expected)
    def new():
        compact(selected,pool,table,lengths,active,actual,actual_lengths)
    def equal():
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        torch.testing.assert_close(actual_lengths,expected_lengths,atol=0,rtol=0)
    selected.fill_(-1)
    lengths.fill_(0)
    old();new();equal()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        old();new()
    patterns = 0
    for limit in (0,37,page-1,page,page+1,4097,width-page//2,width):
        lengths.copy_(torch.tensor([max(min(limit-r*page//2,width),0) for r in range(rows)],device='cuda',dtype=torch.int32))
        for strategy in ('low','high','random','none'):
            selected.fill_(-1)
            for r in range(rows):
                complete = int(lengths[r])//page*buckets
                n = min(complete,budget)
                if strategy == 'low':
                    ids = torch.arange(n,device='cuda')
                elif strategy == 'high':
                    ids = torch.arange(complete-n,complete,device='cuda')
                elif strategy == 'random':
                    ids = torch.randperm(complete,device='cuda',generator=g)[:n].sort().values
                else:
                    ids = torch.empty((0,),device='cuda',dtype=torch.int64)
                    n = 0
                selected[r,:n] = ids.int()
            graph.replay();equal()
            # Independent set-based reference; never part of a query timer.
            for r in range(rows):
                extent = int(lengths[r])
                visible = (extent+7)//8
                complete = extent//page*blocks
                pos = torch.arange(visible,device='cuda')
                labels = (pos//blocks)*buckets+pool[table[r,pos//blocks].long(),page*72+pos%blocks].long()
                keep = (pos >= complete) | torch.isin(labels,selected[r][selected[r] >= 0].long())
                if visible:
                    keep[-1] = True
                ids = pos[keep].int()
                torch.testing.assert_close(actual[r,:len(ids)],ids,atol=0,rtol=0)
                assert int(actual_lengths[r]) == len(ids)*8-((8-extent%8)%8 if extent else 0)
            patterns += 1
    # Timed operands use a full, randomly selected budget, never the empty case.
    lengths.fill_(width)
    selected.copy_(torch.stack([torch.randperm(width//64,device='cuda',generator=g)[:budget].sort().values.int()
                                if budget <= width//64 else torch.cat((torch.arange(width//64,device='cuda',dtype=torch.int32),
                                    torch.full((budget-width//64,),-1,device='cuda',dtype=torch.int32)))
                                for _ in range(rows)]))
    # Every selected page remains identical; table ownership changes on replay.
    table.copy_(table.flip(1));graph.replay();equal()
    return {'width':width,'page':page,'budget':budget,'rows':rows,'patterns':patterns,
            'ordered_candidates_counts_padding_exact':True,'independent_set_reference':True,
            'graph_owner_remap_exact':True,'performance':'separate process required',
            'removed_workspace_bytes':local.numel()*4+counts.numel()*8,
            'token_gate':'unchanged candidate-list contract only; no model qualification claim'}


def main():
    result = {'passed':False,'cases':[],'native_lifecycle':[]}
    try:
        for width,rows in ((8192,9),(65536,9),(1048576,1)):
            for page in (128,256):
                for budget in (64,512):
                    c = case(width,page,budget,rows)
                    result['cases'].append(c)
                    print(json.dumps(c),flush=True)
                    torch.cuda.empty_cache()
        check_pages.PageShortlistPlan = DirectPageShortlistPlan
        for page in (128,256):
            for source,rows,extent in [(False,1,8192),(False,9,8192),(True,2,20001),(False,2,37)]:
                c = check_pages.case(page,source,rows,extent)
                result['native_lifecycle'].append(c)
                print(json.dumps(c),flush=True)
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
