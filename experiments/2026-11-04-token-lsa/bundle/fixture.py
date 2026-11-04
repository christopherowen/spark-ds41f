"""Setup only: native physical layout, one explicit snapshot owner."""
from pathlib import Path
import torch
from metadata import build
from snapshot_layout import restore_layout
from vllm.models.deepseek_v4_1.tilelang.shortlist_indexer import ShortlistPlan


def files():
    return sorted(Path('/snapshots').glob('rank-0-*.pt'))


def setup(path,budget=64,rows=1,candidate=True,extend=0):
    data = torch.load(path,map_location='cpu',weights_only=True)
    if extend:
        page = int(data['page_size'])
        assert extend % page == 0 and extend <= data['width']
        old_pages = (int(data['lengths'][0])+page-1)//page
        # Synthetic geometry screen only: repeat captured page contents into
        # distinct physical pages, preserving the original allocator stride.
        logical = torch.arange(extend//page)
        compact = data['pool'][data['table'][0,:old_pages].long()]
        data['pool'] = compact[logical%old_pages].clone()
        data['table'] = logical.int()[None]
        data['lengths'] = torch.tensor([extend],dtype=torch.int32)
        data['active'] = torch.tensor([extend],dtype=torch.int32)
        data['layout'] = dict(data['layout'],physical_ids=logical.long(),written_prefixes=[page]*len(logical))
    args,span = restore_layout(data)
    args = [value.repeat_interleave(rows,dim=0) if index in (0,1,2,4,5) else value
            for index,value in enumerate(args)]
    source = bool(data['candidate_blocks'])
    if candidate:
        centers,membership,construction = build(data)
        plan = ShortlistPlan(data['width'],data['page_size'],512 if source else budget,
                             centers,membership,rows=rows,source=source)
    else:
        from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan,CANDIDATE_BLOCKS
        plan = IndexerV41Plan(data['width'],data['page_size'],candidate_blocks=source)
        construction = None
        plan.out_buffer = torch.empty((rows,512),dtype=torch.int32,device='cuda')
        plan.scratch_buffer = [torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in plan.scratch_specs(rows)]
        plan.blocks_buffer = torch.empty((rows,CANDIDATE_BLOCKS),dtype=torch.int32,device='cuda')
        plan.counts_buffer = torch.empty((rows,),dtype=torch.int32,device='cuda')
    return data,args,plan,construction,span
