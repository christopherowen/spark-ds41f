"""Adversarial final-candidate equivalence, including uninitialized suffixes."""
import json
import torch
from vllm.models.deepseek_v4_1.tilelang.fused_candidates import FusedCoarse
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerTopK
from vllm.models.deepseek_v4_1.tilelang.block_shortlist import block_candidates

cases=[(n,page,k) for n in (2048,8192,32768) for page in (128,256) for k in (512,4096)]
cases.append((131072,256,4096))
result={'passed':False,'performance_measured':False,'cases':[]}
for width,page,budget in cases:
    raw_width=width*8
    lengths=[0,1,page-1,page,page+1,budget*8-1,budget*8,budget*8+page-1,
             budget*8+page,budget*8+page+1,raw_width-1,raw_width]
    lengths=[min(n,raw_width) for n in lengths];rows=len(lengths)
    native=IndexerTopK(width,budget);fused=FusedCoarse(width,page,budget)
    for cap in (0,budget*8+page+1,raw_width):
        scores=torch.empty(rows,width,dtype=torch.bfloat16,device='cuda')
        extents=[min(n,cap,raw_width) for n in lengths]
        full=[n//page*(page//8) for n in extents]
        counts=[n if n>budget else 0 for n in full]
        g=torch.Generator(device='cuda').manual_seed(width+page+budget+cap)
        for row,n in enumerate(counts):
            if n:
                bits=torch.randint(0,65536,(n,),device='cuda',generator=g,dtype=torch.int32).to(torch.int16)
                if row==len(counts)-1:bits.fill_(-128)  # BF16 negative infinity
                elif row==len(counts)-2:bits.fill_(0x3f80)  # ties
                scores[row,:n]=bits.view(torch.bfloat16)
        lens=torch.tensor(lengths,dtype=torch.int32,device='cuda')
        count=torch.tensor(counts,dtype=torch.int32,device='cuda');active=torch.tensor([cap],dtype=torch.int32,device='cuda')
        selected=torch.empty(rows,budget,dtype=torch.int32,device='cuda')
        values=torch.empty(rows,budget,dtype=torch.bfloat16,device='cuda')
        scratch=[torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in native.scratch_specs(rows)]
        native.run(scores,count,active,selected,values,scratch=scratch)
        expected=torch.empty(rows,budget+64,dtype=torch.int32,device='cuda');expected_n=torch.empty(rows,dtype=torch.int32,device='cuda')
        block_candidates(raw_width,page,budget)(selected,lens,active,expected,expected_n)
        actual=torch.empty_like(expected);actual_n=torch.empty_like(expected_n)
        work=[torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in fused.scratch_specs(rows)]
        fused.run(scores,count,lens,active,actual,actual_n,scratch=work)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        torch.testing.assert_close(actual_n,expected_n,atol=0,rtol=0)
        result['cases'].append({'width':width,'page':page,'budget':budget,'active':cap,'rows':rows,
            'exact_candidates_and_lengths':True,'unwritten_score_and_work_suffixes':True,'nonfinite_and_tie_keys':True})
    print('CASE '+json.dumps({'width':width,'page':page,'budget':budget}),flush=True)
result['passed']=True
print('RESULT '+json.dumps(result),flush=True)
