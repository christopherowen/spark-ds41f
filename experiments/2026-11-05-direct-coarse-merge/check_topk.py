"""Untimed exact comparison on adversarial BF16 keys and unwritten suffixes."""
import json
import torch
from merge import DirectCoarseTopK
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerTopK

result={'passed':False,'performance_measured':False,'cases':[]}
for width in [4097,8192,12288,32768,131072]:
    extents=[-1,0,1,37,4095,4096,4097,8191,8192,8193,width-1,width]
    extents=[min(n,width) for n in extents];rows=len(extents)
    scores=torch.empty(rows,width,dtype=torch.bfloat16,device='cuda')
    g=torch.Generator(device='cuda').manual_seed(width)
    for row,n in enumerate(extents):
        if n>0:
            bits=torch.randint(0,65536,(n,),device='cuda',generator=g,dtype=torch.int32).to(torch.int16)
            for offset,value in enumerate([0x3f80,0xff80,0x7f80,0x7fc1,0xffc1,0x8000,0]):
                bits[offset::17]=value if value<32768 else value-65536
            scores[row,:n]=bits.view(torch.bfloat16)
    lengths=torch.tensor(extents,dtype=torch.int32,device='cuda')
    plans=[IndexerTopK(width,4096),DirectCoarseTopK(width)]
    storage=[]
    for plan in plans:
        storage.append((torch.empty(rows,4096,dtype=torch.int32,device='cuda'),
                        torch.empty(rows,4096,dtype=torch.bfloat16,device='cuda'),
                        [torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in plan.scratch_specs(rows)]))
    for cap in [0,4096,4097,width]:
        active=torch.tensor([cap],dtype=torch.int32,device='cuda')
        for plan,(indices,values,scratch) in zip(plans,storage):
            plan.run(scores,lengths,active,indices,values,scratch=scratch)
        torch.testing.assert_close(storage[0][0],storage[1][0],atol=0,rtol=0)
        torch.testing.assert_close(storage[0][1].view(torch.int16),storage[1][1].view(torch.int16),atol=0,rtol=0)
        result['cases'].append({'width':width,'active':cap,'rows':rows,'exact_positions_and_value_bits':True,
            'unwritten_score_and_work_suffixes':True,'ties_infinities_nan_payloads':True})
    print('WIDTH '+str(width),flush=True)
result['passed']=True
print('RESULT '+json.dumps(result),flush=True)
