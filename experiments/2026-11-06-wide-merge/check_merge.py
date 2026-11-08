"""Independent untimed coarse-selection oracle, including unreadable suffixes."""
import argparse,json
import torch
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerTopK

p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');args=p.parse_args()
torch.set_num_threads(1)
cases=[]
for width in ((32768,) if args.quick else (32768,131072)):
    for topk in ((8192,) if args.quick else (6144,8192)):
        lengths_list=[0,1,4095,4096,4097,8192,16385,width-1]
        for pattern in ('ties','mixed'):
            for active_cap in (0,4097,width):
                rows=len(lengths_list)
                scores=torch.empty(rows,width,dtype=torch.bfloat16,device='cuda')
                lengths=torch.tensor(lengths_list,dtype=torch.int32,device='cuda')
                active=torch.tensor([active_cap],dtype=torch.int32,device='cuda')
                for row,length in enumerate(lengths_list):
                    n=min(length,active_cap)
                    if n:
                        positions=torch.arange(n,device='cuda')
                        values=torch.zeros(n,device='cuda') if pattern=='ties' else ((positions*1103515245+row*12345)%67-33).float()
                        scores[row,:n]=values.bfloat16()
                plan=IndexerTopK(width,topk)
                scratch=[torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in plan.scratch_specs(rows)]
                indices=torch.empty(rows,topk,dtype=torch.int32,device='cuda')
                values=torch.empty(rows,topk,dtype=torch.bfloat16,device='cuda')
                plan.run(scores,lengths,active,indices,values,scratch=scratch)
                expected_indices=torch.full_like(indices,-1)
                expected_values=torch.full_like(values,-float('inf'))
                for row,length in enumerate(lengths_list):
                    n=min(length,active_cap)
                    if n:
                        keep=min(n,topk)
                        selected=torch.argsort(scores[row,:n],descending=True,stable=True)[:keep].sort().values
                        expected_indices[row,:keep]=selected.int()
                        expected_values[row,:keep]=scores[row,selected]
                torch.cuda.synchronize()
                torch.testing.assert_close(indices,expected_indices,atol=0,rtol=0)
                torch.testing.assert_close(values.view(torch.int16),expected_values.view(torch.int16),atol=0,rtol=0)
                cases.append({'width':width,'topk':topk,'rows':rows,'active_cap':active_cap,'pattern':pattern,'independent_stable_sort':True,'score_bits_equal':True,'unwritten_scores_and_scratch':True,'empty_partial_one_and_multiple_lists':True})
                print('CASE '+json.dumps(cases[-1]),flush=True)
print('RESULT '+json.dumps({'passed':True,'performance_measured':False,'source_budget':1024,'cases':cases,'model_quality_tested':False}),flush=True)
