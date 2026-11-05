"""Unwritten compact suffix must not affect the native source/cache reader."""
import argparse,hashlib,json
from pathlib import Path
import torch
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan,CANDIDATE_BLOCKS

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--replay-receipt',type=Path,required=True);args=p.parse_args()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();replay=json.loads(args.replay_receipt.read_text());assert replay['passed']
    result={'passed':False,'performance_measured':False,'model_quality_tested':False,'cases':[]}
    for path in sorted(args.inputs.glob('validation-a-rank-0-*.pt')):
        assert sha(path)==replay['files'][path.name]['sha256'];data=torch.load(path,map_location='cpu',weights_only=True)
        page,width,source=data['page_size'],data['width'],bool(data['candidate_blocks'])
        native=data['pool'].cuda();pool=torch.empty(len(native),page*80+128,dtype=torch.uint8,device='cuda')[:,:page*80]
        pool[:,:page*68].copy_(native)  # compact suffix deliberately unwritten
        plan=IndexerV41Plan(width,page,candidate_blocks=source)
        for rows in (1,8,64):
            q,s,w=[data[k][:rows].cuda() for k in ('q','scales','weights')]
            table=torch.zeros(rows,width//page,dtype=torch.int32,device='cuda');n=min(table.shape[1],data['table'].shape[1]);table[:,:n].copy_(data['table'][:rows,:n])
            lengths,active=data['lengths'][:rows].cuda(),data['active'].cuda()
            outputs=[]
            for storage in (native,pool):
                scratch=[torch.empty(shape,dtype=dtype,device='cuda') for shape,dtype in plan.scratch_specs(rows)]
                out=torch.empty(rows,512,dtype=torch.int32,device='cuda')
                blocks=torch.empty(rows,CANDIDATE_BLOCKS,dtype=torch.int32,device='cuda');counts=torch.empty(rows,dtype=torch.int32,device='cuda')
                options={'candidate_out':blocks,'candidate_out_lengths':counts} if source else {}
                plan.run(q,s,w,storage,table,lengths,active,out=out,scratch=scratch,**options)
                outputs.append((out,blocks,counts))
            for i in range(3 if source else 1):torch.testing.assert_close(outputs[0][i],outputs[1][i],atol=0,rtol=0)
            result['cases'].append({'file':path.name,'sha256':sha(path),'rows':rows,'source':source,'exact_native_keys_and_handoff':True,'compact_suffix_unwritten':True})
    assert len(result['cases'])==12;result['passed']=True;print('RESULT '+json.dumps(result),flush=True)

if __name__=='__main__':main()
