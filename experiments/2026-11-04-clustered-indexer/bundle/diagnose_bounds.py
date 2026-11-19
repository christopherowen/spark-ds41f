"""Untimed real-input bound diagnostic; no changed selector or native benchmark."""
import json
from pathlib import Path
import traceback

import torch
from snapshot_layout import restore_layout
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan


def decode(packed, scales):
    packed = packed.view(torch.uint8)
    scales = scales.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(*packed.shape[:-1],128).long()
    values = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64)
    return values[codes] * torch.exp2(scales.double().repeat_interleave(32,dim=-1)-127)


def diagnose(path):
    data = torch.load(path,map_location='cpu',weights_only=True)
    extent, page = int(data['lengths'][0]), data['page_size']
    q = decode(data['q'],data['scales'].view(torch.uint8).reshape(1,32,4))[0]
    weights = data['weights'][0].double()
    positions = torch.arange(extent)
    physical = data['table'][0,positions//page].long()
    offsets = positions%page
    packed = data['pool'][physical[:,None],offsets[:,None]*64+torch.arange(64)]
    scales = data['pool'][physical[:,None],page*64+offsets[:,None]*4+torch.arange(4)]
    keys = decode(packed,scales)
    clusters = (extent+63)//64
    if extent%64:
        keys = torch.cat((keys,keys[-1:].expand(clusters*64-extent,-1)))
    shaped = keys.reshape(clusters,64,128)
    lower,upper = shaped.amin(dim=1),shaped.amax(dim=1)
    plus,minus = q.clamp(min=0),q.clamp(max=0)
    high = plus @ upper.T + minus @ lower.T
    low = plus @ lower.T + minus @ upper.T
    norm = q.abs() @ torch.maximum(lower.abs(),upper.abs()).T
    high = high + norm/256 + 2**-50
    low = low - norm/256 - 2**-50
    # Certified per-head upper/lower dots, then exactly the native BF16 stages.
    old_terms = (high.to(torch.bfloat16).float().clamp(min=0)*weights.float().clamp(min=0)[:,None]).to(torch.bfloat16).float()
    dot = torch.where(weights[:,None]>=0,high,low)
    signed_terms = (dot.to(torch.bfloat16).float().clamp(min=0)*weights.float()[:,None]).to(torch.bfloat16).float()
    def ordered_sum(terms):
        total = torch.zeros(clusters,dtype=torch.float32)
        for head in range(32):total = total+terms[head]
        result = total.to(torch.bfloat16).float()
        return torch.where(result==0,torch.zeros_like(result),result)
    old_bound,signed_bound = ordered_sum(old_terms),ordered_sum(signed_terms)
    # Only certified encodings participate in the diagnostic. Unsupported input
    # would require infinity, exactly as in the current lossless prototype.
    query_ok = bool(((data['scales'].view(torch.uint8)>=96)&(data['scales'].view(torch.uint8)<=144)).all())
    query_ok &= bool(torch.isfinite(weights).all() and (weights.abs()<=2**20).all())
    valid = ((scales>=96)&(scales<=144)).all(dim=1)
    if extent%64:valid=torch.cat((valid,valid[-1:].expand(clusters*64-extent)))
    valid = valid.reshape(clusters,64).all(dim=1)
    valid &= query_ok
    old_bound[~valid]=float('inf');signed_bound[~valid]=float('inf')
    args,span = restore_layout(data)
    # Native correctness oracle only; no timer surrounds this call.
    _,scores = IndexerV41Plan(data['width'],page).run(*args)
    scores = scores[0,:extent].cpu().float()
    padded = torch.cat((scores,scores[-1:].expand(clusters*64-extent))) if extent%64 else scores
    maxima = padded.reshape(clusters,64).amax(dim=1)
    assert bool((old_bound>=maxima).all()) and bool((signed_bound>=maxima).all())
    key_cut = scores.topk(min(512,extent)).values[-1]
    # Ideal cutoff is stronger than any seed cutoff. Failure to prune against it
    # rules out seed tuning as a fix for this bound on the sampled input.
    blocks = (extent+7)//8
    padded = torch.cat((scores,scores[-1:].expand(blocks*8-extent))) if extent%8 else scores
    block_scores = padded.reshape(blocks,8).amax(dim=1)
    ordinary = block_scores[:-1]
    block_cut = ordinary.topk(min(2047,len(ordinary))).values[-1]
    cutoff = block_cut if data['candidate_blocks'] else key_cut
    sizes = torch.full((clusters,),64,dtype=torch.int64);sizes[-1]=extent-(clusters-1)*64
    return {'snapshot':path.name,'layer':data['capture_context']['layer_id'],'rank':data['rank'],
            'source':data['candidate_blocks'],'extent':extent,'planned_width':data['width'],
            'page_size':page,'native_pool_span_bytes':span,'positive_heads':int((weights>0).sum()),
            'negative_heads':int((weights<0).sum()),'certified_clusters':int(valid.sum()),
            'clusters':clusters,'ideal_native_cutoff':float(cutoff),
            'old_bound_min':float(old_bound.min()),'signed_bound_min':float(signed_bound.min()),
            'old_ideal_retained_keys':int(sizes[old_bound>=cutoff].sum()),
            'signed_ideal_retained_keys':int(sizes[signed_bound>=cutoff].sum()),
            'native_bound_dominance':True,'kind':'untimed diagnostic; no changed selection'}


def main():
    result={'schema_version':1,'passed':False,'cases':[]}
    try:
        files=sorted(Path('/snapshots').glob('rank-0-*.pt'))
        if len(files)!=4:raise RuntimeError('expected four rank0 layer/interface samples')
        for path in files:
            case=diagnose(path);result['cases'].append(case);print(json.dumps(case),flush=True)
        result['passed']=True
    except Exception:
        result['error']=traceback.format_exc();raise
    finally:print('RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':main()
