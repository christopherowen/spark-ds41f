"""Out-of-band exact-summary checks and separate maintenance timings."""
import argparse,hashlib,json
from pathlib import Path
import torch
from vllm.models.deepseek_v4_1.tilelang.compact_shortlist import summarize_compact as summarize_parallel
from benchmark_runtime import graph_of,sample
from vllm.models.deepseek_v4_1.tilelang.compact_reference import summarize_compact
from vllm.models.deepseek_v4_1.tilelang.indexer import quantize_mxfp4

def equal(pool,slots,page):
    expected=pool.clone();actual=pool.clone()
    summarize_compact(page)(expected,expected.view(torch.bfloat16),slots)
    summarize_parallel(page)(actual,actual.view(torch.bfloat16),slots)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    return actual

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--replay-receipt',type=Path,required=True)
    p.add_argument('--timing',action='store_true');args=p.parse_args()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    replay=json.loads(args.replay_receipt.read_text());assert replay['passed']
    result={'passed':False,'request_performance_measured':False,'model_quality_tested':False,'timing':args.timing,'actual':[],'synthetic':[]}
    torch.cuda.reset_peak_memory_stats()
    for path in sorted(args.inputs.glob('validation-a-rank-0-*.pt')):
        assert sha(path)==replay['files'][path.name]['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True);page=data['page_size']
        pool=torch.empty(len(data['pool']),page*80+128,dtype=torch.uint8,device='cuda')[:,:page*80]
        pool.fill_(173);pool[:,:page*68].copy_(data['pool'])
        complete=[i for i,n in enumerate(data['layout']['written_prefixes']) if n==page]
        slots=torch.tensor(complete,dtype=torch.int64,device='cuda')*page+page-1
        actual=equal(pool,slots,page)
        row={'file':path.name,'sha256':sha(path),'page':page,'complete_pages':len(complete),'all_pool_bytes_equal':True}
        if args.timing:
            single=slots[len(slots)//2:len(slots)//2+1]
            skipped=single-1
            row['single_complete_page']=sample(graph_of(lambda:summarize_parallel(page)(pool,pool.view(torch.bfloat16),single)))
            row['all_complete_pages']=sample(graph_of(lambda:summarize_parallel(page)(pool,pool.view(torch.bfloat16),slots)))
            row['skip_only']=sample(graph_of(lambda:summarize_parallel(page)(pool,pool.view(torch.bfloat16),skipped)))
        result['actual'].append(row)
    assert len(result['actual'])==4
    # Native-quantized public random vectors span many scales, zeros and ties.
    # Only complete pages are readable; partial/negative/outside slots must skip.
    for page in (128,256):
        pages=4;pool=torch.empty(pages,page*80+128,dtype=torch.uint8,device='cuda')[:,:page*80];pool.fill_(61)
        g=torch.Generator(device='cuda').manual_seed(page)
        x=torch.randn(pages*page,128,device='cuda',generator=g,dtype=torch.bfloat16)
        exponent=torch.randint(-100,101,(pages*page,4,1),device='cuda',generator=g,dtype=torch.int32)
        x=(x.view(-1,4,32)*torch.exp2(exponent.float())).bfloat16().view(-1,128)
        x[0:8]=0;x[8:16]=1;x[16:24]=-1
        q=torch.empty(pages*page,64,device='cuda',dtype=torch.uint8);sf=torch.empty(pages*page,4,device='cuda',dtype=torch.uint8)
        quantize_mxfp4(x,q,sf)
        for i in range(pages):
            pool[i,:page*64].copy_(q[i*page:(i+1)*page].reshape(-1))
            pool[i,page*64:page*68].copy_(sf[i*page:(i+1)*page].reshape(-1))
        slots=torch.tensor([-1,0,page-2,page-1,2*page-1,3*page-1,4*page-2,5*page-1],device='cuda',dtype=torch.int64)
        equal(pool,slots,page)
        # Reuse the same page with changed keys and preserve unchanged slots.
        pool[0,:page*68].copy_(pool[2,:page*68]);equal(pool,slots,page)
        # Leave skipped pages' keys unwritten for the initialization sanitizer.
        strict=torch.empty(2,page*80,dtype=torch.uint8,device='cuda')
        strict[0,:page*68].copy_(pool[0,:page*68])
        strict[:,page*68:].fill_(91)
        strict_slots=torch.tensor([-1,page-2,page-1,2*page-2,3*page-1],device='cuda',dtype=torch.int64)
        summarize_parallel(page)(strict,strict.view(torch.bfloat16),strict_slots)
        expected=pool[0:1].clone()
        zero_slot=torch.tensor([page-1],device='cuda',dtype=torch.int64)
        summarize_compact(page)(expected,expected.view(torch.bfloat16),zero_slot)
        torch.testing.assert_close(strict[0,page*68:page*68+page//8*68],expected[0,page*68:page*68+page//8*68],atol=0,rtol=0)
        assert bool((strict[1,page*68:]==91).all())
        result['synthetic'].append({'page':page,'scales_exponent_range':[-100,100],'zero_tie_sign_vectors':True,'skip_and_recycle_bytes_equal':True})
    result['memory']={'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved()}
    result['passed']=True;print('RESULT '+json.dumps(result),flush=True)

if __name__=='__main__':main()
