"""Synthetic construction cost only; no model or request speed claim."""
import hashlib
import json
import statistics

import torch

from vllm.models.deepseek_v4_1.tilelang.metric_summary import summarize_pages_metric
from vllm.models.deepseek_v4_1.tilelang.paged_shortlist import summarize_pages,STATE_BYTES


def samples(call):
    for _ in range(3):call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call()
    graph.replay();torch.cuda.synchronize()
    result = []
    for _ in range(9):
        begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(25):graph.replay()
        end.record();end.synchronize()
        result.append(begin.elapsed_time(end)/25)
    return result


def case(page,pages=1024):
    generator = torch.Generator(device='cuda').manual_seed(20261105+page)
    old = torch.zeros(pages,page*STATE_BYTES+128,dtype=torch.uint8,device='cuda')[:,:page*STATE_BYTES]
    new = torch.empty(pages,page*STATE_BYTES+128,dtype=torch.uint8,device='cuda')[:,:page*STATE_BYTES]
    old[:,:page*64].random_(0,256,generator=generator)
    old[:,page*64:page*68].random_(121,130,generator=generator)
    new.copy_(old)
    metric = (torch.randn(128,32,generator=generator,device='cuda')/128**.5).bfloat16()
    slots = torch.arange(pages,device='cuda',dtype=torch.int64)*page+page-1
    original,projected = summarize_pages(page),summarize_pages_metric(page)
    a = lambda:original(old,old.view(torch.bfloat16),slots)
    b = lambda:projected(new,new.view(torch.bfloat16),slots,metric)
    values = {'original_before_ms':samples(a),'projected_ms':samples(b),'original_after_ms':samples(a)}
    torch.testing.assert_close(old[:,:page*68],new[:,:page*68],atol=0,rtol=0)
    for pool in (old,new):
        membership = pool[:,page*72:page*72+page//8].long()
        assert all(bool(((membership == i).sum(1) == 8).all()) for i in range(page//64))
    return {'page_size':page,'completed_pages':pages,'keys':pages*page,'stride_bytes':old.stride(0),
            'synthetic_metric_dimensions':32,'native_records_equal':True,
            'metric_sha256':hashlib.sha256(metric.cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
            'samples':values,'medians_ms':{k:statistics.median(v) for k,v in values.items()},
            'timing':'CUDA graph GPU time; warmup/correctness/host copies outside timer'}


if __name__ == '__main__':
    result = {'passed':True,'scope':'synthetic construction only; no native-model quality or request-performance claim',
              'cases':[case(128),case(256)]}
    print('RESULT '+json.dumps(result),flush=True)
