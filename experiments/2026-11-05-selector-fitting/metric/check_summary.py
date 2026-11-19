"""Separate projected-feature construction gate; synthetic maps only."""
import json

import torch

from vllm.models.deepseek_v4_1.tilelang.metric_summary import summarize_pages_metric
from vllm.models.deepseek_v4_1.tilelang.paged_shortlist import STATE_BYTES


def labels_for(features,buckets):
    labels = torch.zeros(len(features),dtype=torch.int64)
    for level in range(buckets.bit_length()-1):
        updated = labels.clone()
        for group in range(1<<level):
            ids = torch.nonzero(labels == group).flatten().tolist()
            direction = features[ids[0]]-features[ids[-1]]
            for _ in range(4):
                scores = features@direction
                ordered = sorted(ids,key=lambda i:(float(scores[i]),i))
                half = len(ids)//2
                direction = features[ordered[:half]].mean(0)-features[ordered[half:]].mean(0)
            for rank,index in enumerate(ordered):
                updated[index] = group*2+(rank >= half)
        labels = updated
    return labels


def case(page):
    blocks,buckets = page//8,page//64
    pool = torch.full((5,page*STATE_BYTES+128),0xa5,dtype=torch.uint8,device='cuda')[:,:page*STATE_BYTES]
    pool[:,:page*64].zero_();pool[:,page*64:page*68].fill_(127)
    # Two orthogonal partitions. Switching the map must change grouping while
    # retaining raw 128-dimensional centroid arithmetic and native key records.
    signs = torch.tensor([[1 if i%2 == 0 else -1,1 if i<blocks//2 else -1] for i in range(blocks)])
    for physical in (1,3):
        for block in range(blocks):
            low = 7 if signs[block,0] > 0 else 15
            high = 7 if signs[block,1] > 0 else 15
            for token in range(block*8,(block+1)*8):
                pool[physical,token*64] = low|(high<<4)
    native = pool[:,:page*68].clone()
    slots = torch.tensor([2*page-1,4*page-1],dtype=torch.int64,device='cuda')
    metric = torch.zeros(128,32,dtype=torch.bfloat16,device='cuda')
    update = summarize_pages_metric(page)
    def run():update(pool,pool.view(torch.bfloat16),slots,metric)
    metric[0,0] = 1
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    found = []
    for coordinate in (0,1):
        metric.zero_();metric[coordinate,0] = 1
        graph.replay()
        expected = labels_for(signs[:,coordinate:coordinate+1].float(),buckets)
        for physical in (1,3):
            actual = pool[physical,page*72:page*72+blocks].cpu().long()
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            assert all(int((actual == b).sum()) == 8 for b in range(buckets))
            raw = torch.zeros(buckets,128)
            for b in range(buckets):raw[b,:2] = (6*signs[actual == b].float()).mean(0)
            center = pool[physical,page*68:page*72].contiguous().view(torch.bfloat16).reshape(buckets,128).cpu()
            torch.testing.assert_close(center,raw.bfloat16(),atol=0,rtol=0)
        found.append(expected)
    assert not torch.equal(*found)
    torch.testing.assert_close(pool[:,:page*68],native,atol=0,rtol=0)
    before = pool.clone()
    update(pool,pool.view(torch.bfloat16),torch.tensor([-1,0,page,2*page-2],device='cuda',dtype=torch.int64),metric)
    torch.testing.assert_close(pool,before,atol=0,rtol=0)
    pool[:,:page*64].zero_()
    graph.replay()
    for physical in (1,3):
        actual = pool[physical,page*72:page*72+blocks].cpu().long()
        torch.testing.assert_close(actual,torch.arange(blocks)//8,atol=0,rtol=0)
        assert not pool[physical,page*68:page*72].count_nonzero()
    return {'page_size':page,'latent_dimensions':32,'map_changes_expected_partition':True,
            'raw_centroids_exact':True,'native_bytes_unchanged':True,'partial_write_noop':True,
            'graph_and_recycled_page':True,'synthetic_maps':True,'model_quality_tested':False}


if __name__ == '__main__':
    print('RESULT '+json.dumps({'passed':True,'cases':[case(128),case(256)]}),flush=True)
