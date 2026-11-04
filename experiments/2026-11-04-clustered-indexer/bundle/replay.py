"""Replay node-local real indexer snapshots; never time the reference scorer."""
import hashlib
import json
from pathlib import Path
import time

import torch
from vllm.models.deepseek_v4_1.tilelang.clustered_indexer import ClusteredIndexerPlan
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan


def main():
    files = sorted(Path('/snapshots').glob('rank-*.pt'))
    if not files:
        raise RuntimeError('no real indexer snapshots mounted')
    for path in files:
        data = torch.load(path, map_location='cpu', weights_only=True)
        args = [data[k].cuda() for k in ('q', 'scales', 'weights', 'pool', 'table', 'lengths', 'active')]
        width, page_size = data['width'], data['page_size']
        reference = IndexerV41Plan(width, page_size)
        candidate = ClusteredIndexerPlan(width, page_size)
        expected, scores = reference.run(*args)
        actual, evidence = candidate.run(*args)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        extent = int(data['lengths'][0])
        score_cpu, bound_cpu = scores[:, :extent].cpu().float(), evidence['bounds'].cpu().float()
        for cluster in range((extent + 63) // 64):
            observed = score_cpu[0, cluster*64:min((cluster+1)*64, extent)].max()
            bound = bound_cpu[0, cluster]
            if torch.isfinite(observed):
                assert bound >= observed, (cluster, observed, bound)
            else:
                assert torch.isposinf(bound), (cluster, observed, bound)
        samples = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            candidate.run(*args)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        print(json.dumps({'snapshot': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'source': data['source'], 'rank': data['rank'], 'extent': extent,
                          'identical': True, 'candidate_counts': evidence['candidate_counts'].cpu().tolist(),
                          'candidate_wall_ms': samples,
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                          'peak_reserved_bytes': torch.cuda.max_memory_reserved()}), flush=True)


if __name__ == '__main__':
    main()
