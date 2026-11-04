"""Snapshot correctness gate; timings cover only the candidate, never the release."""
import argparse
import hashlib
import json
import time
import traceback

import torch
from vllm.models.deepseek_v4_1.tilelang.clustered_indexer import ClusteredIndexerPlan
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan, page_bytes


def inputs(width, extents, mode, page_size=256):
    rows = len(extents)
    g = torch.Generator(device='cuda').manual_seed(width + rows)
    logical = width // page_size
    pool_pages = rows * logical + 1
    pool = torch.full((pool_pages, page_bytes(page_size) + 64), 255, dtype=torch.uint8, device='cuda')
    table = (torch.randperm(pool_pages - 1, device='cuda', generator=g) + 1).reshape(rows, logical).int()
    for r, extent in enumerate(extents):
        p = torch.arange(extent, device='cuda')
        physical, offset = table[r, p // page_size].long(), p % page_size
        keys = torch.randint(0, 256, (extent, 64), dtype=torch.uint8, device='cuda', generator=g)
        sf = torch.randint(120, 130, (extent, 4), dtype=torch.uint8, device='cuda', generator=g)
        if mode == 'separable':
            keys.fill_(0xaa)
            keys[:512].fill_(0x77)
            sf.fill_(127)
        if mode == 'ties':
            keys.zero_()
            sf.fill_(127)
        if mode == 'unsupported' and extent:
            sf[0, 0] = 150
        if mode == 'nan-key' and extent:
            sf[0, 0] = 255
        if mode == 'extreme':
            sf.random_(96, 145, generator=g)
        for b in range(64):
            pool[physical, offset * 64 + b] = keys[:, b]
        for b in range(4):
            pool[physical, page_size * 64 + offset * 4 + b] = sf[:, b]
    q = torch.randint(0, 256, (rows, 32, 64), dtype=torch.uint8, device='cuda', generator=g)
    scales = torch.randint(120, 130, (rows, 32, 4), dtype=torch.uint8, device='cuda', generator=g)
    weights = (torch.randn((rows, 32), device='cuda', generator=g) / 64).bfloat16()
    if mode in ('separable', 'ties'):
        q.fill_(0x22)
        scales.fill_(127)
        weights.fill_(1 / 64)
    if mode == 'extreme':
        scales.random_(96, 145, generator=g)
    if mode == 'nan-query':
        scales[0, 0, 0] = 255
    if mode == 'nan-weight':
        weights[0, 0] = float('nan')
    lengths = torch.tensor(extents, dtype=torch.int32, device='cuda')
    active = torch.tensor([width], dtype=torch.int32, device='cuda')
    return q, scales, weights, pool, table, lengths, active


def check(width, extents, mode, repeats):
    torch.cuda.reset_peak_memory_stats()
    args = inputs(width, extents, mode)
    candidate = ClusteredIndexerPlan(width, 256, seeds=min(32, width // 64))
    reference = IndexerV41Plan(width, 256)
    expected, scores = reference.run(*args)  # correctness only; never timed
    actual, evidence = candidate.run(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    score_cpu, bound_cpu = scores.cpu(), evidence['bounds'].cpu()
    for r, extent in enumerate(extents):
        if extent:
            for c in range((extent + 63) // 64):
                observed = score_cpu[r, c*64:min((c+1)*64, extent)].float().max()
                bound = bound_cpu[r, c].float()
                if bool(torch.isfinite(observed)):
                    assert bool(bound >= observed), (r, c, bound.item(), observed.item())
                else:
                    assert bool(torch.isposinf(bound)), (r, c, bound.item(), observed.item())
    # Repeat with mutated written cache contents; stale envelopes must not survive.
    if mode == 'separable' and extents[0] > 1024:
        table, pool = args[4], args[3]
        before = actual.clone()
        pool[table[0, 4].long(), :64].fill_(0x77)
        pool[table[0, 4].long(), 256 * 64:256 * 64 + 4].fill_(130)
        expected, _ = reference.run(*args)
        actual, evidence = candidate.run(*args)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert not torch.equal(actual, before), 'mutation did not change the reference winners'
    latencies = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        candidate.run(*args)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    return {'width': width, 'extents': extents, 'mode': mode, 'identical': True,
            'candidate_counts': evidence['candidate_counts'].cpu().tolist(),
            'output_sha256': hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest(),
            'candidate_wall_ms': latencies,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=5)
    options = parser.parse_args()
    result = {'schema_version': 1, 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
              'cases': [], 'passed': False}
    try:
        for width, extents, mode in [
            (1024, [1, 37, 511, 512, 513, 1023], 'random'),
            (1024, [0, 64, 1024], 'ties'),
            (4096, [1025, 4093], 'random'),
            (4096, [4093], 'unsupported'),
            (4096, [4093], 'extreme'),
            (4096, [4093], 'nan-key'),
            (4096, [4093], 'nan-query'),
            (4096, [4093], 'nan-weight'),
            (16384, [16383], 'separable'),
            (16384, [16383], 'random'),
            (262144, [262143], 'separable'),
            (262144, [262143], 'random'),
        ]:
            result['cases'].append(check(width, extents, mode, options.repeats))
            print(json.dumps(result['cases'][-1]), flush=True)
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        with open('/tmp/clustered-indexer-result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
