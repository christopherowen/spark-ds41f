"""One-owner multi-query gates; reference scoring is correctness-only."""
import argparse
import hashlib
import json
import time
import traceback

import torch
from clustered_check import inputs
from vllm.models.deepseek_v4_1.tilelang.clustered_indexer import ClusteredIndexerPlan
from vllm.models.deepseek_v4_1.tilelang.grouped_indexer import GroupedClusteredIndexerPlan
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan


def check(width, rows, mode, grouped, cuda_graph=True, repeats=15, page_size=256):
    torch.cuda.reset_peak_memory_stats()
    args = list(inputs(width, [width - 1], mode, page_size=page_size, unwritten_cache=True))
    generator = torch.Generator(device='cuda').manual_seed(width + rows + 51)
    for index in [0, 1, 2]:
        args[index] = args[index].repeat(rows, *([1] * (args[index].ndim - 1)))
    if mode == 'random':
        args[0].random_(0, 256, generator=generator)
        args[2].copy_((torch.randn((rows, 32), device='cuda', generator=generator) / 64).bfloat16())
    extents = [max(0, width - 1 - 7 * r) for r in range(rows)]
    extents[0] = 0
    extents[-1] = width - 1
    args[5] = torch.tensor(extents, device='cuda', dtype=torch.int32)
    one_table = args[4]
    args[4] = one_table.repeat(rows, 1)
    native = IndexerV41Plan(width, page_size)
    candidate = (GroupedClusteredIndexerPlan(width, page_size) if grouped else
                 ClusteredIndexerPlan(width, page_size, matrix_bounds=True))
    candidate_args = list(args)
    if grouped:
        candidate_args[4] = one_table
    expected, scores = native.run(*args)
    actual, evidence = candidate.run(*candidate_args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    bounds = evidence['bounds'].cpu().float()
    for row, extent in enumerate(extents):
        visible = scores[row, :extent].cpu().float()
        for cluster in range((extent + 63) // 64):
            assert bounds[row, cluster] >= visible[cluster*64:min((cluster+1)*64, extent)].max()
    graph = None
    if cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, evidence = candidate.run(*candidate_args)
    def replay():
        nonlocal actual, evidence
        if graph is None:
            actual, evidence = candidate.run(*candidate_args)
        else:
            graph.replay()
    replay()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    saved = args[5].clone()
    args[5].copy_(saved // 2)
    args[6].fill_(width - 17)
    reduced, _ = native.run(*args)
    replay()
    torch.testing.assert_close(actual, reduced, atol=0, rtol=0)
    args[6].zero_()
    empty, _ = native.run(*args)
    replay()
    torch.testing.assert_close(actual, empty, atol=0, rtol=0)
    args[5].copy_(saved)
    args[6].fill_(width)
    # Mutate a written key in the same cache and replay the existing graph.
    page = one_table[0, 1024 // page_size].long()
    args[3][page, :64].fill_(0x77)
    args[3][page, page_size*64:page_size*64+4].fill_(130)
    changed, _ = native.run(*args)
    if mode == 'separable':
        assert not torch.equal(changed, expected), 'cache mutation did not change the winners'
    replay()
    torch.testing.assert_close(actual, changed, atol=0, rtol=0)
    torch.cuda.synchronize()
    latencies = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        replay()
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    return {'page_size': page_size, 'width': width, 'rows': rows, 'mode': mode, 'grouped_metadata': grouped, 'execution': 'cuda-graph' if cuda_graph else 'eager',
            'identical': True, 'graph_visibility_and_mutation_identical': True if cuda_graph else None,
            'visibility_and_mutation_identical': True,
            'output_sha256': hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest(),
            'metadata_elements': (2 * (1 if grouped else rows) * (width // 64) * 128),
            'candidate_counts': evidence['candidate_counts'].cpu().tolist(),
            'candidate_wall_ms': latencies,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grouped', action='store_true')
    parser.add_argument('--no-graph', action='store_true')
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--page-size', type=int, choices=[128,256], default=256)
    options = parser.parse_args()
    result = {'schema_version': 1, 'passed': False, 'cases': []}
    try:
        cases = [(4096, 1, 'random'), (4096, 8, 'random'), (16384, 32, 'random'),
                                  (65536, 128, 'random'), (65536, 128, 'separable'),
                                  (16384, 256, 'random')]
        if options.quick:
            cases = [(4096, 8, 'random'), (16384, 32, 'separable')]
        for width, rows, mode in cases:
            result['cases'].append(check(width, rows, mode, options.grouped,
                                         not options.no_graph, options.repeats, page_size=options.page_size))
            print(json.dumps(result['cases'][-1]), flush=True)
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
