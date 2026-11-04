"""Gate source winners, native block handoff and a following reindex query."""
import argparse
import hashlib
import json
import time
import traceback

import torch
from clustered_check import inputs
from vllm.models.deepseek_v4_1.tilelang.indexer import CANDIDATE_BLOCKS, IndexerV41Plan
from vllm.models.deepseek_v4_1.tilelang.source_indexer import SourceClusteredIndexerPlan


def check(width, extents, mode, cuda_graph=True, repeats=15, source_reference=False, *, fixture=None, candidate_factory=None):
    torch.cuda.reset_peak_memory_stats()
    args = inputs(width, extents, mode, unwritten_cache=True) if fixture is None else fixture
    rows = len(extents)
    reference = IndexerV41Plan(width, 256, candidate_blocks=True)
    Plan = SourceClusteredIndexerPlan
    if source_reference:
        from vllm.models.deepseek_v4_1.tilelang.source_reference import SourceClusteredIndexerPlan as Plan
    candidate = (candidate_factory or Plan)(width, 256)
    blocks = torch.empty((rows, CANDIDATE_BLOCKS), device='cuda', dtype=torch.int32)
    counts = torch.empty((rows,), device='cuda', dtype=torch.int32)

    def native():
        return reference.run(*args, candidate_out=blocks, candidate_out_lengths=counts)

    expected, scores = native()
    actual, evidence = candidate.run(*args)

    def compare(expected):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(evidence['source_blocks'], blocks, atol=0, rtol=0)
        torch.testing.assert_close(evidence['source_lengths'], counts, atol=0, rtol=0)

    compare(expected)
    if not evidence['native_fallback']:
        bound_cpu = evidence['bounds'].cpu().float()
        # Only written native scores enter the oracle; no row tail is copied.
        for row, extent in enumerate(extents):
            full = scores[row, :extent].cpu().float()
            for cluster in range((extent + 63) // 64):
                observed = full[cluster*64:min((cluster+1)*64, extent)].max()
                bound = bound_cpu[row, cluster]
                if torch.isfinite(observed):
                    assert bound >= observed, (row, cluster, observed, bound)
                else:
                    assert torch.isposinf(bound), (row, cluster, observed, bound)
        # Every evaluated compact score must keep its native BF16 bits, including
        # signed zeros and NaN encodings. Never copy an unwritten score tail.
        for row, count in enumerate(evidence['candidate_counts'].cpu().tolist()):
            slots = torch.arange(count, device='cuda', dtype=torch.int64)
            positions = evidence['candidates'][row, slots // 8].long() * 8 + slots % 8
            observed = evidence['compact_scores'][row, :count].view(torch.int16)
            full = scores[row].index_select(0, positions).view(torch.int16)
            torch.testing.assert_close(observed, full, atol=0, rtol=0)
    reindex = IndexerV41Plan(CANDIDATE_BLOCKS * 8, 256, candidates=True)
    next_args = list(args)
    next_args[0] = args[0].roll(1, dims=1)
    next_expected, _ = reindex.run(*next_args, blocks, counts)
    next_actual, _ = reindex.run(*next_args, evidence['source_blocks'], evidence['source_lengths'])
    torch.testing.assert_close(next_actual, next_expected, atol=0, rtol=0)
    graph = None
    if cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, evidence = candidate.run(*args)
    def replay():
        nonlocal actual, evidence
        if graph is None:
            actual, evidence = candidate.run(*args)
        else:
            graph.replay()
    replay()
    compare(expected)
    saved_lengths = args[5].clone()
    args[5].copy_(saved_lengths // 2)
    args[6].fill_(width - 17)
    reduced, _ = native()
    replay()
    compare(reduced)
    args[6].zero_()
    empty, _ = native()
    replay()
    compare(empty)
    args[5].copy_(saved_lengths)
    args[6].fill_(width)
    restored, _ = native()
    replay()
    compare(restored)
    previous_blocks = blocks.clone()
    row = rows - 1
    position = extents[row] - 65
    if position >= 0:
        page = args[4][row, position // 256].long()
        offset = position % 256
        args[3][page, offset*64:(offset+1)*64].fill_(0x77)
        args[3][page, 256*64+offset*4:256*64+(offset+1)*4].fill_(130)
        changed, _ = native()
        replay()
        compare(changed)
        if mode == 'separable':
            assert not torch.equal(changed, restored)
            assert not torch.equal(blocks, previous_blocks)
    next_expected, _ = reindex.run(*next_args, blocks, counts)
    next_actual, _ = reindex.run(*next_args, evidence['source_blocks'], evidence['source_lengths'])
    torch.testing.assert_close(next_actual, next_expected, atol=0, rtol=0)
    latencies = []
    # The native-only fallback is correctness-only. Never time an unmodified
    # released path as a new reference benchmark.
    if not evidence['native_fallback']:
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            replay()
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
    return {'width': width, 'extents': extents, 'mode': mode,
            'source_winners_identical': True, 'source_blocks_identical': True,
            'source_lengths_identical': True, 'reindex_identical': True,
            'graph_visibility_identical': True if cuda_graph else None, 'cache_mutation_identical': True, 'native_fallback': evidence['native_fallback'],
            'execution': 'cuda-graph' if cuda_graph else 'eager', 'archived_source_reference': source_reference,
            'retained_keys': evidence['candidate_counts'].cpu().tolist()
                             if not evidence['native_fallback'] else None,
            'output_sha256': hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest(),
            'blocks_sha256': hashlib.sha256(evidence['source_blocks'].cpu().numpy().tobytes()).hexdigest(),
            'lengths': evidence['source_lengths'].cpu().tolist(),
            'candidate_wall_ms': latencies,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-width', type=int, default=262144)
    parser.add_argument('--no-graph', action='store_true')
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--source-reference', action='store_true')
    options = parser.parse_args()
    result = {'schema_version': 1, 'passed': False, 'cases': []}
    try:
        cases = [(4096, [0, 37, 4093], 'random'),
                (65536, [0, 37, 32769, 65535], 'random'),
                (65536, [65535], 'separable'), (65536, [65535], 'ties'),
                (65536, [65535], 'extreme'), (65536, [65535], 'nan-key'),
                (65536, [65535], 'nan-weight'), (262144, [262143], 'random'),
                (1048576, [0, 37, 32769, 65535], 'random')]
        if options.quick:
            cases = [(65536, [65535], 'random'), (65536, [65535], 'separable')]
        for width, extents, mode in cases:
            if width > options.max_width:
                continue
            result['cases'].append(check(width, extents, mode, not options.no_graph, options.repeats, options.source_reference))
            print(json.dumps(result['cases'][-1]), flush=True)
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
