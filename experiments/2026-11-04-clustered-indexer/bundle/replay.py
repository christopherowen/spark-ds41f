"""Replay node-local real indexer snapshots; never time the native scorer."""
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch
from snapshot_layout import restore_layout
from vllm.models.deepseek_v4_1.tilelang.clustered_indexer import ClusteredIndexerPlan
from vllm.models.deepseek_v4_1.tilelang.source_indexer import SourceClusteredIndexerPlan
from vllm.models.deepseek_v4_1.tilelang.indexer import CANDIDATE_BLOCKS, IndexerV41Plan


def replay_file(path, cuda_graph=True):
    torch.cuda.reset_peak_memory_stats()
    data = torch.load(path, map_location='cpu', weights_only=True)
    args, address_span = restore_layout(data)
    width, page_size = data['width'], data['page_size']
    source = data['candidate_blocks']
    rows = args[0].shape[0]
    reference = IndexerV41Plan(width, page_size, candidate_blocks=source)
    candidate = SourceClusteredIndexerPlan(width, page_size) if source else ClusteredIndexerPlan(width, page_size, matrix_bounds=True)
    blocks = torch.empty((rows, CANDIDATE_BLOCKS), dtype=torch.int32, device='cuda')
    counts = torch.empty((rows,), dtype=torch.int32, device='cuda')

    def native():
        return reference.run(*args, **({'candidate_out': blocks, 'candidate_out_lengths': counts} if source else {}))

    expected, scores = native()
    actual, evidence = candidate.run(*args)
    fallback = evidence.get('native_fallback', False)

    def compare(expected):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if source:
            torch.testing.assert_close(evidence['source_blocks'], blocks, atol=0, rtol=0)
            torch.testing.assert_close(evidence['source_lengths'], counts, atol=0, rtol=0)

    compare(expected)
    extent = int(data['lengths'][0])
    if not fallback:
        full, bounds = scores[:, :extent].cpu().float(), evidence['bounds'].cpu().float()
        for cluster in range((extent + 63) // 64):
            observed = full[0, cluster*64:min((cluster+1)*64, extent)].max()
            bound = bounds[0, cluster]
            if torch.isfinite(observed):
                assert bound >= observed, (cluster, observed, bound)
            else:
                assert torch.isposinf(bound), (cluster, observed, bound)
        count = int(evidence['candidate_counts'][0])
        slots = torch.arange(count, device='cuda', dtype=torch.int64)
        positions = evidence['candidates'][0, slots // 8].long() * 8 + slots % 8
        compact_scores = evidence['compact_scores'] if source else candidate.final_plan.run(
            *args, evidence['candidates'], evidence['candidate_counts'])[1]
        torch.testing.assert_close(compact_scores[0, :count].view(torch.int16),
            scores[0].index_select(0, positions).view(torch.int16), atol=0, rtol=0)
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
    lengths, active = args[5].clone(), args[6].clone()
    for half, empty in [(True, False), (False, True), (False, False)]:
        args[5].copy_(lengths // 2 if half else lengths)
        args[6].copy_(torch.zeros_like(active) if empty else active)
        changed, _ = native()
        replay()
        compare(changed)
    if source:
        reindex = IndexerV41Plan(CANDIDATE_BLOCKS * 8, page_size, candidates=True)
        # Another deterministic query exercises the exact retained handoff.
        next_args = list(args)
        next_args[0] = args[0].roll(1, dims=1)
        expected_next, _ = reindex.run(*next_args, blocks, counts)
        actual_next, _ = reindex.run(*next_args, evidence['source_blocks'], evidence['source_lengths'])
        torch.testing.assert_close(actual_next, expected_next, atol=0, rtol=0)
    samples = []
    if not fallback:
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
    result = {'snapshot': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source': data['source'], 'capture_context': data.get('capture_context'), 'rank': data['rank'], 'pool_ordinal': data['pool_ordinal'],
              'extent': extent, 'planned_width': width, 'active': int(active[0]),
              'layout': {k: v for k, v in data['layout'].items() if k != 'physical_ids'},
              'address_span_bytes': address_span, 'source_interface': source,
              'identical': True, 'visibility_identical': True, 'execution': 'cuda-graph' if cuda_graph else 'eager', 'reindex_identical': True if source else None,
              'native_fallback': fallback,
              'candidate_counts': evidence['candidate_counts'].cpu().tolist() if not fallback else None,
              'output_sha256': hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest(),
              'candidate_wall_ms': samples, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    if source:
        result['blocks_sha256'] = hashlib.sha256(evidence['source_blocks'].cpu().numpy().tobytes()).hexdigest()
    return result


def main():
    result = {'schema_version': 1, 'passed': False, 'cases': []}
    try:
        files = sorted(Path('/snapshots').glob('rank-*.pt'))
        if not files:
            raise RuntimeError('no real indexer snapshots mounted')
        for path in files:
            case = replay_file(path)
            result['cases'].append(case)
            print(json.dumps(case), flush=True)
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
