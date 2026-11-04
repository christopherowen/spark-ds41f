"""Candidate-only diagnostic traces for the source cutoff; no native timing."""
import argparse
import json
from pathlib import Path
import traceback

import torch
from clustered_check import inputs
from vllm.models.deepseek_v4_1.tilelang.indexer import CANDIDATE_BLOCKS, IndexerV41Plan
from vllm.models.deepseek_v4_1.tilelang.source_indexer import SourceClusteredIndexerPlan
from vllm.models.deepseek_v4_1.tilelang.source_ties_reference import SourceClusteredIndexerPlan as ArchivedPlan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archived', action='store_true')
    options = parser.parse_args()
    result = {'schema_version': 1, 'passed': False, 'instrumented': True}
    try:
        width = 65536
        args = inputs(width, [width-1], 'separable', unwritten_cache=True)
        plan = (ArchivedPlan if options.archived else SourceClusteredIndexerPlan)(width, 256)
        blocks = torch.empty((1, CANDIDATE_BLOCKS), dtype=torch.int32, device='cuda')
        counts = torch.empty((1,), dtype=torch.int32, device='cuda')
        expected, _ = IndexerV41Plan(width, 256, candidate_blocks=True).run(*args,
            candidate_out=blocks, candidate_out_lengths=counts)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, evidence = plan.run(*args)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
        label = 'two-reductions' if options.archived else 'packed'
        path = Path('/c/clustered-indexer-profiles') / f'source-cutoff-{label}-65536-separable.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RuntimeError(f'refusing to overwrite trace {path}')
        profile.export_chrome_trace(str(path))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(evidence['source_blocks'], blocks, atol=0, rtol=0)
        torch.testing.assert_close(evidence['source_lengths'], counts, atol=0, rtol=0)
        result.update(passed=True, trace=str(path), replays=3, width=width, mode='separable',
                      label=label, identical=True, retained_keys=evidence['candidate_counts'].cpu().tolist())
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
