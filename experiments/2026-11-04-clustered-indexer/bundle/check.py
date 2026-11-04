"""Snapshot correctness gate; timings cover only the candidate, never the release."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch
from vllm.models.deepseek_v4_1.tilelang.clustered_indexer import ClusteredIndexerPlan
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan, page_bytes
from envelope_reference import reference_envelopes


def inputs(width, extents, mode, page_size=256, *, unwritten_cache=False):
    rows = len(extents)
    g = torch.Generator(device='cuda').manual_seed(width + rows)
    logical = width // page_size
    pool_pages = rows * logical + 1
    pool = torch.empty((pool_pages, page_bytes(page_size) + 64), dtype=torch.uint8, device='cuda')
    if not unwritten_cache:
        pool.fill_(255)
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
        if mode in ('tiny', 'largest'):
            sf.fill_(96 if mode == 'tiny' else 144)
        if mode == 'cancellation':
            keys.fill_(0x77)
            sf.fill_(127)
        if mode == 'subnormal-key' and extent:
            sf[0, 0] = 0
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
    if mode in ('tiny', 'largest'):
        scales.fill_(96 if mode == 'tiny' else 144)
        weights.fill_(2**20)
    if mode == 'cancellation':
        q.fill_(0x7f)
        scales.fill_(127)
        weights[:, ::2].fill_(2**20)
        weights[:, 1::2].fill_(-2**20)
    if mode == 'unsupported-query':
        scales[0, 0, 0] = 150
    if mode == 'subnormal-query':
        scales[0, 0, 0] = 0
    if mode == 'weight-range':
        weights[0, 0] = 2**21
    if mode == 'inf-weight':
        weights[0, 0] = float('inf')
    if mode == 'nan-query':
        scales[0, 0, 0] = 255
    if mode == 'nan-weight':
        weights[0, 0] = float('nan')
    lengths = torch.tensor(extents, dtype=torch.int32, device='cuda')
    active = torch.tensor([width], dtype=torch.int32, device='cuda')
    return q, scales, weights, pool, table, lengths, active


def check(width, extents, mode, repeats, *, unwritten_cache=False, cuda_graph=False,
          profile_label=None, envelope_reference=False, matrix_bounds=False):
    torch.cuda.reset_peak_memory_stats()
    args = inputs(width, extents, mode, unwritten_cache=unwritten_cache)
    candidate = ClusteredIndexerPlan(width, 256, seeds=min(32, width // 64),
                                     matrix_bounds=matrix_bounds)
    if envelope_reference:
        candidate.envelope = reference_envelopes(width, 256)
    reference = IndexerV41Plan(width, 256)
    expected, scores = reference.run(*args)  # correctness only; never timed
    actual, evidence = candidate.run(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Resource-layout change only: audit both interval endpoints against the
    # frozen previous reduction. Neither implementation is timed here.
    shape = (len(extents), width // 64, 128)
    lower = torch.empty(shape, dtype=torch.float32, device='cuda')
    upper = torch.empty_like(lower)
    old_lower, old_upper = torch.empty_like(lower), torch.empty_like(lower)
    envelope_args = (args[3][:, :page_bytes(256)], args[4], args[5], args[6])
    candidate.envelope(*envelope_args, lower, upper)
    reference_envelopes(width, 256)(*envelope_args, old_lower, old_upper)
    torch.testing.assert_close(lower, old_lower, atol=0, rtol=0)
    torch.testing.assert_close(upper, old_upper, atol=0, rtol=0)
    del lower, upper, old_lower, old_upper
    # The scorer intentionally leaves scores past each extent unwritten. Copy
    # only the visible slice, including when initcheck tracks CUDA copies.
    score_cpu = [scores[r, :extent].cpu() for r, extent in enumerate(extents)]
    bound_cpu = evidence['bounds'].cpu()
    for r, extent in enumerate(extents):
        if extent:
            for c in range((extent + 63) // 64):
                observed = score_cpu[r][c*64:min((c+1)*64, extent)].float().max()
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
    graph = None
    if cuda_graph:
        # Capture the complete candidate, including metadata rebuild and
        # compaction. The reference is used only for correctness outside it.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, evidence = candidate.run(*args)
        graph.replay()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        # Reuse the same graph/storage after a rollback and an empty active
        # domain, then restore the original visibility. All counts are read
        # on the device; no graph is recaptured for these changes.
        saved_lengths = args[5].clone()
        args[5].copy_(saved_lengths // 2)
        args[6].fill_(width - 17)
        reduced, _ = reference.run(*args)
        graph.replay()
        torch.testing.assert_close(actual, reduced, atol=0, rtol=0)
        args[6].zero_()
        empty, _ = reference.run(*args)
        graph.replay()
        torch.testing.assert_close(actual, empty, atol=0, rtol=0)
        args[5].copy_(saved_lengths)
        args[6].fill_(width)
        graph.replay()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    latencies = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if graph is None:
            candidate.run(*args)
        else:
            graph.replay()
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    trace = None
    if profile_label:
        target = Path('/c/clustered-indexer-profiles')
        target.mkdir(parents=True, exist_ok=True)
        trace = target / f'{profile_label}-{width}-{mode}.json'
        if trace.exists():
            raise RuntimeError(f'refusing to overwrite profiler evidence: {trace}')
        # No reference call or bound audit belongs in this timing interval.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        ) as profiler:
            for _ in range(3):
                if graph is None:
                    candidate.run(*args)
                else:
                    graph.replay()
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(trace))
        print(profiler.key_averages().table(
            sort_by='self_device_time_total', row_limit=40), flush=True)
    return {'width': width, 'extents': extents, 'mode': mode, 'identical': True,
            'bound_execution': 'tensor-core' if matrix_bounds else 'scalar',
            'envelope_execution': 'archived-reduction' if envelope_reference else 'streamed',
            'envelopes_identical': True,
            'unwritten_cache': unwritten_cache,
            'execution': 'cuda-graph' if cuda_graph else 'eager',
            'graph_dynamic_counts_identical': True if cuda_graph else None,
            'candidate_trace': str(trace) if trace else None,
            'candidate_counts': evidence['candidate_counts'].cpu().tolist(),
            'output_sha256': hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest(),
            'candidate_wall_ms': latencies,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--max-width', type=int, default=262144)
    parser.add_argument('--unwritten-cache', action='store_true')
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--profile-label')
    parser.add_argument('--matrix-bounds', action='store_true')
    parser.add_argument('--envelope-reference', action='store_true',
                        help='time the archived, unreleased candidate reduction as a control')
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
            (4096, [4093], 'tiny'),
            (4096, [4093], 'largest'),
            (4096, [4093], 'cancellation'),
            (4096, [4093], 'subnormal-key'),
            (4096, [4093], 'subnormal-query'),
            (4096, [4093], 'unsupported-query'),
            (4096, [4093], 'weight-range'),
            (4096, [4093], 'inf-weight'),
        ]:
            if width > options.max_width:
                continue
            result['cases'].append(check(width, extents, mode, options.repeats,
                                         unwritten_cache=options.unwritten_cache,
                                         cuda_graph=options.cuda_graph,
                                         envelope_reference=options.envelope_reference,
                                         matrix_bounds=options.matrix_bounds,
                                         profile_label=(options.profile_label
                                                        if width == options.max_width else None)))
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
