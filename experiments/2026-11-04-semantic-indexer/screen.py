"""Untimed semantic geometry screen; never a serving or timing implementation.

Any negative-head slope in [0,1] gives a valid real-arithmetic majorant.
Optimization only tightens that majorant; convergence is not a correctness
assumption. Native rounding is bounded separately. F64 calculations and observed
dominance do not qualify a future F32/TileLang implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import torch
from diagnose_bounds import decode
from snapshot_layout import restore_layout
from witness import witnessed_keep
from vllm.models.deepseek_v4_1.tilelang.indexer import CANDIDATE_BLOCKS, IndexerV41Plan


from bounds import partition, bound


def diagnose(path, kinds, sizes, selection_gates=False):
    torch.cuda.reset_peak_memory_stats()
    data = torch.load(path, map_location='cpu', weights_only=True)
    n, page = int(data['lengths'][0]), int(data['page_size'])
    q = decode(data['q'], data['scales'].view(torch.uint8).reshape(1, 32, 4))[0].cuda()
    weights = data['weights'][0].double().cuda()
    positions = torch.arange(n)
    physical, offsets = data['table'][0, positions//page].long(), positions % page
    packed = data['pool'][physical[:, None], offsets[:, None]*64+torch.arange(64)]
    scales = data['pool'][physical[:, None], page*64+offsets[:, None]*4+torch.arange(4)]
    assert bool(((scales >= 96) & (scales <= 144)).all())
    assert bool(((data['scales'].view(torch.uint8) >= 96) & (data['scales'].view(torch.uint8) <= 144)).all())
    assert bool(torch.isfinite(weights).all() and (weights.abs() <= 2**20).all())
    keys = decode(packed, scales).cuda()
    args, span = restore_layout(data)
    source = bool(data['candidate_blocks'])
    reference = IndexerV41Plan(data['width'], page, candidate_blocks=source)
    blocks = torch.empty((1, CANDIDATE_BLOCKS), dtype=torch.int32, device='cuda')
    counts = torch.empty((1,), dtype=torch.int32, device='cuda')
    # Native scorer is correctness-only: no timer and no baseline remeasurement.
    expected, native_scores = reference.run(*args, **(
        {'candidate_out': blocks, 'candidate_out_lengths': counts} if source else {}))
    scores = native_scores[0, :n].double()
    if source:
        padded = torch.cat((scores, scores[-1:].expand((-n) % 8)))
        ordinary = padded.reshape(-1, 8).amax(1)[:-1]
        cutoff = ordinary.topk(min(2047, len(ordinary))).values[-1]
    else:
        cutoff = scores.topk(min(512, n)).values[-1]
    result = {'snapshot': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'extent': n, 'source': source, 'ideal_native_cutoff': float(cutoff),
              'native_pool_span_bytes': span, 'variants': []}
    for semantic in (False, True):
        for size in sizes:
            ids = partition(keys, size, semantic)
            points = keys[ids]
            for kind in kinds:
                bounds, error = bound(points, q, weights, kind)
                observed = scores[ids].amax(1)
                if not bool((bounds >= observed).all()):
                    raise AssertionError((path.name, semantic, size, kind, float((bounds-observed).min())))
                retained = torch.zeros(n, dtype=torch.bool, device=keys.device)
                retained[ids[bounds >= cutoff].flatten()] = True
                if source:
                    retained[((n-1)//8)*8:] = True
                block_ids = torch.arange(n, device=keys.device)[retained]//8
                expanded = torch.zeros((n+7)//8, dtype=torch.bool, device=keys.device)
                expanded[block_ids] = True
                expanded_keys = min(n, 8*int(expanded.sum()))
                # All actual winners must survive even before eight-key expansion.
                assert bool(retained[scores >= cutoff].all())
                item = {'semantic': semantic, 'leaf_keys': size, 'groups': len(ids), 'bound': kind,
                        'retained_keys': int(retained.sum()), 'expanded_eight_key_positions_upper': expanded_keys,
                        'bound_min': float(bounds.min()), 'minimum_native_dominance_slack': float((bounds-observed).min()),
                        'maximum_native_rounding_allowance': float(error.max()),
                        'native_bound_dominance': True, 'all_threshold_ties_retained': True}
                center_dots = points.mean(1) @ q.T
                center_estimate = (center_dots.clamp(min=0)*weights[None]).sum(1)
                proposal = torch.full((n,), -float('inf'), device=keys.device, dtype=torch.float64)
                per_key_bound = torch.full_like(proposal, -float('inf'))
                proposal.scatter_reduce_(0, ids.flatten(), center_estimate[:, None].expand_as(ids).flatten(), reduce='amax')
                per_key_bound.scatter_reduce_(0, ids.flatten(), bounds[:, None].expand_as(ids).flatten(), reduce='amax')
                witnessed, witness = witnessed_keep(proposal, per_key_bound, scores, source)
                assert bool(witnessed[scores >= cutoff].all())
                item.update(witness)
                if selection_gates:
                    masked = torch.full_like(native_scores, -float('inf'))
                    masked[0, :n] = torch.where(witnessed, native_scores[0, :n], -float('inf'))
                    actual = torch.empty_like(expected)
                    values = torch.empty_like(expected, dtype=torch.bfloat16)
                    reference.select.run(masked, args[5], args[6], actual, values)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    item['native_selection_identical'] = True
                    if source:
                        actual_blocks, actual_counts = torch.empty_like(blocks), torch.empty_like(counts)
                        block_values = torch.empty_like(blocks, dtype=torch.bfloat16)
                        reference.select_blocks.run(masked, args[5], args[6], actual_blocks,
                                                    block_values, counts=actual_counts)
                        torch.testing.assert_close(actual_blocks, blocks, atol=0, rtol=0)
                        torch.testing.assert_close(actual_counts, counts, atol=0, rtol=0)
                        reindex = IndexerV41Plan(CANDIDATE_BLOCKS*8, page, candidates=True)
                        next_args = [args[0].roll(1, dims=1), *args[1:]]
                        next_expected, _ = reindex.run(*next_args, blocks, counts)
                        next_actual, _ = reindex.run(*next_args, actual_blocks, actual_counts)
                        torch.testing.assert_close(next_actual, next_expected, atol=0, rtol=0)
                        item.update(native_source_blocks_identical=True, native_source_count_identical=True,
                                    native_reindex_identical=True)
                result['variants'].append(item)
                print(json.dumps({'snapshot': path.name, **item}), flush=True)
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bounds', default='box,shape,member-oracle')
    parser.add_argument('--sizes', default='8,64')
    parser.add_argument('--all-ranks', action='store_true')
    parser.add_argument('--selection-gates', action='store_true')
    options = parser.parse_args()
    torch.set_num_threads(8)
    result = {'schema_version': 1, 'passed': False, 'cases': [],
              'scope': 'untimed geometry screen with ideal cutoff; no executable search or speed claim'}
    try:
        files = sorted(Path('/snapshots').glob('rank-*.pt' if options.all_ranks else 'rank-0-*.pt'))
        if len(files) != (12 if options.all_ranks else 4):
            raise RuntimeError('unexpected snapshot corpus size')
        for path in files:
            result['cases'].append(diagnose(path, options.bounds.split(','),
                                           [int(x) for x in options.sizes.split(',')], options.selection_gates))
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
