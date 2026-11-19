"""Untimed query-independent projection screen with exact native score gates.

The map only bounds search. It never replaces checkpoint weights or scores.
This F64 diagnostic deliberately omits metadata construction/query timing.
"""
import hashlib
import json
from pathlib import Path
import traceback

import torch
from diagnose_bounds import decode
from snapshot_layout import restore_layout
from witness import witnessed_keep
from vllm.models.deepseek_v4_1.tilelang.indexer import CANDIDATE_BLOCKS, IndexerV41Plan


def diagnose(path):
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
    plan = IndexerV41Plan(data['width'], page, candidate_blocks=source)
    blocks = torch.empty((1, CANDIDATE_BLOCKS), dtype=torch.int32, device='cuda')
    counts = torch.empty((1,), dtype=torch.int32, device='cuda')
    expected, native = plan.run(*args, **(
        {'candidate_out': blocks, 'candidate_out_lengths': counts} if source else {}))
    scores = native[0, :n].double()
    if source:
        padded = torch.cat((scores, scores[-1:].expand((-n) % 8)))
        ordinary = padded.reshape(-1, 8).amax(1)[:-1]
        cutoff = ordinary.topk(min(2047, len(ordinary))).values[-1]
    else:
        cutoff = scores.topk(min(512, n)).values[-1]
    # Centering improves the retained variance without changing the certificate.
    center = keys.mean(0)
    delta = keys-center
    eigenvalues, eigenvectors = torch.linalg.eigh(delta.T @ delta)
    basis = eigenvectors.flip(1)
    absolute = keys.abs() @ q.abs().T
    native_dot_error = (127*2**-24/(1-127*2**-24))*absolute
    result = {'snapshot': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'extent': n, 'source': source, 'ideal_native_cutoff': float(cutoff),
              'native_pool_span_bytes': span, 'variants': []}
    for rank in (8, 16, 32, 64, 96, 128):
        selected = basis[:, :rank]
        kp, qp = delta @ selected, q @ selected
        kr, qr = delta-kp @ selected.T, q-qp @ selected.T
        # These identities hold even for a computed nonorthogonal basis:
        # q.k = q.center + qp.kp + qr.kr + qp.(kr.B).
        parallel = (kr @ selected).norm(dim=1)
        radius = kr.norm(dim=1)
        error = radius[:, None]*qr.norm(dim=1)[None] + parallel[:, None]*qp.norm(dim=1)[None]
        approximate = kp @ qp.T + (q @ center)[None]
        error += native_dot_error + absolute*2**-40 + 2**-50
        endpoint = approximate + torch.where(weights[None] >= 0, error, -error)
        endpoint += torch.where(weights[None] >= 0, endpoint.abs()*2**-22+2**-40,
                                -endpoint.abs()*2**-22-2**-40)
        dots = endpoint.to(torch.bfloat16).float().clamp(min=0)
        contributions = (dots*weights.float()[None]).to(torch.bfloat16).float()
        total = torch.zeros(n, dtype=torch.float32, device='cuda')
        for head in range(32):
            total = total+contributions[:, head]
        upper = total.to(torch.bfloat16).double()
        assert bool((upper >= scores).all()), (path.name, rank, float((upper-scores).min()))
        retained = upper >= cutoff
        if source:
            retained[((n-1)//8)*8:] = True
        assert bool(retained[scores >= cutoff].all())
        proposal = (approximate.clamp(min=0)*weights[None]).sum(1)
        witnessed, witness = witnessed_keep(proposal, upper, scores, source)
        assert bool(witnessed[scores >= cutoff].all())
        masked = torch.full_like(native, -float('inf'))
        masked[0, :n] = torch.where(witnessed, native[0, :n], -float('inf'))
        actual, values = torch.empty_like(expected), torch.empty_like(expected, dtype=torch.bfloat16)
        plan.select.run(masked, args[5], args[6], actual, values)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if source:
            actual_blocks, actual_counts = torch.empty_like(blocks), torch.empty_like(counts)
            plan.select_blocks.run(masked, args[5], args[6], actual_blocks,
                                   torch.empty_like(blocks, dtype=torch.bfloat16), counts=actual_counts)
            torch.testing.assert_close(actual_blocks, blocks, atol=0, rtol=0)
            torch.testing.assert_close(actual_counts, counts, atol=0, rtol=0)
        expanded = torch.zeros((n+7)//8, dtype=torch.bool, device='cuda')
        expanded[torch.arange(n, device='cuda')[retained]//8] = True
        item = {'rank': rank, 'retained_keys': int(retained.sum()),
                'expanded_eight_key_positions_upper': min(n, 8*int(expanded.sum())),
                'variance_fraction': float(eigenvalues[-rank:].sum()/eigenvalues.sum()),
                'projection_bytes_per_key_if_bf16': rank*2+4,
                'maximum_residual_norm': float(radius.max()),
                'minimum_native_dominance_slack': float((upper-scores).min()),
                'native_bound_dominance': True, 'native_selection_identical': True,
                'native_source_handoff_identical': True if source else None}
        item.update(witness)
        result['variants'].append(item)
        print(json.dumps({'snapshot': path.name, **item}), flush=True)
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    return result


def main():
    torch.set_num_threads(8)
    result = {'schema_version': 1, 'passed': False, 'cases': [],
              'scope': 'untimed ideal-cutoff projection screen; no executable search or speed claim'}
    try:
        files = sorted(Path('/snapshots').glob('rank-0-*.pt'))
        if len(files) != 4:
            raise RuntimeError('expected four rank0 snapshots')
        for path in files:
            torch.cuda.reset_peak_memory_stats()
            result['cases'].append(diagnose(path))
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
