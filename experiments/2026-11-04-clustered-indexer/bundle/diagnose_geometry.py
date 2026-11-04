"""Untimed geometry screen. No selector changes or performance claims.

Affine majorants combine signed heads before taking a support bound. The native
comparison includes a deliberately loose rounding allowance. This diagnostic is
an idea screen, not a machine-arithmetic proof or an executable search candidate.
"""
import json
from pathlib import Path
import traceback

import torch
from diagnose_bounds import decode
from snapshot_layout import restore_layout
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan


def groups_for(keys, size, semantic):
    if not semantic:
        return list(torch.arange(len(keys)).split(size)), {'method': 'consecutive', 'size': size}
    # Cluster native eight-key blocks as indivisible units. A fixed seed and
    # fixed iteration count identify this diagnostic; these are not learned
    # parameters and do not enter the serving model.
    n = (len(keys) + 7) // 8
    padded = torch.cat((keys, keys[-1:].expand(n * 8 - len(keys), -1)))
    x = padded.reshape(n, 8, 128).mean(dim=1).float().cuda()
    count = max(1, n * 8 // size)
    generator = torch.Generator(device='cuda').manual_seed(20261104)
    centers = x[torch.randperm(n, device='cuda', generator=generator)[:count]].clone()
    for _ in range(12):
        distances = x.square().sum(1, keepdim=True) + centers.square().sum(1)[None] - 2 * x @ centers.T
        assignment = distances.argmin(dim=1)
        sums = torch.zeros_like(centers).index_add_(0, assignment, x)
        counts = torch.bincount(assignment, minlength=count)
        centers = torch.where(counts[:, None] > 0, sums / counts.clamp(min=1)[:, None], centers)
    assignment = assignment.cpu()
    groups = []
    for cluster in range(count):
        blocks = torch.where(assignment == cluster)[0]
        if len(blocks):
            positions = (blocks[:, None] * 8 + torch.arange(8)).flatten()
            groups.append(positions[positions < len(keys)])
    return groups, {'method': 'eight-key-block kmeans', 'target_keys': size, 'iterations': 12,
                    'seed': 20261104, 'groups': len(groups), 'maximum_group_keys': max(map(len, groups))}


def screen(keys, q, weights, scores, cutoff, groups):
    bounds = {'independent': [], 'affine_box': [], 'affine_support': []}
    sizes = []
    native_margin_max = 0.
    for positions in groups:
        points = keys[positions]
        lower, upper = points.amin(0), points.amax(0)
        center = points.mean(0)
        plus, minus = q.clamp(min=0), q.clamp(max=0)
        low = plus @ lower + minus @ upper
        high = plus @ upper + minus @ lower
        # A real-arithmetic ReLU secant for positive weights and a supporting
        # plane for negative weights. Their signed weighted sum majorizes f(k).
        cross = (low < 0) & (high > 0)
        slope = torch.where(low >= 0, torch.ones_like(low), torch.zeros_like(low))
        slope = torch.where(cross, high / (high-low).clamp(min=1e-300), slope)
        intercept = torch.where(cross, -low * slope, torch.zeros_like(low))
        slope = torch.where(weights < 0, (q @ center > 0).double(), slope)
        intercept = torch.where(weights < 0, torch.zeros_like(low), intercept)
        a = (weights * slope) @ q
        b = (weights * intercept).sum()
        box = a.clamp(min=0) @ upper + a.clamp(max=0) @ lower + b
        support = (points @ a).max() + b
        independent = (weights * torch.where(weights >= 0, high.clamp(min=0), low.clamp(min=0))).sum()
        # Conservative screening allowance: 1/32 of the absolute product norm
        # covers FP32 dot/sum and both BF16 stages in these certified normal
        # encodings. A runtime implementation would still need its own proof.
        norm = q.abs() @ torch.maximum(lower.abs(), upper.abs())
        margin = (weights.abs() * norm).sum() / 32 + 2**-30
        native_margin_max = max(native_margin_max, float(margin))
        maximum = scores[positions].max().double()
        for name, value in [('independent', independent), ('affine_box', box), ('affine_support', support)]:
            value = float(value + margin)
            if value < float(maximum):
                raise AssertionError((name, value, float(maximum)))
            bounds[name].append(value)
        sizes.append(len(positions))
    return {'retained_keys_at_ideal_native_cutoff': {
        name: sum(size for size, value in zip(sizes, values) if value >= cutoff)
        for name, values in bounds.items()},
        'bound_min': {name: min(values) for name, values in bounds.items()},
        'maximum_native_rounding_allowance': native_margin_max,
        'observed_native_bound_dominance': True}


def diagnose(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    extent, page = int(data['lengths'][0]), data['page_size']
    q = decode(data['q'], data['scales'].view(torch.uint8).reshape(1, 32, 4))[0]
    weights = data['weights'][0].double()
    positions = torch.arange(extent)
    physical, offsets = data['table'][0, positions // page].long(), positions % page
    packed = data['pool'][physical[:, None], offsets[:, None] * 64 + torch.arange(64)]
    scales = data['pool'][physical[:, None], page * 64 + offsets[:, None] * 4 + torch.arange(4)]
    assert bool(((scales >= 96) & (scales <= 144)).all())
    assert bool(((data['scales'].view(torch.uint8) >= 96) & (data['scales'].view(torch.uint8) <= 144)).all())
    assert bool(torch.isfinite(weights).all() and (weights.abs() <= 2**20).all())
    keys = decode(packed, scales)
    args, _ = restore_layout(data)
    _, scores = IndexerV41Plan(data['width'], page).run(*args)  # untimed native oracle
    scores = scores[0, :extent].cpu().float()
    if data['candidate_blocks']:
        blocks = (extent + 7) // 8
        padded = torch.cat((scores, scores[-1:].expand(blocks * 8 - extent)))
        ordinary = padded.reshape(blocks, 8).amax(1)[:-1]
        cutoff = float(ordinary.topk(min(2047, len(ordinary))).values[-1])
    else:
        cutoff = float(scores.topk(min(512, extent)).values[-1])
    result = {'snapshot': path.name, 'layer': data['capture_context']['layer_id'],
              'extent': extent, 'ideal_native_cutoff': cutoff, 'variants': [],
              'scope': 'untimed idea screen; observed dominance is not runtime qualification'}
    for semantic in (False, True):
        for size in (8, 64):
            groups, identity = groups_for(keys, size, semantic)
            value = dict(identity, **screen(keys, q, weights, scores, cutoff, groups))
            result['variants'].append(value)
            print(json.dumps({'snapshot': path.name, **value}), flush=True)
    return result


def main():
    result = {'schema_version': 1, 'passed': False, 'cases': []}
    try:
        files = sorted(Path('/snapshots').glob('rank-0-*.pt'))
        if len(files) != 4:
            raise RuntimeError('expected four rank0 layer/interface samples')
        for path in files:
            result['cases'].append(diagnose(path))
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
