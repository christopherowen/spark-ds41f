"""Untimed semantic geometry screen; never a serving or timing implementation.

Any negative-head slope in [0,1] gives a valid real-arithmetic majorant.
Optimization only tightens that majorant; convergence is not a correctness
assumption. Native rounding is bounded separately. F64 calculations and observed
dominance do not qualify a future F32/TileLang implementation.
"""
import hashlib
import json
from pathlib import Path
import traceback

import torch
from diagnose_bounds import decode
from snapshot_layout import restore_layout
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan


def partition(keys, size, semantic):
    """Balanced, query-independent cosine tree, retaining every original key."""
    n = len(keys)
    padded = 1 << (n - 1).bit_length()
    generator = torch.Generator(device=keys.device).manual_seed(20261104)
    order = torch.randperm(n, generator=generator, device=keys.device)
    ids = torch.cat((order, order[:padded-n]))
    if not semantic:
        ids = torch.arange(padded, device=keys.device) % n
    if semantic:
        features = keys / keys.norm(dim=-1, keepdim=True).clamp(min=2**-50)
        ids = ids.reshape(1, -1)
        while ids.shape[1] > size:
            points = features[ids]
            center = points.mean(1, keepdim=True)
            first = (points-center).square().sum(-1).argmax(1)
            rows = torch.arange(len(ids), device=keys.device)
            a = points[rows, first]
            second = (points-a[:, None]).square().sum(-1).argmax(1)
            b = points[rows, second]
            for _ in range(4):
                projection = (points * (a-b)[:, None]).sum(-1)
                permutation = projection.argsort(dim=1, stable=True)
                half = points.shape[1] // 2
                left, right = permutation[:, :half], permutation[:, half:]
                a = points.gather(1, left[:, :, None].expand(-1, -1, 128)).mean(1)
                b = points.gather(1, right[:, :, None].expand(-1, -1, 128)).mean(1)
            ids = ids.gather(1, permutation).reshape(-1, ids.shape[1] // 2)
        return ids
    return ids.reshape(-1, size)


def geometry(points, q, kind):
    center = points.mean(1)
    delta = points-center[:, None]
    lower, upper = points.amin(1), points.amax(1)
    if kind == 'box':
        low = lower @ q.clamp(min=0).T + upper @ q.clamp(max=0).T
        high = upper @ q.clamp(min=0).T + lower @ q.clamp(max=0).T

        def support(a):
            return (a.clamp(min=0)*upper + a.clamp(max=0)*lower).sum(-1)
    elif kind == 'shape':
        # Full small-leaf shape; residuals are explicitly recomputed rather than
        # assuming the computed singular vectors are orthogonal or exact.
        _, singular, basis = torch.linalg.svd(delta, full_matrices=False)
        rank = min(8, points.shape[1]-1)
        basis, singular = basis[:, :rank], singular[:, :rank]
        coefficients = delta @ basis.transpose(-1, -2)
        residual = delta-coefficients @ basis
        radius = residual.norm(dim=-1).amax(1) + 2**-40*points.abs().amax((1, 2))
        scale = singular.clamp(min=2**-50)
        extent = (coefficients/scale[:, None]).norm(dim=-1).amax(1)
        shape = basis*scale[:, :, None]
        spread = extent[:, None]*(shape @ q.T).square().sum(1).sqrt()
        spread += radius[:, None]*q.norm(dim=-1)[None]
        low, high = center @ q.T-spread, center @ q.T+spread

        def support(a):
            return ((a*center).sum(-1) + extent*(shape @ a[:, :, None]).square().sum((1, 2)).sqrt()
                    + radius*a.norm(dim=-1))
    elif kind == 'member-oracle':
        dots = points @ q.T
        low, high = dots.amin(1), dots.amax(1)

        def support(a):
            return (points @ a[:, :, None]).amax((1, 2))
    else:
        raise ValueError(kind)
    # Cover F64 reduction/cancellation in the diagnostic interval itself.
    absolute = torch.maximum(lower.abs(), upper.abs()) @ q.abs().T
    slack = absolute*2**-40 + 2**-50
    return low-slack, high+slack, absolute, support


def native_upper(real_upper, low, high, absolute, weights):
    """Normal-range FP32 dot/ordered sum and two BF16 rounding stages.

    FP4 products are exact in FP32. Multiplication of two BF16 values is exact
    in FP32 in the supported normal range. Apply final BF16 rounding monotonically
    to a bound on the pre-round sum instead of adding an absolute final error.
    """
    unit = 2**-24
    accumulation = (127*unit/(1-127*unit))*absolute
    dot_error = accumulation + (torch.maximum(low.abs(), high.abs())+accumulation)/256 + 2**-50
    magnitude = weights.abs()[None]*(high.clamp(min=0)+dot_error)
    product_error = magnitude/256 + 2**-50
    error = (weights.abs()[None]*dot_error+product_error).sum(1)
    error += (31*unit/(1-31*unit))*(magnitude+product_error).sum(1)
    upper = real_upper+error
    # Guard conversion through FP32, plus F64 algebra in the affine objective.
    upper += (upper.abs()+magnitude.sum(1))*2**-22 + 2**-40
    return upper.to(torch.bfloat16).double(), error


def bound(points, q, weights, kind, steps=48):
    low, high, absolute, support = geometry(points, q, kind)
    cross = (low < 0) & (high > 0)
    secant = torch.where(low >= 0, torch.ones_like(low), torch.zeros_like(low))
    secant = torch.where(cross, high/(high-low).clamp(min=2**-100), secant).clamp(0, 1)
    intercept = torch.maximum(torch.zeros_like(low), torch.maximum(-secant*low, (1-secant)*high))
    positive = weights >= 0
    intercept = (intercept*weights.clamp(min=0)[None]).sum(1)
    initial = (points.mean(1) @ q.T > 0).double()
    slopes = torch.where(positive[None], secant, initial).detach().requires_grad_()
    optimizer = torch.optim.Adam([slopes], lr=.08)
    best = torch.full((len(points),), float('inf'), device=points.device, dtype=torch.float64)
    for iteration in range(steps+1):
        a = (slopes*weights[None]) @ q
        objective = support(a)+intercept
        best = torch.minimum(best, objective.detach())
        if iteration == steps:
            break
        optimizer.zero_grad()
        objective.sum().backward()
        optimizer.step()
        with torch.no_grad():
            slopes.clamp_(0, 1)
            slopes[:, positive] = secant[:, positive]
    result, error = native_upper(best, low, high, absolute, weights)
    return result.detach(), error.detach()


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
    # Native scorer is correctness-only: no timer and no baseline remeasurement.
    _, scores = IndexerV41Plan(data['width'], page).run(*args)
    scores = scores[0, :n].double()
    source = bool(data['candidate_blocks'])
    if source:
        padded = torch.cat((scores, scores[-1:].expand((-n) % 8)))
        ordinary = padded.reshape(-1, 8).amax(1)[:-1]
        cutoff = ordinary.topk(min(2047, len(ordinary))).values[-1]
    else:
        cutoff = scores.topk(min(512, n)).values[-1]
    result = {'snapshot': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'extent': n, 'source': source, 'ideal_native_cutoff': float(cutoff),
              'native_pool_span_bytes': span, 'variants': []}
    del args
    for semantic in (False, True):
        for size in (8, 64):
            ids = partition(keys, size, semantic)
            points = keys[ids]
            for kind in ('box', 'shape', 'member-oracle'):
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
                result['variants'].append(item)
                print(json.dumps({'snapshot': path.name, **item}), flush=True)
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    return result


def main():
    torch.set_num_threads(8)
    result = {'schema_version': 1, 'passed': False, 'cases': [],
              'scope': 'untimed geometry screen with ideal cutoff; no executable search or speed claim'}
    try:
        files = sorted(Path('/snapshots').glob('rank-0-*.pt'))
        if len(files) != 4:
            raise RuntimeError('expected four rank0 layer/interface snapshots')
        for path in files:
            result['cases'].append(diagnose(path))
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        print('RESULT '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
