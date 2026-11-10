"""Diagnostic real-arithmetic bounds and native rounding allowance.

No model, device, runtime or serving imports; hardware gates live in screen.py.
"""
import torch


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
    elif kind == 'shape' or kind.startswith('shape-'):
        # Full small-leaf shape; residuals are explicitly recomputed rather than
        # assuming the computed singular vectors are orthogonal or exact.
        _, singular, basis = torch.linalg.svd(delta, full_matrices=False)
        requested_rank = int(kind.split('-')[1]) if '-' in kind else 8
        rank = min(requested_rank, points.shape[1]-1)
        basis, singular = basis[:, :rank], singular[:, :rank]
        coefficients = delta @ basis.transpose(-1, -2)
        residual = delta-coefficients @ basis
        radius = residual.norm(dim=-1).amax(1) + 2**-40*points.abs().amax((1, 2))
        scale = singular.clamp(min=2**-50)
        extent = (coefficients/scale[:, None]).norm(dim=-1).amax(1)
        shape = basis*scale[:, :, None]
        # Retain head correlations in the residual complement too. Explicit
        # recomposition covers nonzero computed basis/coefficients residuals.
        q_basis = basis @ q.T
        q_residual = q[None]-q_basis.transpose(1, 2) @ basis
        parallel_radius = (residual @ basis.transpose(1, 2)).norm(dim=-1).amax(1)
        parallel_radius += 2**-40*points.abs().amax((1, 2))
        spread = extent[:, None]*torch.linalg.vector_norm(shape @ q.T, dim=1)
        spread += radius[:, None]*q_residual.norm(dim=-1)
        spread += parallel_radius[:, None]*q_basis.norm(dim=1)
        if kind == 'shape':
            spread = extent[:, None]*torch.linalg.vector_norm(shape @ q.T, dim=1)
            spread += radius[:, None]*q.norm(dim=-1)[None]
        low, high = center @ q.T-spread, center @ q.T+spread

        def support(a):
            if kind == 'shape':
                return ((a*center).sum(-1) + extent*torch.linalg.vector_norm(shape @ a[:, :, None], dim=(1, 2))
                        + radius*a.norm(dim=-1))
            a_basis = (basis @ a[:, :, None]).squeeze(-1)
            a_residual = a-(a_basis[:, None] @ basis).squeeze(1)
            return ((a*center).sum(-1) + extent*torch.linalg.vector_norm(shape @ a[:, :, None], dim=(1, 2))
                    + radius*a_residual.norm(dim=-1) + parallel_radius*a_basis.norm(dim=-1))
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


