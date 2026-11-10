"""Degenerate/signed normal-range CPU checks; hardware gates remain separate."""
import torch
from bounds import bound
from witness import witnessed_keep


def check():
    torch.set_num_threads(8)
    generator = torch.Generator().manual_seed(20261104)
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float64)
    q = values[torch.randint(16, (32, 128), generator=generator)]/32
    keys = values[torch.randint(16, (2, 8, 128), generator=generator)]/32
    weights = torch.randn(32, generator=generator).to(torch.bfloat16).double()
    cases = [('signed', keys, weights), ('constant', keys[:, :1].expand_as(keys), weights),
             ('zero_keys', torch.zeros_like(keys), weights), ('zero_weights', keys, torch.zeros_like(weights)),
             ('positive', keys, weights.abs()), ('negative', keys, -weights.abs())]
    count = 0
    for name, points, w in cases:
        dots = (points.float() @ q.float().T).to(torch.bfloat16).float().clamp(min=0)
        contributions = (dots*w.float()).to(torch.bfloat16).float()
        total = torch.zeros(points.shape[:2], dtype=torch.float32)
        for h in range(32):
            total += contributions[:, :, h]
        observed = total.to(torch.bfloat16).double().amax(1)
        for kind in ('box', 'shape', 'shape-1', 'shape-7', 'member-oracle'):
            upper, _ = bound(points, q, w, kind, steps=8)
            assert bool(torch.isfinite(upper).all()), (name, kind, upper)
            assert bool((upper >= observed).all()), (name, kind, upper, observed)
            count += 1
    # A poor proposal and negative scores must still retain all actual ties.
    native = torch.arange(40000, dtype=torch.float64)/40000-4
    proposal = -native
    for source in (False, True):
        keep, accounting = witnessed_keep(proposal, native, native, source)
        if source:
            cutoff = native.reshape(-1, 8).amax(1)[:-1].topk(2047).values[-1]
            assert bool(keep[-8:].all())
        else:
            cutoff = native.topk(512).values[-1]
        assert bool(keep[native >= cutoff].all())
        assert accounting['total_keys_to_score_once'] == accounting['seed_keys_scored_once']+accounting['additional_keys_to_score']
    return {'cpu_math_cases': count, 'poor_proposal_witness_cases': 2,
            'scope': 'CPU rounding stand-in and degeneracies; not native GPU qualification'}


if __name__ == '__main__':
    import json
    torch.set_num_threads(8)
    print(json.dumps(check()), flush=True)
