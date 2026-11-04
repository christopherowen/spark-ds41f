"""CPU screen: fit auxiliary maps on other ranks' queries, hold rank0 out.

Uses a CPU arithmetic stand-in, not a native GPU oracle or performance test.
No map replaces a model weight. All activations and fitted maps remain node-local.
"""
import json
from pathlib import Path
import torch


def decode(packed, scales):
    packed, scales = packed.view(torch.uint8), scales.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(*packed.shape[:-1], 128).long()
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float64)
    return values[codes]*torch.exp2(scales.double().repeat_interleave(32, dim=-1)-127)


def query(data):
    return decode(data['q'], data['scales'].view(torch.uint8).reshape(1, 32, 4))[0]


def main():
    torch.set_num_threads(2)
    root = Path('/snapshots')
    result = {'passed': True, 'scope': 'CPU held-out projection idea screen; no native GPU/output/performance qualification', 'cases': []}
    for path in sorted(root.glob('rank-0-*.pt')):
        data = torch.load(path, map_location='cpu', weights_only=True)
        others = [torch.load(root/path.name.replace('rank-0-', f'rank-{rank}-'), map_location='cpu', weights_only=True)
                  for rank in (1, 2)]
        assert all(x['capture_context']['layer_id'] == data['capture_context']['layer_id'] for x in others)
        q, weights = query(data), data['weights'][0].double()
        train = torch.cat([query(x)*x['weights'][0].double().abs().sqrt()[:, None] for x in others])
        _, vectors = torch.linalg.eigh(train.T @ train)
        n, page = int(data['lengths'][0]), int(data['page_size'])
        positions = torch.arange(n)
        physical, offsets = data['table'][0, positions//page].long(), positions % page
        packed = data['pool'][physical[:, None], offsets[:, None]*64+torch.arange(64)]
        scales = data['pool'][physical[:, None], page*64+offsets[:, None]*4+torch.arange(4)]
        keys = decode(packed, scales)
        dots = (keys @ q.T).to(torch.bfloat16).float().clamp(min=0)
        products = (dots*weights.float()[None]).to(torch.bfloat16).float()
        score = torch.zeros(n, dtype=torch.float32)
        for h in range(32):
            score += products[:, h]
        score = score.to(torch.bfloat16).float()
        if data['candidate_blocks']:
            padded = torch.cat((score, score[-1:].expand((-n) % 8)))
            ordinary = padded.reshape(-1, 8).amax(1)[:-1]
            cutoff = ordinary.topk(min(2047, len(ordinary))).values[-1]
        else:
            cutoff = score.topk(min(512, n)).values[-1]
        center = keys.mean(0).float().double()
        delta = keys-center
        absolute = keys.abs() @ q.abs().T
        for rank in (8, 16, 32, 64, 96, 128):
            basis = vectors.flip(1)[:, :rank].float().double()
            kp = (delta @ basis).to(torch.bfloat16).double()
            exact_qp = q @ basis
            qp = exact_qp.float().double()
            kr, qr = delta-kp @ basis.T, q-qp @ basis.T
            # Include explicit coefficient conversion and basis reconstruction.
            error = kr.norm(dim=1)[:, None]*qr.norm(dim=1)[None]
            error += (kr @ basis).norm(dim=1)[:, None]*qp.norm(dim=1)[None]
            error += kp.norm(dim=1)[:, None]*(exact_qp-qp).norm(dim=1)[None]
            error += (127*2**-24/(1-127*2**-24))*absolute + absolute*2**-40+2**-40
            approximate = kp @ qp.T + (q @ center)[None]
            endpoint = approximate+torch.where(weights[None] >= 0, error, -error)
            endpoint += torch.where(weights[None] >= 0, endpoint.abs()*2**-22+2**-40, -endpoint.abs()*2**-22-2**-40)
            terms = (endpoint.to(torch.bfloat16).float().clamp(min=0)*weights.float()[None]).to(torch.bfloat16).float()
            upper = torch.zeros(n, dtype=torch.float32)
            for h in range(32):
                upper += terms[:, h]
            upper = upper.to(torch.bfloat16).float()
            assert bool((upper >= score).all())
            case = {'snapshot': path.name, 'map_rank': rank, 'extent': n,
                    'retained_at_cpu_cutoff': int((upper >= cutoff).sum()),
                    'maximum_heldout_query_relative_residual': float((qr.norm(dim=1)/q.norm(dim=1).clamp(min=2**-50)).max()),
                    'coefficient_dtype': 'BF16', 'map_dtype': 'FP32',
                    'nominal_bytes_per_key': rank*2+8, 'cpu_bound_dominance': True}
            result['cases'].append(case)
            print(json.dumps(case), flush=True)
    assert len(result['cases']) == 24
    print('RESULT '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
