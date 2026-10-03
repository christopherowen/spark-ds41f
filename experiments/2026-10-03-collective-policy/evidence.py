#!/usr/bin/env python3
"""Normalize the completed four-path screen; emit evidence, never dispatch rules."""
import hashlib
import json
from pathlib import Path
from statistics import median

SOURCE = Path(__file__).resolve().parents[1] / '2026-10-02-mesh4-fourpaths/hardware/summary.json'
ARMS = {
    'run01-control': 'forward-two', 'run06-control-repeat': 'forward-two',
    'run02-four': 'forward-four', 'run05-four-repeat': 'forward-four',
    'run03-rotate': 'forward-four-rotate', 'run04-rotate-repeat': 'forward-four-rotate',
    'run07-relay': 'cpu-relay',
}

def report():
    raw = SOURCE.read_bytes()
    runs = json.loads(raw)
    cells = {}
    for name, arm in ARMS.items():
        run = runs[name]
        if run['ranks_passed'] != 4 or any(r['exit'] for r in run['results']):
            raise ValueError(f'{name}: incomplete or unsuccessful run')
        for row in run['latency']:
            itemsize = {'torch.bfloat16': 2, 'torch.float32': 4}[row['dtype']]
            shard_bytes = row['elements_per_rank'] * itemsize
            op = row['operation']
            # In this probe length is input elements for AR/AG, output elements for RS.
            input_bytes = shard_bytes * (4 if op == 'reduce_scatter' else 1)
            output_bytes = shard_bytes * (4 if op == 'all_gather' else 1)
            # These cells never exercise the custom backend, regardless of the arm label.
            backend = 'nccl-ring-1-channel' if (op == 'reduce_scatter' or
                      (op == 'all_reduce' and input_bytes > 2 * 1024 * 1024)) else arm
            key = (op, row['dtype'], input_bytes, output_bytes)
            cell = cells.setdefault(key, {})
            cell.setdefault(backend, []).append({
                'run': name, 'median_us': row['median_us'],
                'slowest_rank_samples_us': row['slowest_rank_samples_us'],
                'buffer_drops': row['rx_out_of_buffer'],
                'rdma_errors': row['rdma_error_deltas'],
            })
    rows = []
    for (op, dtype, inp, out), backends in sorted(cells.items()):
        candidates = []
        for backend, launches in sorted(backends.items()):
            times = [v['median_us'] for v in launches]
            candidates.append({'backend': backend, 'launches': launches,
                               'median_of_launch_medians_us': median(times),
                               'launch_median_range_us': [min(times), max(times)],
                               'repeat_screened': len(times) >= 2})
        best = min(candidates, key=lambda c: c['median_of_launch_medians_us'])
        rows.append({'operation': op, 'dtype': dtype, 'input_bytes_per_rank': inp,
                     'output_bytes_per_rank': out, 'candidates': candidates,
                     'lowest_observed_median': best['backend'],
                     'dispatch_qualified': False})
    return {'source': str(SOURCE.relative_to(Path(__file__).resolve().parents[2])),
            'source_sha256': hashlib.sha256(raw).hexdigest(), 'world_size': 4,
            'scope': 'Isolated collective screens; no mixed-backend or serving validation.',
            'restrictions': ['No interpolation or inferred thresholds between sampled sizes.',
                             'No cross-batch or floating-reduction equivalence claim.',
                             'One relay launch in this session; repeat before fitting a policy.',
                             'NCCL-only cells are pooled, not credited to the custom arm.'],
            'cells': rows}

if __name__ == '__main__':
    print(json.dumps(report(), indent=2))
