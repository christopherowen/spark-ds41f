#!/usr/bin/env python3
"""Report candidate TTFT across independent boots; never infer a baseline speedup."""
import json
import math
import statistics
import sys
from pathlib import Path


def report(root, labels):
    if len(labels) != 2:
        raise ValueError('This plan requires exactly two independent boots')
    boots = {}
    workload = None
    for label in labels:
        repeats = sorted(p for p in (root / label).glob('rep-*') if p.is_dir())
        if len(repeats) != 3:
            raise ValueError(f'{label}: expected three repeats, found {len(repeats)}')
        boots[label] = []
        for repeat in repeats:
            if not json.loads(repeat.with_suffix(".conditions.json").read_text())["passed"]:
                raise ValueError(f"{repeat}: identity/memory/thermal guard failed")
            if not json.loads((repeat / 'summary.json').read_text())['passed']:
                raise ValueError(f'{repeat}: serving gate failed; inspect original receipts')
            hashes = json.loads((repeat / 'workload.json').read_text())['prompt_hashes']
            if workload is None:
                workload = hashes
            if hashes != workload:
                raise ValueError(f'{repeat}: workload differs between runs')
            rows = [json.loads(line) for line in (repeat / 'requests.jsonl').read_text().splitlines()]
            if any('error' in row for row in rows):
                raise ValueError(f'{repeat}: request error; inspect original receipts')
            boots[label].append({row['label']: row for row in rows})
    result = {'boots': labels, 'prompt_hashes': workload, 'turns': [],
              'interval': '95% Student t across two boot means, df=1; no old-build speedup measured'}
    for turn in range(4):
        row = {'turn': turn}
        for path in ['cold', 'warm']:
            key = f'{path}-{turn}'
            means = [statistics.mean(rep[key]['ttft_s'] for rep in boots[label]) for label in labels]
            half = 12.706204736 * statistics.stdev(means) / math.sqrt(2)
            per_request = [rep[key] for label in labels for rep in boots[label]]
            row[path] = {'boot_mean_ttft_s': means, 'mean_ttft_s': statistics.mean(means),
                         'ci95_s': [statistics.mean(means) - half, statistics.mean(means) + half],
                         'cached_tokens': [r['cached_tokens'] for r in per_request],
                         'uncached_tokens': [r['uncached_tokens'] for r in per_request],
                         'wall_s': [r['wall_s'] for r in per_request],
                         'output_tokens_per_s': [r.get('output_tokens_per_s') for r in per_request]}
        result['turns'].append(row)
    result['distinct_agent_batches'] = {
        label: [json.loads((root / label / f'rep-{i}/summary.json').read_text()).get(
            'distinct_agent_output_tokens_per_s') for i in range(3)] for label in labels}
    return result


if __name__ == '__main__':
    run, *labels = sys.argv[1:]
    root = Path(__file__).resolve().parents[2] / 'results/private/prefix-endpoint' / run
    print(json.dumps(report(root, labels), indent=2))
