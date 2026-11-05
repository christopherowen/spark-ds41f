"""Compare a new workload on an unreleased control and candidate across two boots."""
import json
import math
import statistics
from pathlib import Path


def paired(before, after):
    if len(before) != 2 or len(after) != 2:
        raise ValueError('Exactly two independent boot pairs are required')
    delta = [a - b for b, a in zip(before, after)]
    half = 12.706204736 * statistics.stdev(delta) / math.sqrt(2)
    mean = statistics.mean(delta)
    result = {'before': statistics.mean(before), 'after': statistics.mean(after),
              'absolute_change': mean, 'absolute_change_ci95': [mean - half, mean + half]}
    if all(b > 0 for b in before):
        relative = [(a / b - 1) * 100 for b, a in zip(before, after)]
        half = 12.706204736 * statistics.stdev(relative) / math.sqrt(2)
        mean = statistics.mean(relative)
        result.update(percent_change=mean, percent_change_ci95=[mean - half, mean + half])
    else:
        result.update(percent_change=None, percent_change_ci95=None)
    return result


def comparison(root, labels):
    from tables_arms import report
    controls = [label for label in labels if label.startswith('control-')]
    candidates = [label for label in labels if label.startswith('candidate-')]
    before = report(root, controls)
    after = report(root, candidates)
    if before['prompt_hashes'] != after['prompt_hashes']:
        raise ValueError('Parent/candidate native workload hashes differ')
    oracle = {}
    receipts = []
    rows = {}
    summaries = {}
    for label in controls + candidates:
        for i in range(3):
            p = root / label / f'rep-{i}'
            summaries[label, i] = json.loads((p / 'summary.json').read_text())
            for line in (p / 'requests.jsonl').read_text().splitlines():
                row = json.loads(line)
                if row.get('cancelled_by_client'):
                    continue
                identity = (row['prompt_sha256'], row['token_ids'], row['logprobs'])
                if row['label'] in oracle and oracle[row['label']] != identity:
                    raise ValueError(f'{p}: parent/candidate or boot output mismatch for {row["label"]}')
                oracle[row['label']] = identity
                rows[label, i, row['label']] = row
            receipts.extend([str(p / 'requests.jsonl'), str(p.with_suffix('.conditions.json'))])
    result = {'passed_output_equality': True, 'receipts': receipts, 'metrics': [],
              'interval': 'paired Student t, two boot pairs, df=1; percent is mean per-pair relative change'}
    for request in ['cold-0', 'warm-1', 'warm-2', 'warm-3']:
        for metric in ['ttft_s', 'wall_s', 'output_tokens_per_s', 'cached_tokens', 'uncached_tokens']:
            means = [[statistics.mean(rows[label, i, request][metric] for i in range(3))
                      for label in group] for group in [controls, candidates]]
            result['metrics'].append({'request': request, 'metric': metric, **paired(*means)})
    means = [[statistics.mean(summaries[label, i]['distinct_agent_output_tokens_per_s'] for i in range(3))
              for label in group] for group in [controls, candidates]]
    result['metrics'].append({'request': 'distinct-agent-batch', 'metric': 'output_tokens_per_s', **paired(*means)})
    return result
