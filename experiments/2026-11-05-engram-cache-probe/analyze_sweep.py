#!/usr/bin/env python3
import json
from pathlib import Path
import statistics as stats

root = Path(__file__).parent
out = {'scope': 'CPU planner design comparisons; intervals over seeded ID batches, not serving boots', 'runs': {}}
for run in ('sweep-a', 'production-sweep-a', 'production-sweep-b', 'expanded-a', 'expanded-b', 'layout-a', 'layout-b'):
    receipt = json.loads((root / 'runs' / run / 'receipt.json').read_text())
    cases = []
    for variants in receipt['runs']:
        control = variants['control']
        case = {k: control[k] for k in ('tokens', 'tp', 'hot_pct', 'warm_rows', 'populated')}
        case['variants'] = {}
        for name, result in variants.items():
            matched_probe = name in ('split96', 'filter96', 'aligned96', 'adaptive96', 'soa96')
            reference = variants['probe96'] if matched_probe else control
            samples, baseline = result['samples'][1:], reference['samples'][1:]
            values = [(a['plan_ns'] - b['plan_ns']) / 1000 for a, b in zip(samples, baseline)]
            t = 2.045229642132703 if len(values) == 30 else 2.2621571628540993
            half = t * stats.stdev(values) / len(values) ** 0.5
            center = stats.mean(values)
            case['variants'][name] = {
                'reference': 'probe96' if matched_probe else 'control',
                'planner_us': stats.mean(x['plan_ns'] / 1000 for x in samples),
                'paired_delta_us': center, 'paired_delta_ci95': [center - half, center + half],
                'jobs': stats.mean(x['planned_jobs'] for x in samples),
                'cache_hits': stats.mean(x['cache_hits'] for x in samples),
                'cache_bytes': result['cache_allocation_bytes'] + result['extra_cache_bytes'],
                'warm_ns': result['warm_ns'], 'peak_rss_bytes': result['peak_rss_bytes'],
            }
        cases.append(case)
        if run.startswith('layout') and case['tp'] == 3 and case['tokens'] in (6, 48) and case['populated'] and case['hot_pct'] in (0, 75, 100):
            print(run, {k: case[k] for k in ('tokens', 'hot_pct', 'warm_rows')},
                  {n: round(v['planner_us'], 2) for n, v in case['variants'].items()
                   if n in ('control', 'split64', 'aligned64', 'filter64', 'adaptive64', 'soa64')})
    out['runs'][run] = cases
(root / 'sweep-results.json').write_text(json.dumps(out, indent=2) + '\n')
