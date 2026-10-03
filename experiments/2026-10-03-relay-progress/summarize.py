"""Attribute host proxy traces to measured CUDA graph sequence windows."""
import argparse
import json
import re
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('runs', type=Path, nargs='+')
a = p.parse_args()
fields = ('rank', 'seq', 'nbytes', 'ready_mask', 'begin', 'local_end',
          'relay_end', 'end', 'early_ns', 'post_ns', 'queue_ns')

def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values)-1, int((len(values)-1)*fraction))]

def summary(values):
    return {'median': statistics.median(values), 'p95': percentile(values, .95),
            'max': max(values), 'mean': statistics.mean(values)} if values else None

report = {}
for run in a.runs:
    ranks = []
    cases = {}
    for rank in range(4):
        traces = []
        final = None
        raw = (run / f'dgx{rank+1}.txt').read_text()
        stderr = run / f'dgx{rank+1}-stderr.txt'
        if stderr.exists():
            raw += '\n' + stderr.read_text()
        trace_pattern = r'ROCE_TRACE,(?:[0-9]+,){10}[0-9]+\n'
        for match in re.finditer(trace_pattern, raw):
            traces.append(dict(zip(fields, map(int, match.group().strip().split(',')[1:]))))
        clean = re.sub(trace_pattern, '', raw)
        for match in re.finditer(r'\{"rank": [0-9]+, "world_size"', clean):
            final = json.JSONDecoder().raw_decode(clean[match.start():])[0]
        for line in []:
            if line.startswith('ROCE_TRACE,'):
                traces.append(dict(zip(fields, map(int, line.split(',')[1:]))))
            elif line.startswith('{'):
                obj = json.loads(line)
                if 'passed' in obj:
                    final = obj
        if final is None or not final['passed']:
            raise ValueError(f'{run}: rank {rank} did not pass')
        ranks.append({'rank': rank, 'trace_count': len(traces), 'proxy': final['proxy']})
        for row in final['timings']:
            key = f"{row['operation']}/{row['dtype']}/{row['elements_per_rank']}"
            case = cases.setdefault(key, {'rank_samples_us': [], 'traces': []})
            case['rank_samples_us'].append(row['microseconds_per_call'])
            if traces and row['expected_backend'] == 'rocenante':
                window = row['proxy_sequence_window']
                chosen = [t for t in traces if window['before'] < t['seq'] <= window['after']]
                if len(chosen) != 1280:
                    raise ValueError(f'{run} {rank} {key}: expected 1280 calls, got {len(chosen)}')
                case['traces'].extend(chosen)
    for key, case in cases.items():
        samples = case.pop('rank_samples_us')
        case['slowest_rank_us'] = [max(s[i] for s in samples) for i in range(5)]
        case['median_us'] = statistics.median(case['slowest_rank_us'])
        traces = case.pop('traces')
        if traces:
            case['trace_count'] = len(traces)
            for name, end, start in [('local_post_us','local_end','begin'),
                                     ('relay_us','relay_end','local_end'),
                                     ('cq_drain_us','end','relay_end'),
                                     ('total_proxy_us','end','begin')]:
                case[name] = summary([(t[end]-t[start])/1000 for t in traces])
            for name in ('post_ns','queue_ns'):
                case[name.replace('_ns','_us')] = summary([t[name]/1000 for t in traces])
            early = [t for t in traces if t['early_ns'] and t['begin'] >= t['early_ns']]
            case['early_fraction'] = len(early)/len(traces)
            case['early_delay_us'] = summary([(t['begin']-t['early_ns'])/1000 for t in early])
            case['ready_at_begin_fraction'] = sum(t['ready_mask'] != 0 for t in traces)/len(traces)
    report[run.name] = {'ranks': ranks, 'cases': cases}
print(json.dumps(report, indent=2))
