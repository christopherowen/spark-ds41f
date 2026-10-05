"""Summarize each boot separately, retaining historical and counter caveats."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
m = runpy.run_path(str(ROOT / 'bin/spark'))
raw, destination = map(Path, sys.argv[1:])
arms = ('fresh', 'pinned-1', 'pinned-2')
reports = {a: json.loads((raw / a / 'bench.json').read_text()) for a in arms}
main = json.loads((ROOT / 'manifests/benchmarks/2026-10-02-karmic-kraken-r5o-64k.json').read_text())
digest = {a: hashlib.sha256((raw / (a + '-costs.json')).read_bytes()).hexdigest() for a in arms}
assert len(set(digest.values())) == 1, digest
assert len({r['identity']['client_commit'] for r in reports.values()}) == 1
assert len({r['identity']['cluster_config_sha256'] for r in reports.values()}) == 1
expected = {h + ': dgx-fan-control.service is enabled/inactive, expected enabled/active; run sudo systemctl enable --now dgx-fan-control.service' for h in ('dgx1', 'dgx2', 'dgx3')}
result = {
    'scope': 'Three boots of one TP3/r5o image sharing the first boot cost table; no old/new image comparison.',
    'cost_sha256': digest,
    'limits': ['step_ms is derived from acceptance and throughput, not independent GPU timing',
               'historical main has only three samples and a different image ID',
               'pinning effect versus unpinned boots is not isolated',
               'one-stream 256-token responses; not full promotion qualification'],
    'arms': {}, 'pairwise': {}, 'output_matches_across_boots': {},
}
grouped = defaultdict(lambda: defaultdict(list))
for arm, r in reports.items():
    assert not r.get('aborted'), r.get('aborted')
    assert set(r['identity']['live_problems']) == expected
    assert not r['identity']['live_warnings']
    assert r['suites']['quality']['ok']
    a = {k: r.get(k) for k in ('identity', 'started_utc', 'finished_utc', 'thermal', 'memory', 'discarded_samples')}
    a['points'] = r['suites']['decode']['points']
    a['quality_passed'] = r['suites']['quality']['passed']
    a['measured_waves'] = {}
    waves = defaultdict(list)
    for line in (raw / arm / 'waves.jsonl').read_text().splitlines():
        w = json.loads(line)
        if w['label'] in a['points']:
            waves[w['label']].append(w)
    for case, entries in r['suites']['decode']['samples'].items():
        assert len(entries) == 12 and len(waves[case]) == 13, (arm, case)
        accepted_at, drafted_at = defaultdict(float), defaultdict(float)
        counter_samples = []
        for s, w in zip(entries, waves[case][1:]):
            request = s['requests'][0]
            assert request['ok'] and request['completion_tokens'] == 256
            assert request['output_sha256'] == w['result']['requests'][0]['output_sha256']
            grouped[case][request['output_sha256']].append({'arm': arm, 'tps': s['tps'],
                'accepted_per_draft': s['accepted_per_draft'], 'verified_per_draft': s['verified_per_draft']})
            before, after = (m['parse_metric_series'](w[k]) for k in ('before', 'after'))
            delta = {k: v - before.get(k, 0) for k, v in after.items()}
            totals = {}
            for metric in ('request_success_total', 'generation_tokens_total', 'prompt_tokens_total',
                           'prompt_tokens_cached_total', 'num_preemptions_total',
                           'spec_decode_num_drafts_total', 'spec_decode_num_accepted_tokens_total',
                           'spec_decode_num_draft_tokens_total', 'iteration_tokens_total_count',
                           'request_decode_time_seconds_sum', 'request_inference_time_seconds_sum',
                           'request_prefill_time_seconds_sum', 'e2e_request_latency_seconds_sum'):
                full = 'vllm:' + metric
                totals[metric] = m['series_sum'](delta, full) if any(k[0] == full for k in delta) else None
            for accumulator, metric in ((accepted_at, 'accepted'), (drafted_at, 'draft')):
                for pos, value in m['series_by_label'](delta, 'vllm:spec_decode_num_' + metric + '_tokens_per_pos_total', 'position').items():
                    accumulator[pos] += value
            counter_samples.append(totals)
        a['measured_waves'][case] = {
            'counter_samples': counter_samples,
            'server_decode_ms_per_counted_step': m['describe']([
                1000 * c['request_decode_time_seconds_sum'] / (c['iteration_tokens_total_count'] - 1)
                for c in counter_samples
                if c['iteration_tokens_total_count'] is not None
                and c['request_decode_time_seconds_sum'] is not None
                and c['iteration_tokens_total_count'] == c['spec_decode_num_drafts_total'] + 1
                and c['request_success_total'] == 1
                and c['iteration_tokens_total_count'] > 1
            ]),
            'server_step_definition': 'Server request decode duration divided by observed iteration count minus one short-prompt prefill; only samples with iteration count = drafts + 1 and exactly one request. Includes runtime overhead; not GPU-only timing.',
            'accepted_by_position': dict(accepted_at),
            'verified_by_position': dict(drafted_at),
            'conditional_acceptance_by_position': {p: accepted_at[p] / n if n else None for p, n in drafted_at.items()},
            'completion_token_lengths': sorted({s['requests'][0]['completion_tokens'] for s in entries}),
            'prompt_token_lengths': sorted({s['requests'][0]['prompt_tokens'] for s in entries}),
        }
    a['historical_main_changes'] = {
        case: {metric: m['difference'](point[metric], main['suites']['decode']['points'][case][metric])
               for metric in ('tps', 'accepted_per_draft', 'verified_per_draft', 'step_ms')}
        for case, point in a['points'].items() if case in main['suites']['decode']['points']
    }
    result['arms'][arm] = a
for first, second in (('fresh', 'pinned-1'), ('fresh', 'pinned-2'), ('pinned-1', 'pinned-2')):
    result['pairwise'][second + '_versus_' + first] = {
        case: {metric: m['difference'](point[metric], result['arms'][first]['points'][case][metric])
               for metric in ('tps', 'accepted_per_draft', 'verified_per_draft', 'step_ms')}
        for case, point in result['arms'][second]['points'].items()
    }
for case, groups in grouped.items():
    result['output_matches_across_boots'][case] = {h: items for h, items in groups.items() if len({i['arm'] for i in items}) > 1}
destination.write_text(json.dumps(result, indent=2) + '\n')
for arm, a in result['arms'].items():
    print(arm)
    for case, p in a['points'].items():
        print(case, 'TPS', p['tps']['mean'], '+/-', p['tps']['ci95'],
              'accepted', p['accepted_per_draft']['mean'], 'verified', p['verified_per_draft']['mean'],
              'derived step ms', p['step_ms']['mean'], 'outputs', p['distinct_outputs'])
