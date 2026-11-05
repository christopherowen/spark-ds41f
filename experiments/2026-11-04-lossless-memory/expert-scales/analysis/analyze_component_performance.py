#!/usr/bin/env python3
"""Read-only step comparisons; preserve different suite definitions and collector limits."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from servebench import suite
from servebench.__main__ import compare_results
from scripts.servebench_suite_table import AA_FLOOR_PCT

PIN = '3c7b455eec27c436f76f08ed28efb30135c86f4f'
METHODS = ['servebench/suite.py', 'servebench/report.py', 'servebench/engine.py',
           'servebench/prefill.py', 'servebench/stall.py', 'servebench/stats.py',
           'servebench/__main__.py', 'scripts/servebench_suite_table.py']
INPUTS = {}
REPEAT_ROOT = None
REPEAT_HASHES = {}
C36_ROOT = ROOT/'experiments/2026-11-03-candidate-38-tp3-qualification/runs/2026-11-03-dgx1-3/previous-candidates'

def read(path):
    logical = Path(path).resolve()
    path = logical
    if not path.exists():
        path = Path(str(path) + '.gz')
    data = path.read_bytes()
    INPUTS[str(path)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    for _ in range(2):
        if data[:2] == b'\x1f\x8b':
            data = gzip.decompress(data)
    if logical.is_relative_to(ROOT / 'manifests') or logical.is_relative_to(C36_ROOT):
        assert data == subprocess.check_output(['git', 'show', PIN+':'+str(logical.relative_to(ROOT))]), 'frozen reference changed'
    if REPEAT_ROOT is not None and logical.is_relative_to(REPEAT_ROOT):
        relative = str(logical.relative_to(REPEAT_ROOT))
        assert hashlib.sha256(data).hexdigest() == REPEAT_HASHES[relative], 'saved repeat differs from original archive'
    return json.loads(data)

def index(directory):
    value = read(Path(directory) / 'suite.json')
    assert value['suite_sha256'] == suite.digest(value['suite']), 'invalid declared suite fingerprint'
    assert value.get('finished_utc') and all(s['exit'] == 0 for s in value['steps']), 'incomplete or failed suite'
    assert [(s['id'], s['command']) for s in value['steps']] == [(s['id'], s['command']) for s in value['suite']['steps']]
    return value

def compare_steps(base, candidate, *, stream_compatible=True):
    a, b = index(base), index(candidate)
    assert [s['id'] for s in a['steps']] == [s['id'] for s in b['steps']], 'different step coverage'
    output = {'scope': 'explicit ordered step comparisons; original whole-suite identities retained',
              'suite': a['suite']['name'], 'base_suite_sha256': a['suite_sha256'],
              'candidate_suite_sha256': b['suite_sha256'], 'labels': [a['label'], b['label']], 'steps': {}}
    for definition_a, definition_b, step_a, step_b in zip(a['suite']['steps'], b['suite']['steps'], a['steps'], b['steps']):
        key = step_a['id']
        if definition_a != definition_b:
            x, y = json.loads(json.dumps(definition_a)), json.loads(json.dumps(definition_b))
            assert x['command'] == y['command'] == 'prefill', 'changed step outside the prefill extent'
            x['args'].pop('sizes'); y['args'].pop('sizes'); assert x == y, 'changed prefill variable beyond sizes'
        if step_a['command'] == 'model':
            output['steps'][key] = {'skipped': 'separate pinned exact model-identity gate'}
            continue
        if not stream_compatible and step_a['command'] in ('run', 'stall'):
            output['steps'][key] = {'skipped': 'different stream event policy; no client throughput/latency or event-gap comparison'}
            continue
        x, y = read(Path(base) / step_a['output']), read(Path(candidate) / step_b['output'])
        for name in ('extra_body', 'question', 'prompts_sha256'):
            assert x.get('workload', {}).get(name) == y.get('workload', {}).get(name), 'different request identity: ' + name
        outcome = compare_results(x, y, 3.0)
        output['steps'][key] = outcome
    return output

def observations(directory):
    value = index(directory)
    rows = {}
    for step in value['steps']:
        if step['command'] == 'model': continue
        data = read(Path(directory) / step['output'])
        if data['schema'] == 'servebench/1':
            summary, requests = data['summary'], data['summary']['requests']
            assert requests['sent'] > 0 and all(requests.get(k) == 0 for k in ('failed', 'rejected', 'foreign', 'approximate_token_times')), 'failed, foreign, or approximate request'
            assert summary['output_tps']['n'] == 10 and len(summary['output_tps']['batches']) == 10
            table = data['request_table']; records = [dict(zip(table['columns'], r)) for r in table['rows']]
            assert len(records) == requests['sent'] and all(not r['error'] and r['completion_tokens'] == data['workload']['max_tokens'] for r in records)
            rows[step['id']] = {'workload': data['workload'], 'summary': summary,
                                'actual_itl_tail': None, 'timed_kv_occupancy': None,
                                'raw_cache_hit_miss_counts': None}
        elif data['schema'] == 'servebench-engine/1':
            assert len(data['engine']) == 15 and all(r['step_ms']['n'] == 100 and len(r['samples']) == 100 for r in data['engine'])
            rows[step['id']] = data['engine']
        elif data['schema'] in ('servebench-prefill/1', 'servebench-stall/1'):
            field = 'prefill' if step['command'] == 'prefill' else 'stall'
            assert not data['workload']['skipped'] and all(r['failed'] == 0 and r['tps']['n'] == 3 for r in data[field])
            rows[step['id']] = {field: data[field], 'workload': data['workload']}
        else:
            raise ValueError('unexpected schema ' + data['schema'])
    return {'identity': value['metadata']['identity'], 'steps': rows,
            'conditions': read(Path(directory) / 'conditions.json')}

def compare_cohorts(references, candidates, *, stream_compatible=True):
    assert len(references) == len(candidates) == 2
    repeat = compare_steps(candidates[0], candidates[1])
    saved = compare_steps(references[0], references[1])
    pairs = {}
    views = []
    for i, reference in enumerate(references):
        results = [compare_steps(reference, c, stream_compatible=stream_compatible) for c in candidates]
        for j, comparison in enumerate(results):
            pairs[f'reference_boot{i+1}_to_candidate_boot{j+1}'] = comparison
            for label, margin in [('candidate_repeat', repeat), ('saved_reference_repeat', saved)]:
                rows = suite.with_floor(comparison, margin, floors=AA_FLOOR_PCT, reference_to_repeat=results[1-j])
                # Non-paired rows keep each boot's own interval+repeat margin.
                # Paired rows intentionally average both boots in the existing method.
                for row in rows:
                    row['plain_verdict'] = {'same': 'no-proven-change'}.get(row['verdict'], row['verdict'])
                    prior = saved['steps'].get(row['step'], {})
                    saved_metrics = set() if 'skipped' in prior else {r['metric'] for r in prior[1].get('rows', [])}
                    row['saved_reference_repeat_available'] = row['metric'] in saved_metrics
                    warnings = comparison['steps'][row['step']][1].get('warnings', [])
                    if row['metric'].startswith('prefill '):
                        size = int(row['metric'].split()[1].replace(',', ''))
                        warnings = [w for w in warnings if w.startswith(f'prefill {size}:')]
                    row['workload_warnings'] = warnings
                    row['eligible_performance_claim'] = row['saved_reference_repeat_available'] and not warnings
                    row['qualification_verdict'] = row['plain_verdict'] if row['eligible_performance_claim'] else 'no-proven-change: incompatible or missing reference evidence'
                views.append({'reference_boot': i+1, 'candidate_boot': j+1, 'repeat_floor': label, 'rows': rows})
    return {'pairs': pairs, 'candidate_repeat': repeat, 'saved_reference_repeat': saved, 'all_pair_and_floor_views': views,
            'limits': ['Eight views reuse four boot pairings; they are not independent measurements.',
                       'Missing saved-reference repeat rows remain absent, never zero-margin wins.',
                       'Every comparison warning and adverse row is retained.',
                       'Long-prefill publication identity/condition limits remain in recorded-performance-reference.json.']}

def main():
    global REPEAT_ROOT, REPEAT_HASHES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True, help='mirror of the node repository root')
    parser.add_argument('--component', choices=['expert-scales', 'engram'], required=True)
    parser.add_argument('--receipt', type=Path, required=True, help='path relative to snapshot')
    parser.add_argument('--reference-repeat-root', type=Path, required=True, help='root containing candidate-14-release-tp3[-suffix]')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.snapshot = args.snapshot.resolve()
    REPEAT_ROOT = args.reference_repeat_root.resolve()
    args.reference_repeat_root = REPEAT_ROOT
    reference_manifest = read(ROOT/'experiments/2026-11-04-lossless-memory/recorded-performance-reference.json')
    for row in reference_manifest['archives']:
        name = row['native_path'].split('/results/private/suites/', 1)[1]
        REPEAT_HASHES[name] = row['uncompressed_sha256']
    methods = {}
    for name in METHODS:
        data = (ROOT/name).read_bytes()
        assert data == subprocess.check_output(['git', 'show', PIN+':'+name]), 'comparison method changed since execution'
        methods[name] = hashlib.sha256(data).hexdigest()
    receipt = read(args.snapshot/args.receipt)
    assert receipt['commit'] == PIN and receipt['component'] == args.component and receipt['client_stream_policy'] == 'recorded-1030'
    assert receipt['completed'] and receipt['restored'] and receipt['gates_passed'] and receipt['two_serving_boots_complete'] and receipt['recorded_model_identity_passed'] and receipt['functional_checks_complete']
    assert not receipt.get('error') and not receipt.get('restore_error')
    spec = read(ROOT/'experiments/2026-11-04-lossless-memory'/args.component/'release-tp3.json');job=spec['jobs'][0]
    labels = [a['label'] for a in job['arms']]
    comparisons, boots, c36_context = {}, {}, {}
    starts = []
    for number, name in enumerate(job['suite']):
        suffix = '' if number == 0 else '-'+name
        candidates = [args.snapshot/'results/private/suites'/(spec['run']+suffix)/label for label in labels]
        for candidate in candidates:
            observed = observations(candidate);ident=observed['identity'];assert ident['client_commit']==PIN and not ident['client_dirty'] and not ident['live_problems'] and ident['client_stream_policy']=='recorded-1030'
            assert set(ident['nodes'])==set(receipt['images'])
            for node, image in receipt['images'].items():assert ident['nodes'][node]['image_id']==image['Id'] and ident['nodes'][node]['image_layers']==image['Layers']
            boots.setdefault(candidate.name,{})[name]=observed
        frozen = ROOT/('manifests/suites/2026-10-30-spark-ds41f-1030-tp3'+suffix)
        repeat = args.reference_repeat_root/('candidate-14-release-tp3'+suffix)/'spark-ds41f-1030-again'
        comparisons[name]=compare_cohorts([frozen,repeat],candidates)
        c36 = C36_ROOT/('candidate-36-screen-tp3'+suffix)/'candidate-36'
        c36_context[name] = {'recorded_boots': 1, 'reference_repeat_available': False,
                            'own_collector_observations': observations(c36),
                            'compatible_step_comparisons': [compare_steps(c36, candidate, stream_compatible=False) for candidate in candidates],
                            'limits': ['Different streaming event policy: no client throughput/latency or event-gap comparison.',
                                       'No C36 repeat; no C36-specific repeat margin is invented.',
                                       'The whole C36 stack already contains both storage ideas; no isolated-component attribution.']}
    for label in labels:
        boot_starts=[]
        for name, observation in boots[label].items():
            boot_starts.append({n:s['container_started_at'] for n,s in observation['identity']['nodes'].items()})
        assert all(v==boot_starts[0] for v in boot_starts), 'companions changed boot'
        starts.append(boot_starts[0])
        memory_dir = args.snapshot/'results/private/determinism'/spec['run']
        startup = read(memory_dir/f'startup-memory-{label}.json')
        assert set(startup) == set(receipt['images'])
        for node, row in startup.items():
            path = memory_dir/f'startup-memory-{label}-{node}.txt'
            data = path.read_bytes();INPUTS[str(path)] = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            lines = data.decode().splitlines()
            assert len(lines) == row['samples'] and lines and row['min_available_gib'] >= 5
            assert round(min(float(line.split()[1]) for line in lines)/1048576,3) == row['min_available_gib']
        boots[label]['startup_memory'] = startup
    assert all(starts[0][node]!=starts[1][node] for node in starts[0]), 'repeat did not reboot all ranks'
    result={'scope':'two-boot component performance; exact-model and functional gate receipts remain separate',
            'execution_commit':PIN,'component':args.component,'method_inputs':methods,'AA_FLOOR_PCT':AA_FLOOR_PCT,
            'new_tolerance':False,'receipt':receipt,'boots':boots,'comparisons':comparisons,'c36_context':c36_context,
            'inputs':INPUTS,'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limits':['Capacity/persistent allocation, startup and host memory need their own original observations; logical encoded bytes are not allocator headroom.',
                      'TPOT is not ITL; missing ITL tails, KV occupancy and raw cache counts remain null.',
                      'C36 already contains both storage ideas and has an incompatible streaming collector plus only one saved boot; no incremental C36 client claim.',
                      'No full-vocabulary equivalence claim is inferred from returned top/given-token score vectors.']}
    with args.out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'component':args.component,'verified_inputs':len(INPUTS),'all_pair_and_floor_views':{k:len(v['all_pair_and_floor_views']) for k,v in comparisons.items()}}))

if __name__=='__main__': main()
