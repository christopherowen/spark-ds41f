"""Reproduce the visible-domain qualification and native-input reports from receipts."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
NOTE = ('Student-t intervals across two fresh-process medians (df=1); the normal '
        'process-median assumption is unverified at n=2. Descriptive prototype '
        'uncertainty, not serving confidence. Ratio intervals use Bonferroni '
        '97.5% component intervals for joint 95%; null if nonpositive.')


def read(name, file='result.json'):
    return json.loads((RUNS / name / file).read_text())


def interval(values, q=.975):
    assert len(values) == 2
    average = statistics.mean(values)
    delta = math.tan(math.pi * (q-.5)) * abs(values[0]-values[1]) / 2
    return [average-delta, average+delta]


def key(case):
    return (case.get('page_size', 256), case['width'], case['mode'],
            case.get('rows', len(case.get('extents', []))), tuple(case.get('extents', [])))


def compare(control_names, candidate_names):
    controls = [{key(c): c for c in read(n)['cases']} for n in control_names]
    candidates = [{key(c): c for c in read(n)['cases']} for n in candidate_names]
    result = []
    for identity in sorted(set.intersection(*(set(x) for x in controls+candidates))):
        if control_names[0] == 'sp8-gsource-map-01' and identity[3] == 1:
            continue  # the separate one-query variable has its own archived control
        cs, ns = [x[identity] for x in controls], [x[identity] for x in candidates]
        if not all(c.get('candidate_wall_ms') for c in cs+ns):
            continue  # correctness-only native fallbacks are never timed
        assert len({c['output_sha256'] for c in cs+ns}) == 1
        for field in ('blocks_sha256', 'lengths', 'candidate_counts', 'retained_keys'):
            values = [c[field] for c in cs+ns if field in c]
            assert not values or all(v == values[0] for v in values)
        before = [statistics.median(c['candidate_wall_ms']) for c in cs]
        after = [statistics.median(c['candidate_wall_ms']) for c in ns]
        a, b = interval(before, .9875), interval(after, .9875)
        result.append({'page_size': identity[0], 'width': identity[1], 'mode': identity[2],
                       'rows': identity[3], 'extents': list(identity[4]),
                       'control_runs': control_names, 'candidate_runs': candidate_names,
                       'control_p50_ms': before, 'candidate_p50_ms': after,
                       'control_t95_ms': interval(before), 'candidate_t95_ms': interval(after),
                       'ratio_joint95_t_interval': [a[0]/b[1], a[1]/b[0]]
                       if min(a+b) > 0 else None})
    return result


prefixes = ('sp8-visible-', 'sp8-p128-', 'sp8-native-capture-')
names = sorted(p.name for p in RUNS.iterdir() if p.name.startswith(prefixes))
assert len(names) == 42
runs = []
for name in names:
    receipt, result = read(name, 'verdict.json'), read(name)
    assert receipt['passed'] and receipt['exit'] == 0 and result['passed']
    assert receipt['commit'] == '1a749338f807c148108d6319e3e44614b7040fb0'
    # Every individual hardware assertion is also retained in the raw result.
    for case in result['cases']:
        for field, value in case.items():
            if 'identical' in field and value is not None:
                assert value is True, (name, field)
    runs.append({'run': name, 'commit': receipt['commit'], 'image': receipt['image'],
                 'passed': True, 'cases': len(result['cases']),
                 'sanitizer': next((s for s in ['initcheck','memcheck','racecheck'] if s in name), None),
                 'memory': receipt['memory'], 'seconds': receipt['seconds']})
pairs = [
    (['sp8-stream-01','sp8-stream-02'], ['sp8-visible-scalar-01','sp8-visible-scalar-02']),
    (['sp8-matrix-01','sp8-matrix-02'], ['sp8-visible-matrix-01','sp8-visible-matrix-02']),
    (['sp8-source-map-01','sp8-source-map-02'], ['sp8-visible-source-01','sp8-visible-source-02']),
    (['sp8-gsource-map-01','sp8-gsource-map-02'], ['sp8-visible-gsource-01','sp8-visible-gsource-02']),
    (['sp8-source-single-01','sp8-source-single-02'], ['sp8-visible-gsource-01','sp8-visible-gsource-02']),
    (['sp8-grouped-03','sp8-grouped-04'], ['sp8-visible-grouped-01','sp8-visible-grouped-02']),
]
visible = {'schema_version': 1, 'status': 'all 42 gates passed; throughput is workload dependent',
           'one_variable': 'visible-domain metadata execution, same native planning allocation',
           'runs': runs, 'comparisons': [c for a,b in pairs for c in compare(a,b)],
           'uncertainty_note': NOTE, 'released_timing_repeated': False,
           'conclusion': 'Wide/short planning improves; full-domain and grouped cases regress. '
                         'Both directions are retained. Real inputs prune zero keys; no promotion.'}
(HERE/'visible-domain-results.json').write_text(json.dumps(visible, indent=2)+'\n')

real_names = ['sp8-real-02','sp8-real-03']
real = [read(n) for n in real_names]
assert all(r['passed'] for r in real)
assert len(real[0]['cases']) == len(real[1]['cases']) == 12
cases = []
for a,b in zip(real[0]['cases'], real[1]['cases']):
    for field in ['snapshot','sha256','extent','planned_width','active','layout','source_interface','candidate_counts','output_sha256']:
        assert a[field] == b[field], field
    for c in [a,b]:
        assert c['identical'] and c['visibility_identical']
        assert c['candidate_counts'] == [c['extent']]
        assert not c['source_interface'] or c['reindex_identical']
    timings = [statistics.median(c['candidate_wall_ms']) for c in [a,b]]
    layout = a['layout']
    cases.append({field:a[field] for field in ['snapshot','sha256','capture_context','rank','extent','planned_width',
                                              'active','source_interface','address_span_bytes','candidate_counts','output_sha256']})
    cases[-1].update(page_size=128 if a['capture_context']['layer_id'] < 20 else 256,
                     original_query_rows=layout['query_rows'], captured_row=layout['captured_row'],
                     pool_pages=layout['pool_pages'], row_stride_bytes=layout['row_stride_bytes'],
                     alignment_mod_256=layout['alignment_mod_256'], pruned_fraction=0.,
                     process_p50_ms=timings, process_median_t95_ms=interval(timings),
                     peak_allocated_bytes=[c['peak_allocated_bytes'] for c in [a,b]],
                     peak_reserved_bytes=[c['peak_reserved_bytes'] for c in [a,b]])
failure = read('sp8-real-01', 'verdict.json')
assert not failure['passed']
bound = read('sp8-bound-diagnostic-01')
geometry = read('sp8-geometry-diagnostic-01')
assert bound['passed'] and geometry['passed']
for c in bound['cases']:
    assert c['old_ideal_retained_keys'] == c['signed_ideal_retained_keys'] == c['extent']
for c in geometry['cases']:
    for v in c['variants']:
        assert set(v['retained_keys_at_ideal_native_cutoff'].values()) == {c['extent']}
report = {'schema_version':1, 'decision':'reject this grouping for serving integration',
          'scope':'12 actual C36 operands, three ranks, layers2/8/14 and source20, one public-source request',
          'capture':'runs/capture-01/receipt.json', 'runs':real_names,
          'runtime_image':read(real_names[0],'verdict.json')['image'],
          'candidate_commit':read(real_names[0],'verdict.json')['commit'],
          'native_oracle':'C36 scorer with independent lazy scratch-read guard; correctness only, never timed',
          'all_12_samples_exact_in_both_processes':True, 'pruned_fraction_every_sample':0.,
          'cases':cases, 'uncertainty_note':NOTE,
          'memory_note':'Original sparse ~5.27GB pool address span is reconstructed; allocation peaks are harness observations, not serving admission evidence.',
          'coverage_limit':'One query saved per layer/interface and rank. Original query-row count is recorded; execution mode was not saved and is not inferred. No full-attention or prompt-corpus quality claim.',
          'operator_failures':[{'run':'sp8-real-01','reason':'Empty input mount following a failed, incorrectly quoted SSH setup; no candidate case executed.',
                                'evidence':'runs/capture-01/transfer-setup-note.json'}],
          'diagnostics': {'signed_head_bound':'runs/sp8-bound-diagnostic-01/result.json',
                          'cancellation_and_eight_key_block_clustering':'runs/sp8-geometry-diagnostic-01/result.json',
                          'result':'No pruning even with ideal native cutoffs. Geometry checks are untimed idea screens, not qualified GPU kernels or machine-arithmetic proofs.'},
          'serving_candidate_integrated':True, 'request_measurement':'runs/serving-window-02/comparison.json', 'released_timing_repeated':False,
          'conclusion':'Exact selected keys and evaluated score bits pass, but these bounds cannot cover their cost without pruning. No production speedup is established. Do not trade output fidelity for synthetic gains.'}
(HERE/'native-input-results.json').write_text(json.dumps(report,indent=2)+'\n')

profile = json.loads((HERE/'source-profiling.json').read_text())
for name in ['sp8-source-visible-profile-01','sp8-source-wide-visible-profile-01']:
    raw=(RUNS/name/'trace.json').read_bytes()
    trace=json.loads(raw)
    counters=collections.defaultdict(lambda:[0,0.])
    for event in trace['traceEvents']:
        if event.get('cat')=='kernel' and 'dur' in event:
            counters[event['name']][0]+=1;counters[event['name']][1]+=event['dur']
    total=sum(x[1] for x in counters.values())
    entry={'run':name,'trace_sha256':hashlib.sha256(raw).hexdigest(),'replays':read(name)['replays'],
           'total_kernel_us':total,'stages':[{'kernel':k,'calls':v[0],'total_us':v[1],'percent':100*v[1]/total}
                       for k,v in sorted(counters.items(),key=lambda x:-x[1][1])]}
    profile['cases']=[x for x in profile['cases'] if x['run']!=name]+[entry]
(HERE/'source-profiling.json').write_text(json.dumps(profile,indent=2)+'\n')
summary=json.loads((HERE/'sp8-results.json').read_text())
summary['status']='Synthetic gates pass; 12 real operands replay exactly twice but prune zero keys. Current grouping rejected for serving integration.'
summary['pending']=['A different exact search with demonstrated real-input benefit before serving integration',
                    'Full serving qualification only if a candidate clears that gate']
summary['latest_reports']=['visible-domain-results.json','native-input-results.json','runs/serving-window-02/comparison.json']
summary['speed_claim']='Only unreleased synthetic-prototype comparisons exist. Real-input pruning is zero; the separate matched C36 request measurement shows higher latency, not a speedup.'
(HERE/'sp8-results.json').write_text(json.dumps(summary,indent=2)+'\n')
print('42 synthetic runs, 12 real operands twice, two untimed diagnostics and two profiles summarized')
