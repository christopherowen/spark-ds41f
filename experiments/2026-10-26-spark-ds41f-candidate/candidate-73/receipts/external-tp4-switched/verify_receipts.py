import json, pathlib
B = pathlib.Path(__file__).resolve().parent
F = B/'full'
rows = [line.split('\t') for line in (F/'status.tsv').read_text().splitlines()]
names=['start','doctor','determinism-wide','determinism-wide-nologprobs','tool-gate-dual','bench','suite-release','suite-single-stream','suite-sampled','stop']
assert [n for n,rc in rows] == [f'{n}-{b}' for b in ['b1','b2'] for n in names]
assert [(n,int(rc)) for n,rc in rows if int(rc)] == [('bench-b1',1),('bench-b2',1)]
report={'protocol_phases':len(rows),'nonzero_phases':[(n,int(rc)) for n,rc in rows if int(rc)],'boots':{}}
for boot in ['b1','b2']:
    gate=json.loads((F/f'tool-gate-dual-{boot}.json').read_text())
    assert gate['temperatures']==[0.0,0.3] and len(gate['models'])==2
    assert len(gate['runs'])==4
    assert all(r['gate_passed'] and r['strict_passed']==20 and r['scenarios']==20 and r['derailment_count']==0 for r in gate['runs'])
    deterministic=[]
    for name in ['determinism-wide','determinism-wide-nologprobs']:
        data=[json.loads(l) for l in (F/f'{name}-{boot}.log').read_text().splitlines() if l.startswith('{')]
        s=[x for x in data if x.get('summary')=='temperature-0 determinism']
        assert len(s)==1
        s=s[0]
        assert s['identical'] is True and s['tokens']==192 and 'mixed15' in s['scenarios']
        assert all(v['token_diffs']==0 and v['logprob_diffs']==0 for v in s['scenarios'].values())
        deterministic.append(s)
    bench=json.loads((F/f'bench-{boot}/bench.json').read_text())
    assert bench['workload']['prefill_text']=='source'
    assert bench['suites']['quality']['ok'] and bench['suites']['quality']['passed']==5
    for suite in ['release','single-stream','sampled']:
        x=json.loads((F/f'suite-{suite}-{boot}/suite.json').read_text())
        assert not x.get('failed'),x.get('failed')
        c=json.loads((F/f'suite-{suite}-{boot}/conditions.json').read_text())
        assert c['halted'] is None
        assert all(v['thermal_slowdown_ms']==0 for v in c['thermal'].values())
    report['boots'][boot]={'tool_strict_passed':sum(r['strict_passed'] for r in gate['runs']),'derailments':0,'determinism':deterministic,'quality_passed':5,'suites_completed':['release','single-stream','sampled']}
a=report['boots']['b1']['determinism']; z=report['boots']['b2']['determinism']
assert [x['reference_sha256'] for x in a]==[x['reference_sha256'] for x in z]
assert [x['long_sha256'] for x in a]==[x['long_sha256'] for x in z]
t=(B/'table-release.txt').read_text()
assert 'greedy responses identical: 256/256' in t and 'KL mean 0.000e+00' in t
report.update(correctness_passed=True,performance_clearance=False,decision='NO_PROMOTION',comparison_caveat='Frozen1120 matching topology/harness reference from upstream hardware; not a same-site causal engine comparison.')
(B/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'protocol_phases':report['protocol_phases'],'correctness_passed':True,'tools_each_boot':80,'suites_each_boot':3,'performance_clearance':False,'decision':'NO_PROMOTION'}))
