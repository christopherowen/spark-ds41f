"""Assemble accepted cases without dropping either boot or historical failures."""
import gzip
import json
from pathlib import Path

from paired import extend

HERE = Path(__file__).resolve().parent


def read(path):
    if not path.exists():
        path = path.with_suffix(path.suffix + '.gz')
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as handle:
        return json.load(handle)


def point(run, operation, boot):
    return read(HERE / 'runs' / run / f'{HERE.name}-{operation}-{boot}' / 'result.json')


boots = {}
for boot in (1, 2):
    common = {r['name']: r for r in point('combined-v4', 'pair', boot)['cases']}
    sampler = {r['name']: r for r in point('final-sampler-v6', 'pair', boot)['cases']}
    tails = {r['name']: r for r in point('final-tails-v9', 'pair', boot)['cases']}
    extensions = {r['name']: r for r in point('logits-extension-v6', 'pair', boot)['cases']}
    rows = {}
    for name, row in common.items():
        run = 'combined-v4'
        if name.startswith('sampler-'):
            row, run = sampler[name], 'final-sampler-v6'
        elif name.startswith('gather-tail-'):
            row, run = tails[name], 'final-tails-v9'
        result = dict(run=run, gate=row['gate'], passed=row['gate']['status'] != 'not_proved',
                      candidate_kernels=row['profiles']['candidate']['kernels'],
                      source_data=row['data'])
        assert not row['profiles']['candidate']['aten_kernels'], name
        if name in extensions:
            followup = extensions[name]
            result['extension'] = extend(row['measured']['log_ratios']['candidate'],
                                         followup['measured']['log_ratios']['candidate'])
            result['passed'] = result['extension']['passed']
        rows[name] = result
    assert len(rows) == 80
    boots[str(boot)] = rows

wire, transfers, tests = {}, {}, {}
for boot in (1, 2):
    wire[str(boot)] = point('final-wire-v9', 'wire', boot)
    transfer = point('final-wire-v9', 'transfer', boot)
    transfers[str(boot)] = dict(cases=[{k: v for k, v in r.items() if k not in ('pairs', 'log_ratios')} for r in transfer['cases']], scope=transfer['scope'])
    assert len(wire[str(boot)]['cases']) == 16
    assert len(transfers[str(boot)]['cases']) == 12
    assert all(not r['aten_kernels'] for r in wire[str(boot)]['cases'])
    run = 'final-tests-v8' if boot == 1 else 'final-tests-v9'
    data = point(run, 'tests', boot)
    tests[str(boot)] = dict(run=run, outcomes=data['outcomes'], source=data['candidate']['head'])
    assert data['pytest_exit'] == 0 and data['outcomes'] == dict(passed=1266)

unresolved = {b: [n for n, r in rows.items() if not r['passed']] for b, rows in boots.items()}
transfer_unresolved = {b: [(r['dtype'], r['rows'], r['stride']) for r in rows['cases']
                          if r['gate']['status'] == 'not_proved'] for b, rows in transfers.items()}
result = dict(candidate=read(HERE / 'candidate.json'), gpu_boots=boots, tests=tests,
              gpu_unresolved=unresolved, gpu_passed=not any(unresolved.values()),
              wire=wire, transfer=transfers, transfer_unresolved=transfer_unresolved,
              transfer_passed=not any(transfer_unresolved.values()),
              scope='Local GPU cases and complete result-transfer timings on sp8. Full model trace, built image, real weights and physical networking are separate release gates.')
(HERE / 'qualification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(gpu_unresolved=unresolved, transfer_unresolved=transfer_unresolved), indent=2))
