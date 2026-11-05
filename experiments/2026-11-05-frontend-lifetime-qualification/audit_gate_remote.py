#!/usr/bin/env python3
"""Read C43's completed gate on DGX4 after a safe restored handoff; no GPU work."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import xml.etree.ElementTree as ET

PIN = '2c4b23afcd834ba2adf6a15ca434e59fa4f49f74'
IMAGE = 'sha256:ac3c5a2ca2e934295ca7be60d69566d4c406aff22057b0a911661d49e5664905'
OUT = Path('/home/swank/projects/spark-ds41f-copy-default/results/private/candidate-rounds/candidate-43-image')
EXPECTED = ['build-prepare.log', 'build-check.log', 'build-image.log', 'build-image-apply.log',
            'gate-manifest.log', 'gate-bundle.log', 'release-tests.log', 'topk-initcheck.log']
HEAD_GUARD = r'''
import json,pathlib,hashlib
h=pathlib.Path.home();root=h/'projects/spark-ds41f';s=h/'tl-logs/frontend-lifetime-qualification/setup'
p=root/'results/private/lab/candidate-43-round/receipt.json';ep=s/'exit.json'
r=json.loads(p.read_text()) if p.exists() else {};e=json.loads(ep.read_text()) if ep.exists() else {}
ready=(r.get('ended') and r.get('restored') and not r.get('restore_error') and
 not r.get('management_loss_latched') and not r.get('management_unreachable') and
 r.get('error')=='RuntimeError: window changed or closure requested' and
 r.get('commit')=='2c4b23afcd834ba2adf6a15ca434e59fa4f49f74' and
 [(x.get('log'),x.get('exit')) for x in r.get('steps',[])]==[('image-and-gate.log',0)] and e.get('actual_wait_exit')==1)
shared={}
for k,n in [('hold','spark-hold.json'),('request','spark-request.json'),('legacy','spark3-request.json')]:
 p=h/n;shared[k]=json.loads(p.read_text()) if p.exists() else None
if not ready or shared['hold'] or shared['legacy']:
 print(json.dumps({'ready':False,'phase':r.get('phase'),'shared':shared}));raise SystemExit(0)
d=s/'gate-handoff-restore-doctor.json';n=s/'gate-handoff-restoration-ranks.json'
if not d.exists() or not n.exists():
 print(json.dumps({'ready':False,'reason':'independent restoration missing'}));raise SystemExit(0)
assert json.loads(d.read_text())['exit']==0 and json.loads(n.read_text())['passed']
f=root/'results/private/candidate-rounds/candidate-43-qualified-image/receipt.json'
print(json.dumps({'ready':True,'frozen_receipt_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'shared':shared}))
'''


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if os.uname().nodename != 'dgx4':
        raise SystemExit('remote DGX4 auditor only; never process the gate on Mac')
    # These limits apply only to this new, read-only audit process.
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    guard = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                            'dgx1', 'python3', '-'], input=HEAD_GUARD,
                           capture_output=True, text=True, timeout=25)
    if guard.returncode:
        raise SystemExit(guard.stderr)
    state = json.loads(guard.stdout)
    if not state['ready']:
        print(json.dumps({'audit_deferred_before_builder_reads': True, **state}))
        return 2
    lock = (Path.home() / 'spark-cpu-work.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({'audit_deferred': True, 'reason': 'remote CPU work lock occupied'}))
        return 2
    receipt_path = OUT / 'receipt.json'
    receipt_hash = sha(receipt_path)
    assert receipt_hash == state['frozen_receipt_sha256']
    r = json.loads(receipt_path.read_text())
    assert r.get('passed') is True and r.get('ended') and not r.get('error')
    assert r['commit'] == PIN and r['image'] == IMAGE
    assert [x['log'] for x in r['steps']] == EXPECTED
    assert all(x.get('exit') == 0 for x in r['steps'])
    assert not r['conditions']['halted'] and not r['conditions']['fan_stop_failures']
    actual = json.loads(subprocess.check_output(['docker', 'image', 'inspect', 'spark-ds41f:candidate-43'], text=True))[0]
    assert actual['Id'] == IMAGE
    labels = actual['Config'].get('Labels') or {}
    assert all(labels.get(k) == v for k, v in r['source']['expected_labels'].items())
    assert labels['local.spark-ds41f.vllm.patch-head'] == '99029e6cbdb966ff198dd13902a0825e8d507e27'
    assert labels['local.spark-ds41f.vllm.tree'] == '082bda0f13e94290f462eaf55d143cdb7931f9e5'
    assert labels['local.spark-ds41f.tilelang.tree'] == '222b895d04ee6cd75179155d06e0c0d450636f51'
    manifest = json.loads((OUT / 'gate-manifest.json').read_text())
    records = [json.loads(line) for line in (OUT / 'gate-results/results.jsonl').read_text().splitlines()]
    assert len(records) == len(manifest['files']) == 138
    assert sorted(x['file'] for x in records) == sorted(manifest['files'])
    assert all(x['status'] == 'ok' and not x['failed'] and not x['error'] and not x['failures'] for x in records)
    totals = {k: sum(x[k] for x in records) for k in ('passed', 'skipped', 'failed', 'error')}
    xml_totals = dict.fromkeys(totals, 0)
    for xml in sorted((OUT / 'gate-results').glob('chunk*.xml')):
        for case in ET.parse(xml).getroot().iter('testcase'):
            kind = next((k for tag, k in [('failure', 'failed'), ('error', 'error'), ('skipped', 'skipped')]
                         if case.find(tag) is not None), 'passed')
            xml_totals[kind] += 1
    assert totals == xml_totals and totals['passed'] > 0
    assert r['release_cache']['absent_before'] is True and r['release_cache']['image'] == IMAGE
    assert sha(OUT / 'bundle/candidate.json') == r['release_cache']['actual_bundle_sha256']
    assert sha(OUT / 'gate-manifest.json') == r['gate_manifest']['sha256']
    assert r['topk_initcheck']['passed'] is True
    assert r['topk_initcheck']['source_sha256'] == '906039870d8dd00d933d1ee1c083510373292e951f971568b4d3cf9c59cbdbfc'
    assert sha(OUT / 'topk-initcheck.log') == r['topk_initcheck']['log_sha256']
    topk = (OUT / 'topk-initcheck.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in topk and 'exact top-k output' in topk
    paths = [p for p in sorted(OUT.rglob('*')) if p.is_file()]
    assert all(not p.is_symlink() for p in paths)
    assert sum(p.stat().st_size for p in paths) < 128 * 1024**2
    hashes = {str(p.relative_to(OUT)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths}
    assert sha(receipt_path) == receipt_hash
    result = {'recorded': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': True,
              'host': 'dgx4', 'execution_commit': PIN, 'image': IMAGE, 'gate_receipt_sha256': receipt_hash,
              'manifest_files': 138, 'per_file_coverage_exact': True, 'totals': totals,
              'independent_xml_totals': xml_totals, 'artifacts': hashes,
              'raw_artifacts_retained_at': str(OUT), 'resource_limits': {'cpu_affinity_count': 2, 'address_space_gib': 4},
              'no_GPU_or_serving_or_source_changes': True}
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'} | {'hashed_artifacts': len(hashes)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
