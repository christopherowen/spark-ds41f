import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

home = Path.home()
root = home / 'projects/spark-ds41f'
native = root / 'results/private/lab/verification-budget-hardware/dgx1-1793927589/screen'
out = home / 'spark-lab/verification-budget-hardware/export-attempt-2'
out.mkdir(exist_ok=False)
receipt = json.loads((native / 'receipt.json').read_text())
assert receipt['commit'] == '1160621844bae0f681318d812e1e11f384844767'
assert receipt['restored'] is True and receipt['completed'] is False
assert receipt['error'] == 'ValueError: sampled: batch/seed invariance failed'
manifest = {'window': receipt['window'], 'native_root': str(native), 'files': {}, 'exports': {}}
def digest(data):
    return hashlib.sha256(data).hexdigest()
def save(relative, data):
    if re.search(rb'hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}', data):
        raise ValueError('possible credential in export')
    target = out / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    manifest['exports'][relative] = {'size': len(data), 'sha256': digest(data)}
def record(path):
    data = path.read_bytes()
    manifest['files'][str(path)] = {'size': len(data), 'sha256': digest(data)}
    return data
engine = {}
for path in sorted(native.rglob('*')):
    if not path.is_file():
        continue
    data = record(path)
    if path.name.endswith('-engine.log'):
        lines = data.decode().splitlines()
        engine[path.name] = {
            'source': str(path), 'sha256': digest(data), 'lines': len(lines),
            'faults': [line for line in lines if re.search(r'ERROR|CRITICAL|Traceback|CUDA error|illegal memory access', line, re.I)],
            'warnings': [line for line in lines if 'WARNING' in line],
        }
        continue
    name = path.relative_to(native).as_posix()
    if path.suffix in ('.log', '.txt'):
        save('native/' + name + '.gz', gzip.compress(data, mtime=0))
    else:
        save('native/' + name, data)
save('engine-audit.json', (json.dumps(engine, indent=2) + '\n').encode())
for path in sorted((home / 'spark-lab/verification-budget-hardware').glob('attempt-2-*')):
    if path.is_file():
        save('handoff/' + path.name, record(path))

published = root / 'experiments/2026-11-04-topk-live-qualification/runs/2026-11-04-dgx/attempt-2/source-receipt.json'
source = json.loads(published.read_text())
controls = {'published_receipt': str(published.relative_to(root)),
            'published_receipt_sha256': digest(published.read_bytes()),
            'execution_commit': source['execution_commit'], 'files': {}}
for label in ('candidate-40', 'candidate-40-repeat'):
    key = f'results/private/determinism/candidate-40-functional-tp3/sampled-{label}.jsonl'
    data = record(root / key)
    expected = source['files'][key]
    assert len(data) == expected['size'] and digest(data) == expected['sha256']
    controls['files'][key] = dict(expected, verified=True)
    save('saved-control/' + Path(key).name, data)
script = 'experiments/2026-10-24-tilelang-completion/sampled_digest.py'
old_script = subprocess.check_output(['git', '-C', str(root), 'show', source['execution_commit'] + ':' + script])
new_script = subprocess.check_output(['git', '-C', str(root), 'show', receipt['commit'] + ':' + script])
assert old_script == new_script
controls['same_script_sha256'] = digest(old_script)
controls['historical_contract'] = 'C40 offline audit explicitly allows sampled batch variation as a limitation and requires repeat-alone equality; do not rewrite that evidence as a batch-invariance pass.'
controls['limitations'] = ['The legacy collector retained aggregate digests and differing case indexes, not request IDs, per-case token arrays, or per-case logprobs.', 'This comparison establishes a pre-existing failed batch-invariance observation, not row-cost causality or a runtime sampling defect.']
save('saved-control/comparison.json', (json.dumps(controls, indent=2) + '\n').encode())
(out / 'source-receipt.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'export': str(out), 'files': len(manifest['files']), 'exported_bytes': sum(x['size'] for x in manifest['exports'].values())}))
