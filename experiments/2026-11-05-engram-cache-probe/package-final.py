import json
import os
from pathlib import Path
import shutil

root = Path('/home/chris/projects/engram-cache-probe')
repo = root / 'deployment'
base = repo / 'experiments/2026-10-26-spark-ds41f-candidate/candidate-38'
parent = Path('experiments/2026-11-05-engram-cache-probe')
for name in ('aligned', 'filter'):
    relative = parent / name
    dest = repo / relative
    (dest / 'vllm').mkdir(parents=True, exist_ok=True)
    patch = '0001-engram-cache-' + name + '.patch'
    shutil.copyfile(root / 'final-candidates' / (name + '.patch'), dest / 'vllm' / patch)
    lines = ['# Candidate-38 source stack, followed by one cache-metadata layout change.']
    for raw in (base / 'vllm/series').read_text().splitlines():
        raw = raw.strip()
        if raw and not raw.startswith('#'):
            lines.append(os.path.relpath((base / 'vllm' / raw).resolve(), dest / 'vllm'))
    lines.append(patch)
    (dest / 'vllm/series').write_text('\n'.join(lines) + '\n')
    identity = dict(line.split(' ', 1) for line in (root / 'final-candidates' / (name + '-identity.txt')).read_text().splitlines())
    lock = json.loads((base / 'upstreams.lock.json').read_text())
    manifest = json.loads((base / 'source.json').read_text())
    lock['captured_at'] = manifest['captured_at'] = '2026-11-06'
    lock['source_manifest'] = str(relative / 'source.json')
    for record in (lock['sources']['vllm'], manifest['vllm']):
        record['patch_series'] = str(relative / 'vllm/series')
    manifest['vllm']['verification'] = 'Experimental exact source; fresh ordered application, image build and GPU/serving qualification are recorded separately.'
    (dest / 'upstreams.lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    (dest / 'source.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for shape in ('tp3', 'tp4'):
        profile = {
            'notes': 'Unpromoted candidate-38 experiment with ' + name + ' Engram cache metadata. Image qualification pending.',
            'extends': {'config': str(base.relative_to(repo) / (shape + '.json'))},
            'set': {'upstreams_config': str(relative / 'upstreams.lock.json'),
                    'deployment': {'branch': 'feat/engram-cache-probe-20261105'},
                    'container': {'image': 'spark-ds41f:engram-cache-' + name + '-20261106',
                                  'expected_labels': {'local.spark-ds41f.vllm.tree': identity['tree']}}}}
        (dest / (shape + '.json')).write_text(json.dumps(profile, indent=2) + '\n')
    print(name, identity)
