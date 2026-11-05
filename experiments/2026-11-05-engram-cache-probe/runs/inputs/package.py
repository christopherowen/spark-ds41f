import json
import os
from pathlib import Path
import shutil

root = Path('/home/chris/projects/engram-cache-probe')
repo = root / 'deployment'
relative = Path('experiments/2026-11-05-engram-cache-probe')
dest = repo / relative
base = repo / 'experiments/2026-10-26-spark-ds41f-candidate/candidate-38'
(dest / 'vllm').mkdir(parents=True, exist_ok=True)
shutil.copyfile(root / '0001-engram-bounded-cache-probes.patch', dest / 'vllm/0001-engram-bounded-cache-probes.patch')
lines = ['# Candidate-38 ordered source stack, followed by the single cache-probe variable.']
for raw in (base / 'vllm/series').read_text().splitlines():
    raw = raw.strip()
    if raw and not raw.startswith('#'):
        lines.append(os.path.relpath((base / 'vllm' / raw).resolve(), dest / 'vllm'))
lines.append('0001-engram-bounded-cache-probes.patch')
(dest / 'vllm/series').write_text('\n'.join(lines) + '\n')
lock = json.loads((base / 'upstreams.lock.json').read_text())
manifest = json.loads((base / 'source.json').read_text())
lock['captured_at'] = manifest['captured_at'] = '2026-11-05'
lock['source_manifest'] = str(relative / 'source.json')
for record in (lock['sources']['vllm'], manifest['vllm']):
    record['patch_series'] = str(relative / 'vllm/series')
manifest['vllm']['verification'] = 'Fresh ordered application and expected source tree only; image build and GPU/serving qualification pending.'
(dest / 'upstreams.lock.json').write_text(json.dumps(lock, indent=2) + '\n')
(dest / 'source.json').write_text(json.dumps(manifest, indent=2) + '\n')
for shape in ('tp3', 'tp4'):
    profile = {
        'notes': 'Unbuilt experiment: candidate-38 with bounded Engram cache probes. No serving qualification or promotion.',
        'extends': {'config': str(base.relative_to(repo) / (shape + '.json'))},
        'set': {
            'upstreams_config': str(relative / 'upstreams.lock.json'),
            'container': {
                'image': 'spark-ds41f:engram-cache-probe-20261105',
                'expected_labels': {'local.spark-ds41f.vllm.tree': 'd2f16b4f0697452766593281f7412b5f72a59073'},
            },
        },
    }
    (dest / (shape + '.json')).write_text(json.dumps(profile, indent=2) + '\n')
