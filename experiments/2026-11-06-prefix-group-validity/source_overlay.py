"""Verify the effective Python source on every serving rank without trusting tags."""
import hashlib
import json
from pathlib import Path


def validate_snapshot(manifest, policy, snapshot):
    if snapshot['image_id'] != manifest['parent_image_id']:
        raise RuntimeError('Serving dependency image differs from candidate-38')
    expected = {f['container_path']: f[policy + '_sha256'] for f in manifest['files']}
    if snapshot['hashes'] != expected:
        raise RuntimeError(f'{policy} source files differ from the published source export')
    changed = set(expected)
    mounted = {m['Destination']: m for m in snapshot['mounts'] if m['Destination'] in changed}
    if policy == 'candidate':
        if set(mounted) != changed or any(m['RW'] for m in mounted.values()):
            raise RuntimeError('Candidate source must be mounted read-only on every changed path')
        for file in manifest['files']:
            if not mounted[file['container_path']]['Source'].endswith('/' + file['export']):
                raise RuntimeError('Candidate source mount is not the published experiment export')
    elif mounted:
        raise RuntimeError('Control unexpectedly has candidate source mounts')


def verify(cluster, nodes, policy, manifest_path):
    from spark_ds41f.config import run_ssh
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parents[2]
    for file in manifest['files']:
        if hashlib.sha256((root / file['export']).read_bytes()).hexdigest() != file['candidate_sha256']:
            raise RuntimeError('Local source export differs from the recorded source commit')
    script = 'import hashlib,json,sys;print(json.dumps({p:hashlib.sha256(open(p,"rb").read()).hexdigest() for p in sys.argv[1:]}))'
    snapshots = {}
    for node in nodes['nodes']:
        inspected = run_ssh(nodes, node, 'docker', 'inspect', cluster['container']['name'])
        if inspected.returncode:
            raise RuntimeError(f"{node['name']}: serving container unavailable")
        container = json.loads(inspected.stdout)[0]
        result = run_ssh(nodes, node, 'docker', 'exec', cluster['container']['name'],
                         'python3', '-c', script, *[f['container_path'] for f in manifest['files']])
        if result.returncode:
            raise RuntimeError(f"{node['name']}: source hashing failed")
        snapshot = {'image_id': container['Image'], 'mounts': container['Mounts'],
                    'hashes': json.loads(result.stdout)}
        validate_snapshot(manifest, policy, snapshot)
        snapshots[node['name']] = snapshot
    return {'policy': policy, 'head': manifest['source_head'] if policy == 'candidate' else manifest['parent_head'],
            'nodes': snapshots}
