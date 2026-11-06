"""Build a Python-only candidate from a verified immutable C36 dependency image.

This experimental recipe reuses compiled dependencies; it is not a native rebuild.
No GPU is opened. Normal build check and later GPU smoke/serving gates still apply.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import spark_ds41f as spark

PARENT = 'sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713'
PROFILE = str(EXP.relative_to(ROOT) / 'tp3.json')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--apply', action='store_true')
args = parser.parse_args()
source = args.source.resolve()
identity = json.loads((EXP / 'identity.json').read_text())


def output(*cmd, cwd=ROOT):
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def image(name):
    return json.loads(output('docker', 'image', 'inspect', name))[0]


def manifest(head):
    entries = output('git', 'ls-tree', '-r', head, cwd=source).splitlines()
    files = []
    for line in entries:
        metadata, path = line.split('\t', 1)
        mode, kind, blob = metadata.split()
        assert kind == 'blob', (path, kind)
        assert not path.endswith(('.so', '.a', '.o')), path
        files.append({'path': path, 'mode': mode, 'blob': blob})
    return {'head': head, 'tree': output('git', 'rev-parse', head+'^{tree}', cwd=source),
            'files': files}


cluster, _, upstreams = spark.configuration(argparse.Namespace(cluster_config=PROFILE))
inputs = spark.build_inputs(upstreams)
directory = spark.build_directory(inputs)
problems = spark.build_problems(directory, inputs)
assert not problems, problems
assert not output('git', 'status', '--porcelain', cwd=source), 'source checkout is dirty'
assert output('git', 'rev-parse', 'HEAD', cwd=source) == identity['candidate']['patch_head']
parent = image(PARENT)
labels = parent['Config']['Labels']
expected = json.loads((EXP / 'source.json').read_text())
assert labels['local.spark-ds41f.vllm.tree'] == identity['unreleased_control']['tree']
for project in ('b12x', 'nccl', 'tilelang', 'tile_kernels', 'sparknet'):
    assert labels[f'local.spark-ds41f.{project}.tree'] == expected[project]['expected_tree'], project
changes = output('git', 'diff', '--name-status', identity['unreleased_control']['head'], 'HEAD', cwd=source)
for change in changes.splitlines():
    status, name = change.split('\t')
    assert status in ('M', 'A') and name.endswith('.py'), change
    assert name.startswith(('vllm/', 'tests/')), name
context = directory / 'context/vllm-source'
assert not (context / '.git').exists(), 'build only the checked Git export'
head = output('git', 'rev-parse', 'HEAD')
tag = cluster['container']['image'] + '-reuse-' + PARENT[7:15] + '-git'
command = ['docker', 'build', '--network=none', '--memory=4g', '--memory-swap=4g',
           '--cpu-period=100000', '--cpu-quota=100000', '--cpuset-cpus=4',
           '--file', 'Dockerfile', '--tag', tag,
           '--build-arg', 'PARENT_IMAGE='+PARENT,
           '--build-arg', 'DEPLOYMENT_COMMIT='+head,
           '--build-arg', 'VLLM_PATCH_HEAD='+identity['candidate']['patch_head'],
           '--build-arg', 'VLLM_TREE='+identity['candidate']['expected_tree'], '-']
stream_command = [sys.executable, str(EXP/'stream_image_context.py'), '--source', str(source),
                  '--head', identity['candidate']['patch_head'], '--dockerfile', str(EXP/'Dockerfile.native-reuse'),
                  '--', *command]
print(json.dumps({'command': command, 'stream_command': stream_command, 'parent': PARENT, 'tag': tag,
                  'gpu_smoke': 'required later in an owned window; not run by this CPU recipe'}, indent=2))
if not args.apply:
    raise SystemExit(0)
assert not output('git', 'status', '--porcelain'), 'deployment checkout is dirty'
refs = output('git', 'branch', '-r', '--contains', 'HEAD', '--format=%(refname:short)').splitlines()
assert 'origin/feat/ced-local-boundary' in refs, 'publish this recipe before building'
build_lock = spark.BUILD_LOCK.open('w')
fcntl.flock(build_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
spark.require_idle_build_host(cluster)
assert spark.image_labels(tag) is None, 'candidate tag already exists; do not overwrite'
assert spark.available_memory_gib() >= 48
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
receipt_dir = ROOT / 'results/private/ced-local-projection' / ('image-'+stamp)
receipt_dir.mkdir(parents=True)
record = {'source_head': identity['candidate']['patch_head'], 'deployment_commit': head,
          'parent_image': PARENT, 'parent_labels': labels, 'changes': changes.splitlines(),
          'command': command, 'stream_command': stream_command, 'build_kind': 'Python-only immutable native reuse',
          'gpu_smoke': 'pending', 'serving_qualification': 'pending'}


def check_image(tag, source_head, stem):
    path = receipt_dir / (stem+'-source.json')
    path.write_text(json.dumps(manifest(source_head), indent=2)+'\n')
    command = ['docker', 'run', '--rm', '--runtime=runc', '--cpus=1', '--memory=4g',
               '--memory-swap=4g', '--pids-limit=128', '--network=none',
               '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'CUDA_VISIBLE_DEVICES=',
               '-v', str(EXP)+':/check:ro', '-v', str(receipt_dir)+':/receipts:ro',
               '--entrypoint', 'python3', tag, '/check/verify_image_source.py', '/receipts/'+path.name]
    record[stem+'_check_command'] = command
    raw = output(*command)
    (receipt_dir / (stem+'-check.txt')).write_text(raw+'\n')
    return json.loads(raw)


try:
    record['parent_check'] = check_image(PARENT, identity['unreleased_control']['head'], 'parent')
    # The exported source was validated by the normal build-input checker above.
    # The legacy builder exposes explicit per-container memory/CPU limits and
    # accepts this host's immutable image ID without another large image copy.
    os.environ['DOCKER_BUILDKIT'] = '0'
    record['min_available_gib'] = spark.run_memory_guarded(stream_command, 48)
    built = image(tag)
    record['image_id'] = built['Id']
    record['candidate_check'] = check_image(built['Id'], identity['candidate']['patch_head'], 'candidate')
    assert record['parent_check']['native_sha256'] == record['candidate_check']['native_sha256']
    layers = parent['RootFS']['Layers']
    assert built['RootFS']['Layers'][:len(layers)] == layers
    record['layers'] = built['RootFS']['Layers']
    for project in ('b12x', 'nccl', 'tilelang', 'tile_kernels', 'sparknet'):
        key = f'local.spark-ds41f.{project}.tree'
        assert built['Config']['Labels'][key] == labels[key]
    assert built['Config']['Labels']['local.spark-ds41f.vllm.tree'] == identity['candidate']['expected_tree']
    record['status'] = 'CPU source/native identity passed; GPU smoke and serving pending'
except BaseException as error:
    record['failure'] = repr(error)
    raise
finally:
    (receipt_dir / 'receipt.json').write_text(json.dumps(record, indent=2)+'\n')
    print('receipt', receipt_dir, flush=True)
