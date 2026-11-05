import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path('/home/chris/projects/engram-cache-probe')
repo = root / 'deployment'
dest = repo / 'experiments/2026-11-05-engram-cache-probe'
sys.path.insert(0, str(repo))
from spark_ds41f.config import read_cluster

for shape in ('tp3', 'tp4'):
    profile = f'experiments/2026-11-05-engram-cache-probe/{shape}.json'
    cluster = read_cluster(profile)
    path = repo / cluster.get('nodes_config', 'config/nodes.json')
    assert path.name.endswith('.local.json') or path.name == 'nodes.json'
    nodes = json.loads(path.read_text())
    for node in nodes['nodes']:
        if node.get('head'):
            node['management_ip'] = cluster['distributed']['master_addr']
    path.write_text(json.dumps(nodes, indent=2) + '\n')
    result = subprocess.run([str(repo / 'bin/spark'), '--cluster-config', profile, 'doctor'],
                            cwd=repo, capture_output=True, text=True)
    (root / f'doctor-{shape}.log').write_text(result.stdout + result.stderr)
    print(shape, result.returncode, result.stdout, result.stderr)
    result.check_returncode()
for run in ('sp8-a', 'sp8-b', 'bounded-a', 'pinned-a', 'pinned-b', 'sanitizers', 'gcc-sanitizers'):
    target = dest / 'runs' / run
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / 'runs' / run / 'receipt.json', target / 'receipt.json')
for name in ('probe.c', 'analyze.py', 'results.json'):
    shutil.copyfile(root / name, dest / name)
shutil.copyfile(root / 'run-sanitizers.py', dest / 'run.py')
inputs = dest / 'runs/inputs'
inputs.mkdir(exist_ok=True)
for name in ('run.py', 'run-bounded.py', 'source-patch.sh', 'package.py', 'lock.sh', 'prepare-retry.sh'):
    shutil.copyfile(root / name, inputs / name)
logs = dest / 'runs/setup'
logs.mkdir(exist_ok=True)
for name in ('doctor-initial.log', 'doctor.log', 'doctor-tp3.log', 'doctor-tp4.log', 'prepare-retry.log', 'source-patch.log', 'lock.log'):
    shutil.copyfile(root / name, logs / name)
rejected = subprocess.run(['diff', '-u', '--label', 'a/vllm/models/deepseek_v4_1/engram_rows/_row_plan.h',
                          '--label', 'b/vllm/models/deepseek_v4_1/engram_rows/_row_plan.h',
                          str(root / 'base/_row_plan.h'), str(root / 'candidate/_row_plan.h')], capture_output=True, text=True)
assert rejected.returncode == 1
(inputs / 'rejected-unbounded.diff').write_text(rejected.stdout)
