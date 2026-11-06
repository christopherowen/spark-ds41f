import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path('/home/chris/projects/engram-cache-probe')
repo = root / 'deployment'
sys.path.insert(0, str(repo))
from spark_ds41f import build

cache = Path('/home/chris/projects/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-9b160482')
for name in ('aligned', 'filter'):
    relative = Path('experiments/2026-11-05-engram-cache-probe') / name
    upstreams = json.loads((repo / relative / 'upstreams.lock.json').read_text())
    inputs = build.build_inputs(upstreams)
    directory = build.build_directory(inputs)
    for project in build.build_projects(inputs):
        target = directory / 'src' / project
        source = cache / 'src' / project
        if not target.exists() and source.exists() and not build.project_problems(source, inputs[project]):
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['cp', '-a', '--reflink=auto', str(source), str(target)], check=True)
            print(name, project, 'copied only after exact locked source validation', flush=True)
    target = directory / 'src/cutlass'
    if not target.exists() and build.git_output(cache / 'src/cutlass', 'rev-parse', 'HEAD') == inputs['cutlass']['revision']:
        subprocess.run(['cp', '-a', '--reflink=auto', str(cache / 'src/cutlass'), str(target)], check=True)
    wheel_source = cache / 'context/cutlass-dsl-wheels'
    wheel_target = directory / 'context/cutlass-dsl-wheels'
    if not wheel_target.exists() and not build.wheel_problems(wheel_source):
        wheel_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(wheel_source, wheel_target)
    for command in ('prepare', 'check'):
        subprocess.run([str(repo / 'bin/spark'), '--cluster-config', str(relative / 'tp3.json'),
                        'build', command], cwd=repo, check=True)
    print(name, 'complete build inputs validated', directory, flush=True)
