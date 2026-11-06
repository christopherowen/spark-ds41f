import json
from pathlib import Path
import shutil
import subprocess
import time

root = Path('/home/chris/projects/engram-cache-probe')
base = root / 'deployment/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-05dd5db6/src/vllm/vllm/models/deepseek_v4_1/engram_rows'
record = {'scope': 'CPU-driven native reader component tests; no CUDA or serving', 'started_unix': time.time(), 'commands': []}
try:
    for name, bound in (('control', 64), ('probe96', 96), ('probe128', 128), ('split64', 64), ('split96', 96), ('filter64', 64), ('filter96', 96)):
        source = root / 'runs/production-sweep-a' / name
        for p in [*base.glob('*.[ch]'), base / '_native.py']:
            if p.name != '_row_plan.h':
                shutil.copyfile(p, source / p.name)
        argv = ['python3', str(root / 'reader_cpu.py'), '--source', str(source), '--bound', str(bound),
                '--out', str(root / 'runs' / ('reader-' + name))]
        start = time.time()
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        record['commands'].append({'argv': argv, 'exit': result.returncode, 'elapsed_s': time.time() - start,
                                   'stdout': result.stdout, 'stderr': result.stderr})
        print(name, result.returncode, result.stdout, result.stderr, flush=True)
        result.check_returncode()
finally:
    record['finished_unix'] = time.time()
    (root / 'reader-commands.json').write_text(json.dumps(record, indent=2) + '\n')
