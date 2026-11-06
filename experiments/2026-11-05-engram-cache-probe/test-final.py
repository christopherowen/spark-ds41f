import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

root = Path('/home/chris/projects/engram-cache-probe')
out = root / 'runs/final-source-cpu'
out.mkdir(exist_ok=False)
record = {'scope': 'Exact final source native CPU reader and ASan/UBSan planner; no GPU or serving',
          'started_unix': time.time(), 'commands': [], 'cases': [], 'sources': {}}

def command(argv):
    begin = time.time()
    result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=180)
    record['commands'].append({'argv': list(map(str, argv)), 'exit': result.returncode,
                              'elapsed_s': time.time() - begin, 'stdout': result.stdout, 'stderr': result.stderr})
    result.check_returncode()
    return result.stdout

try:
    for name, define in (('aligned', 'PLE_SPLIT_TAGS'), ('filter', 'PLE_FILTER_TAGS')):
        source = root / ('vllm-' + name) / 'vllm/models/deepseek_v4_1/engram_rows'
        record['sources'][name] = {'header_sha256': hashlib.sha256((source / '_row_plan.h').read_bytes()).hexdigest(),
                                   'identity': (root / 'final-candidates' / (name + '-identity.txt')).read_text()}
        print(name, command(['python3', root / 'reader_cpu.py', '--source', source, '--bound', 64,
                             '--out', out / ('reader-' + name)]), flush=True)
        executable = out / ('sanitize-' + name)
        command(['cc', '-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                 '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-D' + define + '=1', '-I', source,
                 root / 'runs/layout-a/audit-probe.c', '-o', executable])
        cases = [(n, hot, warm, True) for n in (6, 48, 64, 65, 96, 129) for hot in (0, 75, 100)
                 for warm in (1024, 16384)]
        cases += [(6144, 75, 1024, True), (16384, 0, 1024, False)]
        for tp in (3, 4):
            for tokens, hot, warm, populated in cases:
                value = json.loads(command([executable, tokens, tp, hot, int(populated), 2, warm, 1]))
                record['cases'].append({'variant': name, 'result': value})
        print(name, 'sanitizers passed', flush=True)
    record['passed'] = True
finally:
    record['finished_unix'] = time.time()
    (out / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
