"""Bootstrap the existing image environment and run one bounded screen process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('gate', 'smoke', 'full'), required=True)
parser.add_argument('--process-id', required=True)
args = parser.parse_args()
manifest = json.loads(Path('/b/input-manifest.json').read_text())
for name, expected in manifest.items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'input hash mismatch: {name}')
print(json.dumps({'kind': 'verified_inputs', 'sha256': manifest,
                  'mode': args.mode, 'process_id': args.process_id}), flush=True)
subprocess.run(['uv', 'venv', '--system-site-packages', '--python', '/usr/bin/python3',
                '/tmp/expert-venv'], check=True)
python = '/tmp/expert-venv/bin/python'
if args.mode == 'gate':
    command = [python, '-m', 'pytest',
               'tests/kernels/moe/test_deepseek_v41_tilelang_moe.py',
               '-k', 'emits_requested_prefill_swizzle or prefill_swizzle_panel_matches_unswizzled_bits',
               '-v', '-o', 'addopts=']
else:
    command = [python, '/experiment/benchmark_swizzle.py',
               '--source-commit', 'f3bfdf8b8c08870754cc3c756b75faa9156595e0',
               '--source-sha256', manifest['/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py'],
               '--process-id', args.process_id]
    if args.mode == 'smoke':
        command.append('--smoke')
os.execv(python, command)
