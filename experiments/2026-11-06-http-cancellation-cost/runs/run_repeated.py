"""Run the HTTP lifetime regressions against installed or replacement source."""
import argparse
import hashlib
import importlib.util
import json
import os
import resource
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--replacement', type=Path)
parser.add_argument('--boundaries', action='store_true')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
os.environ['VLLM_PLUGINS'] = ''
args.output.mkdir(parents=True, exist_ok=True)
import vllm.entrypoints.serve.utils.api_utils as original
original_path = Path(original.__file__)
source = args.replacement or original_path
if args.replacement:
    spec = importlib.util.spec_from_file_location(original.__name__, args.replacement)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

import importlib.metadata
receipt = {
    'original_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest(),
    'tested_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'replacement': bool(args.replacement),
    'packages': {k: importlib.metadata.version(k) for k in
                 ('starlette', 'fastapi', 'httpx', 'anyio', 'pytest', 'uvicorn')},
}
import pytest
status = pytest.main(['-q',
    '-p', 'no:cacheprovider', '--tb=short', str(root / 'test_repeated_cancellation.py'),
    '--junitxml=' + str(args.output / 'junit.xml')])
receipt['exit_code'] = int(status)
receipt['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
(args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
raise SystemExit(status)
