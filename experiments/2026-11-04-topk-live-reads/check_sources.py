"""Check the isolated fix against the exact recorded original source."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
RELATIVE = Path('vllm/models/deepseek_v4_1/tilelang/indexer.py')
base = json.loads((HERE / 'base.json').read_text())
provenance = json.loads((HERE / 'provenance.json').read_text())
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    target = root / RELATIVE
    target.parent.mkdir(parents=True)
    expected = (HERE / 'bundle/overlay' / RELATIVE).read_bytes()
    target.write_bytes(expected)
    patch = HERE / 'vllm/0001-topk-merge-live-reads.patch'
    subprocess.run(['git', 'apply', '--reverse', str(patch)], cwd=root, check=True)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == base['indexer_sha256']
    subprocess.run(['git', 'apply', str(patch)], cwd=root, check=True)
    assert target.read_bytes() == expected
    assert hashlib.sha256(expected).hexdigest() == provenance['copied_evidence']['fixed_indexer_sha256']
for path in HERE.rglob('*.py'):
    ast.parse(path.read_text(), filename=str(path), feature_version=(3, 12))
assert hashlib.sha256((HERE / 'bundle/topk_reads.py').read_bytes()).hexdigest() == provenance['copied_evidence']['script_sha256']
assert not json.loads((HERE / 'runs/original-01/verdict.json').read_text())['passed']
assert json.loads((HERE / 'runs/fixed-01/verdict.json').read_text())['passed']
print('original hash, patch roundtrip, recorded executable identity and Python 3.12 syntax pass')
