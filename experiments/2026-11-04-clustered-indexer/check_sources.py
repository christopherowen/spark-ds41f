"""Verify the review patch and executable bundle contain the same source."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
RELATIVE = Path('vllm/models/deepseek_v4_1/tilelang/clustered_indexer.py')
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    subprocess.run(['git', 'apply',
                    str(HERE / 'vllm/0001-clustered-indexer.patch')], check=True,
                   cwd=root)
    actual = (root / RELATIVE).read_bytes()
    expected = (HERE / 'bundle/overlay' / RELATIVE).read_bytes()
    assert actual == expected, 'patch and executable overlay differ'
    ast.parse(actual)
print('patch and overlay match; Python syntax OK')

# Applying the capture patch backwards to its executable copies must recover the
# exact recorded candidate-36 indexer, without requiring an image or Git clone.
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    target = root / RELATIVE.parent
    target.mkdir(parents=True)
    for source in (HERE / 'capture/overlay').glob('*.py'):
        ast.parse(source.read_text())
        shutil.copy2(source, target / source.name)
    subprocess.run(['git', 'apply', '--reverse',
                    str(HERE / 'capture/0001-capture.patch')], cwd=root, check=True)
    expected_hash = json.loads((HERE / 'capture/base.json').read_text())['indexer_sha256']
    assert hashlib.sha256((target / 'indexer.py').read_bytes()).hexdigest() == expected_hash
    assert not (target / 'clustered_capture.py').exists()
print('capture patch recovers the exact candidate-36 indexer; syntax OK')
