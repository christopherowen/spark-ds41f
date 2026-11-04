"""Verify the review patch and executable bundle contain the same source."""
from pathlib import Path
import ast
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
