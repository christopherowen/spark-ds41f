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
    subprocess.run(['git', 'apply',
                    str(HERE / 'vllm/0003-stream-envelope.patch')], check=True,
                   cwd=root)
    subprocess.run(['git', 'apply', str(HERE / 'vllm/0004-tensor-interval-bounds.patch')],
                   cwd=root, check=True)
    subprocess.run(['git', 'apply', str(HERE / 'vllm/0005-share-owned-snapshot.patch')],
                   cwd=root, check=True)
    grouped = RELATIVE.parent / 'grouped_indexer.py'
    assert (root / grouped).read_bytes() == (HERE / 'bundle/overlay' / grouped).read_bytes()
    ast.parse((root / grouped).read_text())
    subprocess.run(['git', 'apply', str(HERE / 'vllm/0006-preserve-source-handoff.patch')],
                   cwd=root, check=True)
    subprocess.run(['git', 'apply', str(HERE / 'vllm/0007-source-tie-cutoff.patch')],
                   cwd=root, check=True)
    subprocess.run(['git', 'apply', str(HERE / 'vllm/0008-pack-source-cutoff.patch')],
                   cwd=root, check=True)
    subprocess.run(['git', 'apply', str(HERE / 'vllm/0009-share-source-snapshot.patch')],
                   cwd=root, check=True)
    subprocess.run(['git', 'apply', str(HERE / 'vllm/0010-map-source-handoff.patch')],
                   cwd=root, check=True)
    subprocess.run(['git', 'apply', str(HERE / 'vllm/0011-single-source-snapshot.patch')],
                   cwd=root, check=True)
    grouped_source = RELATIVE.parent / 'grouped_source_indexer.py'
    assert (root / grouped_source).read_bytes() == (HERE / 'bundle/overlay' / grouped_source).read_bytes()
    ast.parse((root / grouped_source).read_text())
    source = RELATIVE.parent / 'source_indexer.py'
    assert (root / source).read_bytes() == (HERE / 'bundle/overlay' / source).read_bytes()
    ast.parse((root / source).read_text())
    subprocess.run(['git', 'apply', str(HERE / 'vllm/0012-visible-metadata-domain.patch')],
                   cwd=root, check=True)
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
    for patch in reversed((HERE / 'capture/series').read_text().splitlines()):
        subprocess.run(['git', 'apply', '--reverse',
                        str(HERE / 'capture' / patch)], cwd=root, check=True)
    expected_hash = json.loads((HERE / 'capture/base.json').read_text())['indexer_sha256']
    assert hashlib.sha256((target / 'indexer.py').read_bytes()).hexdigest() == expected_hash
    assert not (target / 'clustered_capture.py').exists()
    attention_hash = json.loads((HERE / 'capture/base.json').read_text())['attention_sha256']
    assert hashlib.sha256((target / 'attention.py').read_bytes()).hexdigest() == attention_hash
print('capture patch recovers the exact candidate-36 indexer; syntax OK')

# The independent top-k dependency owns its patch, executable copy and regression.
import sys
subprocess.run([sys.executable, str(HERE.parent / '2026-11-04-topk-live-reads/check_sources.py')], check=True)

# Real-input replay uses C36's own API plus the same independent scratch guard.
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    target = root / RELATIVE.parent / 'indexer.py'
    target.parent.mkdir(parents=True)
    target.write_bytes((HERE / 'replay/overlay/indexer.py').read_bytes())
    subprocess.run(['git', 'apply', '--reverse', str(HERE.parent /
                    '2026-11-04-topk-live-reads/vllm/0001-topk-merge-live-reads.patch')],
                   cwd=root, check=True)
    expected_hash = json.loads((HERE / 'capture/base.json').read_text())['indexer_sha256']
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected_hash
print('replay dependency recovers the exact candidate-36 indexer; syntax OK')

# Freeze the preceding unreleased source prototype independently of current edits.
control_hash = json.loads((HERE / 'source-ties.json').read_text())['control_sha256']
assert hashlib.sha256((HERE / 'bundle/source_reference.py').read_bytes()).hexdigest() == control_hash
print('archived source control hash matches')

tie_hash = json.loads((HERE / 'source-cutoff-packing.json').read_text())['archived_tie_source_sha256']
assert hashlib.sha256((HERE / 'bundle/source_ties_reference.py').read_bytes()).hexdigest() == tie_hash
print('archived tie-cutoff source hash matches')
