"""CPU-only image/source check; does not initialize a GPU or import vLLM."""
import hashlib
import json
import os
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
root = Path('/opt/spark-ds41f/candidate/vllm')
failures = []
for entry in manifest['files']:
    path = root / entry['path']
    try:
        if entry['mode'] == '120000':
            data = os.readlink(path).encode()
        else:
            data = path.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != entry['blob']:
            failures.append([entry['path'], entry['blob'], actual])
    except OSError as error:
        failures.append([entry['path'], str(error)])
assert not failures, failures[:20]
binaries = {}
for name in ('_C_stable_libtorch', '_moe_C_stable_libtorch'):
    path = root / 'vllm' / (name + '.abi3.so')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        assert stream.read(4) == b'\x7fELF'
        stream.seek(0)
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    binaries[name] = h.hexdigest()
print(json.dumps({'head': manifest['head'], 'tree': manifest['tree'],
                  'source_files_verified': len(manifest['files']), 'native_sha256': binaries,
                  'scope': 'CPU source/native identity only; GPU smoke remains required'}))
