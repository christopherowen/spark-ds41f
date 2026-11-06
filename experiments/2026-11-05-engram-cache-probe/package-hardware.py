import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

root = Path('/home/chris/projects/engram-cache-probe')
dest = root / 'deployment/experiments/2026-11-05-engram-cache-probe/hardware'
dest.mkdir(exist_ok=True)
reader = 'vllm/models/deepseek_v4_1/engram_rows'
test = 'tests/kernels/test_deepseek_v41_engram_rows.py'
sources = [('control', '1f33102547e4da5c64712390604695d1abd577c4', root / 'vllm'),
           ('aligned', '745da6586fdebf8d48133639002ed41858831e13', root / 'vllm-aligned'),
           ('filter', 'e6fe9fb614d1b6488bb1310044a408822feaa6f6', root / 'vllm-filter')]
index = {'scope': 'Git exports of exact source commits for an unreleased native component fixture', 'variants': {}}
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w', format=tarfile.USTAR_FORMAT) as archive:
    for name, revision, repo in sources:
        paths = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '--name-only', revision,
                                         '--', reader, test], text=True).splitlines()
        files = {}
        for path in paths:
            data = subprocess.check_output(['git', '-C', str(repo), 'show', revision + ':' + path])
            relative = path.removeprefix(reader + '/') if path.startswith(reader + '/') else 'tests/' + Path(path).name
            item = tarfile.TarInfo(name + '/' + relative)
            item.size = len(data); item.mode = 0o644; item.mtime = 0
            archive.addfile(item, io.BytesIO(data))
            files[relative] = hashlib.sha256(data).hexdigest()
        index['variants'][name] = {'commit': revision, 'files': files}
payload = gzip.compress(buffer.getvalue(), compresslevel=6, mtime=0)
(dest / 'native-sources.tar.gz').write_bytes(payload)
index['archive_sha256'] = hashlib.sha256(payload).hexdigest()
(dest / 'sources.json').write_text(json.dumps(index, indent=2) + '\n')
print(json.dumps({'archive_bytes': len(payload), 'sha256': index['archive_sha256']}))
