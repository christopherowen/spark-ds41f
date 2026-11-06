import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

root = Path('/home/chris/projects/engram-cache-probe')
exp = root / 'deployment/experiments/2026-11-05-engram-cache-probe'
out = root / 'runs/image-cpu-reader-minimal'
out.mkdir(exist_ok=False)
native = out / 'native'
native.mkdir()
metadata = json.loads((exp / 'hardware/sources.json').read_text())
archive_path = exp / 'hardware/native-sources.tar.gz'
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == metadata['archive_sha256']
with tarfile.open(archive_path) as archive:
    archive.extractall(native, filter='data')
image = 'sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b'
report = {'scope': 'CPU-only native direct-I/O validation in the pinned image; runc, no GPU devices',
          'image': image, 'started_unix': time.time(), 'commands': []}
try:
    for name in ('aligned', 'filter'):
        target = out / name
        target.mkdir()
        for relative, sha in metadata['variants'][name]['files'].items():
            assert hashlib.sha256((native / name / relative).read_bytes()).hexdigest() == sha
        argv = ['docker', 'run', '--rm', '--runtime=runc', '--name', 'engram-image-cpu-' + name,
                '--cpus=1', '--cpuset-cpus=19', '--memory=1g', '--memory-swap=1g',
                '--security-opt=seccomp=unconfined', '--ulimit=memlock=-1:-1',
                '--network=none', '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'OMP_NUM_THREADS=1',
                '-v', str(native / name) + ':/source:ro', '-v', str(exp / 'reader_cpu.py') + ':/reader_cpu.py:ro',
                '-v', str(target) + ':/out', '--entrypoint', 'python3', image, '-c',
                'from pathlib import Path; import runpy,sys; assert not list(Path("/dev").glob("nvidia*"));'
                'sys.argv=["reader_cpu.py","--source","/source","--out","/out/reader","--bound","64"];'
                'runpy.run_path("/reader_cpu.py",run_name="__main__")']
        begin = time.time()
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        report['commands'].append({'argv': argv, 'exit': result.returncode, 'elapsed_s': time.time() - begin,
                                   'stdout': result.stdout, 'stderr': result.stderr})
        print(name, result.returncode, result.stdout, result.stderr, flush=True)
        result.check_returncode()
    report['passed'] = True
finally:
    report['finished_unix'] = time.time()
    (out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
