#!/usr/bin/env python3
"""Owner-authorized ext4 checkpoint maintenance; run in a bounded systemd unit."""
import argparse
import datetime
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import subprocess
import time


def emit(event, **fields):
    print(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), event=event, **fields)), flush=True)


def available():
    return int(re.search(r'^MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text(), re.M)[1]) * 1024


def memory_guard():
    value = available()
    if value < 4 * 2**30:
        raise RuntimeError(f'MemAvailable below 4 GiB: {value}')
    return value


def counts(path):
    p = subprocess.run(['e4defrag', '-c', str(path)], text=True, capture_output=True, check=True, timeout=30, env=dict(os.environ, LC_ALL='C'))
    m = re.search(r'Total/best extents\s+(\d+)/(\d+)', p.stdout)
    if not m:
        raise RuntimeError(p.stdout + p.stderr)
    return list(map(int, m.groups()))


def digest(path):
    """Aligned O_DIRECT reads avoid filling the host page cache with weights."""
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        with mmap.mmap(-1, 8 * 2**20) as buf:
            view = memoryview(buf)
            try:
                while True:
                    n = os.readv(fd, [view])
                    if not n:
                        break
                    h.update(view[:n])
                    # A partial EOF read leaves the offset unaligned for another O_DIRECT read.
                    if n < len(view):
                        break
            finally:
                view.release()
    finally:
        os.close(fd)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path('/home/swank/projects/spark3-vllm-ds41f')
    config = json.loads((repo / 'config/cluster.json').read_text())
    root = Path(next(s for s, t, _ in config['container']['mounts'] if t == '/models').format(home=config['host']['home']))
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    files = sorted(set(index['weight_map'].values()))
    if args.only:
        if args.only not in files:
            raise RuntimeError('requested shard is absent from index')
        files = [args.only]
    baseline = []
    for name in files:
        path = (root / name).resolve(strict=True)
        st = path.stat()
        baseline.append(dict(name=name, path=str(path), bytes=st.st_size, inode=st.st_ino, mtime_ns=st.st_mtime_ns, before=counts(path)))
    save(out / 'before.json', baseline)
    emit('started', files=len(files), source_commit=subprocess.check_output(['git', '-c', 'safe.directory=' + str(repo), '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(), mem_available=memory_guard())
    for number, record in enumerate(baseline, 1):
        path = Path(record['path'])
        current, best = record['before']
        if current <= best:
            emit('already_ideal', name=record['name'], current=current, best=best)
            continue
        memory_guard()
        started = time.monotonic()
        emit('checksum_before', file=number, name=record['name'], bytes=record['bytes'], current=current, best=best)
        before_hash = digest(path)
        record['sha256_before'] = before_hash
        if before_hash != path.name:
            raise RuntimeError(f'Checkpoint hash differs from content-addressed blob name: {path}')
        record['hash_before_seconds'] = time.monotonic() - started
        memory_guard()
        emit('defragmenting', name=record['name'], checksum_seconds=record['hash_before_seconds'])
        with (out / (record['name'] + '.e4defrag.log')).open('w') as log:
            child = subprocess.Popen(['e4defrag', '-v', str(path)], stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, LC_ALL='C'))
            low_memory = False
            try:
                while child.poll() is None:
                    if available() < 4 * 2**30:
                        low_memory = True
                        child.terminate()
                        child.wait(timeout=30)
                        break
                    time.sleep(0.5)
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=30)
        record['defrag_exit_code'] = child.returncode
        emit('checksum_after', name=record['name'], exit_code=child.returncode)
        record['sha256_after'] = digest(path)
        record['after'] = counts(path)
        record['elapsed_seconds'] = time.monotonic() - started
        st = path.stat()
        if (st.st_ino, st.st_size, st.st_mtime_ns) != (record['inode'], record['bytes'], record['mtime_ns']):
            raise RuntimeError(f'Unexpected inode/size/mtime change: {path}')
        record['verified'] = record['sha256_after'] == before_hash
        save(out / (record['name'] + '.json'), record)
        emit('file_complete', name=record['name'], before=record['before'], after=record['after'], verified=record['verified'], elapsed_seconds=record['elapsed_seconds'], mem_available=available())
        if not record['verified']:
            raise RuntimeError(f'CHECKSUM MISMATCH: {path}')
        if low_memory or child.returncode:
            raise RuntimeError(f'Defrag stopped: low_memory={low_memory}, exit={child.returncode}')
    final = [dict(name=r['name'], counts=counts(r['path'])) for r in baseline]
    save(out / 'after.json', final)
    emit('complete', files=len(final), current=sum(f['counts'][0] for f in final), best=sum(f['counts'][1] for f in final), nonideal=sum(f['counts'][0] > f['counts'][1] for f in final))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        emit('error', detail=str(error))
        raise
