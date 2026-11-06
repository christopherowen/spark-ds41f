"""Run an unchanged functional client with exact HTTP request-body evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import threading
import time
import urllib.request


class RequestArchive:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()
        self.records = []

    def add(self, request):
        body = request.data
        if body is None:
            return
        assert isinstance(body, bytes)
        assert request.full_url.startswith('http://127.0.0.1:8000/')
        with self.lock:
            index = len(self.records)
            filename = f'{index:04d}.body.json'
            (self.directory / filename).write_bytes(body)
            self.records.append({'order': index, 'url': request.full_url,
                                 'method': request.get_method(), 'at': time.time(),
                                 'body_file': filename, 'body_bytes': len(body),
                                 'body_sha256': hashlib.sha256(body).hexdigest()})
            (self.directory / 'manifest.json').write_text(
                json.dumps({'requests': self.records}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--token', required=True)
    parser.add_argument('script', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    hold = json.loads((Path.home() / 'spark-hold.json').read_text())
    assert hold.get('token') == args.token and hold.get('holder') == 'ced-sp567-native-correctness'
    assert hold.get('nodes') == ['sp5', 'sp6', 'sp7'] and hold.get('phase') == 'serving'
    assert len(os.sched_getaffinity(0)) == 1, 'functional client requires one CPU'
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    script = args.script.resolve()
    archive = RequestArchive(args.out)
    identity = {'script': str(script), 'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
                'arguments': args.arguments, 'token': args.token, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                'address_space_limit_bytes': 1024**3, 'scope': 'request correctness; no timing qualification'}
    (args.out / 'client.json').write_text(json.dumps(identity, indent=2) + '\n')
    original = urllib.request.Request.__init__

    def recorded(request, *positional, **keyword):
        original(request, *positional, **keyword)
        archive.add(request)

    urllib.request.Request.__init__ = recorded
    sys.argv = [str(script), *args.arguments]
    try:
        runpy.run_path(str(script), run_name='__main__')
    finally:
        urllib.request.Request.__init__ = original
        (args.out / 'finished.json').write_text(json.dumps(
            {'requests': len(archive.records), 'at': time.time()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
