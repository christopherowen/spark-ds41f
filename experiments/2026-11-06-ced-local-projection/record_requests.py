"""Run an unchanged functional client with exact HTTP request-body evidence and stock-cooling pacing."""
import argparse
import builtins
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import subprocess
import sys
import threading
import time
import urllib.request

NODES = ('10.30.0.13', '10.30.0.14', '10.30.0.15')
HOLDER = 'ced-sp567-native-correctness'
NAMES = ['sp5', 'sp6', 'sp7']
SSH_OPTS = (
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=10',
    '-o', 'ControlMaster=auto',
    '-o', 'ControlPath=/tmp/spark-ssh-%C',
    '-o', 'ControlPersist=600',
)
WAVE1_SIZE = 8
SINGLE_WAVE_MAX = 9
NODE_READ = """from pathlib import Path
import json,subprocess,time
h=json.loads((Path.home()/'spark-hold.json').read_text())
zones={p.parent.name:int(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
print(json.dumps({'at':time.time(),'hold':h,'thermal_zones_c':zones,'host_available_bytes':available,'gpu_temperature_c':float(gpu)}))
"""


def local_hottest_c():
    zones = [int(p.read_text()) / 1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')]
    return max(zones) if zones else 0.0


def read_nodes(token, floor_gib=3):
    records = {}
    for node in NODES:
        out = subprocess.check_output(
            ['ssh', *SSH_OPTS, 'chris@' + node, 'python3 -'],
            input=NODE_READ,
            text=True,
            timeout=30,
        )
        row = json.loads(out)
        hold = row['hold']
        assert hold.get('holder') == HOLDER and hold.get('token') == token and hold.get('nodes') == NAMES, node + ': lease changed'
        assert hold.get('phase') == 'serving', node + ': phase changed'
        assert row['thermal_zones_c'], node + ': no thermal sensors'
        assert max(row['thermal_zones_c'].values()) <= 70 and row['gpu_temperature_c'] <= 70, node + ': thermal guard'
        assert row['host_available_bytes'] >= floor_gib * 2**30, node + ': memory reserve'
        records[node] = row
    return records


class TimedResponse:
    def __init__(self, response, finish):
        self._response = response
        self._finish = finish

    def __getattr__(self, name):
        return getattr(self._response, name)

    def __iter__(self):
        return iter(self._response)

    def __enter__(self):
        self._response.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._response.__exit__(exc_type, exc, tb)
        finally:
            if self._finish is not None:
                finish, self._finish = self._finish, None
                finish(exc_type is None)


class RequestArchive:
    def __init__(self, directory, *, token=None, below_c=48.5, settle_polls=2, poll_sleep=3.0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()
        self.records = []
        self.cooldowns = []
        self.token = token
        self.below_c = below_c
        self.settle_polls = settle_polls
        self.poll_sleep = poll_sleep

    def _write_manifest(self):
        (self.directory / 'manifest.json').write_text(
            json.dumps({'requests': self.records}, indent=2) + '\n')

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
            record = {'order': index, 'url': request.full_url,
                      'method': request.get_method(), 'at': time.time(),
                      'body_file': filename, 'body_bytes': len(body),
                      'body_sha256': hashlib.sha256(body).hexdigest()}
            self.records.append(record)
            request._archive_order = index
            self._write_manifest()

    def complete(self, request, started_monotonic, ok):
        order = getattr(request, '_archive_order', None)
        if order is None:
            return
        duration = round(time.monotonic() - started_monotonic, 6)
        with self.lock:
            self.records[order]['completed_at'] = time.time()
            self.records[order]['duration_seconds'] = duration
            self.records[order]['ok'] = ok
            self._write_manifest()

    def cool(self, reason, *, settle_polls=None):
        if not self.token:
            return
        needed = self.settle_polls if settle_polls is None else settle_polls
        began = time.monotonic()
        started_at = time.time()
        settled = 0
        polls = 0
        peak_hottest = 0.0
        peak_gpu = 0.0
        final_hottest = {}
        final_gpu = {}
        while True:
            current = read_nodes(self.token)
            polls += 1
            hottest = {node: max(row['thermal_zones_c'].values()) for node, row in current.items()}
            gpus = {node: row['gpu_temperature_c'] for node, row in current.items()}
            peak_hottest = max(peak_hottest, max(hottest.values()))
            peak_gpu = max(peak_gpu, max(gpus.values()))
            if all(hottest[node] < self.below_c and gpus[node] < self.below_c for node in NODES):
                settled += 1
                if settled >= needed:
                    final_hottest = hottest
                    final_gpu = gpus
                    break
            else:
                settled = 0
            assert time.monotonic() - began < 600, reason + ': passive cooling deadline'
            time.sleep(self.poll_sleep)
        with self.lock:
            self.cooldowns.append({
                'reason': reason,
                'started_at': started_at,
                'ended_at': time.time(),
                'elapsed_seconds': round(time.monotonic() - began, 3),
                'polls': polls,
                'settle_polls': needed,
                'below_c': self.below_c,
                'peak_hottest_c': peak_hottest,
                'peak_gpu_c': peak_gpu,
                'final_hottest_c': final_hottest,
                'final_gpu_c': final_gpu,
            })
            (self.directory / 'cooldowns.json').write_text(
                json.dumps({'cooldowns': self.cooldowns}, indent=2) + '\n')

    def after_main_request(self, body_bytes):
        if not self.token:
            return
        if body_bytes > 4096:
            self.cool('main-long-request', settle_polls=self.settle_polls)
        elif local_hottest_c() >= self.below_c:
            self.cool('main-short-request', settle_polls=self.settle_polls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--token', required=True)
    parser.add_argument('script', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    hold = json.loads((Path.home() / 'spark-hold.json').read_text())
    assert hold.get('token') == args.token and hold.get('holder') == HOLDER
    assert hold.get('nodes') == NAMES and hold.get('phase') == 'serving'
    assert len(os.sched_getaffinity(0)) == 1, 'functional client requires one CPU'
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    script = args.script.resolve()
    archive = RequestArchive(args.out, token=args.token)
    identity = {'script': str(script), 'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
                'arguments': args.arguments, 'token': args.token, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                'address_space_limit_bytes': 1024**3, 'cooldown_below_c': archive.below_c,
                'cooldown_settle_polls': archive.settle_polls,
                'scope': 'request correctness; stock-cooled inter-request pacing; no timing qualification'}
    (args.out / 'client.json').write_text(json.dumps(identity, indent=2) + '\n')
    original_request_init = urllib.request.Request.__init__
    original_urlopen = urllib.request.urlopen
    original_thread_start = threading.Thread.start
    original_thread_join = threading.Thread.join
    original_print = builtins.print
    worker_threads = []
    worker_lock = threading.Condition()
    batch_all_started = False
    wave1_remaining = 0
    wave2_released = False

    def complete_wave1_slot():
        nonlocal wave1_remaining, wave2_released
        should_cool_wave = False
        with worker_lock:
            if wave1_remaining > 0:
                wave1_remaining -= 1
                if wave1_remaining == 0 and len(worker_threads) > SINGLE_WAVE_MAX and not wave2_released:
                    should_cool_wave = True
        if should_cool_wave:
            try:
                archive.cool('batch-wave')
            finally:
                with worker_lock:
                    wave2_released = True
                    worker_lock.notify_all()

    def recorded_init(request, *positional, **keyword):
        original_request_init(request, *positional, **keyword)
        archive.add(request)

    def recorded_urlopen(url, *positional, **keyword):
        current = threading.current_thread()
        request = url if isinstance(url, urllib.request.Request) else None
        body_bytes = len(request.data) if request is not None and request.data is not None else 0
        batch_index = getattr(current, '_batch_index', None)
        is_wave1 = False
        if current is not threading.main_thread() and batch_index is not None and script.name != 'ced_cross_shape.py':
            if batch_index < WAVE1_SIZE:
                is_wave1 = True
            else:
                need_long_delay = False
                with worker_lock:
                    worker_lock.wait_for(lambda: batch_all_started, timeout=5.0)
                    if len(worker_threads) > SINGLE_WAVE_MAX:
                        worker_lock.wait_for(lambda: wave2_released, timeout=900.0)
                        need_long_delay = body_bytes > 4096
                if need_long_delay:
                    time.sleep(0.3)
        started = time.monotonic()
        try:
            response = original_urlopen(url, *positional, **keyword)
        except BaseException:
            if is_wave1:
                complete_wave1_slot()
            raise

        def finish(ok):
            if request is not None:
                archive.complete(request, started, ok)
            if is_wave1:
                complete_wave1_slot()
            if (ok and script.name != 'ced_cross_shape.py'
                    and current is threading.main_thread()
                    and body_bytes > 0):
                archive.after_main_request(body_bytes)

        return TimedResponse(response, finish)

    def tracked_start(thread, *positional, **keyword):
        nonlocal wave1_remaining
        if threading.current_thread() is threading.main_thread():
            with worker_lock:
                thread._batch_index = len(worker_threads)
                worker_threads.append(thread)
                if len(worker_threads) <= WAVE1_SIZE:
                    wave1_remaining += 1
        return original_thread_start(thread, *positional, **keyword)

    def tracked_join(thread, *positional, **keyword):
        nonlocal batch_all_started, wave1_remaining, wave2_released
        if threading.current_thread() is threading.main_thread() and script.name != 'ced_cross_shape.py':
            with worker_lock:
                if not batch_all_started:
                    batch_all_started = True
                    worker_lock.notify_all()
        result = original_thread_join(thread, *positional, **keyword)
        if threading.current_thread() is threading.main_thread() and script.name != 'ced_cross_shape.py':
            should_cool = False
            with worker_lock:
                if worker_threads and all(not t.is_alive() for t in worker_threads):
                    worker_threads.clear()
                    batch_all_started = False
                    wave1_remaining = 0
                    wave2_released = False
                    should_cool = True
            if should_cool:
                archive.cool('batch-join')
        return result

    def tracked_print(*values, **keyword):
        original_print(*values, **keyword)
        if (script.name == 'ced_cross_shape.py' and len(values) == 1
                and isinstance(values[0], str) and values[0].startswith('{')):
            row = json.loads(values[0])
            if 'length' in row and 'repeat' in row:
                archive.cool(f"ced-{row['length']}-{row['repeat']}")

    urllib.request.Request.__init__ = recorded_init
    urllib.request.urlopen = recorded_urlopen
    threading.Thread.start = tracked_start
    threading.Thread.join = tracked_join
    builtins.print = tracked_print
    sys.argv = [str(script), *args.arguments]
    try:
        runpy.run_path(str(script), run_name='__main__')
    finally:
        urllib.request.Request.__init__ = original_request_init
        urllib.request.urlopen = original_urlopen
        threading.Thread.start = original_thread_start
        threading.Thread.join = original_thread_join
        builtins.print = original_print
        (args.out / 'finished.json').write_text(json.dumps(
            {'requests': len(archive.records), 'cooldowns': len(archive.cooldowns), 'at': time.time()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
