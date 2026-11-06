"""Independent cleanup for one labeled native component job, never another task."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--window', required=True)
parser.add_argument('--run', required=True)
parser.add_argument('--unit', required=True)
parser.add_argument('--fixture-hold', type=Path)
args = parser.parse_args()
assert args.run.replace('-', '').replace('_', '').isalnum()
assert args.unit.replace('-', '').replace('_', '').removesuffix('.service').isalnum()
root = Path.home() / args.run
ready = Path.home() / ('.engram-watch-' + args.run + '.json')
record = {'window': args.window, 'run': args.run, 'unit': args.unit, 'pid': os.getpid(),
          'mode': 'fixture' if args.fixture_hold else 'production', 'started_unix': time.time(), 'commands': []}

def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    record['commands'].append({'argv': argv, 'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    return result

def hold():
    if args.fixture_hold:
        return json.loads(args.fixture_hold.read_text())
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'dgx1',
                             'cat', '/home/swank/spark-hold.json'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError('Cannot verify window: ' + result.stderr)
    return json.loads(result.stdout)

def persist():
    if root.exists():
        (root / 'watchdog.json').write_text(json.dumps(record, indent=2) + '\n')

ready.write_text(json.dumps({k: record[k] for k in ('window', 'run', 'unit', 'pid', 'mode')}) + '\n')
reason = None
state = {}
try:
    deadline = time.monotonic() + 30
    while not (root / 'watch.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Primary job did not start within 30 seconds')
        time.sleep(1)
    while True:
        state = json.loads((root / 'watch.json').read_text())
        assert state['window'] == args.window and state['unit'] == args.unit
        if state.get('complete'):
            record['result'] = 'primary completed and cleaned up'
            break
        current = hold()
        if current.get('window') != args.window:
            raise RuntimeError('Exclusive window changed')
        if time.time() - state['heartbeat_unix'] > 60:
            raise RuntimeError('Primary heartbeat is stale')
        if command(['systemctl', '--user', 'is-active', '--quiet', args.unit]).returncode:
            raise RuntimeError('Primary unit is not active')
        persist()
        time.sleep(5)
except BaseException as error:
    reason = repr(error)
    record['result'] = 'cleanup: ' + reason
    command(['systemctl', '--user', 'stop', args.unit])
    result = command(['docker', 'ps', '-aq', '--filter', 'label=local.engram.window=' + args.window,
                      '--filter', 'label=local.engram.run=' + args.run])
    for container in result.stdout.split():
        command(['docker', 'stop', '-t', '5', container])
    try:
        current = hold()
        owns_window = current.get('window') == args.window
    except Exception:
        owns_window = False
    if state.get('fan_restore_required'):
        if owns_window and not args.fixture_hold:
            result = command(['sudo', '-n', 'systemctl', 'stop', 'dgx-fan-control.service'])
            record['fan_restored'] = result.returncode == 0
        else:
            record['fan_restored'] = False
            record['fan_note'] = 'No fan change without verified window ownership'
finally:
    record['finished_unix'] = time.time()
    persist()
    ready.unlink(missing_ok=True)
print(json.dumps({'result': record.get('result'), 'record': str(root / 'watchdog.json')}), flush=True)
if reason:
    raise SystemExit(1)
