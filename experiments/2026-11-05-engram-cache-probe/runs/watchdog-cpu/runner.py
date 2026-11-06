import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/home/chris/projects/engram-cache-probe')
out = root / 'runs/watchdog-cpu'
out.mkdir(exist_ok=False)
image = 'sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b'
report = {'scope': 'CPU-only fault injection for independent cleanup; no GPU devices or fan/serving changes',
          'image': image, 'started_unix': time.time(), 'cases': []}

def call(argv, check=True):
    r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    current['commands'].append({'argv': argv, 'exit': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr})
    if check:
        r.check_returncode()
    return r

try:
    for failure in ('unit-killed', 'heartbeat-stale', 'window-changed'):
        name = 'engram-watch-fixture-' + failure + '-20261106'
        unit = name + '.service'
        window = 'cpu-fixture-' + failure
        work = Path.home() / name
        work.mkdir(exist_ok=False)
        hold = work / 'fixture-hold.json'
        hold.write_text(json.dumps({'window': window}))
        state = {'window': window, 'unit': unit, 'heartbeat_unix': time.time(),
                 'fan_restore_required': False, 'complete': False}
        (work / 'watch.json').write_text(json.dumps(state))
        current = {'failure': failure, 'commands': []}
        report['cases'].append(current)
        primary = None
        own, control = name + '-own', name + '-control'
        try:
            for container, label in ((own, name), (control, name + '-other')):
                call(['docker', 'run', '-d', '--rm', '--runtime=runc', '--name', container,
                      '--cpus=0.1', '--memory=32m', '--memory-swap=32m', '--network=none',
                      '-e', 'NVIDIA_VISIBLE_DEVICES=void', '--label', 'local.engram.window=' + window,
                      '--label', 'local.engram.run=' + label, '--entrypoint', '/bin/sh', image,
                      '-c', 'test ! -e /dev/nvidia0 && sleep 120'])
            call(['systemd-run', '--user', '--unit=' + unit, '--collect', '-p', 'MemoryMax=32M',
                  '-p', 'CPUQuota=5%', '-p', 'TasksMax=4', '/bin/sleep', '120'])
            watcher = [sys.executable, str(root / 'watch-hardware.py'), '--window', window, '--run', name,
                       '--unit', unit, '--fixture-hold', str(hold)]
            primary = subprocess.Popen(watcher, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + 15
            while not (work / 'watchdog.json').exists():
                assert primary.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.1)
            if failure == 'unit-killed':
                call(['systemctl', '--user', 'kill', '--signal=KILL', unit])
            elif failure == 'heartbeat-stale':
                state['heartbeat_unix'] = time.time() - 120
                (work / 'watch.json').write_text(json.dumps(state))
            else:
                hold.write_text(json.dumps({'window': 'different-window'}))
            stdout, stderr = primary.communicate(timeout=25)
            current.update(watcher_argv=watcher, watcher_exit=primary.returncode, stdout=stdout, stderr=stderr)
            assert primary.returncode == 1, current
            assert call(['docker', 'inspect', own], check=False).returncode != 0, 'Own container survived'
            assert call(['docker', 'inspect', '--format', '{{.State.Running}}', control]).stdout.strip() == 'true'
            current['watchdog'] = json.loads((work / 'watchdog.json').read_text())
            assert not any('dgx-fan-control.service' in c['argv'] for c in current['watchdog']['commands'])
            current['passed'] = True
            print(failure, 'passed; only the labeled fixture container stopped', flush=True)
        finally:
            if primary is not None and primary.poll() is None:
                primary.terminate(); primary.wait(timeout=15)
            call(['systemctl', '--user', 'stop', unit], check=False)
            call(['docker', 'stop', '-t', '1', own, control], check=False)
    report['passed'] = True
finally:
    report['finished_unix'] = time.time()
    (out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
