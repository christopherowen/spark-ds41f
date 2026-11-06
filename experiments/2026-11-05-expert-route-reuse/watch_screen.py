"""Independent user-unit watchdog for one labelled experimental container.

Start separately from the screen unit. An absent main unit, lost hold or a
120-second-old heartbeat stops only the container with the matching name AND
window label. This does not restore serving or close the global cluster window;
that remains the existing lab watchdog/controller's responsibility.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--unit', required=True)
parser.add_argument('--process-id', required=True)
parser.add_argument('--holder', required=True)
parser.add_argument('--window', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
name = 'expert-swizzle-' + args.process_id
hold_path = Path.home() / 'spark-hold.json'
out = Path(args.out)
began = time.monotonic()


def run(*argv):
    return subprocess.run(argv, capture_output=True, text=True, timeout=20)


while True:
    # Read only this exact container identity; never stop a borrowed GPU job.
    inspected = run('docker', 'inspect', name)
    container = json.loads(inspected.stdout)[0] if inspected.returncode == 0 else None
    active = run('systemctl', '--user', 'show', args.unit, '-p', 'ActiveState', '--value').stdout.strip()
    try:
        hold = json.loads(hold_path.read_text())
        ours = hold.get('holder') == args.holder and hold.get('window') == args.window
        beat = dt.datetime.fromisoformat(hold['heartbeat'].replace('Z', '+00:00'))
        stale = (dt.datetime.now(dt.timezone.utc) - beat).total_seconds() > 120
    except (OSError, ValueError, KeyError):
        ours, stale = False, True
    if container is None and active not in ('active', 'activating') and time.monotonic() - began > 30:
        break
    reason = None
    if not ours or stale:
        reason = 'lost hold or heartbeat older than 120 seconds'
    elif active not in ('active', 'activating') and time.monotonic() - began > 30:
        reason = 'main screen unit ended with its container still present'
    if container is not None and reason:
        labels = container.get('Config', {}).get('Labels') or {}
        if labels.get('expert-route-window') != args.window:
            raise RuntimeError('container label differs; refusing to stop it')
        stopped = run('docker', 'stop', '--time', '10', name)
        if stopped.returncode:
            run('docker', 'kill', name)
        out.mkdir(parents=True, exist_ok=True)
        fan_restore = []
        if ours:
            try:
                resource = json.loads((out / 'resources.json').read_text())
                if resource.get('performance_mode') and not resource.get('finished'):
                    if resource.get('fan_service_before') == 'active':
                        fan_restore.append(run('sudo', '-n', 'systemctl', 'start', 'dgx-fan-control.service').returncode)
                    elif resource.get('fan_service_before') == 'inactive':
                        fan_restore.append(run('sudo', '-n', 'systemctl', 'stop', 'dgx-fan-control.service').returncode)
                        fan_restore.append(run('sudo', '-n', 'dgx-fan-control', 'automatic').returncode)
            except (OSError, ValueError):
                fan_restore.append('resource receipt unavailable; no fan setting guessed')
        (out / 'independent-watchdog.json').write_text(json.dumps({
            'reason': reason, 'container': name, 'window': args.window, 'fan_restore': fan_restore,
            'at': dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2) + '\n')
        break
    time.sleep(5)
