"""Run the native benchmark while preserving full replies and metric snapshots."""
import json
from pathlib import Path
import runpy
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
m = runpy.run_path(str(ROOT / 'bin/spark3'))
args = m['parser']().parse_args(sys.argv[1:])
cluster, nodes, _ = m['configuration'](args)
errors, warnings = m['split_findings'](m['live_doctor'](cluster, nodes))
expected = {n['name'] + ': dgx-fan-control.service is enabled/inactive, expected enabled/active; run sudo systemctl enable --now dgx-fan-control.service' for n in nodes['nodes']}
assert set(errors) == expected and not warnings, (errors, warnings)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
(out / 'preflight.json').write_text(json.dumps({'errors': errors, 'warnings': warnings}, indent=2))
original = m['Bench'].run


def observed(self, label, payloads, concurrency=None, keep_content=False, retries=3):
    def snapshot():
        with urllib.request.urlopen(self.base + '/metrics', timeout=10) as response:
            return response.read().decode()
    before = snapshot()
    result = original(self, label, payloads, concurrency, True, retries)
    after = snapshot()
    with (out / 'waves.jsonl').open('a') as f:
        f.write(json.dumps({'time': time.time(), 'label': label, 'payloads': payloads,
                            'result': result, 'before': before, 'after': after}) + '\n')
    return result


m['Bench'].run = observed
raise SystemExit(args.func(args))
