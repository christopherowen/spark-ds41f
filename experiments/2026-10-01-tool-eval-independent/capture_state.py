"""Read-only serving receipts; run from this experiment directory."""

import concurrent.futures
import datetime
import json
import pathlib
import shlex
import subprocess
import urllib.request


REMOTE = r'''
import datetime, json, pathlib, socket, subprocess
repo = pathlib.Path('/home/swank/projects/spark3-vllm-ds41f')
def run(args):
    return subprocess.check_output(args, text=True).strip()
d = json.loads(run(['docker', 'inspect', 'dsv41-karmic-kraken']))[0]
environment = dict(e.split('=', 1) for e in d['Config']['Env'] if '=' in e)
print(json.dumps({
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'hostname': socket.gethostname(),
    'deployment_commit': run(['git', '-C', str(repo), 'rev-parse', 'HEAD']),
    'git_status': run(['git', '-C', str(repo), 'status', '--porcelain']),
    'container_id': d['Id'],
    'image_id': d['Image'],
    'image_tag': d['Config']['Image'],
    'started_at': d['State']['StartedAt'],
    'state': d['State']['Status'],
    'labels': d['Config'].get('Labels', {}),
    'selected_environment': {k:v for k,v in environment.items()
        if k.startswith(('B12X_', 'VLLM_DS41_', 'SPARK3_DSPARK_'))},
    'config': json.loads((repo/'config/cluster.json').read_text()),
    'upstreams': json.loads((repo/'upstreams.lock.json').read_text())
}))
'''


def capture(node):
    output = subprocess.check_output(
        ['ssh', 'swank@' + node, 'python3 -c ' + shlex.quote(REMOTE)],
        text=True,
        timeout=40,
    )
    receipt = json.loads(output)
    path = pathlib.Path('runs') / ('after-' + node + '.json')
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    before = json.loads(path.with_name('before-' + node + '.json').read_text())
    checked = ['deployment_commit', 'git_status', 'container_id', 'image_id',
               'image_tag', 'started_at', 'state', 'selected_environment', 'config']
    mismatches = [key for key in checked if before[key] != receipt[key]]
    return {'node': node, 'checked': checked, 'mismatches': mismatches}


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        comparisons = list(pool.map(capture, ['dgx1', 'dgx2', 'dgx3']))
    with urllib.request.urlopen('http://10.0.1.71:8000/metrics', timeout=20) as response:
        metrics = response.read().decode()
    pathlib.Path('runs/metrics-after.txt').write_text(metrics)
    counters = [line for line in metrics.splitlines()
                if line.startswith(('vllm:num_requests_running{',
                                    'vllm:num_requests_waiting{'))]
    assert len(counters) >= 2, 'Missing request counters'
    idle = all(float(line.rsplit(' ', 1)[-1]) == 0 for line in counters)
    report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'comparisons': comparisons, 'request_counters': counters, 'idle': idle}
    pathlib.Path('runs/after-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert idle and not any(row['mismatches'] for row in comparisons)
