#!/usr/bin/env python3
"""Save every rank's identity, bounded engine log and memory after one boot's work.

Lab extra: collect_runtime.py HEAD_URL --config ARM --run RUN --label LABEL
Only selected Docker fields are retained; environment values are never written out.
Logs stay under ignored results/private, with explicit truncation metadata. A repeat
collection cannot overwrite a boot's evidence. This does not send inference requests.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spark_ds41f import config as spark

def expected_image(args, cluster):
    """Resolve an explicit immutable identity; an image tag is never authority."""
    value = args.expected_image_id
    if args.image_receipt:
        path = (ROOT / args.image_receipt).resolve()
        if not path.is_relative_to((ROOT / "results/private").resolve()):
            raise ValueError("image receipt must be retained under this checkout's private results")
        receipt = json.loads(path.read_text())
        gate, conditions = receipt.get("gate", {}), receipt.get("conditions", {})
        if (not receipt.get("ended") or receipt.get("passed") is not True
                or receipt.get("error") or gate.get("passed") is not True
                or gate.get("failed") or gate.get("errors")
                or conditions.get("halted") or conditions.get("fan_stop_failures")):
            raise ValueError("image receipt lacks a clean completed gate")
        steps = receipt.get("steps", [])
        if (not steps or any(x.get("exit") != 0 for x in steps)
                or not {"build-image-apply.log", "release-tests.log"}.issubset({x.get("log") for x in steps})):
            raise ValueError("image build/test execution is incomplete")
        labels = receipt.get("source", {}).get("expected_labels", {})
        expected = cluster["container"].get("expected_labels", {})
        if not expected or any(labels.get(k) != v for k, v in expected.items()):
            raise ValueError("image receipt source labels differ from the requested profile")
        value = receipt.get("image")
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("an explicit sha256 immutable image ID is required")
    return value


def collect_on_node(container, tail_lines, max_bytes):
    import base64
    import datetime
    import hashlib
    import json
    import os
    import pathlib
    import selectors
    import subprocess
    import time

    def docker(*args):
        p = subprocess.run(['docker', *args], capture_output=True, timeout=15)
        if p.returncode:
            raise RuntimeError('Docker command failed: ' + args[0])
        return json.loads(p.stdout)[0]

    def selected_labels(labels):
        return {k: v for k, v in (labels or {}).items()
                if (k.startswith('local.spark-ds41f.') and
                    k.rsplit('.', 1)[-1] in ('tree', 'revision', 'patch-head'))
                or k in ('org.opencontainers.image.revision',
                         'org.opencontainers.image.base.digest')}

    record = {'collected_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        before = docker('inspect', container)
        cid = before['Id']
        image = docker('image', 'inspect', before['Image'])
        record['container'] = {
            'id': cid, 'name': before['Name'], 'image_id': before['Image'],
            'image_ref': before['Config']['Image'],
            'profile': (before['Config'].get('Labels') or {}).get('spark.cluster-config'),
            'status': before['State']['Status'], 'started_at': before['State']['StartedAt'],
            'restart_count': before['RestartCount'],
            'labels': selected_labels(before['Config'].get('Labels')),
            'mount_destinations': sorted(m['Destination'] for m in before['Mounts']),
        }
        record['image'] = {
            'id': image['Id'], 'labels': selected_labels(image['Config'].get('Labels')),
            'layers': image['RootFS']['Layers'],
            'config_sha256': hashlib.sha256(json.dumps(
                image['Config'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        }
        record['meminfo'] = pathlib.Path('/proc/meminfo').read_text()
        # Request one extra line to distinguish a full log from a bounded tail.
        p = subprocess.Popen(['docker', 'logs', '--timestamps', '--tail',
                              str(tail_lines + 1), cid], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        retained, total_bytes, newlines = bytearray(), 0, 0
        deadline = time.monotonic() + 30
        selector = selectors.DefaultSelector()
        selector.register(p.stdout, selectors.EVENT_READ)
        try:
            while selector.get_map():
                if time.monotonic() > deadline:
                    raise TimeoutError('Docker log collection exceeded 30 seconds')
                for key, _ in selector.select(timeout=0.2):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    total_bytes += len(chunk)
                    newlines += chunk.count(b'\n')
                    retained.extend(chunk)
                    if len(retained) > max_bytes:
                        del retained[:-max_bytes]
            code = p.wait(timeout=5)
        finally:
            selector.close()
            if p.poll() is None:
                p.kill()
                p.wait(timeout=5)
            p.stdout.close()
        byte_truncated = total_bytes > len(retained)
        lines = bytes(retained).splitlines(keepends=True)
        log = b''.join(lines[-tail_lines:])
        record['log'] = {
            'exit_code': code, 'requested_tail_lines': tail_lines,
            'max_bytes': max_bytes, 'received_bytes': total_bytes,
            'received_newlines': newlines, 'saved_bytes': len(log),
            'saved_lines': len(log.splitlines()),
            'truncated': byte_truncated or newlines > tail_lines,
            'byte_limit_applied': byte_truncated,
            'line_limit_reached': newlines > tail_lines,
            'sha256': hashlib.sha256(log).hexdigest(),
        }
        record['log_base64'] = base64.b64encode(log).decode()
        after = docker('inspect', container)
        record['same_container_after_collection'] = (
            cid == after['Id'] and before['State']['StartedAt'] == after['State']['StartedAt']
            and before['RestartCount'] == after['RestartCount'])
        record['status_after_collection'] = after['State']['Status']
        if code or not log or not record['same_container_after_collection']:
            record['error'] = 'log missing/failed or container changed during collection'
    except Exception as error:
        # Do not serialize command output or the full Docker object on an error.
        record['error'] = type(error).__name__ + ': ' + str(error)[:300]
    return record


def main(args):
    for value in (args.run, args.label):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
            raise SystemExit('run and label must be simple names')
    cluster, nodes, _ = spark.configuration(argparse.Namespace(cluster_config=args.config))
    image_id = expected_image(args, cluster)
    out = ROOT / 'results/private/determinism' / args.run / ('runtime-' + args.label)
    out.mkdir(parents=True, exist_ok=False)
    remote = (inspect.getsource(collect_on_node) + '\nimport json,sys\n'
              'print(json.dumps(collect_on_node(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))))\n')
    expected = cluster['container'].get('expected_labels', {})

    def one(node):
        name = node['name']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            return {'node': name, 'error': 'invalid node name'}
        directory = out / name
        directory.mkdir()
        try:
            result = spark.run_ssh(nodes, node, 'timeout', '90', 'python3', '-c', remote,
                                   cluster['container']['name'], str(args.tail_lines),
                                   str(args.max_log_bytes))
            if result.returncode:
                raise RuntimeError(f'remote collector exited {result.returncode}')
            record = json.loads(result.stdout)
            record['node'] = name
            raw = base64.b64decode(record.pop('log_base64', ''), validate=True)
            (directory / 'engine.log').write_bytes(raw)
            issues = [record['error']] if record.get('error') else []
            for field in ('container', 'image', 'meminfo', 'log'):
                if not record.get(field):
                    issues.append('missing ' + field)
            if not raw or hashlib.sha256(raw).hexdigest() != record.get('log', {}).get('sha256'):
                issues.append('missing log or log hash mismatch')
            container = record.get('container', {})
            if (container.get('image_id') != image_id
                    or record.get('image', {}).get('id') != image_id):
                issues.append('image ID differs from the explicitly qualified image')
            if container.get('profile') != args.config:
                issues.append('live profile differs from requested arm')
            if container.get('status') != 'running':
                issues.append('container is not running')
            if record.get('status_after_collection') != 'running':
                issues.append('container stopped during collection')
            for key, value in expected.items():
                if record.get('image', {}).get('labels', {}).get(key) != value:
                    issues.append('image label mismatch: ' + key)
            record['issues'] = issues
        except Exception as error:
            record = {'node': name, 'issues': [type(error).__name__ + ': ' + str(error)[:300]]}
        (directory / 'runtime.json').write_text(json.dumps(record, indent=2) + '\n')
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes['nodes'])) as pool:
        records = list(pool.map(one, nodes['nodes']))
    fingerprints = {(tuple(r.get('image', {}).get('layers', [])),
                     r.get('image', {}).get('config_sha256')) for r in records}
    issues = {r['node']: r.get('issues', [r.get('error', 'missing evidence')])
              for r in records if r.get('issues') or r.get('error')}
    if len(fingerprints) != 1:
        issues['image_agreement'] = ['image layers or Config hash differ between ranks']
    summary = {'summary': 'runtime evidence', 'profile': args.config, 'label': args.label,
               'expected_image_id': image_id,
               'image_receipt': args.image_receipt,
               'image_receipt_sha256': hashlib.sha256((ROOT / args.image_receipt).read_bytes()).hexdigest() if args.image_receipt else None,
               'profile_sha256': hashlib.sha256((ROOT / args.config).read_bytes()).hexdigest(),
               'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'expected_nodes': [n['name'] for n in nodes['nodes']],
               'collected_nodes': [r['node'] for r in records],
               'output': str(out.relative_to(ROOT)), 'issues': issues,
               'passed': not issues and len(records) == len(nodes['nodes'])}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('head_url', help='lab interface; target ranks come from --config')
    parser.add_argument('--config', required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument('--expected-image-id', help='explicit qualified sha256 image ID')
    identity.add_argument('--image-receipt', help='repository-relative immutable passing image-gate receipt')
    parser.add_argument('--run', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--tail-lines', type=int, default=20000)
    parser.add_argument('--max-log-bytes', type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    if not 1 <= args.tail_lines <= 100000 or not 1024 <= args.max_log_bytes <= 32 * 1024 * 1024:
        parser.error('log limits must be 1..100000 lines and 1 KiB..32 MiB')
    raise SystemExit(main(args))
