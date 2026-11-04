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
