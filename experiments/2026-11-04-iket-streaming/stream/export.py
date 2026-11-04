#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stream binary chunks into Chrome Trace JSON readable by Perfetto."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct

HEADER = struct.Struct('<8sIIQQ4Q')
RECORD = struct.Struct('<QQQQQIIIIII')


def records(root):
    expected = 0
    for path in sorted(Path(root).glob('events-*.bin')):
        with path.open('rb') as handle:
            magic, version, size, chunk, first, *_ = HEADER.unpack(handle.read(HEADER.size))
            if (magic, version, size, first) != (b'SPKIKET1', 1, 64, expected):
                raise ValueError(f'invalid or discontinuous chunk {path}')
            count, checksum = 0, 14695981039346656037
            while raw := handle.read(RECORD.size):
                if len(raw) != RECORD.size:
                    raise ValueError(f'partial record in {path}')
                row = RECORD.unpack(raw)
                if row[0] != expected:
                    raise ValueError(f'sequence gap at {path}: {row[0]} != {expected}')
                for byte in raw:
                    checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64) - 1)
                expected += 1
                count += 1
                yield row
        receipt = json.loads(path.with_suffix('.bin.json').read_text())
        if receipt != {'chunk': chunk, 'records': count, 'fnv1a64': f'{checksum:016x}'}:
            raise ValueError(f'checksum/count mismatch in {path}')


def export(root, output, grid=None):
    root = Path(root)
    summary = json.loads((root / 'summary.json').read_text())
    if not summary['complete']:
        raise ValueError('capture is incomplete; retain native chunks for diagnosis')
    names, kernels = {}, {}
    for path in (root / 'sources').glob('*.json'):
        manifest = json.loads(path.read_text())
        identity = manifest['kernel_id']
        kernels[identity] = manifest['kernel']
        for site in manifest['sites']:
            if 'range_id' in site:
                names[identity, site['range_id']] = site['range']
    counts = Counter()
    with Path(output).open('x') as handle:
        handle.write('{"traceEvents":[')
        comma = ''
        for seq, timestamp, launch, kernel, cta, warp, sm, event, payload, kind, _ in records(root):
            counts['records'] += 1
            counts[f'kind_{kind}'] += 1
            if grid is not None and launch != grid:
                continue
            name = kernels[kernel] if kind == 0 else names.get((kernel, event), 'phase')
            item = {'ph': 'B' if kind in (0, 1) else 'E', 'pid': launch,
                    'tid': cta * 32 + warp, 'ts': timestamp / 1000,
                    'args': {'sequence': seq, 'sm': sm, 'payload': payload}}
            if kind in (0, 1):
                item['name'] = name
            handle.write(comma + json.dumps(item, separators=(',', ':')))
            comma = ','
        handle.write(']}\n')
    if counts['records'] != summary['written_records']:
        raise ValueError('summary record count mismatch')
    if counts['kind_0'] != counts['kind_3'] or counts['kind_1'] != counts['kind_2']:
        raise ValueError('unbalanced lifetime/phase totals')
    return dict(counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('output')
    parser.add_argument('--grid', type=int)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.output, args.grid)))
