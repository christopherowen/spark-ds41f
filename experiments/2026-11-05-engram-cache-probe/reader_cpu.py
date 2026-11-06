#!/usr/bin/env python3
"""Exercise the full native reader with CPU control words and real O_DIRECT I/O.

No CUDA, model arithmetic, serving, GPU wait measurement or mapped CUDA memory.
Run one source variant per process: the native module owns a shared worker.
"""
import argparse
from array import array
import ctypes as ct
import ctypes.util
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import struct
import time


def payload(row, scale=False):
    # The first eight bytes encode the full row ID, so different rows cannot
    # alias just because their low byte matches.
    prefix = bytes(x ^ (0x5A if scale else 0xA5) for x in struct.pack('<Q', row))
    return prefix if scale else prefix + bytes((row * 131 + j * 17) & 255 for j in range(8, 256))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bound', type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['XDG_CACHE_HOME'] = str(args.out / 'native-cache')
    report = {'scope': 'CPU-driven native reader with real direct reads; no CUDA/model/serving qualification',
              'bound': args.bound, 'started_unix': time.time(), 'cases': [],
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(args.source.glob('*.[ch]'))}}
    atomic = ct.CDLL(ctypes.util.find_library('atomic'))
    store = getattr(atomic, '__atomic_store_4')
    store.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_int]
    load = getattr(atomic, '__atomic_load_4')
    load.argtypes = [ct.c_void_p, ct.c_int]
    load.restype = ct.c_uint32
    table_rows = 4099
    paths = []
    for scale, offset in ((False, 4093), (True, 123)):
        path = args.out / ('scales.bin' if scale else 'weights.bin')
        with path.open('wb') as f:
            f.write(bytes(offset))
            for row in range(table_rows):
                f.write(payload(row, scale))
            f.flush()
            os.fsync(f.fileno())
        paths.append(path)
    try:
        spec = importlib.util.spec_from_file_location('native_loader', args.source / '_native.py')
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        native = loader.load()
        report['extension'] = str(loader._build())
        report['source_digest'] = loader._source_digest()
        report['abi'] = native.ABI_VERSION

        class Reader:
            def __init__(self, mask, rows=129, cache=64, timeout=1.0, scale=True, files=paths):
                self.mask, self.seq = mask, 0
                self.ids = (ct.c_int64 * (rows * 24))()
                self.weights = (ct.c_ubyte * (rows * 24 * 256))()
                self.scales = (ct.c_ubyte * (rows * 24 * 8))()
                self.ctrl = (ct.c_uint32 * native.CTRL_WORDS)()
                self.cap = native.reader(table_rows, 0, table_rows, mask, rows * 24, 16,
                                         ct.addressof(self.ids), ct.addressof(self.weights),
                                         ct.addressof(self.scales), ct.addressof(self.ctrl),
                                         timeout, 0.0, True, True, cache)
                native.add(self.cap, str(files[0]), 4093, False)
                if scale:
                    native.add(self.cap, str(files[1]), 123, True)

            def put(self, slot, value):
                store(ct.addressof(self.ctrl) + slot * 4, value, 3)  # release

            def get(self, slot):
                return load(ct.addressof(self.ctrl) + slot * 4, 2)  # acquire

            def wait(self, predicate, deadline=3.0):
                end = time.monotonic() + deadline
                while not predicate():
                    assert time.monotonic() < end, ('reader wait timed out', native.status(self.cap))
                    time.sleep(0.0001)

            def check(self, values):
                weights = bytes(self.weights)[:len(values) * 256]
                scales = bytes(self.scales)[:len(values) * 8]
                for i, row in enumerate(values):
                    owned = row >= 0 and (self.mask >> (i % 24)) & 1
                    assert weights[i * 256:(i + 1) * 256] == (payload(row) if owned else b'\x7f' * 256)
                    assert scales[i * 8:(i + 1) * 8] == (payload(row, True) if owned else b'\xff' * 8)
                return hashlib.sha256(weights + scales).hexdigest()

            def step(self, values, mode='late'):
                values = array('q', values)
                self.seq += 1
                ct.memmove(ct.addressof(self.ids), values.buffer_info()[0], len(values) * 8)
                ct.memset(ct.addressof(self.weights), 0x7F, ct.sizeof(self.weights))
                ct.memset(ct.addressof(self.scales), 0xFF, ct.sizeof(self.scales))
                self.put(native.CTRL_IDS_SEQ, self.seq)
                early = None if mode == 'late' else array('q', values)
                if mode == 'mixed':
                    for i in range(0, len(early), 7):
                        early[i] = -(1 << 63)
                native.submit(self.cap, self.seq, len(values), 1.0, early)
                self.wait(lambda: self.get(native.CTRL_ROWS_SEQ) == self.seq or native.status(self.cap)['failure'])
                assert native.status(self.cap)['failure'] is None
                digest = self.check(values)
                self.put(native.CTRL_CONSUMED_SEQ, self.seq)
                self.wait(lambda: native.status(self.cap)['queued'] == 0)
                return native.status(self.cap), digest

            def close(self):
                self.wait(lambda: native.status(self.cap)['queued'] == 0)
                self.cap = None
                gc.collect()

        masks = [(1 << 24) - 1, sum(1 << c for c in range(1, 24, 3)), sum(1 << c for c in range(1, 24, 4))]
        for mask in masks:
            for mode in ('late', 'early', 'mixed'):
                for tokens in (65, 96, 128, 129):
                    reader = Reader(mask)
                    try:
                        hot = list(range(24)) * tokens
                        hot[1] = -1
                        _, first_digest = reader.step(hot, mode)
                        warm, _ = reader.step(list(range(24)) * 16, mode)
                        repeat, digest = reader.step(hot, mode)
                        assert digest == first_digest
                        if tokens <= args.bound:
                            assert repeat['read_calls'] == warm['read_calls']
                            assert repeat['cache_hits'] > warm['cache_hits']
                        else:
                            assert repeat['read_calls'] > warm['read_calls']
                        cold = list(range(1000, 1024)) * tokens
                        miss1, _ = reader.step(cold, mode)
                        miss2, _ = reader.step(cold, mode)
                        assert miss2['read_calls'] > miss1['read_calls'] > repeat['read_calls']
                        assert miss2['cache_hits'] == miss1['cache_hits']
                        final, digest = reader.step(hot, mode)
                        assert digest == first_digest
                        if tokens <= args.bound:
                            assert final['read_calls'] == miss2['read_calls']
                        report['cases'].append({'tokens': tokens, 'mask': mask, 'mode': mode,
                                                'warm': warm, 'hot': repeat, 'miss1': miss1, 'miss2': miss2,
                                                'final': final, 'bytes_exact': True,
                                                'timeline': native.timeline(reader.cap)})
                    finally:
                        reader.close()

        # A queued early step may read into cached staging while the previous
        # mapped rows remain unconsumed. Preserve those rows until release.
        reader = Reader(masks[1])
        try:
            first = array('q', list(range(24)) * 16)
            second = array('q', list(range(1000, 1024)) * 96)
            ct.memset(ct.addressof(reader.weights), 0x7F, ct.sizeof(reader.weights))
            ct.memset(ct.addressof(reader.scales), 0xFF, ct.sizeof(reader.scales))
            ct.memmove(ct.addressof(reader.ids), first.buffer_info()[0], len(first) * 8)
            reader.put(native.CTRL_IDS_SEQ, 1)
            native.submit(reader.cap, 1, len(first), 1.0, first)
            reader.wait(lambda: reader.get(native.CTRL_ROWS_SEQ) == 1)
            first_digest = reader.check(first)
            ct.memmove(ct.addressof(reader.ids), second.buffer_info()[0], len(second) * 8)
            reader.put(native.CTRL_IDS_SEQ, 2)
            native.submit(reader.cap, 2, len(second), 1.0, second)
            reader.wait(lambda: native.status(reader.cap)['early_prefetched_steps'] == 1)
            assert reader.get(native.CTRL_ROWS_SEQ) == 1
            assert reader.check(first) == first_digest
            reader.put(native.CTRL_CONSUMED_SEQ, 1)
            reader.wait(lambda: reader.get(native.CTRL_ROWS_SEQ) == 2)
            reader.check(second)
            reader.put(native.CTRL_CONSUMED_SEQ, 2)
            reader.wait(lambda: native.status(reader.cap)['queued'] == 0)
            report['early_overlap'] = native.status(reader.cap)
        finally:
            reader.close()

        # Registration succeeds before truncation; an actual short read must fail
        # without publishing a new row sequence. A distinct reader has no hot cache.
        short = args.out / 'short-weights.bin'
        short.write_bytes(paths[0].read_bytes())
        reader = Reader(masks[0], cache=0, files=[short, paths[1]])
        try:
            with short.open('r+b') as f:
                f.truncate(4093 + 100 * 256)
            values = array('q', [table_rows - 1] * 24)
            ct.memmove(ct.addressof(reader.ids), values.buffer_info()[0], len(values) * 8)
            reader.put(native.CTRL_IDS_SEQ, 1)
            native.submit(reader.cap, 1, len(values), 1.0, None)
            reader.wait(lambda: bool(native.status(reader.cap)['failure']))
            assert reader.get(native.CTRL_ROWS_SEQ) == 0
            report['short_read'] = native.status(reader.cap)
        finally:
            reader.close()

        for kind in ('missing_scale', 'ids_timeout'):
            reader = Reader(masks[0], cache=0, timeout=0.05, scale=kind != 'missing_scale')
            try:
                if kind == 'missing_scale':
                    reader.put(native.CTRL_IDS_SEQ, 1)
                native.submit(reader.cap, 1, 24, 1.0, None)
                reader.wait(lambda: bool(native.status(reader.cap)['failure']))
                assert reader.get(native.CTRL_ROWS_SEQ) == 0
                report[kind] = native.status(reader.cap)
            finally:
                reader.close()
        report['passed'] = True
    finally:
        report['finished_unix'] = time.time()
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        (args.out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'cases': len(report['cases']), 'passed': report['passed'], 'receipt': str(args.out / 'receipt.json')}))


if __name__ == '__main__':
    main()
