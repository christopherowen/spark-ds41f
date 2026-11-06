#!/usr/bin/env python3
"""Unreleased component fixture: actual mapped buffers, direct reads and GPU gather.

This selects an exact native source snapshot inside a fixed existing environment.
It is not image qualification or a rerun of a released serving benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['SPARK_DS41F_ENGRAM_TIMELINE'] = '1'
    from vllm.models.deepseek_v4_1.engram_rows import EngramDiskRows, _native
    _native._SOURCE = args.native_source / '_reader.c'
    assert not _native.load.cache_info().currsize
    report = {'scope': __doc__, 'started_unix': time.time(), 'cases': [],
              'native_sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(args.native_source.glob('*.[ch]'))},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'conditions': [], 'environment': {k: os.getenv(k) for k in
                  ('SPARK_DS41F_ENGRAM_TIMELINE', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'CC')}}

    def conditions():
        result = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True)
        report['conditions'].append({'at_unix': time.time(), 'exit': result.returncode,
                                     'stdout': result.stdout, 'stderr': result.stderr,
                                     'meminfo': Path('/proc/meminfo').read_text()})

    def idle(reader):
        deadline = time.monotonic() + 60
        while reader.stats()['queued']:
            reader.raise_if_failed()
            assert time.monotonic() < deadline, reader.stats()
            time.sleep(0.0001)
        reader.raise_if_failed()

    columns, stride, warm_rows = 24, 16384, 1024
    table_rows = columns * stride
    tables = []
    try:
        conditions()
        # Encode the complete row index into finite E4M3 bytes. No low-byte alias.
        row = np.arange(table_rows, dtype=np.uint32)
        for layer in (0, 1):
            codes = np.empty((table_rows, 256), dtype=np.uint8)
            for byte in range(256):
                codes[:, byte] = ((row >> ((byte % 4) * 5)) + byte * 7 + layer * 11) % 120
            weights = args.out / ('weights-' + str(layer) + '.bin')
            scales = args.out / ('scales-' + str(layer) + '.bin')
            for path, offset, data in ((weights, 4093, codes),
                                       (scales, 123, np.full((table_rows, 8), 127, dtype=np.uint8))):
                with path.open('wb') as handle:
                    handle.write(bytes(offset)); handle.write(data.tobytes())
                    handle.flush(); os.fsync(handle.fileno())
            tables.append((torch.from_numpy(codes), weights, scales))
        report['tables'] = [{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (w, s)}
                            for _, w, s in tables]
        cases = [(3, tokens, hot, False) for tokens in (6, 48) for hot in (0, 75, 100)]
        cases += [(4, 96, hot, False) for hot in (0, 75, 100)]
        cases += [(3, tokens, 75, early) for tokens in (129, 6144) for early in (False, True)]
        if args.reverse:
            cases.reverse()
        for tp, tokens, hot, early in cases:
            owned = list(range(1, columns, tp))
            mask = sum(1 << c for c in owned)
            readers = []
            case = {'tp': tp, 'tokens': tokens, 'requested_hot_pct': hot, 'early': early,
                    'warm_rows_per_column': warm_rows, 'layers': 2, 'samples': [], 'timeline': []}
            report['cases'].append(case)
            try:
                for _, weight, scale in tables:
                    r = EngramDiskRows(device='cuda', table_rows=table_rows, shard_start=0,
                                       shard_end=table_rows, max_lookups=max(tokens, 64) * columns,
                                       columns=mask, cache_rows=65536, timeline=True)
                    r.add_source(str(weight), 4093, scale=False)
                    r.add_source(str(scale), 123, scale=True)
                    readers.append(r)
                    for start in range(0, warm_rows, 64):
                        ids = np.arange(start, start + 64)[:, None] + np.arange(columns)[None, :] * stride
                        r.prefetch(ids.astype(np.int64))
                        idle(r)
                rng = np.random.default_rng(87403 + tp * 10000 + tokens * 10 + hot)
                stream = torch.cuda.current_stream()
                live = torch.tensor([tokens], dtype=torch.int32, device='cuda')
                output = [torch.empty((tokens, len(owned) * 256), dtype=torch.bfloat16, device='cuda')
                          for _ in readers]
                rounds = 32 if tokens <= 128 else 12
                for iteration in range(rounds + 1):
                    hot_mask = rng.integers(0, 100, (tokens, columns)) < hot
                    ids = np.where(hot_mask, rng.integers(0, warm_rows, (tokens, columns)),
                                   rng.integers(warm_rows, stride, (tokens, columns)))
                    ids = np.ascontiguousarray(ids + np.arange(columns)[None, :] * stride, dtype=np.int64)
                    ids.reshape(-1)[::97] = -1
                    ids_gpu = torch.from_numpy(ids).cuda()
                    torch.cuda.synchronize()
                    before = [r.stats() for r in readers]
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    wall = time.perf_counter_ns()
                    begin.record()
                    for r in readers:
                        r.stage(stream, ids_gpu, ids.size, ids if early else None)
                    for r, dest in zip(readers, output):
                        r.gather(dest, ids_gpu, live)
                    end.record(); end.synchronize()
                    wall = time.perf_counter_ns() - wall
                    for r in readers:
                        idle(r)
                    after = [r.stats() for r in readers]
                    for (codes, _, _), actual in zip(tables, output):
                        selected = torch.from_numpy(ids[:, owned].copy())
                        expected = codes[selected.clamp_min(0)].view(torch.float8_e4m3fn).to(torch.bfloat16)
                        expected[selected < 0] = 0
                        assert torch.equal(actual.cpu().view(torch.int16), expected.flatten(1).view(torch.int16))
                    case['samples'].append({'iteration': iteration, 'warmup': iteration < 2,
                                            'drain': iteration == rounds,
                                            'ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                                            'gpu_interval_ms': begin.elapsed_time(end), 'wall_ns': wall,
                                            'before': before, 'after': after})
                    for index, r in enumerate(readers):
                        case['timeline'].extend({'layer': index, **entry} for entry in r.timeline())
                # After timing, compare the same rows in singleton batches on the
                # actual gather path. This does not alter any measured cache state.
                expected_batch = [value.cpu().clone() for value in output]
                gate_indices = sorted({0, 1, tokens // 2, tokens - 1})
                single_live = torch.tensor([1], dtype=torch.int32, device='cuda')
                single_output = [torch.empty((1, len(owned) * 256), dtype=torch.bfloat16, device='cuda')
                                 for _ in readers]
                for index in gate_indices:
                    single_ids = ids_gpu[index:index + 1].contiguous()
                    for r in readers:
                        r.stage(stream, single_ids, columns)
                    for r, value in zip(readers, single_output):
                        r.gather(value, single_ids, single_live)
                    torch.cuda.synchronize()
                    for r, value, expected in zip(readers, single_output, expected_batch):
                        idle(r)
                        assert torch.equal(value.cpu().view(torch.int16), expected[index:index + 1].view(torch.int16))
                case['batch_invariance_rows'] = gate_indices
                print(tp, tokens, hot, early, 'passed', flush=True)
            finally:
                torch.cuda.synchronize()
                for r in readers:
                    r.close()
            conditions()
        report['passed'] = True
    finally:
        report['finished_unix'] = time.time()
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        (args.out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
