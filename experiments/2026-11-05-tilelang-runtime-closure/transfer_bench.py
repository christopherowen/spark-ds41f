"""Balanced complete nonblocking transfer timings, including host work and wait."""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import math

import torch
from paired import classify
from vllm.v1.outputs import LogprobsTensors

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('previous_outputs', here / 'control/outputs.py')
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
torch.manual_seed(20261105)
rng = random.Random(20261105)
results = []
for dtype in (torch.int32, torch.int64):
    for count in (8, 96, 1025):
        for stride in (2, 33):
            ids = torch.arange(count * stride + 7, device='cuda', dtype=dtype)[7::stride, None]
            assert ids.shape == (count, 1)
            gpu = LogprobsTensors(ids, torch.randn((count, 1), device='cuda'),
                                  torch.arange(count, device='cuda', dtype=torch.int64))
            launch = dict(reference=lambda: previous.LogprobsTensors.to_cpu_nonblocking(gpu),
                          candidate=gpu.to_cpu_nonblocking)
            for _ in range(10):
                for fn in launch.values():
                    out = fn()
                    torch.cuda.synchronize()
                    assert torch.equal(out.logprob_token_ids, ids.cpu())
            pairs, logs = [], []
            for iteration in range(120):
                order = list(launch)
                rng.shuffle(order)
                order += order[::-1]
                samples = {role: [] for role in launch}
                for role in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    out = launch[role]()
                    torch.cuda.current_stream().synchronize()
                    samples[role].append((time.perf_counter_ns() - start) / 1000)
                means = {role: sum(v) / len(v) for role, v in samples.items()}
                logs.append(math.log(means['candidate'] / means['reference']))
                pairs.append(dict(order=order, times_us=samples))
            row = dict(dtype=str(dtype), rows=count, stride=stride,
                       gate=classify(logs), log_ratios=logs, pairs=pairs,
                       logical_bytes=ids.numel() * ids.element_size(),
                       physical_bytes=((count - 1) * stride + 1) * ids.element_size())
            results.append(row)
            print(dtype, count, stride, row['gate'], flush=True)
            (args.out / 'result.json').write_text(json.dumps(dict(cases=results,
                scope='Complete host call and originating-stream wait; both roles include all three result fields. No released build is rerun.'), indent=2) + '\n')
