#!/usr/bin/env python3
"""Finite numerical replay with far more trace records than ring slots."""
import argparse
import hashlib
import json
from pathlib import Path
import time
parser = argparse.ArgumentParser()
parser.add_argument('operation', choices=['gemm'])
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--ring-bytes', type=int, default=4096)
parser.add_argument('--delay-us', type=int, default=0)
parser.add_argument('--fail-after', type=int, default=0)
parser.add_argument('--repeats', type=int, default=100)
parser.add_argument('--graph', action='store_true')
parser.add_argument('--patched', action='store_true')
parser.add_argument('--streams', type=int, default=1)
args = parser.parse_args()
import torch
from stream.collector import enable
from stream.export import export
collector = enable(args.out / 'stream', ring_bytes=args.ring_bytes, chunk_bytes=16384,
                   writer_delay_us=args.delay_us, fail_after_records=args.fail_after, patched=args.patched)
from vllm.models.deepseek_v4_1.tilelang.gemm import bf16_gemm
gemm = bf16_gemm(128, 512, out_dtype='float32')
x = torch.full((6, 512), .5, device='cuda', dtype=torch.bfloat16)
w = torch.ones((128, 512), device='cuda', dtype=torch.bfloat16)
outputs = [torch.empty((6, 128), device='cuda', dtype=torch.float32) for _ in range(args.streams)]
streams = [torch.cuda.Stream() for _ in outputs]
torch.cuda.synchronize()
start = time.monotonic()
if args.graph:
    gemm(x, w, outputs[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm(x, w, outputs[0])
    for _ in range(args.repeats):
        graph.replay()
else:
    for _ in range(args.repeats):
        for stream, output in zip(streams, outputs):
            with torch.cuda.stream(stream):
                gemm(x, w, output)
torch.cuda.synchronize()
for output in outputs:
    assert bool((output == 256).all())
collector.close()
counts = export(args.out / 'stream', args.out / 'trace.json')
launches = args.repeats * (1 if args.graph else args.streams) + int(args.graph)
assert counts['kind_0'] == counts['kind_3'] == 16 * launches
assert counts['kind_1'] == counts['kind_2'] == 3 * launches
summary = json.loads((args.out / 'stream/summary.json').read_text())
assert summary['written_records'] > collector.capacity * 10
result = {'operation':args.operation, 'seconds':time.monotonic()-start,
          'sha256':hashlib.sha256(outputs[0].cpu().numpy().tobytes()).hexdigest(),
          'graph':args.graph, 'streams':args.streams, 'repeats':args.repeats,
          'summary':summary, 'validation':counts}
(args.out / 'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result), flush=True)
