"""Per-step cost of the shadow learner's capture: CUDA graphs of a draft-step tail stand-in
(the last position's exchange and selection: a fixed-length kernel) with the capture forked
onto a side stream before it and joined after it, with the capture inline before it, and
without it, replayed many times. Also the capture alone in a graph."""
import json

import torch

from vllm.v1.worker.gpu.spec_decode.dspark import turn_learning as tl
from vllm.v1.worker.gpu.spec_decode.dspark import turn_learning_shadow as sh

N_SPEC, PARTS, SLOTS, CAP = 5, 16, 16, 512
NUM_TILES, COLS = 505, 64
dev = "cuda"
ring_values = torch.zeros((SLOTS, CAP, N_SPEC, PARTS), dtype=torch.float32, device=dev)
ring_index = torch.zeros((SLOTS, CAP, N_SPEC, PARTS), dtype=torch.int32, device=dev)
ring_meta = torch.full((SLOTS, CAP, tl._META), -1, dtype=torch.int64, device=dev)
ring_drafts = torch.zeros((SLOTS, CAP, N_SPEC), dtype=torch.int32, device=dev)
ring_count = torch.zeros(SLOTS, dtype=torch.int64, device=dev)
part_values = torch.randn((N_SPEC, SLOTS * PARTS), device=dev)
part_indices = torch.randint(0, 32320, (N_SPEC, SLOTS * PARTS), dtype=torch.int32, device=dev)
tile_values = torch.randn(SLOTS * NUM_TILES, device=dev)
tile_indices = torch.randint(0, 32320, (SLOTS * NUM_TILES,), dtype=torch.int32, device=dev)
drafts = torch.randint(0, 129280, (SLOTS, N_SPEC), dtype=torch.int64, device=dev)
input_ids = torch.randint(0, 129280, (SLOTS * 8,), dtype=torch.int32, device=dev)
anchor_idx = torch.arange(SLOTS, dtype=torch.int64, device=dev) * 8
kern = tl._kernel(sh.shadow_capture_kernel, N_SPEC, PARTS)
side = torch.cuda.Stream()


def timed(fn, reps=2000):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        fn()
    for _ in range(50):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000.0 / reps


# The tail stand-in: a copy long enough to stand for the exchange (sized by the
# measured graph time of the tail alone, printed).
src = torch.randn(1 << 21, device=dev)
dst = torch.empty_like(src)

for num_reqs in (1, 4, 16):
    rows = num_reqs * N_SPEC
    smap = torch.arange(num_reqs, device=dev, dtype=torch.int32).repeat_interleave(N_SPEC)
    spos = torch.arange(rows, dtype=torch.int64, device=dev) + 1000

    def capture():
        kern(part_values, part_indices, tile_values, tile_indices, smap, spos, drafts,
             input_ids, anchor_idx, ring_values, ring_index, ring_meta, ring_drafts,
             ring_count, N_SPEC, 1, NUM_TILES, sh.GREEDY_BLOCK // COLS, 32320)

    def tail():
        dst.copy_(src)

    def inline():
        capture()
        tail()

    def forked():
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            capture()
        tail()
        torch.cuda.current_stream().wait_stream(side)

    only = timed(capture)
    base = timed(tail)
    inl = timed(inline)
    frk = timed(forked)
    print(json.dumps({
        "num_reqs": num_reqs,
        "capture_graph_us": round(only, 2),
        "tail_us": round(base, 2),
        "inline_adds_us": round(inl - base, 2),
        "forked_adds_us": round(frk - base, 2),
    }), flush=True)
