"""Per-step cost of the turn learner's in-graph capture: CUDA graphs of N draft-step-sized
kernels with and without the capture node, replayed many times; the difference over N is
what the capture adds to a draft step. Also the capture alone in a graph."""
import json
import torch
from vllm.v1.worker.gpu.spec_decode.dspark import turn_learning as tl

N_SPEC, HIDDEN, SLOTS, CAP = 5, 5120, 16, 256
dev = "cuda"
ring_h = torch.zeros((SLOTS, CAP, N_SPEC, HIDDEN), dtype=torch.bfloat16, device=dev)
words = ring_h.view(torch.int64)
meta = torch.full((SLOTS, CAP, 2), -1, dtype=torch.int64, device=dev)
rdr = torch.zeros((SLOTS, CAP, N_SPEC), dtype=torch.int32, device=dev)
count = torch.zeros(SLOTS, dtype=torch.int64, device=dev)
kern = tl._kernel(tl.capture_kernel, N_SPEC, HIDDEN // 4)


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


for num_reqs in (1, 4, 16):
    rows = num_reqs * N_SPEC
    hidden = torch.randn((rows, HIDDEN), dtype=torch.bfloat16, device=dev)
    smap = torch.full((SLOTS * N_SPEC,), -1, dtype=torch.int32, device=dev)
    smap[:rows] = torch.arange(num_reqs, device=dev, dtype=torch.int32).repeat_interleave(N_SPEC)
    spos = torch.arange(SLOTS * N_SPEC, dtype=torch.int64, device=dev) + 1000
    drafts = torch.randint(0, 129280, (SLOTS, N_SPEC), dtype=torch.int64, device=dev)
    # A stand-in for the draft step's tail: a row copy of the same hidden rows (one kernel).
    sink = torch.empty_like(hidden)

    def tail():
        sink.copy_(hidden)

    def capture():
        kern(hidden.view(torch.int64), smap, spos, drafts, words, meta, rdr, count, 5)

    def tail_and_capture():
        tail()
        capture()

    only = timed(capture)
    base = timed(tail)
    both = timed(tail_and_capture)
    print(json.dumps({"num_reqs": num_reqs, "bytes_a_step": rows * HIDDEN * 2,
                      "capture_graph_us": round(only, 2), "tail_us": round(base, 2),
                      "tail_plus_capture_us": round(both, 2),
                      "capture_adds_us": round(both - base, 2)}), flush=True)
