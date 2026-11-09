#!/usr/bin/env python3
"""#984 supplement (candidate-53 own marks): host time of the one-shot post-prefill flush the
candidate runs on the first FULL decode step after an eager prefill
(vllm.utils.mem_utils.reclaim_uma_memory(device, synchronize=False): gc.collect() +
empty_cache() + trim_host_heap()). The base runs nothing there. After a startup-like
gc.freeze(), a synthetic prefill leaves 1.5 GiB of freed CUDA caching-allocator blocks and
64 x 1 MiB of freed CPU tensors, with N live unfrozen Python container objects (0, 1e5, 1e6,
standing in for per-request host state). Each case: 5 repeats, the flush timed on the host
while a decode-like GPU sleep (~5 ms) is in flight, as in sample_tokens."""
import gc, json, statistics, sys, time
import torch
from vllm.utils import mem_utils
label, out = sys.argv[1], sys.argv[2]
dev = torch.device("cuda:0")
_ = torch.empty(1, device=dev)
fn = getattr(mem_utils, "reclaim_uma_memory", None)
res = {"label": label, "has_reclaim_uma_memory": fn is not None, "cases": []}
gc.collect(); gc.freeze()
for n_live in (0, 100_000, 1_000_000):
    live = [{"i": i, "l": [i]} for i in range(n_live)]
    times, slack_before, slack_after = [], [], []
    for _ in range(5):
        a = torch.empty((1536 << 20,), dtype=torch.uint8, device=dev); del a
        h = [torch.ones((512, 512), dtype=torch.float32) for _ in range(64)]; del h
        torch.cuda.synchronize()
        slack_before.append((torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) >> 20)
        torch.cuda._sleep(10_000_000)
        t0 = time.perf_counter()
        if fn is not None:
            fn(dev, synchronize=False)
        times.append((time.perf_counter() - t0) * 1e3)
        torch.cuda.synchronize()
        slack_after.append((torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) >> 20)
    res["cases"].append({"live_unfrozen_objects": n_live, "flush_host_ms": [round(t, 3) for t in times],
                         "flush_host_ms_median": round(statistics.median(times), 3),
                         "slack_mib_before": slack_before, "slack_mib_after": slack_after})
    del live; gc.collect()
open(out, "w").write(json.dumps(res, indent=1) + "\n"); print(json.dumps(res))
