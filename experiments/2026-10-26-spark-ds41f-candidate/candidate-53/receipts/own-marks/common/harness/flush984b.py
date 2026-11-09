#!/usr/bin/env python3
"""#984 supplement v2 (candidate-53 own marks): the post-prefill flush's host time by component,
under the recipe's allocator (run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
garbage_collection_threshold:0.6 and MALLOC_ARENA_MAX=2, MALLOC_MMAP_THRESHOLD_=1048576 as
config/cluster.json sets them). Same synthetic state as flush984.py (startup-like gc.freeze(),
then per repeat a freed 1.5 GiB CUDA block and 64 x 1 MiB freed CPU tensors, with N live
unfrozen Python containers). Per repeat, with a decode-like GPU sleep in flight: the candidate's
reclaim_uma_memory(device, synchronize=False) whole, then (fresh state each) gc.collect() alone,
torch.accelerator.empty_cache() alone, trim_host_heap() alone. Also the cost the next prefill pays
to take the 1.5 GiB back after the flush (one torch.empty + synchronize), against reusing cached
blocks without a flush."""
import gc, json, statistics, sys, time
import torch
from vllm.utils import mem_utils
label, out = sys.argv[1], sys.argv[2]
dev = torch.device("cuda:0"); _ = torch.empty(1, device=dev)
fn = mem_utils.reclaim_uma_memory; trim = mem_utils.trim_host_heap
def prefill_state():
    a = torch.empty((1536 << 20,), dtype=torch.uint8, device=dev); a.fill_(1); del a
    h = [torch.ones((512, 512), dtype=torch.float32) for _ in range(64)]; del h
    torch.cuda.synchronize()
def timed(f):
    prefill_state(); torch.cuda._sleep(10_000_000)
    t0 = time.perf_counter(); f(); dt = (time.perf_counter() - t0) * 1e3
    torch.cuda.synchronize(); return dt
def realloc_ms():
    torch.cuda.synchronize(); t0 = time.perf_counter()
    a = torch.empty((1536 << 20,), dtype=torch.uint8, device=dev); a.fill_(1); torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1e3; del a; return dt
gc.collect(); gc.freeze()
res = {"label": label, "alloc_conf": __import__("os").environ.get("PYTORCH_CUDA_ALLOC_CONF"), "cases": []}
for n_live in (0, 100_000, 1_000_000):
    live = [{"i": i, "l": [i]} for i in range(n_live)]
    c = {"live_unfrozen_objects": n_live}
    for name, f in (("reclaim_uma_memory", lambda: fn(dev, synchronize=False)), ("gc_collect", gc.collect),
                    ("empty_cache", torch.accelerator.empty_cache), ("trim_host_heap", trim)):
        v = [timed(f) for _ in range(5)]
        c[name + "_ms"] = [round(x, 3) for x in v]; c[name + "_ms_median"] = round(statistics.median(v), 3)
    # next prefill's 1.5 GiB: after a flush (fresh mapping) vs from the cache (no flush)
    after_flush, cached = [], []
    for _ in range(5):
        prefill_state(); fn(dev, synchronize=False); torch.cuda.synchronize(); after_flush.append(realloc_ms())
        prefill_state(); cached.append(realloc_ms())
    c["next_prefill_1p5GiB_after_flush_ms_median"] = round(statistics.median(after_flush), 3)
    c["next_prefill_1p5GiB_cached_ms_median"] = round(statistics.median(cached), 3)
    res["cases"].append(c); del live; gc.collect()
open(out, "w").write(json.dumps(res, indent=1) + "\n"); print(json.dumps(res))
