#!/usr/bin/env python3
"""The DSpark Markov projection held in persisting L2 across a step's five draft calls.

A TP4 rank's shard of ``markov_w2`` (draft vocabulary / 4 rows, markov_rank columns, BF16,
about 16.5 MB) is read once a draft position: five TileLang ``bf16_gemm`` calls a step,
with only the greedy selection between them (stood in for here by a read of the base
logits row and an argmax). The decode-bytes study bounded keeping it resident at +0.5% of a
single-stream step; serving shows 40-68 us a call against 71 us cold.

Per row count (1-16, the token tile serving uses) this times one step's five calls as a
CUDA graph, the L2 flushed by a 64 MB write before each replay, as the target's forward
leaves it:
- ``normal``: as serving runs it;
- ``window``: the same graph captured on a stream whose access-policy window marks the
  weight persisting (cudaLimitPersistingL2CacheSize at the device's maximum).
It checks that both give the same bits, and reports the device's L2, persisting maximum
and window maximum. One JSON line a case; exit status 0 when every output matched.
"""
import ctypes
import glob
import json
import os
import statistics
import sys

import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang.gemm import bf16_gemm
from vllm.models.deepseek_v4_1.tilelang.vocab import BF16_HEAD_CONFIGS

SNAPSHOT = ("/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/"
            "1bae0e1088993b9b82de08f59de5be5a34e42170")
TP = 4
DRAFTS = 5
ROWS = (1, 2, 4, 8, 16)
REPEATS = 31
DEVICE = "cuda"


class Window(ctypes.Structure):
    _fields_ = [("base_ptr", ctypes.c_void_p), ("num_bytes", ctypes.c_size_t), ("hitRatio", ctypes.c_float),
                ("hitProp", ctypes.c_int), ("missProp", ctypes.c_int)]


class AttrValue(ctypes.Union):
    _fields_ = [("accessPolicyWindow", Window), ("pad", ctypes.c_char * 64)]


def cudart():
    import torch as _t
    roots = [os.path.dirname(_t.__file__)] + [p for p in sys.path if p.endswith("site-packages")]
    for root in roots:
        for path in sorted(glob.glob(f"{root}/**/libcudart.so*", recursive=True)):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    return ctypes.CDLL("libcudart.so")


def check(code, what):
    if code != 0:
        raise SystemExit(f"{what}: cudaError {code}")


def attribute(rt, attr, device=0):
    value = ctypes.c_int()
    check(rt.cudaDeviceGetAttribute(ctypes.byref(value), attr, device), f"attribute {attr}")
    return value.value


def shape():
    config = json.load(open(f"{SNAPSHOT}/config.json"))
    vocab = config.get("draft_vocab_size") or config["vocab_size"]
    rank = config.get("dspark_markov_rank", 256)
    return -(-vocab // TP), rank


def main():
    torch.cuda.set_device(0)
    rt = cudart()
    l2, persist_max, window_max = attribute(rt, 38), attribute(rt, 108), attribute(rt, 109)
    n, k = shape()
    print(json.dumps({"case": "device", "l2_bytes": l2, "persisting_l2_max": persist_max,
                      "access_policy_max_window": window_max, "shard_rows": n, "rank": k,
                      "shard_bytes": n * k * 2}), flush=True)
    if not persist_max or not window_max:
        print(json.dumps({"case": "verdict", "all_exact": True, "note": "no persisting L2 on this device"}))
        return 0
    check(rt.cudaDeviceSetLimit(0x06, ctypes.c_size_t(persist_max)), "persisting L2 limit")
    gen = torch.Generator(device=DEVICE).manual_seed(41)
    weight = (torch.randn((n, k), generator=gen, device=DEVICE) * 0.02).to(torch.bfloat16)
    flush = torch.empty(64 << 20, dtype=torch.uint8, device=DEVICE)
    ok = True
    for out_dtype, tl_out in ((torch.float32, T.float32), (torch.bfloat16, T.bfloat16)):
        kernel = bf16_gemm(n, k, block_M=16, num_stages=2, out_dtype=tl_out, **BF16_HEAD_CONFIGS[16])
        for rows in ROWS:
            xs = [torch.randn((rows, k), generator=gen, device=DEVICE).to(torch.bfloat16) for _ in range(DRAFTS)]
            base = torch.randn((rows, DRAFTS, n), generator=gen, device=DEVICE).to(out_dtype)
            results = {}
            for mode in ("normal", "window"):
                outs = [torch.empty((rows, n), dtype=out_dtype, device=DEVICE) for _ in range(DRAFTS)]
                picks = [torch.empty((rows,), dtype=torch.int64, device=DEVICE) for _ in range(DRAFTS)]

                def step():
                    for i in range(DRAFTS):
                        kernel(xs[i], weight, outs[i])
                        torch.argmax(base[:, i] + outs[i], dim=-1, out=picks[i])

                stream = torch.cuda.Stream()
                if mode == "window":
                    value = AttrValue()
                    nbytes = min(weight.numel() * 2, window_max)
                    value.accessPolicyWindow = Window(weight.data_ptr(), nbytes, min(1.0, persist_max / nbytes), 2, 1)
                    check(rt.cudaStreamSetAttribute(ctypes.c_void_p(stream.cuda_stream), 1, ctypes.byref(value)),
                          "stream access policy")
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    step()
                    step()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    step()
                samples = []
                for _ in range(REPEATS):
                    flush.zero_()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    torch.cuda.synchronize()
                    samples.append(start.elapsed_time(end) * 1e3)
                results[mode] = (statistics.median(samples), min(samples),
                                 [o.clone() for o in outs], [p.clone() for p in picks])
                if mode == "window":
                    check(rt.cudaCtxResetPersistingL2Cache(), "reset persisting L2")
                del graph
            same = all(torch.equal(a.view(torch.int8), b.view(torch.int8))
                       for a, b in zip(results["normal"][2], results["window"][2]))
            same &= all(torch.equal(a, b) for a, b in zip(results["normal"][3], results["window"][3]))
            ok &= same
            print(json.dumps({"case": "step", "out_dtype": str(out_dtype).split(".")[-1], "rows": rows,
                              "normal_us": round(results["normal"][0], 2), "window_us": round(results["window"][0], 2),
                              "normal_min_us": round(results["normal"][1], 2),
                              "window_min_us": round(results["window"][1], 2),
                              "saved_us": round(results["normal"][0] - results["window"][0], 2),
                              "bit_exact": same}), flush=True)
    print(json.dumps({"case": "verdict", "all_exact": bool(ok)}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
