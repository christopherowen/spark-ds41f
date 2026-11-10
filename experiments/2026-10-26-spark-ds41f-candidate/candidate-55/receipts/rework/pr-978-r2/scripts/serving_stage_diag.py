#!/usr/bin/env python3
"""Diagnostic for the JPEG slowdown: round 1's serving_path_bench.py (a real AsyncLLM
frontend, the recipe's flags, a fake engine core that msgpack-encodes each request), with
every stage of the request wrapped to record its wall time, the CPU time of the thread
that ran it, and the CPU it started and ended on (cpus 0-4 and 10-14 are Cortex-A725,
5-9 and 15-19 Cortex-X925 on GB10). Wrapping is done here, at runtime; no vLLM file
changes. The wrappers cost a few microseconds per call.

Per stage: wall, thread CPU, thread system time, minor page faults, involuntary context
switches (getrusage RUSAGE_THREAD), share of samples on little cores.
Stages: load (ImageMediaIO.load_base64, on the event loop, as the bench does), hash
(MultiModalHasher.hash_kwargs, renderer's MM executor), hf (the DS4.1 processor's
_apply_hf_processor_main), mat (_LazyPILImage materialization, where it exists), proc
(renderer._process_multimodal, the whole processor call), prep (AsyncLLM's
_prepare_mm_features_for_prefix_cache, if any), enc (msgpack encode).
Usage: serving_stage_diag.py <label> <out.json>
"""
import asyncio
import ctypes
import functools
import json
import os
import resource
import statistics
import sys
import threading
import time
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, "/opt/exp")
import bench_mm_prefix_fastpath as bench  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_PLACEHOLDER  # noqa: E402
from vllm.multimodal import hasher as mm_hasher  # noqa: E402
from vllm.multimodal.media import image as mm_image  # noqa: E402
from vllm.multimodal.media.image import ImageMediaIO  # noqa: E402
from vllm.multimodal.processing.processor import BaseMultiModalProcessor  # noqa: E402
from vllm.renderers.base import BaseRenderer  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.usage.usage_lib import UsageContext  # noqa: E402
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402
from vllm.v1.engine.core_client import EngineCoreClient  # noqa: E402
from vllm.v1.serial_utils import MsgpackEncoder  # noqa: E402

label, out_path = sys.argv[1], sys.argv[2]
WORKLOADS = [
    ("1302x1302_jpeg_1img", 1302, 1302, "JPEG", "image/jpeg", 1),
    ("1302x1302_jpeg_4img", 1302, 1302, "JPEG", "image/jpeg", 4),
    ("1920x1080_png_1img", 1920, 1080, "PNG", "image/png", 1),
    ("4032x3024_jpeg_4img", 4032, 3024, "JPEG", "image/jpeg", 4),
]
CONVERSATIONS, TURNS = 3, 5
LITTLE = set(range(0, 5)) | set(range(10, 15))

_libc = ctypes.CDLL(None, use_errno=True)
_libc.sched_getcpu.restype = ctypes.c_int


def cpu():
    return _libc.sched_getcpu()


REC = defaultdict(list)


def timed(stage, fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        r0 = resource.getrusage(resource.RUSAGE_THREAD)
        c0, t0, p0 = cpu(), time.perf_counter(), time.thread_time()
        try:
            return fn(*a, **k)
        finally:
            r1 = resource.getrusage(resource.RUSAGE_THREAD)
            REC[stage].append({
                "wall": (time.perf_counter() - t0) * 1e3,
                "cpu": (time.thread_time() - p0) * 1e3,
                "sys": (r1.ru_stime - r0.ru_stime) * 1e3,
                "minflt": r1.ru_minflt - r0.ru_minflt,
                "nivcsw": r1.ru_nivcsw - r0.ru_nivcsw,
                "little": int(c0 in LITTLE) + int(cpu() in LITTLE),
                "thread": threading.current_thread().name,
            })
    return wrapper


def atimed(stage, fn):
    @functools.wraps(fn)
    async def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return await fn(*a, **k)
        finally:
            REC[stage].append({"wall": (time.perf_counter() - t0) * 1e3, "cpu": None,
                               "little": None, "thread": threading.current_thread().name})
    return wrapper


ImageMediaIO.load_base64 = timed("load", ImageMediaIO.load_base64)
mm_hasher.MultiModalHasher.hash_kwargs = classmethod(
    timed("hash", mm_hasher.MultiModalHasher.hash_kwargs.__func__))
BaseMultiModalProcessor._apply_hf_processor_main = timed(
    "hf", BaseMultiModalProcessor._apply_hf_processor_main)
BaseRenderer._process_multimodal = timed("proc", BaseRenderer._process_multimodal)
if hasattr(mm_image, "_LazyPILImage"):
    mm_image._LazyPILImage._materialize = timed("mat", mm_image._LazyPILImage._materialize)
if hasattr(AsyncLLM, "_prepare_mm_features_for_prefix_cache"):
    AsyncLLM._prepare_mm_features_for_prefix_cache = atimed(
        "prep", AsyncLLM._prepare_mm_features_for_prefix_cache)


class FakeEngineCore:
    def __init__(self):
        self.resources = SimpleNamespace(engine_dead=False)
        self.engine_ranks_managed = [0]
        self.encoder = MsgpackEncoder()
        self.encoder.encode = timed("enc", self.encoder.encode)
        self.outputs = asyncio.Queue()
        self.last = None

    async def add_request_async(self, req):
        frames = self.encoder.encode(req)
        self.last = {"t": time.perf_counter(),
                     "bytes": sum(memoryview(f).nbytes for f in frames),
                     "stripped": sum(1 for f in (req.mm_features or []) if f.data is None)}
        self.outputs.put_nowait(EngineCoreOutputs(
            outputs=[EngineCoreOutput(request_id=req.request_id, new_token_ids=[100],
                                      finish_reason=FinishReason.STOP)],
            finished_requests={req.request_id}))

    async def get_output_async(self):
        return await self.outputs.get()

    async def abort_requests_async(self, request_ids):
        pass

    async def get_supported_tasks_async(self):
        return ("generate",)

    def shutdown(self, timeout=None):
        pass


def summarize(rows):
    out = {}
    for stage, recs in rows.items():
        wall = [sum(r["wall"] for r in rr) for rr in recs]
        d = {"wall_ms": round(statistics.median(wall), 3)}
        cpus = [[r["cpu"] for r in rr if r["cpu"] is not None] for rr in recs]
        if all(cpus):
            d["thread_cpu_ms"] = round(statistics.median(sum(c) for c in cpus), 3)
        litt = [[r["little"] for r in rr if r["little"] is not None] for rr in recs]
        if all(litt):
            d["little_core_share"] = round(sum(sum(x) for x in litt) / (2 * sum(len(x) for x in litt)), 3)
        for key in ("sys", "minflt", "nivcsw"):
            vals = [[r[key] for r in rr if key in r] for rr in recs]
            if all(vals):
                d[key] = round(statistics.median(sum(v) for v in vals), 3)
        d["calls"] = statistics.median(len(rr) for rr in recs)
        out[stage] = d
    return out


async def run():
    args = AsyncEngineArgs(
        model=os.environ["DS41F_SNAPSHOT"], tokenizer_mode="deepseek_v41", dtype="bfloat16",
        block_size=256, enable_prefix_caching=True, max_model_len=32768,
        limit_mm_per_prompt={"image": 67}, mm_processor_cache_gb=0, enforce_eager=True)
    vllm_config = args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    fake = FakeEngineCore()
    with patch.object(EngineCoreClient, "make_async_mp_client", return_value=fake):
        llm = AsyncLLM(vllm_config, executor_class=object, log_stats=False, log_requests=False)
    media_io = ImageMediaIO()
    result = {"label": label, "mm_prefix_fastpath": getattr(llm, "_mm_prefix_fastpath", None),
              "affinity": sorted(os.sched_getaffinity(0)), "workloads": {}}
    rid = 0
    for name, w, h, fmt, mtype, count in WORKLOADS:
        per = {1: defaultdict(list), 2: defaultdict(list)}
        tot = {1: [], 2: []}
        for conv in range(CONVERSATIONS):
            b64 = [bench._make_encoded_image(w, h, fmt, seed=1000 * conv + 100 + i) for i in range(count)]
            for turn in range(1, TURNS + 1):
                text = ("System: analyze images.\n"
                        + "".join(f"Image {i}: {IMAGE_PLACEHOLDER}\n" for i in range(count))
                        + "".join(f"Turn {t} question and assistant reply.\n" for t in range(1, turn + 1)))
                REC.clear()
                t0 = time.perf_counter()
                images = [media_io.load_base64(mtype, b) for b in b64]
                rid += 1
                async for _ in llm.generate({"prompt": text, "multi_modal_data": {"image": images}},
                                            SamplingParams(max_tokens=1, temperature=0.0), f"r{rid}"):
                    pass
                k = 1 if turn == 1 else 2
                tot[k].append((fake.last["t"] - t0) * 1000.0)
                for stage, recs in REC.items():
                    per[k][stage].append(list(recs))
        result["workloads"][name] = {
            "turn1_total_ms": round(statistics.median(tot[1]), 3),
            "turn2plus_total_ms": round(statistics.median(tot[2]), 3),
            "turn1": summarize(per[1]), "turn2plus": summarize(per[2]),
            "stripped_last": fake.last["stripped"], "bytes_last": fake.last["bytes"],
        }
        r = result["workloads"][name]
        print(f"[{label}] {name}: t1 {r['turn1_total_ms']} t2+ {r['turn2plus_total_ms']} | turn1 "
              + " ".join(f"{s}={v['wall_ms']}/{v.get('thread_cpu_ms')}/sys{v.get('sys')}/pf{v.get('minflt')}/L{v.get('little_core_share')}"
                         for s, v in r["turn1"].items()), flush=True)
    llm.shutdown()
    json.dump(result, open(out_path, "w"), indent=1)


asyncio.run(run())
