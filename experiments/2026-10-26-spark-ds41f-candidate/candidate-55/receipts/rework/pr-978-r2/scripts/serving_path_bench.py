#!/usr/bin/env python3
"""#978's mark on the path the API server takes: multi-turn image prompts through a
real AsyncLLM frontend (VllmConfig from the served DS4.1 checkpoint with the recipe's
--enable-prefix-caching, --mm-processor-cache-gb 0, --limit-mm-per-prompt, block size
256; the real renderer, DS4.1 processor, InputProcessor and AsyncLLM.generate). Images
arrive as base64 and are loaded with ImageMediaIO.load_base64, as the chat parser does.
Only the engine-core client is an in-process fake: it msgpack-encodes each request (the
IPC serialization add_request_async does) and finishes it at once.

Per workload, 3 conversations (new images each, so turn 1 is cold) of 5 turns. Times:
load_base64 start -> request encoded for the engine. Records payload bytes, whether any
item went stripped, and per-turn digests of prompt token ids and mm hashes.
Usage: serving_path_bench.py <label> <out.json>
"""
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, "/opt/exp")
import bench_mm_prefix_fastpath as bench  # noqa: E402  (image generator only)
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_PLACEHOLDER  # noqa: E402
from vllm.multimodal.media.image import ImageMediaIO  # noqa: E402
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
    ("3840x2160_png_1img", 3840, 2160, "PNG", "image/png", 1),
    ("3840x2160_png_4img", 3840, 2160, "PNG", "image/png", 4),
    ("4032x3024_jpeg_4img", 4032, 3024, "JPEG", "image/jpeg", 4),
]
CONVERSATIONS, TURNS = 3, 5


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


class FakeEngineCore:
    def __init__(self):
        self.resources = SimpleNamespace(engine_dead=False)
        self.engine_ranks_managed = [0]
        self.encoder = MsgpackEncoder()
        self.outputs = asyncio.Queue()
        self.last = None

    async def add_request_async(self, req):
        frames = self.encoder.encode(req)
        self.last = {
            "t": time.perf_counter(),
            "bytes": sum(memoryview(f).nbytes for f in frames),
            "stripped": sum(1 for f in (req.mm_features or []) if f.data is None),
            "tokens": digest(list(req.prompt_token_ids)),
            "mm_hashes": [f.mm_hash for f in (req.mm_features or [])],
        }
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
    result = {"label": label, "conversations": CONVERSATIONS, "turns": TURNS,
              "mm_prefix_fastpath": getattr(llm, "_mm_prefix_fastpath", None), "workloads": {}}
    rid = 0
    for name, w, h, fmt, mtype, count in WORKLOADS:
        cold, warm, warm_bytes, cold_bytes, stripped, ident = [], [], [], [], 0, []
        for conv in range(CONVERSATIONS):
            b64 = [bench._make_encoded_image(w, h, fmt, seed=1000 * conv + 100 + i) for i in range(count)]
            for turn in range(1, TURNS + 1):
                text = ("System: analyze images.\n"
                        + "".join(f"Image {i}: {IMAGE_PLACEHOLDER}\n" for i in range(count))
                        + "".join(f"Turn {t} question and assistant reply.\n" for t in range(1, turn + 1)))
                t0 = time.perf_counter()
                images = [media_io.load_base64(mtype, b) for b in b64]
                rid += 1
                async for _ in llm.generate({"prompt": text, "multi_modal_data": {"image": images}},
                                            SamplingParams(max_tokens=1, temperature=0.0), f"r{rid}"):
                    pass
                ms = (fake.last["t"] - t0) * 1000.0
                (cold if turn == 1 else warm).append(ms)
                (cold_bytes if turn == 1 else warm_bytes).append(fake.last["bytes"])
                stripped += fake.last["stripped"]
                if conv == 0:
                    ident.append({"turn": turn, "tokens": fake.last["tokens"],
                                  "mm_hashes": digest(fake.last["mm_hashes"])})
        result["workloads"][name] = {
            "turn1_median_ms": round(statistics.median(cold), 3),
            "turn2plus_median_ms": round(statistics.median(warm), 3),
            "turn2plus_ms": [round(x, 3) for x in warm],
            "turn1_bytes": cold_bytes[0], "turn2plus_bytes": warm_bytes[0],
            "stripped_items": stripped, "identity_conv0": ident,
        }
        r = result["workloads"][name]
        print(f"[{label}] {name}: turn1 {r['turn1_median_ms']:.2f} ms, turn2+ {r['turn2plus_median_ms']:.2f} ms, "
              f"bytes {r['turn1_bytes']} -> {r['turn2plus_bytes']}, stripped {stripped}", flush=True)
    llm.shutdown()
    json.dump(result, open(out_path, "w"), indent=1)


asyncio.run(run())
