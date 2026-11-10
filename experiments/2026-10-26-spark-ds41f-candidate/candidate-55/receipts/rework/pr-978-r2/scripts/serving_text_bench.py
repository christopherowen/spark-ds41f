#!/usr/bin/env python3
"""Text-only requests on the path the API server takes, for #978 round 2's mark ("text-only
requests must not be slower"): round 1's serving-path harness (a real AsyncLLM frontend built
from the served DS4.1 checkpoint with the recipe's flags, --enable-prefix-caching,
--mm-processor-cache-gb 0, --limit-mm-per-prompt, block size 256; the real renderer and
InputProcessor; only the engine-core client is an in-process fake that msgpack-encodes each
request and finishes it with one token). Times generate() call -> request encoded for the
engine (submission) and -> final output received (round trip), per request.

Workloads: a short prompt (~60 tokens) and a long one (~4,000 tokens), 3 rounds of 40 requests
each (prompts differ per request), plus n=4 sampling on the short prompt (one submission, four
children). Records the per-request prompt token-id digest of round 0 for identity.
Usage: serving_text_bench.py <label> <out.json>
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

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.serial_utils import MsgpackEncoder

label, out_path = sys.argv[1], sys.argv[2]
ROUNDS, PER_ROUND = 3, 40
SENTENCE = "The quick brown fox jumps over the lazy dog while the cat watches. "


class FakeEngineCore:
    def __init__(self):
        self.resources = SimpleNamespace(engine_dead=False)
        self.engine_ranks_managed = [0]
        self.encoder = MsgpackEncoder()
        self.outputs = asyncio.Queue()
        self.sent = []

    async def add_request_async(self, req):
        self.encoder.encode(req)
        self.sent.append((time.perf_counter(), req))
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
    result = {"label": label, "mm_prefix_fastpath": getattr(llm, "_mm_prefix_fastpath", None),
              "workloads": {}}
    rid = 0
    for name, repeat, n in (("short_prompt", 5, 1), ("long_prompt", 330, 1), ("short_prompt_n4", 5, 4)):
        submit, total, ident = [], [], []
        for rnd in range(ROUNDS):
            for i in range(PER_ROUND):
                rid += 1
                text = f"Request {rid}: " + SENTENCE * repeat + "Answer briefly."
                params = SamplingParams(max_tokens=1, temperature=0.0 if n == 1 else 1.0, n=n)
                before = len(fake.sent)
                t0 = time.perf_counter()
                async for _ in llm.generate({"prompt": text}, params, f"t{rid}"):
                    pass
                t1 = time.perf_counter()
                submit.append((fake.sent[-1][0] - t0) * 1e3)
                total.append((t1 - t0) * 1e3)
                if rnd == 0 and i < 5:
                    ids = [list(r.prompt_token_ids) for _, r in fake.sent[before:]]
                    ident.append(hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16])
        result["workloads"][name] = {
            "submit_median_ms": round(statistics.median(submit), 4),
            "roundtrip_median_ms": round(statistics.median(total), 4),
            "submit_p90_ms": round(sorted(submit)[int(0.9 * len(submit))], 4),
            "requests": len(submit), "identity_round0": ident,
        }
        r = result["workloads"][name]
        print(f"[{label}] {name}: submit {r['submit_median_ms']} ms, round trip "
              f"{r['roundtrip_median_ms']} ms (p90 submit {r['submit_p90_ms']})", flush=True)
    llm.shutdown()
    json.dump(result, open(out_path, "w"), indent=1)


asyncio.run(run())
