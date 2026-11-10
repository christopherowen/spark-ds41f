#!/usr/bin/env python3
"""Construct a real AsyncLLM frontend for DS4.1 with the recipe's multimodal flags.

Everything in the API-server process is real: VllmConfig from AsyncEngineArgs
(the served DS4.1 checkpoint, --tokenizer-mode deepseek_v41, --enable-prefix-caching,
--mm-processor-cache-gb 0, --limit-mm-per-prompt {"image":67}, --block-size 256),
the renderer and DS4.1 multimodal processor, InputProcessor, OutputProcessor and
AsyncLLM.__init__ inside a running event loop, which starts the output handler
(where candidate-55 raised NameError). Only the engine-core client is replaced by
an in-process fake, because one GB10 cannot load the TP4 model.

Then it drives a text request and a two-turn image chat through generate(): the
second turn's image goes stripped, the fake engine answers with a retryable cache
miss once, and the output handler must materialize and resubmit it.
Usage: construct_async_llm.py <label> <out.json>
"""
import asyncio
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

label, out_path = sys.argv[1], sys.argv[2]
result = {"label": label, "steps": []}


def step(name, ok, **info):
    result["steps"].append({"step": name, "ok": bool(ok), **info})
    print(f"[{label}] {name}: {'ok' if ok else 'FAILED'} {json.dumps(info, default=str)[:300]}", flush=True)


def save():
    result["ok"] = all(s["ok"] for s in result["steps"]) and len(result["steps"]) > 0
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, default=str)


try:
    import vllm.v1.engine.async_llm as async_llm_module  # noqa: F401
    step("import vllm.v1.engine.async_llm", True)
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_PLACEHOLDER
    from vllm.sampling_params import SamplingParams
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core_client import EngineCoreClient
except Exception as e:  # noqa: BLE001
    step("import", False, error=repr(e), tb=traceback.format_exc())
    save()
    sys.exit(1)


class FakeEngineCore:
    """In-process stand-in for the engine-core client: records each submission."""

    def __init__(self):
        self.resources = SimpleNamespace(engine_dead=False)
        self.engine_ranks_managed = [0]
        self.added = []
        self.aborted = []
        self.outputs = asyncio.Queue()
        self.miss_on_stripped = True

    async def add_request_async(self, req):
        feats = [
            {"mm_hash": f.mm_hash, "data": None if f.data is None else sorted(f.data.keys())}
            for f in (req.mm_features or [])
        ]
        self.added.append({"request_id": req.request_id, "features": feats})
        stripped = [f.mm_hash for f in (req.mm_features or []) if f.data is None]
        if stripped and self.miss_on_stripped:
            # P1 has evicted the image from its encoder and prefix caches.
            self.miss_on_stripped = False
            out = EngineCoreOutput(request_id=req.request_id, new_token_ids=[],
                                   finish_reason=FinishReason.ERROR,
                                   mm_cache_miss_hashes=stripped)
            self.outputs.put_nowait(EngineCoreOutputs(outputs=[out], finished_requests={req.request_id}))
        else:
            out = EngineCoreOutput(request_id=req.request_id, new_token_ids=[100],
                                   finish_reason=FinishReason.STOP)
            self.outputs.put_nowait(EngineCoreOutputs(outputs=[out], finished_requests={req.request_id}))

    async def get_output_async(self):
        return await self.outputs.get()

    async def abort_requests_async(self, request_ids):
        self.aborted.extend(request_ids)

    async def get_supported_tasks_async(self):
        return ("generate",)

    def shutdown(self, timeout=None):
        pass


async def run():
    snapshot = os.environ["DS41F_SNAPSHOT"]
    args = AsyncEngineArgs(
        model=snapshot, served_model_name=["deepseek-v4.1-flash"], tokenizer_mode="deepseek_v41",
        dtype="bfloat16", block_size=256, enable_prefix_caching=True, max_model_len=16384,
        limit_mm_per_prompt={"image": 67}, mm_processor_cache_gb=0, enforce_eager=True,
    )
    t0 = time.time()
    vllm_config = args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    step("create_engine_config", True, seconds=round(time.time() - t0, 1),
         enable_prefix_caching=vllm_config.cache_config.enable_prefix_caching,
         mm_processor_cache_gb=vllm_config.model_config.multimodal_config.mm_processor_cache_gb)
    fake = FakeEngineCore()
    with patch.object(EngineCoreClient, "make_async_mp_client", return_value=fake):
        try:
            llm = AsyncLLM(vllm_config, executor_class=object, log_stats=False, log_requests=False)
        except Exception as e:  # noqa: BLE001
            step("AsyncLLM.__init__ in a running loop", False, error=repr(e), tb=traceback.format_exc())
            return
    handler = llm.output_handler
    step("AsyncLLM.__init__ in a running loop", handler is not None and not handler.done(),
         mm_prefix_fastpath=getattr(llm, "_mm_prefix_fastpath", "absent"),
         mm_processor_cache_is_none=llm.renderer.mm_processor_cache is None)

    async def one(prompt, request_id, max_tokens=1):
        outs = []
        async for out in llm.generate(prompt, SamplingParams(max_tokens=max_tokens, temperature=0.0),
                                      request_id):
            outs.append(out)
        return outs[-1]

    out = await one("Say one word.", "text-1")
    step("text request", out.finished and not fake.added[-1]["features"],
         finished=out.finished, features=fake.added[-1]["features"],
         pending_retries=len(llm._pending_mm_retries))

    rng = np.random.default_rng(7)
    x = np.linspace(0, 255, 1302, dtype=np.float32)
    arr = np.clip(0.5 * x[None, :, None] + 0.5 * x[:, None, None]
                  + rng.integers(0, 64, (1302, 1302, 3)), 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    turn1 = f"User: {IMAGE_PLACEHOLDER}\nWhat is in this image?\nAssistant:"
    n_before = len(fake.added)
    out = await one({"prompt": turn1, "multi_modal_data": {"image": image}}, "image-turn-1")
    sent = fake.added[n_before:]
    full = len(sent) == 1 and all(f["data"] for f in sent[0]["features"]) and sent[0]["features"]
    step("image turn 1 (cold: full data, then marked as sent to P1)",
         out.finished and bool(full) and len(llm._p1_cached_mm_identifiers) == 1,
         submissions=sent, marked=sorted(llm._p1_cached_mm_identifiers))

    turn2 = turn1 + " A gradient.\nUser: Describe it again.\nAssistant:"
    n_before = len(fake.added)
    out = await one({"prompt": turn2, "multi_modal_data": {"image": image}}, "image-turn-2")
    sent = fake.added[n_before:]
    ok = (len(sent) == 2 and sent[0]["features"][0]["data"] is None
          and sent[1]["features"][0]["data"] and sent[0]["request_id"] == sent[1]["request_id"])
    step("image turn 2 (stripped; retryable miss; materialized and resubmitted)",
         out.finished and bool(ok) and not llm._pending_mm_retries,
         submissions=sent, pending_retries=len(llm._pending_mm_retries),
         marked=sorted(llm._p1_cached_mm_identifiers))

    llm.shutdown()
    step("shutdown", True, aborted=fake.aborted)


try:
    asyncio.run(run())
except Exception as e:  # noqa: BLE001
    step("run", False, error=repr(e), tb=traceback.format_exc())
save()
sys.exit(0 if result["ok"] else 1)
