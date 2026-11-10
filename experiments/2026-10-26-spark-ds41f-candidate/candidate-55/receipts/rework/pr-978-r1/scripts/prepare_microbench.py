#!/usr/bin/env python3
"""Time AsyncLLM._prepare_mm_features_for_prefix_cache alone, the only code on the
#978 bench's path that round 1 changed. Features come from the PR bench's DS4.1
processor (1302x1302 JPEG, 1 and 4 images): 'warm' = turn 2+, lazy items already
sent to P1 (stripped); 'cold' = full items not yet sent (marked).
Usage: prepare_microbench.py <label> <out.json> [iterations]
"""
import asyncio
import json
import statistics
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, "/opt/exp")
import bench_mm_prefix_fastpath as bench  # noqa: E402
from vllm.multimodal.inputs import MultiModalFeatureSpec  # noqa: E402
from vllm.multimodal.media.image import ImageMediaIO  # noqa: E402
from vllm.multimodal.processing.context import TimingContext  # noqa: E402
from vllm.multimodal.processing.inputs import ProcessorInputs  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.engine import EngineCoreRequest  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402

label, out_path = sys.argv[1], sys.argv[2]
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 300
proc = bench._build_ds41_processor()
media_io = ImageMediaIO()
tok = proc.info.get_tokenizer()
loop = asyncio.new_event_loop()
result = {"label": label, "iterations": iters, "cases": {}}


def request(b64, rid):
    text = "".join(f"Image {i}: {bench.IMAGE_PLACEHOLDER}\n" for i in range(len(b64))) + "Q?\n"
    items = proc.info.parse_mm_data({"image": [media_io.load_base64("image/jpeg", b) for b in b64]})
    out = proc.apply(ProcessorInputs(prompt=tok.encode(text), mm_data_items=items), TimingContext())
    feats = [MultiModalFeatureSpec(data=k, modality="image", identifier=h, mm_position=p, mm_hash=h)
             for k, h, p in zip(out["mm_kwargs"]["image"], out["mm_hashes"]["image"],
                                out["mm_placeholders"]["image"])]
    return EngineCoreRequest(request_id=rid, prompt_token_ids=out["prompt_token_ids"], mm_features=feats,
                             sampling_params=SamplingParams(max_tokens=1), pooling_params=None,
                             arrival_time=0.0, lora_request=None, cache_salt=None, data_parallel_rank=None)


for count in (1, 4):
    b64 = [bench._make_encoded_image(1302, 1302, "JPEG", seed=100 + i) for i in range(count)]
    request(b64, "prime")  # fills the processor's metadata cache: later requests get lazy items
    for case in ("warm", "cold"):
        times = []
        for i in range(iters):
            req = request(b64, f"{case}-{i}")
            if case == "cold":  # full items, as turn 1 sends them
                for f in req.mm_features:
                    f.data.materialize()
            hashes = {f.identifier for f in req.mm_features}
            dummy = SimpleNamespace(
                _p1_cached_mm_identifiers=set(hashes) if case == "warm" else set(),
                _pending_mm_retries={}, model_config=SimpleNamespace(uses_mrope=False),
                renderer=SimpleNamespace(_mm_executor=None))
            t0 = time.perf_counter_ns()
            loop.run_until_complete(AsyncLLM._prepare_mm_features_for_prefix_cache(dummy, req))
            times.append((time.perf_counter_ns() - t0) / 1000.0)
            if case == "warm":
                assert all(f.data is None for f in req.mm_features)
            else:
                assert dummy._p1_cached_mm_identifiers == hashes
        key = f"{count}img_{case}"
        result["cases"][key] = {"median_us": round(statistics.median(times), 2),
                                "p10_us": round(sorted(times)[len(times) // 10], 2),
                                "mean_us": round(statistics.mean(times), 2)}
        print(f"[{label}] {key}: median {result['cases'][key]['median_us']} us", flush=True)
json.dump(result, open(out_path, "w"), indent=1)
