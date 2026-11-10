#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import statistics
import sys
import time
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pybase64
import torch
from PIL import Image

from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config
from vllm.models.deepseek_v41.common.mm_preprocess import (
    IMAGE_PLACEHOLDER,
    IMAGE_SENTINEL_BASE_ID,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.multimodal.media.image import ImageMediaIO
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.serial_utils import MsgpackEncoder


class _DS41BenchTokenizer:
    def __init__(self) -> None:
        self._tok2id = {IMAGE_PLACEHOLDER: IMAGE_SENTINEL_BASE_ID}

    def convert_tokens_to_ids(self, tok: str) -> int | None:
        return self._tok2id.get(tok)

    def encode(self, text: str, **kwargs) -> list[int]:
        out: list[int] = []
        parts = text.split(IMAGE_PLACEHOLDER)
        for idx, part in enumerate(parts):
            out.extend(1000 + (ord(c) % 256) for c in part)
            if idx + 1 < len(parts):
                out.append(IMAGE_SENTINEL_BASE_ID)
        return out

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return "".join(
            IMAGE_PLACEHOLDER if t == IMAGE_SENTINEL_BASE_ID else chr(max(32, (t - 1000) % 127))
            for t in token_ids
        )


def _build_ds41_processor() -> DeepseekV4VLMultiModalProcessor:
    hf_config = DeepseekV41Config()
    tokenizer = _DS41BenchTokenizer()
    ctx = SimpleNamespace(
        model_config=SimpleNamespace(model="deepseek-ai/DeepSeek-V4.1", uses_mrope=False),
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=True)
        ),
        tokenizer=tokenizer,
        get_hf_config=lambda cls=None: hf_config,
        get_mm_config=lambda: SimpleNamespace(mm_hasher_algorithm="blake3"),
        call_hf_processor=lambda hf_proc, data, kwargs: hf_proc(**data, **kwargs),
    )
    info = DeepseekV4VLProcessingInfo(ctx)
    info.get_tokenizer = lambda: tokenizer
    info.get_data_parser = lambda: MultiModalDataParser()
    info.parse_mm_data = lambda mm_data, validate=True: info.get_data_parser().parse_mm_data(mm_data)
    return DeepseekV4VLMultiModalProcessor(info, dummy_inputs=MagicMock(), cache=None)


def _make_encoded_image(width: int, height: int, fmt: str, seed: int) -> str:
    rng = np.random.default_rng(seed)
    # Smooth gradient + noise for realistic PNG/JPEG payload sizes
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None, None]
    noise = rng.integers(0, 64, size=(height, width, 3), dtype=np.uint8)
    arr = np.clip(0.4 * x + 0.4 * y + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=92)
    else:
        img.save(buf, format="PNG", compress_level=3)
    return pybase64.b64encode(buf.getvalue()).decode("ascii")


def _run_case(
    proc: DeepseekV4VLMultiModalProcessor,
    media_io: ImageMediaIO,
    encoder: MsgpackEncoder,
    b64_images: list[str],
    media_type: str,
    turns: int = 5,
    repeats: int = 6,
) -> dict:
    tokenizer = proc.info.get_tokenizer()
    has_fastpath = hasattr(AsyncLLM, "_prepare_mm_features_for_prefix_cache")

    turn1_totals = []
    turn_warm_totals = []
    stage_turn1 = {"load_base64_ms": [], "processor_apply_ms": [], "ipc_encode_ms": []}
    stage_warm = {"load_base64_ms": [], "processor_apply_ms": [], "ipc_encode_ms": []}
    ipc_bytes_turn1 = 0
    ipc_bytes_warm = 0
    prompt_token_count = 0

    loop = asyncio.new_event_loop()
    for rep in range(repeats):
        # Reset per-conversation state before Turn 1
        if hasattr(proc, "clear_cache"):
            proc.clear_cache()
        try:
            from vllm.multimodal.media.image import _VALIDATED_IMAGE_META_CACHE

            _VALIDATED_IMAGE_META_CACHE.clear()
        except ImportError:
            pass

        dummy_llm = SimpleNamespace(
            _p1_cached_mm_identifiers=set(),
            _pending_mm_retries={},
            model_config=SimpleNamespace(uses_mrope=False),
            renderer=SimpleNamespace(_mm_executor=None),
        )

        for turn in range(1, turns + 1):
            prompt_text = (
                "System: analyze images.\n"
                + "".join(f"Image {i}: {IMAGE_PLACEHOLDER}\n" for i in range(len(b64_images)))
                + "".join(f"Turn {t} question and assistant reply.\n" for t in range(1, turn + 1))
            )
            prompt_ids = tokenizer.encode(prompt_text)

            t0 = time.perf_counter()
            loaded_imgs = [media_io.load_base64(media_type, b64) for b64 in b64_images]
            t1 = time.perf_counter()

            mm_items = proc.info.parse_mm_data({"image": loaded_imgs})
            p_inputs = ProcessorInputs(prompt=prompt_ids, mm_data_items=mm_items)
            out = proc.apply(p_inputs, TimingContext())
            t2 = time.perf_counter()

            features = []
            for idx, (kw_item, mm_hash, pos) in enumerate(
                zip(
                    out["mm_kwargs"]["image"],
                    out["mm_hashes"]["image"],
                    out["mm_placeholders"]["image"],
                )
            ):
                features.append(
                    MultiModalFeatureSpec(
                        data=kw_item,
                        modality="image",
                        identifier=mm_hash,
                        mm_position=pos,
                        mm_hash=mm_hash,
                    )
                )

            req = EngineCoreRequest(
                request_id=f"req-r{rep}-t{turn}",
                prompt_token_ids=out["prompt_token_ids"],
                mm_features=features,
                sampling_params=SamplingParams(max_tokens=16),
                 pooling_params=None,
                arrival_time=0.0,
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
            )
            if has_fastpath:
                loop.run_until_complete(
                    AsyncLLM._prepare_mm_features_for_prefix_cache(dummy_llm, req)
                )

            encoded_frames = encoder.encode(req)
            t3 = time.perf_counter()

            load_ms = (t1 - t0) * 1000.0
            proc_ms = (t2 - t1) * 1000.0
            ipc_ms = (t3 - t2) * 1000.0
            total_ms = (t3 - t0) * 1000.0
            total_bytes = sum(memoryview(b).nbytes for b in encoded_frames)

            if turn == 1:
                prompt_token_count = len(out["prompt_token_ids"])
                ipc_bytes_turn1 = total_bytes
                if rep >= 1:
                    turn1_totals.append(total_ms)
                    stage_turn1["load_base64_ms"].append(load_ms)
                    stage_turn1["processor_apply_ms"].append(proc_ms)
                    stage_turn1["ipc_encode_ms"].append(ipc_ms)
            else:
                ipc_bytes_warm = total_bytes
                if rep >= 1:
                    turn_warm_totals.append(total_ms)
                    stage_warm["load_base64_ms"].append(load_ms)
                    stage_warm["processor_apply_ms"].append(proc_ms)
                    stage_warm["ipc_encode_ms"].append(ipc_ms)

    return {
        "prompt_tokens_turn1": prompt_token_count,
        "ipc_bytes_turn1": ipc_bytes_turn1,
        "ipc_bytes_warm": ipc_bytes_warm,
        "turn1_Cold": {
            "total_median_ms": round(statistics.median(turn1_totals), 3),
            "total_mean_ms": round(statistics.mean(turn1_totals), 3),
            "load_base64_median_ms": round(statistics.median(stage_turn1["load_base64_ms"]), 3),
            "processor_apply_median_ms": round(statistics.median(stage_turn1["processor_apply_ms"]), 3),
            "ipc_encode_median_ms": round(statistics.median(stage_turn1["ipc_encode_ms"]), 3),
        },
        "turn2plus_Warm": {
            "total_median_ms": round(statistics.median(turn_warm_totals), 3),
            "total_mean_ms": round(statistics.mean(turn_warm_totals), 3),
            "total_stdev_ms": round(statistics.stdev(turn_warm_totals), 3),
            "load_base64_median_ms": round(statistics.median(stage_warm["load_base64_ms"]), 3),
            "processor_apply_median_ms": round(statistics.median(stage_warm["processor_apply_ms"]), 3),
            "ipc_encode_median_ms": round(statistics.median(stage_warm["ipc_encode_ms"]), 3),
        },
    }


def main() -> None:
    mode_label = sys.argv[1] if len(sys.argv) > 1 else "run"
    proc = _build_ds41_processor()
    media_io = ImageMediaIO()
    encoder = MsgpackEncoder()

    workloads = [
        ("1302x1302_jpeg_1img", 1302, 1302, "JPEG", "image/jpeg", 1),
        ("1302x1302_jpeg_4img", 1302, 1302, "JPEG", "image/jpeg", 4),
        ("1920x1080_png_1img", 1920, 1080, "PNG", "image/png", 1),
        ("1920x1080_png_4img", 1920, 1080, "PNG", "image/png", 4),
        ("3840x2160_png_1img", 3840, 2160, "PNG", "image/png", 1),
        ("3840x2160_png_4img", 3840, 2160, "PNG", "image/png", 4),
        ("4032x3024_jpeg_1img", 4032, 3024, "JPEG", "image/jpeg", 1),
        ("4032x3024_jpeg_4img", 4032, 3024, "JPEG", "image/jpeg", 4),
    ]

    results = {"mode": mode_label, "workloads": {}}
    for name, w, h, fmt, mtype, count in workloads:
        b64_list = [_make_encoded_image(w, h, fmt, seed=100 + i) for i in range(count)]
        res = _run_case(proc, media_io, encoder, b64_list, mtype)
        results["workloads"][name] = res
        print(
            f"[{mode_label}] {name}: "
            f"Turn1={res['turn1_Cold']['total_median_ms']:.2f}ms "
            f"(ipc={res['ipc_bytes_turn1']}B), "
            f"Turn2+={res['turn2plus_Warm']['total_median_ms']:.2f}ms "
            f"[load={res['turn2plus_Warm']['load_base64_median_ms']:.2f}ms, "
            f"proc={res['turn2plus_Warm']['processor_apply_median_ms']:.2f}ms, "
            f"ipc={res['turn2plus_Warm']['ipc_encode_median_ms']:.2f}ms, "
            f"bytes={res['ipc_bytes_warm']}B]",
            flush=True,
        )

    out_file = sys.argv[2] if len(sys.argv) > 2 else None
    if out_file:
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
