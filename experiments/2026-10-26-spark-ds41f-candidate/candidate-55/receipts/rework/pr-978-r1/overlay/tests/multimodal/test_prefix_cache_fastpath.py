# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pybase64
import pytest
import torch
from blake3 import blake3
from PIL import Image, ImageOps

from vllm.multimodal.hasher import MultiModalHasher, _get_hasher_factory
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.multimodal.media.image import (
    _VALIDATED_IMAGE_META_CACHE,
    ImageMediaIO,
    _LazyPILImage,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, FinishReason
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.structured_output.backend_types import StructuredOutputKey


def _make_png_bytes(width: int = 256, height: int = 256, seed: int = 42, mode: str = "RGB") -> bytes:
    rng = np.random.default_rng(seed)
    channels = 4 if mode == "RGBA" else 3
    arr = rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
    img = Image.fromarray(arr, mode=mode)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_parallel_blake3_hasher_equivalence():
    _get_hasher_factory.cache_clear()
    factory = _get_hasher_factory("blake3")
    raw_bytes = _make_png_bytes(512, 512, seed=7)
    b64_data = pybase64.b64encode(raw_bytes).decode("ascii")

    media_io = ImageMediaIO()
    loaded = media_io.load_base64("image/png", b64_data)

    digest_fast = MultiModalHasher.hash_kwargs(
        "blake3",
        model_id="deepseek-v4.1",
        image=loaded,
    )

    expected_hasher = blake3(max_threads=1)
    for k, v in sorted({"model_id": "deepseek-v4.1", "image": loaded}.items()):
        for chunk in MultiModalHasher.iter_item_to_bytes(k, v):
            expected_hasher.update(chunk)
    assert digest_fast == expected_hasher.hexdigest()
    assert factory().hexdigest() == blake3().hexdigest()


def test_image_media_io_validated_header_cache_and_lazy_pil():
    _VALIDATED_IMAGE_META_CACHE.clear()
    media_io = ImageMediaIO()

    with pytest.raises(ValueError, match="Failed to load image"):
        media_io.load_bytes(b"not-a-valid-image-payload")

    raw_bytes = _make_png_bytes(320, 240, seed=99, mode="RGBA")
    # Turn 1: cold validation -> eager PIL decode + RGBA->RGB conversion
    turn1 = media_io.load_bytes(raw_bytes)
    assert isinstance(turn1.media, Image.Image)
    assert not isinstance(turn1.media, _LazyPILImage)
    assert turn1.media.size == (320, 240)
    assert turn1.media.mode == "RGB"
    assert turn1.io_config == {
        "image_mode": "RGB",
        "rgba_background_color": (255, 255, 255),
    }

    # Turn 2: warm validation cache -> returns _LazyPILImage without Image.open
    with patch("vllm.multimodal.media.image.Image.open", side_effect=AssertionError("should not decode")) as mock_open:
        turn2 = media_io.load_bytes(raw_bytes)
        assert mock_open.call_count == 0
        assert isinstance(turn2.media, Image.Image)
        assert isinstance(turn2.media, _LazyPILImage)
        assert not turn2.media.is_materialized
        assert turn2.media.size == (320, 240)
        assert turn2.media.width == 320
        assert turn2.media.height == 240
        assert turn2.media.mode == "RGB"
        assert len(turn2.media.getexif()) == 0
        assert turn2.io_config == turn1.io_config
        assert not turn2.media.is_materialized

    # On-demand materialization when pixel data is requested
    padded = ImageOps.pad(turn2.media, (320, 240))
    assert turn2.media.is_materialized
    np.testing.assert_array_equal(np.asarray(turn1.media), np.asarray(turn2.media))
    np.testing.assert_array_equal(np.asarray(turn1.media), np.asarray(padded))


def test_lazy_multimodal_kwargs_item_materialization_and_stripping():
    calls = 0

    def _materialize():
        nonlocal calls
        calls += 1
        return {
            "pixel_values": MultiModalFieldConfig.flat("image", [slice(0, 4)]).build_elems(
                " pixel_values", torch.ones((4, 8), dtype=torch.float32)
            )[0],
            "image_grid_thw": MultiModalFieldConfig.shared(
                "image", 1, keep_on_cpu=True
            ).build_elems("image_grid_thw", torch.tensor([1, 2, 2]))[0],
        }

    cpu_elem = MultiModalFieldConfig.shared("image", 1, keep_on_cpu=True).build_elems(
        "image_grid_thw", torch.tensor([1, 2, 2])
    )[0]

    lazy_item = MultiModalKwargsItem(
        lazy_materializer=_materialize,
        lazy_mm_hash="hash-abc",
        keep_on_cpu_elems={"image_grid_thw": cpu_elem},
    )
    assert lazy_item.is_lazy
    assert calls == 0

    # Stripping for non-MRoPE returns None without materializing
    assert lazy_item.get_stripped_for_prefix_cache(uses_mrope=False) is None
    assert lazy_item.is_lazy
    assert calls == 0

    # Stripping for MRoPE returns CPU-only metadata without materializing pixel_values
    stripped_mrope = lazy_item.get_stripped_for_prefix_cache(uses_mrope=True)
    assert stripped_mrope is not None
    assert not stripped_mrope.is_lazy
    assert list(stripped_mrope.keys()) == ["image_grid_thw"]
    assert calls == 0

    # Accessing keys/items/getitem materializes automatically
    assert "pixel_values" in lazy_item
    assert not lazy_item.is_lazy
    assert calls == 1
    assert lazy_item["pixel_values"].data.shape == (4, 8)


def test_async_llm_and_engine_core_strip_and_self_heal():
    from vllm.multimodal.cache import MultiModalCacheMissError
    from vllm.multimodal.inputs import MultiModalFeatureSpec

    materialized_calls = 0

    def _make_lazy_feature():
        def _mat():
            nonlocal materialized_calls
            materialized_calls += 1
            return {
                "image_patches": MultiModalFieldConfig.shared("image", 1).build_elems(
                    "image_patches", torch.ones((2, 16), dtype=torch.bfloat16)
                )[0]
            }

        item = MultiModalKwargsItem(
            lazy_materializer=_mat,
            lazy_mm_hash="mm-hash-1",
            keep_on_cpu_elems={},
        )
        return MultiModalFeatureSpec(
            data=item,
            modality="image",
            identifier="mm-hash-1",
            mm_position=PlaceholderRange(offset=4, length=10),
            mm_hash="mm-hash-1",
        )

    class _DummyAsyncLLM:
        def __init__(self):
            self._p1_cached_mm_identifiers = set()
            self._pending_mm_retries = {}
            self.model_config = SimpleNamespace(uses_mrope=False)
            self.renderer = SimpleNamespace(_mm_executor=None)

        _prepare_mm_features_for_prefix_cache = (
            AsyncLLM._prepare_mm_features_for_prefix_cache
        )

    dummy_llm = _DummyAsyncLLM()

    # Turn 1: not yet in _p1_cached_mm_identifiers -> materializes before send
    req1 = SimpleNamespace(
        request_id="req-1",
        mm_features=[_make_lazy_feature()],
    )
    asyncio.run(dummy_llm._prepare_mm_features_for_prefix_cache(req1))
    assert materialized_calls == 1
    assert req1.mm_features[0].data is not None
    assert not req1.mm_features[0].data.is_lazy
    assert "mm-hash-1" in dummy_llm._p1_cached_mm_identifiers
    assert "req-1" not in dummy_llm._pending_mm_retries

    # Turn 2: identifier is in _p1_cached_mm_identifiers -> strips data to None!
    req2 = SimpleNamespace(
        request_id="req-2",
        mm_features=[_make_lazy_feature()],
    )
    asyncio.run(dummy_llm._prepare_mm_features_for_prefix_cache(req2))
    assert materialized_calls == 1  # Still 1: zero HF processor work on Turn 2!
    assert req2.mm_features[0].data is None
    assert "req-2" in dummy_llm._pending_mm_retries

    # Verify EngineCore._verify_or_resolve_stripped_mm_features behavior.
    # Its lookup stub returns what KVCacheManager.get_computed_blocks does.
    from typing import get_args

    from vllm.v1.core.kv_cache_manager import KVCacheManager

    lookup = KVCacheManager.get_computed_blocks.__annotations__["return"]
    assert get_args(lookup)[1] is int and len(get_args(lookup)) == 3

    # 1) Hit in encoder_cache_manager -> accepted and pinned
    pinned = []
    dummy_core = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=True)
        ),
        scheduler=SimpleNamespace(
            kv_cache_manager=SimpleNamespace(
                # KVCacheManager.get_computed_blocks: (blocks, computed tokens,
                # shared-prefix boundary)
                get_computed_blocks=lambda r: (None, 0, 0)
            ),
            encoder_cache_manager=SimpleNamespace(
                cached={"mm-hash-1": set()},
                check_and_update_cache=lambda r, idx: pinned.append((r.request_id, idx)),
            ),
            requests={},
            ced_Prompt_replay_tokens=128,
        ),
        _has_stripped_mm_data=EngineCore._has_stripped_mm_data,
    )
    req2_core = SimpleNamespace(
        request_id="req-2",
        skip_reading_prefix_cache=False,
        num_prompt_tokens=200,
        mm_features=req2.mm_features,
    )
    EngineCore._verify_or_resolve_stripped_mm_features(dummy_core, req2_core)
    assert pinned == [("req-2", 0)]

    # 2) Evicted from encoder_cache_manager, covered by KV prefix cache beyond CED replay window -> accepted
    dummy_core.scheduler.encoder_cache_manager.cached.clear()
    dummy_core.scheduler.kv_cache_manager.get_computed_blocks = lambda r: (None, 160, 0)
    # offset=4, length=10 => end=14 <= min(160, 200-128=72) => covered!
    EngineCore._verify_or_resolve_stripped_mm_features(dummy_core, req2_core)

    # 3) Evicted from both encoder_cache_manager and kv_cache_manager -> raises MultiModalCacheMissError
    dummy_core.scheduler.kv_cache_manager.get_computed_blocks = lambda r: (None, 0, 0)
    with pytest.raises(MultiModalCacheMissError) as exc_info:
        EngineCore._verify_or_resolve_stripped_mm_features(dummy_core, req2_core)
    assert exc_info.value.mm_hashes == ["mm-hash-1"]

    # And the saved retry materializer in _pending_mm_retries self-heals req2 on demand:
    saved_req, materializers = dummy_llm._pending_mm_retries.pop("req-2")
    saved_req.mm_features[0].data = materializers["mm-hash-1"]()
    assert materialized_calls == 2
    assert saved_req.mm_features[0].data is not None
    assert "image_patches" in saved_req.mm_features[0].data


def test_base_mm_processor_prefix_metadata_cache_turn2_skip():
    from vllm.multimodal.parse import MultiModalDataParser
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.processor import (
        BaseMultiModalProcessor,
        PromptReplacement,
        PromptUpdateDetails,
    )
    from transformers.feature_extraction_utils import BatchFeature

    hf_calls = 0

    class _DummyTokenizer:
        def encode(self, text, **kwargs):
            return [ord(c) for c in text]

        def decode(self, ids, **kwargs):
            return "".join(chr(i) for i in ids)

    class _DummyInfo:
        model_id = "dummy-vl"

        def __init__(self):
            self.ctx = SimpleNamespace(
                tokenizer=_DummyTokenizer(),
                vllm_config=SimpleNamespace(
                    cache_config=SimpleNamespace(enable_prefix_caching=True)
                ),
                get_mm_config=lambda: SimpleNamespace(mm_hasher_algorithm="blake3"),
            )

        def get_tokenizer(self):
            return self.ctx.tokenizer

        def get_data_parser(self):
            return MultiModalDataParser()

        def parse_mm_data(self, mm_data, validate=True):
            parser = self.get_data_parser()
            return parser.parse_mm_data(mm_data)

    class _DummyProcessor(BaseMultiModalProcessor):
        def _apply_hf_processor_main(self, mm_items, hf_processor_mm_kwargs, **kwargs):
            nonlocal hf_calls
            hf_calls += 1
            n = mm_items.get_count("image", strict=False)
            return BatchFeature(
                {
                    "patches": torch.ones((n * 4, 8), dtype=torch.bfloat16),
                    "patch_sizes": torch.tensor([4] * n, dtype=torch.long),
                    "types": torch.ones((n * 6,), dtype=torch.long),
                    "types_sizes": torch.tensor([6] * n, dtype=torch.long),
                }
            )

        def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
            patch_sizes = hf_inputs.get("patch_sizes", torch.empty(0, dtype=torch.long))
            types_sizes = hf_inputs.get("types_sizes", torch.empty(0, dtype=torch.long))
            return {
                "patches": MultiModalFieldConfig.flat_from_sizes("image", patch_sizes),
                "types": MultiModalFieldConfig.flat_from_sizes(
                    "image", types_sizes, keep_on_cpu=True
                ),
            }

        def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
            def _repl(item_idx: int):
                n_toks = out_mm_kwargs["image"][item_idx]["types"].data.numel()
                return PromptUpdateDetails.select_token_id([999] * n_toks, 999)

            return [
                PromptReplacement(
                    modality="image",
                    target=[ord("@")],
                    replacement=_repl,
                )
            ]

    proc = _DummyProcessor(_DummyInfo(), dummy_inputs=MagicMock(), cache=None)
    media_io = ImageMediaIO()
    raw_bytes = _make_png_bytes(128, 128, seed=123)
    img1 = media_io.load_bytes(raw_bytes)

    # Turn 1: runs _apply_hf_processor_main once
    inputs1 = ProcessorInputs(
        prompt=[ord("A"), ord("@"), ord("B")],
        mm_data_items=proc.info.parse_mm_data({"image": [img1]}),
    )
    out1 = proc.apply(inputs1, TimingContext())
    assert hf_calls == 1
    assert out1["prompt_token_ids"] == [ord("A"), *([999] * 6), ord("B")]
    assert not out1["mm_kwargs"]["image"][0].is_lazy

    # Turn 2 with same image: skips _apply_hf_processor_main completely!
    img2 = media_io.load_bytes(raw_bytes)
    inputs2 = ProcessorInputs(
        prompt=[ord("A"), ord("@"), ord("B"), ord("C")],
        mm_data_items=proc.info.parse_mm_data({"image": [img2]}),
    )
    out2 = proc.apply(inputs2, TimingContext())
    assert hf_calls == 1  # Zero HF processor calls on Turn 2!
    assert out2["prompt_token_ids"] == [ord("A"), *([999] * 6), ord("B"), ord("C")]
    assert out2["mm_hashes"] == out1["mm_hashes"]
    assert out2["mm_placeholders"] == out1["mm_placeholders"]
    lazy_kw = out2["mm_kwargs"]["image"][0]
    assert lazy_kw.is_lazy

    # Materializes on demand if accessed directly
    assert torch.equal(lazy_kw["patches"].data, out1["mm_kwargs"]["image"][0]["patches"].data)
    assert not lazy_kw.is_lazy
    assert hf_calls == 2
