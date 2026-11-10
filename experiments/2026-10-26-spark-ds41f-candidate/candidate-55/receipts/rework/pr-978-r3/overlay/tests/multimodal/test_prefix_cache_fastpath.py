# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
import threading
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
from vllm.multimodal.media import image as media_image
from vllm.multimodal.media.image import (
    _VALIDATED_IMAGE_META_CACHE,
    ImageMediaIO,
    _LazyPILImage,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, FinishReason
from vllm.v1.engine.async_llm import AsyncLLM, SentMMIdentifiers
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


@pytest.fixture
def header_cache_on():
    media_image.set_validated_image_meta_cache_enabled(False)  # empties it
    media_image.set_validated_image_meta_cache_enabled(True)
    yield
    media_image.set_validated_image_meta_cache_enabled(False)


def test_image_bytes_load_as_before_without_the_fast_path():
    # Without the fast path the processor decodes repeat images anyway, so
    # loading must not defer the decode or hash the bytes.
    media_image.set_validated_image_meta_cache_enabled(False)
    media_io = ImageMediaIO()
    raw_bytes = _make_png_bytes(64, 48, seed=5)
    with patch.object(media_image.hashlib, "sha256", side_effect=AssertionError):
        for _ in range(2):
            loaded = media_io.load_bytes(raw_bytes)
            assert type(loaded.media) is not _LazyPILImage
            assert loaded.media.size == (64, 48)
    assert not _VALIDATED_IMAGE_META_CACHE


def test_image_media_io_validated_header_cache_and_lazy_pil(header_cache_on):
    media_io = ImageMediaIO()

    with pytest.raises(ValueError, match="Failed to load image"):
        media_io.load_bytes(b"not-a-valid-image-payload")

    raw_bytes = _make_png_bytes(320, 240, seed=99, mode="RGBA")
    # Turn 1: cold validation -> eager PIL decode + RGBA->RGB conversion, and
    # the image is recorded.
    with patch.object(
        media_image,
        "_store_validated_image",
        wraps=media_image._store_validated_image,
    ) as store:
        turn1 = media_io.load_bytes(raw_bytes)
    assert store.call_count == 1
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
            self._p1_cached_mm_identifiers = SentMMIdentifiers(16)
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

    # 2) Evicted from encoder_cache_manager, covered by the KV prefix cache -> accepted
    dummy_core.scheduler.encoder_cache_manager.cached.clear()
    dummy_core.scheduler.kv_cache_manager.get_computed_blocks = lambda r: (None, 16, 0)
    # offset=4, length=10 => end=14 <= 16 => covered
    EngineCore._verify_or_resolve_stripped_mm_features(dummy_core, req2_core)
    # Covered only up to its middle -> not covered
    dummy_core.scheduler.kv_cache_manager.get_computed_blocks = lambda r: (None, 13, 0)
    with pytest.raises(MultiModalCacheMissError):
        EngineCore._verify_or_resolve_stripped_mm_features(dummy_core, req2_core)

    # 3) Evicted from both encoder_cache_manager and kv_cache_manager -> raises MultiModalCacheMissError
    dummy_core.scheduler.kv_cache_manager.get_computed_blocks = lambda r: (None, 0, 0)
    with pytest.raises(MultiModalCacheMissError) as exc_info:
        EngineCore._verify_or_resolve_stripped_mm_features(dummy_core, req2_core)
    assert exc_info.value.mm_hashes == ["mm-hash-1"]

    # And the saved retry materializer in _pending_mm_retries self-heals req2 on demand:
    saved_req, materializers = dummy_llm._pending_mm_retries.pop("req-2")
    assert list(materializers) == [0]
    saved_req.mm_features[0].data = materializers[0]()
    assert materialized_calls == 2
    assert saved_req.mm_features[0].data is not None
    assert "image_patches" in saved_req.mm_features[0].data


class _DummyTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class _DummyInfo:
    model_id = "dummy-vl"

    def __init__(self, fastpath=True, calls=None):
        # As registry.create_processor builds it for the serving renderer.
        self.ctx = SimpleNamespace(
            tokenizer=_DummyTokenizer(),
            mm_prefix_metadata_cache=fastpath,
            get_mm_config=lambda: SimpleNamespace(mm_hasher_algorithm="blake3"),
        )
        self.calls = calls if calls is not None else [0]

    def get_tokenizer(self):
        return self.ctx.tokenizer

    def get_data_parser(self):
        from vllm.multimodal.parse import MultiModalDataParser

        return MultiModalDataParser()

    def parse_mm_data(self, mm_data, validate=True):
        parser = self.get_data_parser()
        return parser.parse_mm_data(mm_data)


def _dummy_processor(fastpath=True, calls=None, cache=None):
    """A processor with DS4.1's field layout whose HF processor counts its
    calls in calls[0]."""
    from transformers.feature_extraction_utils import BatchFeature

    from vllm.multimodal.processing.processor import (
        BaseMultiModalProcessor,
        PromptReplacement,
        PromptUpdateDetails,
    )

    class _DummyProcessor(BaseMultiModalProcessor):
        def _apply_hf_processor_main(self, mm_items, hf_processor_mm_kwargs, **kwargs):
            self.info.calls[0] += 1
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

    return _DummyProcessor(
        _DummyInfo(fastpath, calls), dummy_inputs=MagicMock(), cache=cache
    )


def test_base_mm_processor_prefix_metadata_cache_turn2_skip(header_cache_on):
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    calls = [0]

    # A context without the setting, or a processor with a processor cache,
    # keeps the processor as it was.
    assert not _dummy_processor(fastpath=False).prefix_metadata_cache_enabled
    assert not _dummy_processor(cache=MagicMock()).prefix_metadata_cache_enabled
    off = _dummy_processor(fastpath=False, calls=calls)

    proc = _dummy_processor(calls=calls)
    assert proc.prefix_metadata_cache_enabled
    media_io = ImageMediaIO()
    raw_bytes = _make_png_bytes(128, 128, seed=123)
    img1 = media_io.load_bytes(raw_bytes)

    # Turn 1: runs _apply_hf_processor_main once
    inputs1 = ProcessorInputs(
        prompt=[ord("A"), ord("@"), ord("B")],
        mm_data_items=proc.info.parse_mm_data({"image": [img1]}),
    )
    out1 = proc.apply(inputs1, TimingContext())
    assert calls[0] == 1
    assert out1["prompt_token_ids"] == [ord("A"), *([999] * 6), ord("B")]
    assert not out1["mm_kwargs"]["image"][0].is_lazy

    # Turn 2 with same image (loaded lazily from bytes seen before): skips
    # _apply_hf_processor_main completely!
    img2 = media_io.load_bytes(raw_bytes)
    assert isinstance(img2.media, _LazyPILImage)
    inputs2 = ProcessorInputs(
        prompt=[ord("A"), ord("@"), ord("B"), ord("C")],
        mm_data_items=proc.info.parse_mm_data({"image": [img2]}),
    )
    out2 = proc.apply(inputs2, TimingContext())
    assert calls[0] == 1  # Zero HF processor calls on Turn 2!
    assert out2["prompt_token_ids"] == [ord("A"), *([999] * 6), ord("B"), ord("C")]
    assert out2["mm_hashes"] == out1["mm_hashes"]
    assert out2["mm_placeholders"] == out1["mm_placeholders"]
    lazy_kw = out2["mm_kwargs"]["image"][0]
    assert lazy_kw.is_lazy

    # Materializes on demand if accessed directly
    assert torch.equal(lazy_kw["patches"].data, out1["mm_kwargs"]["image"][0]["patches"].data)
    assert not lazy_kw.is_lazy
    assert calls[0] == 2

    # Without the setting, every turn runs the HF processor.
    for _ in range(2):
        out_off = off.apply(
            ProcessorInputs(
                prompt=[ord("A"), ord("@"), ord("B")],
                mm_data_items=off.info.parse_mm_data({"image": [img2]}),
            ),
            TimingContext(),
        )
        assert not out_off["mm_kwargs"]["image"][0].is_lazy
    assert calls[0] == 4


def test_renderer_enables_the_fast_path_only_where_it_applies():
    from vllm.renderers.base import BaseRenderer

    def config(prefix=True, kv=None, ec=None, dp=1, external_lb=False):
        return SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=prefix),
            kv_transfer_config=kv,
            ec_transfer_config=ec,
            parallel_config=SimpleNamespace(
                data_parallel_size=dp, data_parallel_external_lb=external_lb
            ),
        )

    uses = BaseRenderer._uses_mm_prefix_fastpath
    assert uses(config(), None)
    assert not uses(config(), MagicMock())  # a processor cache
    assert not uses(config(prefix=False), None)
    assert not uses(config(kv=object()), None)
    assert not uses(config(ec=object()), None)
    assert not uses(config(dp=2), None)
    assert uses(config(dp=2, external_lb=True), None)


class _RecordingEngineCore:
    """The engine-core client of a real AsyncLLM: msgpack-encodes each request,
    as the IPC send does, keeps it, and finishes it at once."""

    def __init__(self):
        from vllm.v1.serial_utils import MsgpackEncoder

        self.resources = SimpleNamespace(engine_dead=False)
        self.engine_ranks_managed = [0]
        self.encoder = MsgpackEncoder()
        self.outputs = asyncio.Queue()
        self.sent = []

    async def add_request_async(self, request):
        frames = self.encoder.encode(request)
        self.sent.append((request, sum(memoryview(f).nbytes for f in frames)))
        self.outputs.put_nowait(
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[100],
                        finish_reason=FinishReason.STOP,
                    )
                ],
                finished_requests={request.request_id},
            )
        )

    async def get_output_async(self):
        return await self.outputs.get()

    async def abort_requests_async(self, request_ids):
        pass

    async def get_supported_tasks_async(self):
        return ("generate",)

    def shutdown(self, timeout=None):
        pass


@pytest.mark.parametrize("mm_processor_cache_gb", [0, 4])
def test_serving_frontend_sends_repeat_images_without_data(mm_processor_cache_gb):
    """The API server's path: the renderer builds the processor from the
    served DS4.1 config with the recipe's multimodal flags, and AsyncLLM sends
    a conversation's repeated image without its tensors from the second turn
    (with --mm-processor-cache-gb 0 only)."""
    snapshot = os.environ.get("DS41F_SNAPSHOT")
    if not snapshot:
        pytest.skip("needs the DS4.1 checkpoint (DS41F_SNAPSHOT)")
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_PLACEHOLDER
    from vllm.sampling_params import SamplingParams
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.core_client import EngineCoreClient

    args = AsyncEngineArgs(
        model=snapshot,
        tokenizer_mode="deepseek_v41",
        dtype="bfloat16",
        block_size=256,
        enable_prefix_caching=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 67},
        mm_processor_cache_gb=mm_processor_cache_gb,
        enforce_eager=True,
    )
    vllm_config = args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    fake = _RecordingEngineCore()

    async def run():
        with patch.object(EngineCoreClient, "make_async_mp_client", return_value=fake):
            llm = AsyncLLM(
                vllm_config, executor_class=object, log_stats=False, log_requests=False
            )
        try:
            fastpath = mm_processor_cache_gb == 0
            processor = llm.renderer.mm_processor
            assert processor.info.ctx.mm_prefix_metadata_cache is fastpath
            assert processor.prefix_metadata_cache_enabled is fastpath
            assert llm._mm_prefix_fastpath is fastpath
            if not fastpath:
                return
            b64 = pybase64.b64encode(_make_png_bytes(320, 240, seed=11)).decode()
            media_io = ImageMediaIO()
            for turn in range(1, 4):
                text = f"Image: {IMAGE_PLACEHOLDER}\n" + "Question.\n" * turn
                prompt = {
                    "prompt": text,
                    "multi_modal_data": {
                        "image": [media_io.load_base64("image/png", b64)]
                    },
                }
                params = SamplingParams(max_tokens=1, temperature=0.0)
                async for _ in llm.generate(prompt, params, f"turn-{turn}"):
                    pass
            requests = [request for request, _ in fake.sent]
            [first, *later] = requests
            assert first.mm_features[0].data is not None
            for request in later:
                [feature] = request.mm_features
                assert feature.data is None
                assert feature.mm_hash == first.mm_features[0].mm_hash
                assert feature.mm_position == first.mm_features[0].mm_position
                n = first.mm_features[0].mm_position.offset + feature.mm_position.length
                assert list(request.prompt_token_ids[:n]) == list(
                    first.prompt_token_ids[:n]
                )
            sizes = [nbytes for _, nbytes in fake.sent]
            assert max(sizes[1:]) * 100 < sizes[0]
        finally:
            llm.shutdown()

    asyncio.run(run())


def test_header_cache_tells_images_with_the_same_edges_apart(header_cache_on):
    # Same length, first and last 16 KiB (the pre-key); different middle.
    media_io = ImageMediaIO()
    side = 400
    base = np.random.default_rng(3).integers(0, 256, (side, side, 3), dtype=np.uint8)
    other = base.copy()
    other[side // 2, side // 2] ^= 0xFF

    def encode(arr):
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="BMP")  # uncompressed: same length
        return buf.getvalue()

    a, b = encode(base), encode(other)
    assert len(a) == len(b) and a != b
    edge = media_image._VALIDATED_IMAGE_PREKEY_EDGE
    assert a[:edge] == b[:edge] and a[-edge:] == b[-edge:]
    media_io.load_bytes(a)
    assert isinstance(media_io.load_bytes(a).media, _LazyPILImage)
    loaded_b = media_io.load_bytes(b)
    assert not isinstance(loaded_b.media, _LazyPILImage)
    np.testing.assert_array_equal(np.asarray(loaded_b.media), other)


def test_new_image_digest_runs_alongside_its_decode(header_cache_on):
    # A new image of 64 KiB or more is hashed on a helper thread while the
    # loading thread decodes it, and is recorded before the load returns.
    media_io = ImageMediaIO()
    raw_bytes = _make_png_bytes(700, 700, seed=21)
    assert len(raw_bytes) >= media_image._VALIDATED_IMAGE_DIGEST_OVERLAP_MIN
    real_sha256 = media_image.hashlib.sha256
    full_hash_threads = []
    target = [raw_bytes]

    def sha256(data=b"", *args, **kwargs):
        if len(data) == len(target[0]):
            full_hash_threads.append(threading.current_thread().name)
        return real_sha256(data, *args, **kwargs)

    with patch.object(media_image.hashlib, "sha256", sha256):
        first = media_io.load_bytes(raw_bytes)
        assert not isinstance(first.media, _LazyPILImage)
        [thread] = full_hash_threads
        assert thread.startswith("vllm-image-digest")
        second = media_io.load_bytes(raw_bytes)
    assert isinstance(second.media, _LazyPILImage)
    assert second.media.size == first.media.size
    np.testing.assert_array_equal(np.asarray(second.media), np.asarray(first.media))

    # A small image hashes on the loading thread.
    small = _make_png_bytes(64, 64, seed=22)
    assert len(small) < media_image._VALIDATED_IMAGE_DIGEST_OVERLAP_MIN
    full_hash_threads.clear()
    target[0] = small
    with patch.object(media_image.hashlib, "sha256", sha256):
        media_io.load_bytes(small)
    assert full_hash_threads == [threading.current_thread().name]
    assert isinstance(media_io.load_bytes(small).media, _LazyPILImage)


def test_new_items_are_hashed_alongside_the_hf_processor(header_cache_on):
    """Turn 1 of new images under the fast path does what the processor does
    without it, with the hashes computed on a helper thread while the HF
    processor runs, and records the items' metadata."""
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    proc = _dummy_processor()
    off = _dummy_processor(fastpath=False)
    media_io = ImageMediaIO()
    raw = [_make_png_bytes(96 + 16 * i, 80, seed=300 + i) for i in range(2)]
    hash_threads = []
    real_get_mm_hashes = ProcessorInputs.get_mm_hashes

    def get_mm_hashes(self, *args, **kwargs):
        hash_threads.append(threading.current_thread().name)
        return real_get_mm_hashes(self, *args, **kwargs)

    prompt = [ord("A"), ord("@"), ord("B"), ord("@"), ord("C")]
    with patch.object(ProcessorInputs, "get_mm_hashes", get_mm_hashes):
        images = [media_io.load_bytes(b) for b in raw]
        assert not any(isinstance(i.media, _LazyPILImage) for i in images)
        out = proc.apply(
            ProcessorInputs(
                prompt=list(prompt),
                mm_data_items=proc.info.parse_mm_data({"image": images}),
            ),
            TimingContext(),
        )
        [thread] = hash_threads
        assert thread.startswith("vllm-mm-hash")
        expected = off.apply(
            ProcessorInputs(
                prompt=list(prompt),
                mm_data_items=off.info.parse_mm_data(
                    {"image": [media_io.load_bytes(b) for b in raw]}
                ),
            ),
            TimingContext(),
        )
    assert out["prompt_token_ids"] == expected["prompt_token_ids"]
    assert out["mm_hashes"] == expected["mm_hashes"]
    assert out["mm_placeholders"] == expected["mm_placeholders"]
    for got, want in zip(out["mm_kwargs"]["image"], expected["mm_kwargs"]["image"]):
        assert not got.is_lazy
        assert set(got) == set(want)
        for key in got:
            assert torch.equal(got[key].data, want[key].data)
    assert set(proc._prefix_metadata_cache) == set(out["mm_hashes"]["image"])
