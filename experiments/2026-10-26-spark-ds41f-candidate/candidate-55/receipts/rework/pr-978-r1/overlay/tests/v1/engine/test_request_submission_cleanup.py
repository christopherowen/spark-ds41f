# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Submission failures must not retain frontend prompts or orphan sampled children."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import OutputProcessor

pytestmark = pytest.mark.cpu_test


class SubmissionCore:
    def __init__(self):
        self.resources = SimpleNamespace(engine_dead=False)
        self.submitted = []
        self.aborted = []
        self.fail_at = None
        self.block_at = None
        self.abort_error = False
        self.entered = asyncio.Event()
        self.resume = asyncio.Event()

    async def add_request_async(self, request):
        self.submitted.append(request.request_id)
        if len(self.submitted) == self.block_at:
            self.entered.set()
            await self.resume.wait()
        if len(self.submitted) == self.fail_at:
            raise RuntimeError("submission failed")

    async def abort_requests_async(self, request_ids):
        self.aborted.extend(request_ids)
        if self.abort_error:
            raise RuntimeError("abort send failed")

    def shutdown(self, timeout=None):
        pass


def make_llm():
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.engine_core = SubmissionCore()
    llm.output_processor = OutputProcessor(None, log_stats=False)
    llm.scheduler_config = SimpleNamespace(
        max_num_queued_reqs=None, max_num_queued_tokens=None
    )
    llm.admission_stats = None
    llm.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    llm.log_requests = False
    llm.output_handler = None
    llm._run_output_handler = lambda: None
    llm._request_stall_warning_s = 0
    llm._log_request_timeline = False
    sequence = 0

    def assign_request_id(request):
        nonlocal sequence
        sequence += 1
        request.external_req_id = request.request_id
        request.request_id = f"{request.request_id}-internal-{sequence}"

    llm.input_processor = SimpleNamespace(assign_request_id=assign_request_id)
    return llm


def make_request(n=1, request_id="shared-external-id"):
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(range(65536)),
        mm_features=None,
        sampling_params=SamplingParams(n=n, max_tokens=16, detokenize=False),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def assert_empty(llm):
    processor = llm.output_processor
    assert not processor.request_states
    assert not processor.parent_requests
    assert not processor.external_req_ids
    assert processor.get_num_queued_tokens() == 0


@pytest.mark.parametrize("abort_error", [False, True])
def test_failed_submission_rolls_back_registered_prompt(abort_error):
    async def run():
        llm = make_llm()
        llm.engine_core.fail_at = 1
        llm.engine_core.abort_error = abort_error
        request = make_request()
        with pytest.raises(RuntimeError, match="^submission failed$"):
            await llm.add_request(request.request_id, request, request.sampling_params)
        assert_empty(llm)
        assert llm.engine_core.aborted == llm.engine_core.submitted

    asyncio.run(run())


def test_repeated_failed_submissions_do_not_accumulate_prompt_history():
    async def run():
        llm = make_llm()
        for index in range(16):
            llm.engine_core.fail_at = len(llm.engine_core.submitted) + 1
            request = make_request(request_id=f"failed-{index}")
            with pytest.raises(RuntimeError, match="submission failed"):
                await llm.add_request(
                    request.request_id, request, request.sampling_params
                )
        # The uncorrected code retains sixteen prompts (1,048,576 token IDs)
        # even though none reached the engine and no request can finish them.
        assert_empty(llm)

    asyncio.run(run())


@pytest.mark.parametrize("n, block_at", [(1, 1), (3, 2), (3, 3)])
def test_disconnect_during_submission_rolls_back_all_children(n, block_at):
    async def run():
        llm = make_llm()
        llm.engine_core.block_at = block_at
        request = make_request(n)
        # Exercise generate(): its queue handle has not yet been assigned when
        # cancellation interrupts add_request(). No engine output will arrive.
        generator = llm.generate(
            request, request.sampling_params, request.request_id
        )
        task = asyncio.create_task(anext(generator))
        await asyncio.wait_for(llm.engine_core.entered.wait(), timeout=2)
        assert len(llm.output_processor.request_states) == block_at
        assert llm.output_processor.get_num_queued_tokens() == block_at * 65536
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await generator.aclose()
        assert_empty(llm)
        assert set(llm.engine_core.aborted) == set(llm.engine_core.submitted)

    asyncio.run(run())


@pytest.mark.parametrize("n", [1, 3])
def test_failure_does_not_abort_another_request_reusing_external_id(n):
    async def run():
        llm = make_llm()
        first = make_request()
        queue = await llm.add_request(
            first.request_id, first, first.sampling_params
        )
        original_id = first.request_id
        llm.engine_core.fail_at = 1 + n
        request = make_request(n)
        with pytest.raises(RuntimeError, match="submission failed"):
            await llm.add_request(request.request_id, request, request.sampling_params)
        assert set(llm.output_processor.request_states) == {original_id}
        assert not llm.output_processor.parent_requests
        assert llm.output_processor.external_req_ids == {
            "shared-external-id": [original_id]
        }
        assert original_id not in llm.engine_core.aborted
        await llm.abort(queue.request_id, internal=True)
        assert_empty(llm)

    asyncio.run(run())


@pytest.mark.parametrize("n", [1, 3])
def test_successful_submission_and_completion_keep_normal_cleanup(n):
    async def run():
        llm = make_llm()
        request = make_request(n)
        queue = await llm.add_request(
            request.request_id, request, request.sampling_params
        )
        assert len(llm.output_processor.request_states) == n
        assert not llm.engine_core.aborted
        outputs = [
            EngineCoreOutput(
                request_id=internal_id,
                new_token_ids=[1],
                finish_reason=FinishReason.STOP,
            )
            for internal_id in llm.engine_core.submitted
        ]
        llm.output_processor.process_outputs(outputs)
        assert_empty(llm)
        result = await queue.get()
        assert result.finished
        assert len(result.outputs) == n
        assert not llm.engine_core.aborted

    asyncio.run(run())


# Multimodal prefix-cache fast path (--mm-processor-cache-gb 0, prefix caching).
# Its materialization await sits between local registration and the engine send,
# so it must keep the guarantees above, and its retry entries and P1 marks must
# not outlive a request that never reached the engine.


class Materializer:
    """Builds an image item on the MM executor once released."""

    def __init__(self, released=True):
        self.started = threading.Event()
        self.release = threading.Event()
        if released:
            self.release.set()
        self.calls = 0

    def __call__(self):
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=10)
        return {
            "image_patches": MultiModalFieldConfig.shared("image", 1).build_elems(
                "image_patches", torch.ones((2, 16), dtype=torch.bfloat16)
            )[0]
        }


def make_mm_llm(executor):
    llm = make_llm()
    llm._mm_prefix_fastpath = True
    llm._p1_cached_mm_identifiers = set()
    llm._pending_mm_retries = {}
    llm.model_config = SimpleNamespace(uses_mrope=False)
    llm.renderer = SimpleNamespace(_mm_executor=executor, shutdown=lambda: None)
    return llm


def make_mm_request(materializer, n=1):
    # A repeat image as the processor returns it under the fast path.
    request = make_request(n)
    request.mm_features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem(
                lazy_materializer=materializer,
                lazy_mm_hash="image-1",
                keep_on_cpu_elems={},
            ),
            modality="image",
            identifier="image-1",
            mm_position=PlaceholderRange(offset=4, length=10),
            mm_hash="image-1",
        )
    ]
    return request


def run_with_mm_executor(run):
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


async def drain(executor):
    # The executor has one worker: this returns once earlier work finished.
    await asyncio.get_running_loop().run_in_executor(executor, lambda: None)


async def until(condition):
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


class NoRenderer:
    def shutdown(self):
        pass

    def __getattr__(self, name):
        raise AssertionError(f"a text-only request read renderer.{name}")


@pytest.mark.parametrize("fail", [False, True])
def test_text_only_requests_skip_the_multimodal_fast_path(fail):
    async def run():
        llm = make_llm()
        llm._mm_prefix_fastpath = True
        llm._p1_cached_mm_identifiers = set()
        llm._pending_mm_retries = {}
        llm.renderer = NoRenderer()
        request = make_request()
        if fail:
            llm.engine_core.fail_at = 1
            with pytest.raises(RuntimeError, match="^submission failed$"):
                await llm.add_request(
                    request.request_id, request, request.sampling_params
                )
        else:
            queue = await llm.add_request(
                request.request_id, request, request.sampling_params
            )
            await llm.abort(queue.request_id, internal=True)
        assert_empty(llm)
        assert llm.engine_core.aborted == llm.engine_core.submitted

    asyncio.run(run())


@pytest.mark.parametrize("n", [1, 3])
def test_disconnect_during_mm_materialize_rolls_back_all_children(n):
    async def run(executor):
        llm = make_mm_llm(executor)
        materializer = Materializer(released=False)
        request = make_mm_request(materializer, n)
        generator = llm.generate(
            request, request.sampling_params, request.request_id
        )
        task = asyncio.create_task(anext(generator))
        try:
            assert await asyncio.to_thread(materializer.started.wait, 2)
            # The first child is registered and materializing; nothing is sent.
            registered = set(llm.output_processor.request_states)
            assert len(registered) == 1
            assert llm.output_processor.get_num_queued_tokens() == 65536
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await generator.aclose()
        finally:
            materializer.release.set()
        await drain(executor)
        assert_empty(llm)
        assert not llm.engine_core.submitted
        assert set(llm.engine_core.aborted) == registered
        assert not llm._pending_mm_retries
        assert not llm._p1_cached_mm_identifiers

    run_with_mm_executor(run)


@pytest.mark.parametrize("n, block_at", [(1, 1), (3, 2), (3, 3)])
def test_disconnect_during_stripped_mm_send_leaves_no_retry_entry(n, block_at):
    async def run(executor):
        llm = make_mm_llm(executor)
        # An earlier turn sent the image to P1, so this one goes stripped and
        # keeps a retry materializer in case P1 has since evicted it.
        llm._p1_cached_mm_identifiers.add("image-1")
        llm.engine_core.block_at = block_at
        materializer = Materializer()
        request = make_mm_request(materializer, n)
        generator = llm.generate(
            request, request.sampling_params, request.request_id
        )
        task = asyncio.create_task(anext(generator))
        await asyncio.wait_for(llm.engine_core.entered.wait(), timeout=2)
        assert len(llm.output_processor.request_states) == block_at
        assert llm._pending_mm_retries
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await generator.aclose()
        assert_empty(llm)
        assert set(llm.engine_core.aborted) == set(llm.engine_core.submitted)
        assert not llm._pending_mm_retries
        # The mark is the earlier turn's delivered request's, not this one's.
        assert llm._p1_cached_mm_identifiers == {"image-1"}
        assert materializer.calls == 0

    run_with_mm_executor(run)


@pytest.mark.parametrize("n", [1, 3])
def test_failed_mm_submission_unmarks_items_it_did_not_deliver(n):
    async def run(executor):
        llm = make_mm_llm(executor)
        llm.engine_core.fail_at = n
        materializer = Materializer()
        request = make_mm_request(materializer, n)
        with pytest.raises(RuntimeError, match="^submission failed$"):
            await llm.add_request(request.request_id, request, request.sampling_params)
        assert_empty(llm)
        assert set(llm.engine_core.aborted) == set(llm.engine_core.submitted)
        assert not llm._pending_mm_retries
        assert materializer.calls == 1
        if n == 1:
            # Its only send failed: P1 never received the image.
            assert not llm._p1_cached_mm_identifiers
        else:
            # The first child delivered the image before the last send failed.
            # As for any aborted request, the mark stays a hint: P1 verifies a
            # stripped item and returns a retryable miss if it lacks it.
            assert llm._p1_cached_mm_identifiers == {"image-1"}

    run_with_mm_executor(run)


def start_output_handler(llm):
    """Run the real output handler on engine outputs the test feeds it."""
    outputs = asyncio.Queue()
    polls = [0]

    async def get_output_async():
        polls[0] += 1
        return await outputs.get()

    llm.engine_core.get_output_async = get_output_async
    llm.log_stats = False
    llm.logger_manager = None
    llm.renderer.mm_processor_cache = None
    llm.renderer.stat_mm_cache = lambda: None
    AsyncLLM._run_output_handler(llm)
    return outputs, polls


async def stop_output_handler(llm):
    llm.output_handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await llm.output_handler


def test_output_handler_starts_on_a_fast_path_engine():
    # Starting the handler defines the cache-miss retry coroutine, and Python
    # evaluates its annotations then: a name the module does not import makes
    # every server start fail.
    async def run(executor):
        llm = make_mm_llm(executor)
        start_output_handler(llm)
        try:
            await asyncio.sleep(0)
            assert not llm.output_handler.done()
        finally:
            await stop_output_handler(llm)

    run_with_mm_executor(run)


def test_abort_during_mm_retry_materialize_does_not_resubmit():
    async def run(executor):
        llm = make_mm_llm(executor)
        llm._p1_cached_mm_identifiers.add("image-1")
        materializer = Materializer(released=False)
        request = make_mm_request(materializer)
        queue = await llm.add_request(
            request.request_id, request, request.sampling_params
        )
        request_id = queue.request_id
        assert request.mm_features[0].data is None
        assert request_id in llm._pending_mm_retries

        outputs, polls = start_output_handler(llm)
        try:
            # P1 has evicted the image from both caches: a retryable miss.
            outputs.put_nowait(
                EngineCoreOutputs(
                    outputs=[
                        EngineCoreOutput(
                            request_id=request_id,
                            new_token_ids=[],
                            finish_reason=FinishReason.ERROR,
                            mm_cache_miss_hashes=["image-1"],
                        )
                    ],
                    finished_requests={request_id},
                )
            )
            try:
                assert await asyncio.to_thread(materializer.started.wait, 2)
                # The client disconnects while the retry materializes the image.
                await llm.abort(request_id, internal=True)
            finally:
                materializer.release.set()
            await drain(executor)
            # The handler has finished the batch once it asks for the next one.
            await until(lambda: polls[0] == 2)
            assert llm.engine_core.submitted == [request_id]
            assert "image-1" not in llm._p1_cached_mm_identifiers
            assert not llm._pending_mm_retries
            assert_empty(llm)
            assert not llm.output_handler.done()
        finally:
            await stop_output_handler(llm)

    run_with_mm_executor(run)
