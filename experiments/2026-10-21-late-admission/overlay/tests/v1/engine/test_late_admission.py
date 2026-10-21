# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Late admission: with async scheduling a request that arrives while a step runs
joins the next step, not the one after it.

The engine core decides step N+1 when step N's execute_model reply arrives (under
late admission, when N's GPU work reaches VLLM_LATE_ADMISSION_LAYER), after taking
the requests that arrived meanwhile. Without it, N+1 is decided as soon as N is under
way and an arrival during N can only join N+2.
"""

import queue
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.engine.core import EngineCore, EngineCoreProc, late_admission_enabled
from vllm.v1.outputs import ModelRunnerOutput

from ..core.utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


class ReachedLayer(Future):
    """An execute_model reply; reading it means the step reached the layer."""

    def __init__(self, log: list, batch: int):
        super().__init__()
        self.log, self.batch = log, batch

    def result(self, timeout=None):
        if not self.done():
            self.log.append(("reached layer", self.batch))
            self.set_result(None)
        return super().result()


class FakeExecutor:
    """Records each batch's requests; every request samples one token a step."""

    def __init__(self, log: list):
        self.log = log
        self.batches: list[list[str]] = []
        self.pending = None

    def execute_model(self, scheduler_output, non_block=False):
        self.batches.append(sorted(scheduler_output.num_scheduled_tokens))
        self.log.append(("execute", len(self.batches)))
        self.pending = scheduler_output
        return ReachedLayer(self.log, len(self.batches))

    def sample_tokens(self, grammar_output, non_block=False):
        req_ids = list(self.pending.num_scheduled_tokens)
        future: Future = Future()
        future.set_result(
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
                sampled_token_ids=[[7] for _ in req_ids],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )
        )
        return future


@pytest.fixture
def opt_model_path(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"architectures":["OPTForCausalLM"],"model_type":"opt",'
        '"max_position_embeddings":32768}'
    )
    return str(tmp_path)


def _engine(model: str, late_admission: bool):
    log: list = []
    arrivals: list = []
    scheduler = create_scheduler(
        model=model,
        skip_tokenizer_init=True,
        device="cpu",
        async_scheduling=True,
        max_model_len=256,
    )
    engine = object.__new__(EngineCore)
    engine.scheduler = scheduler
    engine.model_executor = FakeExecutor(log)
    engine.batch_queue_size = 2
    engine.batch_queue = deque(maxlen=2)
    engine.late_admission = late_admission
    engine.is_ec_consumer = True
    engine.is_pooling_model = False
    engine.check_for_draft_tokens = False
    engine._last_model_completion_time = None
    engine._should_throttle_prefills = Mock(return_value=False)
    engine.log_error_detail = lambda _: nullcontext()
    engine.capture_iteration_details = lambda _: nullcontext()
    engine._wait_for_boundary_checkpoint_copies = Mock()
    engine._process_aborts_queue = Mock()
    engine._attach_iteration_details = Mock()

    def admit():
        log.append(("admit", [r.request_id for r in arrivals]))
        for request in arrivals:
            scheduler.add_request(request)
        arrivals.clear()

    engine._admit_arrivals = admit
    return engine, log, arrivals


@pytest.mark.parametrize("late_admission", [True, False])
def test_an_arrival_joins_the_next_step(opt_model_path, late_admission):
    engine, log, arrivals = _engine(opt_model_path, late_admission)
    first, second = create_requests(2, num_tokens=8, max_tokens=8, req_ids=["a", "b"])
    engine.scheduler.add_request(first)

    # Idle engine: the first step is decided at once, nothing to wait for.
    assert engine.step_with_batch_queue() == (None, True)
    assert log == [("execute", 1)]

    # "b" arrives while step 1 runs.
    arrivals.append(second)
    engine.step_with_batch_queue()
    batches = engine.model_executor.batches
    if late_admission:
        # Step 2 is decided once step 1 reaches the layer, with "b" in it.
        assert log[1:4] == [("reached layer", 1), ("admit", ["b"]), ("execute", 2)]
        assert batches[1] == ["a", "b"]
    else:
        # Step 2 was decided without waiting; "b" waits for step 3.
        assert log[1] == ("execute", 2)
        assert batches[1] == ["a"]


def test_no_wait_when_nothing_is_in_flight(opt_model_path):
    engine, log, _ = _engine(opt_model_path, late_admission=True)
    engine.scheduler.add_request(create_requests(1, num_tokens=8, req_ids=["a"])[0])
    engine.step_with_batch_queue()
    assert ("reached layer", 1) not in log
    assert not any(entry[0] == "admit" for entry in log)


def test_enabled_only_with_the_batch_queue_and_no_pipeline(monkeypatch):
    config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1))
    monkeypatch.delenv("VLLM_LATE_ADMISSION_LAYER", raising=False)
    assert not late_admission_enabled(config, batch_queue=True)
    monkeypatch.setenv("VLLM_LATE_ADMISSION_LAYER", "28")
    assert late_admission_enabled(config, batch_queue=True)
    with pytest.raises(ValueError, match="async scheduling"):
        late_admission_enabled(config, batch_queue=False)
    config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="pipeline"):
        late_admission_enabled(config, batch_queue=True)


def test_the_engine_process_admits_every_queued_request_without_blocking():
    engine = object.__new__(EngineCoreProc)
    engine.input_queue = queue.Queue()
    engine.input_queue.put(("add", 1))
    engine.input_queue.put(("abort", 2))
    engine._handle_client_request = Mock()
    engine._admit_arrivals()
    assert [c.args for c in engine._handle_client_request.call_args_list] == [
        ("add", 1),
        ("abort", 2),
    ]
    engine._admit_arrivals()  # an empty queue returns at once
    assert engine._handle_client_request.call_count == 2
