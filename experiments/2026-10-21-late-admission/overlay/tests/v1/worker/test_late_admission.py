# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Late admission on the worker: the step records an event when it reaches the
admission layer, and its execute_model reply waits for that event."""

import time
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.worker.gpu.late_admission import LayerProgress, StepReachedLayer


class FakeEvent:
    def __init__(self):
        self.records = 0
        self.synchronized = 0

    def record(self):
        self.records += 1

    def synchronize(self):
        self.synchronized += 1


class Layer(nn.Module):
    def __init__(self, trace: list, index: int):
        super().__init__()
        self.trace, self.index = trace, index

    def forward(self, x):
        self.trace.append(self.index)
        return x + 1


class Inner(nn.Module):
    def __init__(self, trace: list, n: int):
        super().__init__()
        self.layers = nn.ModuleList(Layer(trace, i) for i in range(n))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CausalLM(nn.Module):
    def __init__(self, trace: list, n: int = 4):
        super().__init__()
        self.model = Inner(trace, n)

    def forward(self, x):
        return self.model(x)


class VisionLanguage(nn.Module):
    """The language model under ``language_model``, as DS4.1's vision wrapper."""

    def __init__(self, trace: list, n: int = 4):
        super().__init__()
        self.language_model = CausalLM(trace, n)

    def forward(self, x):
        return self.language_model(x)


@pytest.mark.cpu_test
@pytest.mark.parametrize("wrapper", [CausalLM, VisionLanguage])
def test_the_event_is_recorded_as_the_layer_starts(wrapper):
    trace: list = []
    model = wrapper(trace)
    event = FakeEvent()
    marks: list = []
    event.record = lambda: marks.append(len(trace))  # layers run so far
    progress = LayerProgress(model, 2, event_factory=lambda: event)
    model(torch.zeros(1))
    model(torch.zeros(1))
    assert trace == [0, 1, 2, 3, 0, 1, 2, 3]
    assert marks == [2, 6]  # before layer 2 of each forward
    assert progress.layer == 2


@pytest.mark.cpu_test
def test_a_layer_outside_the_model_or_no_layers_is_refused():
    with pytest.raises(ValueError, match="outside the model's 4 decoder layers"):
        LayerProgress(CausalLM([]), 4, event_factory=FakeEvent)
    with pytest.raises(ValueError, match="no `layers` ModuleList"):
        LayerProgress(nn.Linear(2, 2), 0, event_factory=FakeEvent)


@pytest.mark.cpu_test
def test_the_reply_waits_for_the_event_and_carries_nothing():
    event = FakeEvent()
    reply = LayerProgress(CausalLM([]), 1, event_factory=lambda: event).reply()
    assert isinstance(reply, AsyncModelRunnerOutput)
    assert reply.get_output() is None
    assert event.synchronized == 1


@pytest.mark.cpu_test
def test_the_worker_sends_the_reply_once_the_event_completes():
    """WorkerProc's output thread resolves the reply before enqueueing it, so the
    engine core receives it when the step reaches the layer."""
    event = FakeEvent()
    worker = object.__new__(WorkerProc)
    worker.worker_response_mq = Mock()
    worker.enqueue_output(StepReachedLayer(event))
    assert event.synchronized == 1
    worker.worker_response_mq.enqueue.assert_called_once_with(
        (WorkerProc.ResponseStatus.SUCCESS, None)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_captured_graph_records_the_event_at_the_layer_on_every_replay():
    """The record is captured as an external event node: after each replay the
    reply is ready while the layers after the admission point still run."""
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)

    class Slow(nn.Module):
        def forward(self, x):
            for _ in range(40):
                x = torch.mm(x, a) * 1e-3
            return x

    model = CausalLM([])
    model.model.layers = nn.ModuleList(Slow() for _ in range(4))
    progress = LayerProgress(model, 1)
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        model(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        model(x)
    torch.cuda.synchronize()
    for _ in range(3):
        start = time.perf_counter()
        graph.replay()
        progress.reply().get_output()
        reached = time.perf_counter() - start
        still_running = not torch.cuda.current_stream().query()
        torch.cuda.synchronize()
        total = time.perf_counter() - start
        assert still_running
        # Layer 1 starts a quarter of the way through four equal layers.
        assert 0.1 * total < reached < 0.5 * total
