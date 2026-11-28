# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Late admission: tell the engine core when a step reaches a fixed decoder layer.

With async scheduling the engine core decides step N+1 as soon as step N is under
way, so a request that arrives while N runs can only join N+2 and waits a whole
step for nothing. Under late admission (``VLLM_LATE_ADMISSION_LAYER``) the engine
core decides N+1 when N's GPU work reaches the start of that decoder layer. The
rest of N (the later layers, sampling and drafting) still hides the host's
preparation of N+1, and the requests that arrived meanwhile join N+1.

The signal is one CUDA event recorded at the layer by a forward pre-hook. In a
full CUDA graph the record is captured as an external event node, so every replay
records it at that point; eager steps record it as they pass. ``execute_model``
returns an asynchronous output that waits on the event, so the output rank's reply
to ``execute_model`` reaches the engine core when the step reaches the layer, in
order before the step's sampled output. No tensor is read or written: outputs
cannot change.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.v1.outputs import AsyncModelRunnerOutput


class StepReachedLayer(AsyncModelRunnerOutput):
    """The reply to ``execute_model``: ready when the step reaches the layer."""

    def __init__(self, event: torch.cuda.Event):
        self._event = event

    def get_output(self) -> None:
        self._event.synchronize()
        return None


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The target model's decoder layers (``model.layers`` of the language model)."""
    for candidate in (
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ):
        layers = getattr(candidate, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers
    raise ValueError(
        f"late admission needs the decoder layers of {type(model).__name__} "
        "(no `layers` ModuleList on the model or its language model)"
    )


class LayerProgress:
    """Records an event when the GPU reaches decoder layer ``layer``.

    Args:
        model: the target model.
        layer: index of the decoder layer whose start is the admission point.
        event_factory: builds the event (tests pass a host-side stand-in).
    """

    def __init__(self, model: nn.Module, layer: int, event_factory=None):
        layers = decoder_layers(model)
        if not 0 <= layer < len(layers):
            raise ValueError(
                f"VLLM_LATE_ADMISSION_LAYER={layer} is outside the model's "
                f"{len(layers)} decoder layers"
            )
        self.layer = layer
        self.event = (
            event_factory()
            if event_factory is not None
            else torch.cuda.Event(blocking=True, external=True)
        )
        self._handle = layers[layer].register_forward_pre_hook(self._record)

    def _record(self, module: nn.Module, args) -> None:
        self.event.record()

    def reply(self) -> StepReachedLayer:
        """The ``execute_model`` reply of a step whose forward was launched."""
        return StepReachedLayer(self.event)
