# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DS4.1 image path's row copies against the torch they replace, bit for bit:
the span (delimiter rows at their roles, aligner rows at the IMAGE slots) and the
spans' merge into the text embeddings at the host mask's slots."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from vllm.models.deepseek_v4_1.nvidia.vl_model import (  # noqa: E402
    DeepseekV41ForCausalLM,
)
from vllm.models.deepseek_v41.common.mm_preprocess import (  # noqa: E402
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    image_token_types,
)

DEVICE = torch.device("cuda")
HIDDEN = 5120


def _bits(x):
    return x.view(torch.int16)


def _model(gen):
    """The class's image-path state: the learned delimiters (FP32) and their table."""
    vectors = [torch.randn(HIDDEN, generator=gen, device=DEVICE) for _ in range(3)]
    model = SimpleNamespace(
        image_start=vectors[0],
        image_end=vectors[1],
        image_newline=vectors[2],
        _DELIMITERS=DeepseekV41ForCausalLM._DELIMITERS,
    )
    model._image_delimiters = torch.stack([v.to(torch.bfloat16) for v in vectors])
    return model


def _torch_span(model, image_embeds, types):
    """The torch span builder the row copies replace."""
    types = types.to(image_embeds.device)
    span = image_embeds.new_empty(types.numel(), image_embeds.shape[-1])
    dtype = image_embeds.dtype
    span[types == IMAGE_START] = model.image_start.to(dtype)
    span[types == IMAGE_END] = model.image_end.to(dtype)
    span[types == IMAGE_NEW_LINE] = model.image_newline.to(dtype)
    span[types == IMAGE] = image_embeds
    return span


@pytest.mark.parametrize("grid", [(1, 1), (3, 5), (27, 27), (16, 48)])
def test_image_span_matches_masked_assignments(grid):
    gen = torch.Generator(device=DEVICE).manual_seed(grid[0] * 97 + grid[1])
    model = _model(gen)
    types = image_token_types(*grid)
    image_embeds = torch.randn(
        grid[0] * grid[1], HIDDEN, generator=gen, device=DEVICE
    ).to(torch.bfloat16)
    image_embeds[0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
    ours = DeepseekV41ForCausalLM._build_image_span(model, image_embeds, types)
    expected = _torch_span(model, image_embeds, types)
    assert ours.shape == expected.shape
    assert torch.equal(_bits(ours), _bits(expected))


def test_image_span_refuses_device_roles_and_miscounts():
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    model = _model(gen)
    types = image_token_types(2, 3)
    embeds = torch.zeros(6, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(ValueError, match="host data"):
        DeepseekV41ForCausalLM._build_image_span(model, embeds, types.to(DEVICE))
    with pytest.raises(ValueError, match="image slots"):
        DeepseekV41ForCausalLM._build_image_span(model, embeds[:5], types)


@pytest.mark.parametrize("tokens", [17, 4096])
def test_merge_matches_index_put(tokens):
    """Two spans (one cut short, as a prefill chunk cuts an image) scattered at the
    host mask's slots, against torch's index_put of the flattened spans."""
    gen = torch.Generator(device=DEVICE).manual_seed(tokens)
    text = torch.randn(tokens, HIDDEN, generator=gen, device=DEVICE).to(torch.bfloat16)
    mask = torch.zeros(tokens, dtype=torch.bool, pin_memory=True)
    slots = torch.randperm(tokens, generator=torch.Generator().manual_seed(tokens))
    count = min(tokens - 1, 600)
    mask[slots[:count]] = True
    spans = [
        torch.randn(n, HIDDEN, generator=gen, device=DEVICE).to(torch.bfloat16)
        for n in (count // 3, count - count // 3)
    ]
    expected = text.clone()
    expected[mask.to(DEVICE)] = torch.cat(spans)
    model = SimpleNamespace(
        _embed_text_input_ids=lambda ids, embed, is_multimodal: text.clone(),
        get_language_model=lambda: SimpleNamespace(embed_input_ids=None),
    )
    ours = DeepseekV41ForCausalLM.embed_input_ids(
        model,
        torch.zeros(tokens, dtype=torch.int32, device=DEVICE),
        spans,
        is_multimodal=mask,
    )
    assert torch.equal(_bits(ours), _bits(expected))
    with pytest.raises(ValueError, match="placeholders"):
        DeepseekV41ForCausalLM.embed_input_ids(
            model,
            torch.zeros(tokens, dtype=torch.int32, device=DEVICE),
            spans[:1],
            is_multimodal=mask,
        )
