# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TileLang vocabulary-parallel embedding against vLLM's CUDA kernel
(``ops.vocab_parallel_embedding``), which it replaces: the same bits for every
rank's shard, including ids another rank owns, ids no rank owns and LoRA's added
vocabulary."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.vocab_parallel_embedding import (  # noqa: E402
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    pad_vocab_size,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import embedding, prepare  # noqa: E402

# DeepSeek-V4.1: a 129280-token vocabulary of 5120-wide BF16 rows.
VOCAB, HIDDEN = 129280, 5120


def _shard_indices(vocab: int, org_vocab: int, rank: int, tp: int):
    """A rank's indices as VocabParallelEmbedding lays them out (its padding is
    ``lcm(64, tp)``)."""
    pad = math.lcm(64, tp)
    org_padded = pad_vocab_size(org_vocab, pad)
    padded = pad_vocab_size(org_padded + vocab - org_vocab, pad)
    return (
        VocabParallelEmbedding._get_indices(
            padded, org_padded, vocab, org_vocab, rank, tp
        ),
        padded // tp,
    )


def _random_bits(rows: int, width: int, dtype: torch.dtype, seed: int):
    """A table of random bit patterns: NaN payloads, infinities, subnormals and
    signed zeros included, so only a bit copy reproduces it."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    nbytes = rows * width * dtype.itemsize
    bits = torch.randint(
        0, 256, (nbytes,), generator=generator, dtype=torch.uint8, device="cuda"
    )
    return bits.view(dtype).view(rows, width)


def _ids(tokens: int, vocab: int, dtype: torch.dtype, seed: int):
    """Ids over the whole vocabulary and past it on both sides (no rank's)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(
        -8, vocab + 200, (tokens,), generator=generator, dtype=dtype, device="cuda"
    )


def _cuda(ids, weight, shard, out=None):
    return ops.vocab_parallel_embedding(
        ids,
        weight,
        shard.org_vocab_start_index,
        shard.org_vocab_end_index,
        shard.num_org_vocab_padding,
        shard.added_vocab_start_index,
        shard.added_vocab_end_index,
        out=out,
    )


def _tilelang(ids, weight, shard, out=None):
    if out is None:
        out = torch.empty(
            (ids.shape[0], weight.shape[1]), dtype=weight.dtype, device=weight.device
        )
    embedding.vocab_embedding(
        ids,
        weight,
        out,
        shard.org_vocab_start_index,
        shard.org_vocab_end_index,
        shard.num_org_vocab_padding,
        shard.added_vocab_start_index,
        shard.added_vocab_end_index,
    )
    return out


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.uint8)


@pytest.mark.parametrize("tp,rank", [(4, 0), (4, 3), (3, 1), (3, 2)])
@pytest.mark.parametrize("tokens", [1, 5, 6, 80, 96, 8192])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_serving_shard_bit_identical_to_the_cuda_kernel(tp, rank, tokens, id_dtype):
    """DS4.1's own shards (TP3's last rank has padding rows), decode and prefill
    token counts, ids from every rank and from none."""
    shard, rows = _shard_indices(VOCAB, VOCAB, rank, tp)
    weight = _random_bits(rows, HIDDEN, torch.bfloat16, seed=tp * 10 + rank)
    ids = _ids(tokens, VOCAB, id_dtype, seed=tokens)
    # The ids this rank owns, at both ends of its range, among the others.
    ends = torch.tensor(
        [shard.org_vocab_start_index, shard.org_vocab_end_index - 1], dtype=id_dtype
    )[:tokens]
    ids[: ends.numel()] = ends
    want = _cuda(ids, weight, shard)
    got = _tilelang(ids, weight, shard)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(_bits(got), _bits(want))


@pytest.mark.parametrize("tp", [2, 3, 4, 8])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_added_vocabulary_bit_identical_to_the_cuda_kernel(tp, id_dtype):
    """LoRA's added rows sit after the padded base rows of each shard."""
    vocab, org_vocab, width = 32256, 32000, 1024
    for rank in range(tp):
        shard, rows = _shard_indices(vocab, org_vocab, rank, tp)
        weight = _random_bits(rows, width, torch.bfloat16, seed=rank)
        ids = _ids(1031, vocab, id_dtype, seed=rank)
        ids[:64] = torch.randint(org_vocab, vocab, (64,), dtype=id_dtype)
        assert torch.equal(
            _bits(_tilelang(ids, weight, shard)), _bits(_cuda(ids, weight, shard))
        ), rank


@pytest.mark.parametrize(
    "dtype,width",
    [(torch.float32, 1028), (torch.float16, 2056), (torch.float8_e4m3fn, 4112)],
)
def test_every_dtype_is_a_bit_copy(dtype, width):
    """Rows copy as words: any dtype whose rows are whole 16-byte pieces, rows not a
    multiple of the block's 2 KiB included."""
    shard, rows = _shard_indices(VOCAB, VOCAB, 1, 4)
    weight = _random_bits(rows, width, dtype, seed=3)
    ids = _ids(257, VOCAB, torch.int32, seed=4)
    assert torch.equal(
        _bits(_tilelang(ids, weight, shard)), _bits(_cuda(ids, weight, shard))
    )


@pytest.mark.parametrize("tp", [3, 4])
def test_rank_sum_is_the_full_lookup(tp):
    """What the all-reduce sees: one rank's row plus the others' zeros, per token."""
    vocab, width = 4099, 512
    generator = torch.Generator(device="cuda").manual_seed(tp)
    table = torch.randn(vocab, width, generator=generator, device="cuda").bfloat16()
    ids = torch.randint(0, vocab, (333,), dtype=torch.int32, device="cuda")
    total = torch.zeros(ids.shape[0], width, dtype=torch.float32, device="cuda")
    for rank in range(tp):
        shard, rows = _shard_indices(vocab, vocab, rank, tp)
        weight = torch.zeros(rows, width, dtype=torch.bfloat16, device="cuda")
        owned = shard.org_vocab_end_index - shard.org_vocab_start_index
        weight[:owned] = table[shard.org_vocab_start_index : shard.org_vocab_end_index]
        total += _tilelang(ids, weight, shard).float()
    assert torch.equal(total.bfloat16(), table[ids.long()])


def test_writes_only_its_rows_of_a_larger_buffer():
    """Prefill SP's reduce-scatter embeds into the head of a padded buffer."""
    shard, rows = _shard_indices(VOCAB, VOCAB, 2, 4)
    weight = _random_bits(rows, HIDDEN, torch.bfloat16, seed=5)
    ids = _ids(1000, VOCAB, torch.int32, seed=6)
    buffer = torch.full((1024, HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda")
    _tilelang(ids, weight, shard, out=buffer[:1000])
    assert torch.equal(_bits(buffer[:1000]), _bits(_cuda(ids, weight, shard)))
    assert (buffer[1000:] == 7.0).all()


def test_the_method_matches_the_unquantized_method():
    """``TileLangEmbeddingMethod.parallel_embedding`` reads the layer's shard as
    the CUDA kernel's method does."""
    shard, rows = _shard_indices(VOCAB, VOCAB, 3, 4)
    layer = SimpleNamespace(
        weight=_random_bits(rows, HIDDEN, torch.bfloat16, seed=7), shard_indices=shard
    )
    ids = _ids(96, VOCAB, torch.int32, seed=8)
    got = embedding.TileLangEmbeddingMethod().parallel_embedding(layer, ids)
    want = UnquantizedEmbeddingMethod().parallel_embedding(layer, ids)
    assert torch.equal(_bits(got), _bits(want))


def test_no_tokens_launch_nothing():
    shard, rows = _shard_indices(VOCAB, VOCAB, 0, 4)
    weight = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    ids = torch.empty(0, dtype=torch.int32, device="cuda")
    assert _tilelang(ids, weight, shard).shape == (0, HIDDEN)


def test_refuses_layouts_it_cannot_copy_in_pieces():
    shard, rows = _shard_indices(VOCAB, VOCAB, 0, 4)
    ids = _ids(4, VOCAB, torch.int32, seed=9)
    narrow = torch.zeros(rows, 12, dtype=torch.bfloat16, device="cuda")  # 24 bytes
    with pytest.raises(ValueError):
        _tilelang(ids, narrow, shard)
    weight = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    shifted = torch.empty(4 * HIDDEN + 2, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):  # rows 4 bytes off a 16-byte boundary
        _tilelang(ids, weight, shard, out=shifted[2:].view(4, HIDDEN))
    with pytest.raises(ValueError):
        _tilelang(ids.float(), weight, shard)


def test_declared_as_the_lookup_calls_it():
    registry = JitWarmupRegistry(vllm_config=None)
    layer = SimpleNamespace(weight=torch.empty(16, HIDDEN, dtype=torch.bfloat16))
    with registry.activate():
        embedding.TileLangEmbeddingMethod().declare_layer_kernels(layer)
    calls = list(prepare.declared_calls(registry))
    words = HIDDEN * 2 // 4
    full = -(-(words // 4) // 128)
    assert calls == [
        (embedding.vocab_embedding_kernel, (words, d), {"per_thread": p})
        for d in ("int32", "int64")
        for p in (1, full)
    ]
