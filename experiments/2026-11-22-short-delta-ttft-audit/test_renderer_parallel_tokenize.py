import concurrent.futures
import pytest
from types import SimpleNamespace
from vllm.tokenizers import deepseek_v41 as dsv41_tok


def test_subchunk_cache_reuses_prefix_subchunks_bit_exact():
    calls = []

    def encode_batch(texts):
        calls.append(list(texts))
        return [[ord(c) for c in t[:4]] + [len(t)] for t in texts]

    cache = dsv41_tok._BoundedPieceCache()
    # Build >65536 char text with unique lines and BPE-invariant newline breaks
    prefix = "".join(f"line_{i:04d} alpha beta gamma delta epsilon zeta\n" for i in range(1600))
    p1 = prefix + "tail_one\n"
    p2 = prefix + "tail_two_longer\n"

    r1 = dsv41_tok._resolve_cached_pieces([p1], cache, encode_batch)
    first_batch_count = len(calls[0])
    assert first_batch_count >= 4

    r2 = dsv41_tok._resolve_cached_pieces([p2], cache, encode_batch)
    # Only the final changed subchunk should be encoded on the second call!
    assert len(calls) == 2
    assert len(calls[1]) == 1

    # Verify exact identity against a fresh uncached run
    fresh_cache = dsv41_tok._BoundedPieceCache()
    r2_fresh = dsv41_tok._resolve_cached_pieces([p2], fresh_cache, encode_batch)
    assert r2[p2] == r2_fresh[p2]


def test_deepseek_v4_renderer_initializes_thread_pool_when_workers_gt_1(monkeypatch):
    from vllm.renderers import deepseek_v4 as dsv4_rend

    pooled = []
    monkeypatch.setattr(
        dsv4_rend,
        "maybe_make_thread_pool",
        lambda tok, copies: pooled.append((tok, copies)) or tok,
    )
    monkeypatch.setattr(
        dsv4_rend.BaseRenderer,
        "__init__",
        lambda self, cfg, tok: (
            setattr(self, "tokenizer", tok),
            setattr(self, "_executor", None),
        ) and None,
    )
    tok = object()
    cfg1 = SimpleNamespace(model_config=SimpleNamespace(renderer_num_workers=1))
    dsv4_rend.DeepseekV4Renderer(cfg1, tok)
    assert pooled == []

    cfg8 = SimpleNamespace(model_config=SimpleNamespace(renderer_num_workers=8))
    dsv4_rend.DeepseekV4Renderer(cfg8, tok)
    assert pooled == [(tok, 9)]
