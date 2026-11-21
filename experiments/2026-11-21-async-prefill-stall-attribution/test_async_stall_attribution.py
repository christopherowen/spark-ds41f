
import functools
import json
from types import SimpleNamespace
import pytest
import vllm.v1.core.sched.scheduler as sched_mod

_OPT_CONFIG = {
    "architectures": ["OPTForCausalLM"],
    "model_type": "opt",
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "ffn_dim": 3072,
    "max_position_embeddings": 8192,
    "vocab_size": 50272,
    "word_embed_proj_dim": 768,
    "torch_dtype": "float16",
    "do_layer_norm_before": True,
    "pad_token_id": 1,
    "bos_token_id": 2,
    "eos_token_id": 2,
}

BUDGET = 1024
SEQS = 8
THRESHOLD = BUDGET - SEQS


def _async_scheduler(monkeypatch, tmp_path):
    utils = pytest.importorskip("tests.v1.core.utils")
    monkeypatch.setattr(
        utils,
        "SchedulerConfig",
        functools.partial(
            utils.SchedulerConfig,
            align_prefill_chunks=True,
            prefill_yield="short",
            prefill_yield_limit_s=10.0,
                    ),
    )
    (tmp_path / "config.json").write_text(json.dumps(_OPT_CONFIG))
    scheduler = utils.create_scheduler(
        model=str(tmp_path),
        max_num_seqs=SEQS,
        max_num_batched_tokens=BUDGET,
        long_prefill_token_threshold=THRESHOLD,
        max_model_len=8192,
        skip_tokenizer_init=True,
        device="cpu",
        async_scheduling=True,
    )
    return utils, scheduler


def test_async_inflight_prefill_does_not_mistake_55ms_decode_wait_for_chunk_time(
    monkeypatch, tmp_path
):
    clock = [100.0]
    monkeypatch.setattr(sched_mod.time, "monotonic", lambda: clock[0])
    utils, scheduler = _async_scheduler(monkeypatch, tmp_path)

    for i in range(6):
        r = utils.create_requests(num_requests=1, num_tokens=20, req_ids=[f"d{i}"])[0]
        r.max_tokens = 100
        scheduler.add_request(r)

    # Prefill the 6 decode streams
    out0 = scheduler.schedule()
    assert len(out0.num_scheduled_tokens) == 6

    # Add a long prompt (multi-chunk)
    long_req = utils.create_requests(num_requests=1, num_tokens=4000, req_ids=["long"])[0]
    long_req.max_tokens = 10
    scheduler.add_request(long_req)

    # Step N: schedules Chunk 1 alongside the 6 decode streams.
    out1 = scheduler.schedule()
    assert out1.num_scheduled_tokens["long"] == THRESHOLD

    # Under async_scheduling (batch_queue_size=2), Step N-1 (decode, 55 ms)
    # finishes while Step N (Chunk 1, 0.91 s) is STILL IN FLIGHT on the GPU!
    # So schedule() for Step N+1 runs after only 55 ms (> _PREFILL_WALL_CLOCK_MIN_S=50 ms),
    # and update_from_output has NOT been called for out1 yet.
    clock[0] += 0.055
    out2 = scheduler.schedule()
    assert "long" not in out2.num_scheduled_tokens, (
        "Async scheduler mistook 55ms prior decode wait for in-flight prefill chunk "
        f"duration and double-queued chunk 2: {out2.num_scheduled_tokens}"
    )
    assert len(out2.num_scheduled_tokens) == 6

    # Step N+2 (2nd interleaved decode): short probe arrives now and must be admitted!
    clock[0] += 0.910
    probe = utils.create_requests(num_requests=1, num_tokens=500, req_ids=["probe"])[0]
    probe.max_tokens = 16
    scheduler.add_request(probe)
    out3 = scheduler.schedule()
    assert "long" not in out3.num_scheduled_tokens
    assert out3.num_scheduled_tokens.get("probe") == 500

    # Step N+3: after K=2 interleaved steps, Chunk 2 resumes!
    clock[0] += 0.150
    out4 = scheduler.schedule()
    assert out4.num_scheduled_tokens["long"] == THRESHOLD


def test_async_completed_fast_prefill_still_uses_wall_clock(monkeypatch, tmp_path):
    """When the prefill step HAS completed before the next schedule() (e.g.
    batch queue drained), wall_dt is its real duration and is respected."""
    clock = [100.0]
    monkeypatch.setattr(sched_mod.time, "monotonic", lambda: clock[0])
    utils, scheduler = _async_scheduler(monkeypatch, tmp_path)

    dec = utils.create_requests(num_requests=1, num_tokens=20, req_ids=["dec"])[0]
    dec.max_tokens = 100
    scheduler.add_request(dec)
    scheduler.schedule()

    long_req = utils.create_requests(num_requests=1, num_tokens=4000, req_ids=["long"])[0]
    long_req.max_tokens = 10
    scheduler.add_request(long_req)

    out1 = scheduler.schedule()
    assert out1.num_scheduled_tokens["long"] == THRESHOLD
    # Mark out1 completed (simulating update_from_output on out1)
    assert scheduler._last_prefill_output_id == id(out1)
    scheduler._last_prefill_output_id = None

    # Actual fast chunk took 0.45 s (< 1.0 s) -> no interleave yet
    clock[0] += 0.45
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens["long"] == THRESHOLD
