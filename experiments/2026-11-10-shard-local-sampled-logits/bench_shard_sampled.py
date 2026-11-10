import time
import torch

from vllm.v1.worker.gpu.spec_decode.tilelang_rejection import (
    RESAMPLE_BLOCK,
    STATS_BLOCK,
    verify_and_resample,
    verify_and_resample_from_shards,
    resample_from_shards,
)
from vllm.v1.worker.gpu.spec_decode.tilelang_resampled import insert_resampled
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

DEVICE = torch.device("cuda:0")
VOCAB_SIZE = 129280
SPEC_STEPS = 5


def bench_shape(tp_size: int, num_reqs: int, warmup: int = 20, iters: int = 200):
    tokens_per_req = SPEC_STEPS + 1
    num_logits = num_reqs * tokens_per_req
    base_shard = -(-VOCAB_SIZE // tp_size)
    padded_stride = ((base_shard + 63) // 64) * 64
    org_vocab = min(VOCAB_SIZE, base_shard)

    torch.manual_seed(42)
    full_logits = torch.randn(num_logits, VOCAB_SIZE, dtype=torch.bfloat16, device=DEVICE)
    shard_buf = torch.randn(num_logits, padded_stride, dtype=torch.bfloat16, device=DEVICE)
    shard_logits = shard_buf[:, :org_vocab]

    drafts = torch.randint(0, VOCAB_SIZE, (num_logits,), dtype=torch.int32, device=DEVICE)
    cu = torch.arange(0, (num_reqs + 1) * tokens_per_req, tokens_per_req, dtype=torch.int32, device=DEVICE)
    pos = torch.arange(num_logits, dtype=torch.int64, device=DEVICE)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE).repeat_interleave(tokens_per_req)
    expanded_local_pos = torch.arange(tokens_per_req, dtype=torch.int32, device=DEVICE).repeat(num_reqs)
    temperature = torch.full((num_reqs,), 1.0, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(1, num_reqs + 1, dtype=torch.int64, device=DEVICE)

    local_blocks = -(-padded_stride // STATS_BLOCK)
    resample_blocks = -(-padded_stride // RESAMPLE_BLOCK)
    local_stats = torch.empty(num_logits, 4 * local_blocks, dtype=torch.float32, device=DEVICE)
    gathered_stats_buf = torch.empty(num_logits, 4 * local_blocks * tp_size, dtype=torch.float32, device=DEVICE)
    loc_argmax = torch.empty(num_reqs, resample_blocks, dtype=torch.int64, device=DEVICE)
    loc_max = torch.empty(num_reqs, resample_blocks, dtype=torch.float32, device=DEVICE)
    local_pairs = torch.empty(num_reqs, 4, dtype=torch.float32, device=DEVICE)
    gathered_pairs_buf = torch.empty(num_reqs, 4 * tp_size, dtype=torch.float32, device=DEVICE)
    sampled_out = torch.empty(num_reqs, tokens_per_req, dtype=torch.int64, device=DEVICE)
    num_sampled_out = torch.empty(num_reqs, dtype=torch.int32, device=DEVICE)

    def mock_ag(t: torch.Tensor) -> torch.Tensor:
        if t.shape[1] == 4 * local_blocks:
            for r in range(tp_size):
                gathered_stats_buf[:, r * 4 * local_blocks : (r + 1) * 4 * local_blocks].copy_(t)
            return gathered_stats_buf
        for r in range(tp_size):
            gathered_pairs_buf[:, r * 4 : (r + 1) * 4].copy_(t)
        return gathered_pairs_buf

    def run_full():
        # Simulate local_logits -> full_logits gather copy + full-vocab verify_and_resample + insert_resampled
        for r in range(tp_size):
            start = r * base_shard
            end = min(VOCAB_SIZE, (r + 1) * base_shard)
            full_logits[:, start:end].copy_(shard_logits[:, : end - start])
        s, ns, la, lm = verify_and_resample(
            full_logits,
            drafts,
            cu,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seed,
            SPEC_STEPS,
            apply_temperature=False,
        )
        insert_resampled(s, ns, la, lm, cu, expanded_idx_mapping, temperature)
        return s, ns

    def run_shard():
        return verify_and_resample_from_shards(
            target_logits=shard_logits,
            draft_sampled=drafts,
            cu_num_logits=cu,
            pos=pos,
            idx_mapping=idx_mapping,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            temperature=temperature,
            seed=seed,
            num_speculative_steps=SPEC_STEPS,
            vocab_start=0,
            vocab_size=VOCAB_SIZE,
            tp_size=tp_size,
            all_gather_fn=mock_ag,
            apply_temperature=False,
            local_stats=local_stats,
            resampled_local_argmax=loc_argmax,
            resampled_local_max=loc_max,
            local_pairs=local_pairs,
            sampled=sampled_out,
            num_sampled=num_sampled_out,
        )

    for _ in range(warmup):
        run_full()
        run_shard()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    start_ev.record()
    for _ in range(iters):
        run_full()
    end_ev.record()
    torch.cuda.synchronize()
    full_us = (start_ev.elapsed_time(end_ev) * 1000.0) / iters

    start_ev.record()
    for _ in range(iters):
        run_shard()
    end_ev.record()
    torch.cuda.synchronize()
    shard_us = (start_ev.elapsed_time(end_ev) * 1000.0) / iters

    full_ag_bytes = num_logits * padded_stride * 2 * (tp_size - 1)
    shard_ag_bytes = (num_logits * 4 * local_blocks * 4 + num_reqs * 4 * 4) * (tp_size - 1)
    # RoCE wire time at 9.5 GB/s per neighbour link (plus 2x8us one-shot latency vs chunked NCCL/Sparknet)
    roce_full_us = (full_ag_bytes / 9.5e9) * 1e6 + 35.0
    roce_shard_us = (shard_ag_bytes / 9.5e9) * 1e6 + 16.0

    print(
        f"TP{tp_size} reqs={num_reqs:2d} rows={num_logits:2d} | "
        f"kernel_full={full_us:6.1f}us kernel_shard={shard_us:6.1f}us | "
        f"payload: {full_ag_bytes/1024:7.1f}KB -> {shard_ag_bytes/1024:5.2f}KB "
        f"({full_ag_bytes/shard_ag_bytes:5.0f}x) | "
        f"est_step_verify: {full_us + roce_full_us:6.1f}us -> {shard_us + roce_shard_us:6.1f}us "
        f"(-{(full_us + roce_full_us) - (shard_us + roce_shard_us):5.1f}us)"
    )


if __name__ == "__main__":
    for tp in (4, 3):
        for reqs in (1, 2, 4, 8, 16):
            bench_shape(tp, reqs)
