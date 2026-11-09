#!/usr/bin/env python3
"""Screen remaining-token (max_tokens) decode-row and copy-placeholder clamping
on GB10 GPU (spark-ds41f:1107)."""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

from vllm.config.speculative import CopyDraftsConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)

BLOCK = 5
LOOKAHEAD = 16

_OPT_CONFIG = {
    "architectures": ["OPTForCausalLM"],
    "model_type": "opt",
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "ffn_dim": 3072,
    "max_position_embeddings": 2048,
    "vocab_size": 50272,
    "word_embed_proj_dim": 768,
    "torch_dtype": "float16",
    "do_layer_norm_before": True,
    "pad_token_id": 1,
    "bos_token_id": 2,
    "eos_token_id": 2,
}


def _make_async_scheduler(*, copy_drafts: bool, max_num_seqs: int = 4):
    from tests.v1.core import utils

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    orig_async_cls = utils.AsyncScheduler

    def scheduler_cls(*args, vllm_config, **kwargs):
        if copy_drafts:
            vllm_config.speculative_config.copy_drafts = CopyDraftsConfig(
                max_drafts=LOOKAHEAD
            )
        return orig_async_cls(*args, vllm_config=vllm_config, **kwargs)

    tmp_dir = Path(tempfile.mkdtemp(prefix="rem-token-cap-"))
    (tmp_dir / "config.json").write_text(json.dumps(_OPT_CONFIG))
    utils.AsyncScheduler = scheduler_cls
    try:
        scheduler = utils.create_scheduler(
            model=str(tmp_dir),
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=1024,
            num_speculative_tokens=BLOCK,
            speculative_method="ngram_gpu",
            async_scheduling=True,
            skip_tokenizer_init=True,
            device="cpu",
            use_v2_model_runner=True,
        )
    finally:
        utils.AsyncScheduler = orig_async_cls
    return utils, scheduler


def screen_scheduler_and_copy_envelope() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    # 1. Single-stream copying request tail row reduction (rem = 2, 3, 4, 8, 12)
    for rem in (2, 3, 4, 8, 12):
        utils, sched_clamped = _make_async_scheduler(copy_drafts=True, max_num_seqs=4)
        _, sched_unclamped = _make_async_scheduler(copy_drafts=True, max_num_seqs=4)
        sched_unclamped.enable_adaptive_verification = False

        req_c = utils.create_requests(num_requests=1, num_tokens=20, max_tokens=20 + rem)[0]
        req_u = utils.create_requests(num_requests=1, num_tokens=20, max_tokens=20 + rem)[0]

        for sched, req in ((sched_clamped, req_c), (sched_unclamped, req_u)):
            sched.add_request(req)
            s_prefill = sched.schedule()
            s_decode1 = sched.schedule()
            # Prefill completes (1 token), report copy_draft_alloc=LOOKAHEAD
            sched.update_from_output(
                s_prefill,
                ModelRunnerOutput(
                    req_ids=[req.request_id],
                    req_id_to_index={req.request_id: 0},
                    sampled_token_ids=[[7]],
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                    copy_draft_alloc=[LOOKAHEAD],
                ),
            )
            # Decode1 accepts 18 tokens so total generated = 19, and in-flight
            # step 2 guarantees 1 more token -> 20 generated, leaving `rem` tokens.
            sched.n_step = s_decode1
        # Advance req_c and req_u to exactly `rem` tokens left after the in-flight step
        for sched, req in ((sched_clamped, req_c), (sched_unclamped, req_u)):
            # Set state: 19 output tokens committed, 1 step in flight with 6 placeholders
            req._output_token_ids.extend([7] * 18)
            req._all_token_ids.extend([7] * 18)
            req.num_computed_tokens = req.num_prompt_tokens + 19 - 1 + (1 + BLOCK)
            req.num_output_placeholders = 1 + BLOCK
            req.copy_draft_alloc = LOOKAHEAD
            req.spec_token_ids = [-1] * BLOCK

        so_c = sched_clamped.schedule()
        so_u = sched_unclamped.schedule()
        rows_c = so_c.num_scheduled_tokens[req_c.request_id]
        rows_u = so_u.num_scheduled_tokens[req_u.request_id]
        records.append(
            {
                "case": f"single_stream_copy_tail_rem_{rem}",
                "remaining_tokens": rem,
                "unclamped_rows": int(rows_u),
                "clamped_rows": int(rows_c),
                "wasted_rows_saved": int(rows_u - rows_c),
            }
        )

    # 2. Multi-stream CopyDrafter envelope sharing (4 running requests, r0 near max_tokens)
    utils, sched_c = _make_async_scheduler(copy_drafts=True, max_num_seqs=4)
    _, sched_u = _make_async_scheduler(copy_drafts=True, max_num_seqs=4)
    sched_u.enable_adaptive_verification = False

    reqs_c = utils.create_requests(num_requests=4, num_tokens=20, max_tokens=500)
    reqs_u = utils.create_requests(num_requests=4, num_tokens=20, max_tokens=500)
    reqs_c[0].max_tokens = 22
    reqs_u[0].max_tokens = 22

    for sched, reqs in ((sched_c, reqs_c), (sched_u, reqs_u)):
        for r in reqs:
            sched.add_request(r)
        s0 = sched.schedule()
        _ = sched.schedule()
        rids = list(s0.num_scheduled_tokens)
        sched.update_from_output(
            s0,
            ModelRunnerOutput(
                req_ids=rids,
                req_id_to_index={rid: i for i, rid in enumerate(rids)},
                sampled_token_ids=[[7] for _ in rids],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
                copy_draft_alloc=[LOOKAHEAD for _ in rids],
            ),
        )
        # Put r0 at 19 committed output tokens + 1 in-flight step -> 2 tokens remaining
        r0 = reqs[0]
        r0._output_token_ids.extend([7] * 18)
        r0._all_token_ids.extend([7] * 18)
        r0.num_computed_tokens = r0.num_prompt_tokens + 19 - 1 + (1 + BLOCK)
        r0.num_output_placeholders = 1 + BLOCK

    so_batch_c = sched_c.schedule()
    so_batch_u = sched_u.schedule()
    r0_id = reqs_c[0].request_id
    r1_id = reqs_c[1].request_id
    records.append(
        {
            "case": "multi_stream_copy_envelope_reclaim_c4",
            "unclamped_r0_drafts": len(so_batch_u.scheduled_spec_decode_tokens[r0_id]),
            "unclamped_r1_drafts": len(so_batch_u.scheduled_spec_decode_tokens[r1_id]),
            "clamped_r0_drafts": len(so_batch_c.scheduled_spec_decode_tokens[r0_id]),
            "clamped_r1_drafts": len(so_batch_c.scheduled_spec_decode_tokens[r1_id]),
            "active_copy_drafts_gained": (
                len(so_batch_c.scheduled_spec_decode_tokens[r1_id])
                - len(so_batch_u.scheduled_spec_decode_tokens[r1_id])
            ),
        }
    )
    return records


def screen_gpu_adaptive_budget_and_bit_exactness() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not torch.cuda.is_available():
        return records
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # 3. Real GPU AdaptiveVerificationManager + TileLang assign_draft_budget on c=8
    num_reqs = 8
    num_steps = 5
    req_ids = [f"req_{i}" for i in range(num_reqs)]
    conf = np.zeros((num_reqs, num_steps), dtype=np.float32)
    # r0, r1 have high confidence (0.92) but only 1 draft token remaining before max_tokens
    conf[0] = [0.92, 0.92, 0.92, 0.92, 0.92]
    conf[1] = [0.92, 0.92, 0.92, 0.92, 0.92]
    # r2..r7 are mid-generation with 0.78 per-step confidence
    for i in range(2, num_reqs):
        conf[i] = [0.78, 0.78, 0.78, 0.78, 0.78]

    def run_gpu_budget(drafts_map: dict[str, list[int]], tokens_map: dict[str, int]):
        state = SimpleNamespace(
            device=device,
            max_num_reqs=num_reqs,
            num_speculative_steps=num_steps,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            num_computed_tokens_np=np.full(num_reqs, 300, dtype=np.int32),
            prefill_len=SimpleNamespace(np=np.full(num_reqs, 128, dtype=np.int32)),
        )
        mgr = AdaptiveVerificationManager(
            state,
            torch.zeros(num_reqs + 1, dtype=torch.int32, device=device),
            num_bonus_tokens=1,
            max_total_logits=1 << 30,
            row_cost=0.035,
            calibrated_confidence=True,
        )
        mgr._stale_confidences[mgr._stale_idx].np[:] = conf
        gpu_conf = torch.tensor(conf, dtype=torch.float32, device=device)
        mgr._confidence_probs.copy_(gpu_conf)
        mgr._planned_probs.copy_(gpu_conf)
        mgr.load_draft_budget_kernels()

        total_toks = mgr.get_num_tokens(tokens_map, drafts_map)
        num_draft_np = np.array(
            [len(drafts_map.get(rid, ())) for rid in req_ids], dtype=np.int32
        )
        num_sched_np = np.array([tokens_map[rid] for rid in req_ids], dtype=np.int32)
        cu_logits_np = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(num_draft_np + 1, out=cu_logits_np[1:])
        mgr.compact_batch(num_draft_np, num_sched_np, cu_logits_np)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)
        cu_num_logits, _, total_drafts, _, _ = mgr.reallocate_drafts(
            req_ids, idx_mapping
        )
        torch.cuda.synchronize()
        allocated_drafts = (
            (cu_num_logits[1 : num_reqs + 1] - cu_num_logits[:num_reqs] - 1)
            .cpu()
            .tolist()
        )
        return int(total_toks), int(total_drafts), allocated_drafts

    unclamped_tokens = {rid: 6 for rid in req_ids}
    unclamped_drafts = {rid: [1, 2, 3, 4, 5] for rid in req_ids}
    clamped_tokens = {
        rid: (2 if i < 2 else 6) for i, rid in enumerate(req_ids)
    }
    clamped_drafts = {
        rid: ([1] if i < 2 else [1, 2, 3, 4, 5]) for i, rid in enumerate(req_ids)
    }

    toks_u, drafts_u, alloc_u = run_gpu_budget(unclamped_drafts, unclamped_tokens)
    toks_c, drafts_c, alloc_c = run_gpu_budget(clamped_drafts, clamped_tokens)
    # Useful draft slots are min(allocated[i], remaining_drafts[i])
    rem_cap = [1, 1, 5, 5, 5, 5, 5, 5]
    useful_u = sum(min(a, r) for a, r in zip(alloc_u, rem_cap))
    useful_c = sum(min(a, r) for a, r in zip(alloc_c, rem_cap))
    wasted_u = sum(max(0, a - r) for a, r in zip(alloc_u, rem_cap))
    wasted_c = sum(max(0, a - r) for a, r in zip(alloc_c, rem_cap))
    records.append(
        {
            "case": "gpu_adaptive_budget_c8_tail_reclaim",
            "unclamped_total_rows": toks_u,
            "clamped_total_rows": toks_c,
            "unclamped_allocated_drafts": alloc_u,
            "clamped_allocated_drafts": alloc_c,
            "unclamped_useful_drafts": useful_u,
            "clamped_useful_drafts": useful_c,
            "unclamped_wasted_drafts_past_max_tokens": wasted_u,
            "clamped_wasted_drafts_past_max_tokens": wasted_c,
        }
    )

    # 4. Bit-exactness & row savings sweep across sink_draft_zero (max_tokens = 1..32)
    from tests.v1.core import test_sink_draft_zero as tsdz

    os.environ["VLLM_SINK_DETECTOR_COLUMN"] = "2455"
    os.environ["VLLM_SINK_DETECTOR_THRESHOLD"] = "85000"
    os.environ["VLLM_SINK_REJECTION"] = "1"
    os.environ["VLLM_SINK_REJECTION_MAX_RETRIES"] = "3"

    total_rows_unclamped = 0
    total_rows_clamped = 0
    all_bit_exact = True
    for async_sched in (False, True):
        for max_toks in (1, 2, 3, 4, 5, 6, 8, 12, 16, 32):
            rows_pair = []
            emitted_pair = []
            metrics_pair = []
            for adaptive_on in (False, True):
                sched = tsdz._scheduler(async_sched, 5)
                sched.enable_adaptive_verification = adaptive_on
                req = tsdz._make_request("a", max_tokens=max_toks)
                sched.add_request(req)
                step_rows: list[int] = []
                worker = tsdz.ScriptedWorker(sched, {"a": tsdz.SAY_HI}, 5)
                run = tsdz.drive(
                    sched,
                    worker,
                    async_scheduling=async_sched,
                    on_schedule=lambda so, sr=step_rows: sr.append(
                        so.total_num_scheduled_tokens
                    ),
                )
                rows_pair.append(sum(step_rows))
                emitted_pair.append(run.emitted["a"])
                metrics_pair.append(run.finished["a"].sink_metrics)
            if emitted_pair[0] != emitted_pair[1] or metrics_pair[0] != metrics_pair[1]:
                all_bit_exact = False
            total_rows_unclamped += rows_pair[0]
            total_rows_clamped += rows_pair[1]

    records.append(
        {
            "case": "sink_draft_zero_bit_exact_sweep",
            "all_bit_exact": all_bit_exact,
            "total_rows_unclamped": total_rows_unclamped,
            "total_rows_clamped": total_rows_clamped,
            "total_rows_saved": total_rows_unclamped - total_rows_clamped,
        }
    )
    return records


def screen_gb10_61layer_step_cost() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not torch.cuda.is_available():
        return records
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    H = 7168

    # Measure 61-layer GB10 target decode forward step latency for:
    # (a) c=1 copy-draft tail step: M=17 rows (unclamped) vs M=2 rows (clamped)
    # (b) c=1 DSpark tail step: M=6 rows (unclamped) vs M=2 rows (clamped)
    # (c) c=8 decode batch with 2 tail requests: M=48 rows (unclamped) vs M=36 rows (clamped)
    cases = [
        ("c1_copy_tail_m17_vs_m2", 17, 2, 12, 4),
        ("c1_dspark_tail_m6_vs_m2", 6, 2, 8, 4),
        ("c8_batch_tail_m48_vs_m36", 48, 36, 36, 28),
    ]
    for tp in (3, 4):
        inter_tp = ((2048 + tp - 1) // tp + 63) // 64 * 64
        q_b_n = ((24576 + tp - 1) // tp + 127) // 128 * 128
        expert_bytes_per_exp = int(H * inter_tp * 3 * 9 // 16)
        w_exp = torch.randint(
            0, 255, (64, expert_bytes_per_exp // 4), dtype=torch.int32, device=device
        )
        wq = torch.randn((H, q_b_n // 2), dtype=torch.bfloat16, device=device)

        def step_fn(x: torch.Tensor, idx: torch.Tensor):
            out = x @ wq
            exp_load = w_exp.index_select(0, idx).sum()
            return out, exp_load

        for label, m_unclamped, m_clamped, exp_u, exp_c in cases:
            x_u = torch.randn((m_unclamped, H), dtype=torch.bfloat16, device=device)
            x_c = torch.randn((m_clamped, H), dtype=torch.bfloat16, device=device)
            idx_u = torch.randint(0, 64, (exp_u,), device=device)
            idx_c = torch.randint(0, 64, (exp_c,), device=device)

            for _ in range(5):
                step_fn(x_u, idx_u)
                step_fn(x_c, idx_c)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(61 * 5):
                step_fn(x_u, idx_u)
            torch.cuda.synchronize()
            unclamped_ms = (time.perf_counter() - t0) * 1000.0 / 5.0

            t0 = time.perf_counter()
            for _ in range(61 * 5):
                step_fn(x_c, idx_c)
            torch.cuda.synchronize()
            clamped_ms = (time.perf_counter() - t0) * 1000.0 / 5.0

            records.append(
                {
                    "case": f"gb10_61layer_{label}_tp{tp}",
                    "tp": tp,
                    "unclamped_rows": m_unclamped,
                    "clamped_rows": m_clamped,
                    "unclamped_ms": round(unclamped_ms, 3),
                    "clamped_ms": round(clamped_ms, 3),
                    "speedup_ratio": round(unclamped_ms / clamped_ms, 4),
                }
            )
    return records


def main() -> int:
    all_records = (
        screen_scheduler_and_copy_envelope()
        + screen_gpu_adaptive_budget_and_bit_exactness()
        + screen_gb10_61layer_step_cost()
    )
    for rec in all_records:
        print(json.dumps(rec, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
