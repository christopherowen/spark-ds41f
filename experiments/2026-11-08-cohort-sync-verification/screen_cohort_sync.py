#!/usr/bin/env python3
"""Screen cohort-atomic adaptive verification, 16..40-row cost band, and idle-wakeup
burst coalescing on GB10 GPU (spark-ds41f:1107)."""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)


def make_manager(
    confidences: np.ndarray, row_cost: float = 0.035
) -> AdaptiveVerificationManager:
    num_reqs, num_steps = confidences.shape
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_steps
    manager._stale_confidences = [SimpleNamespace(np=confidences)]
    manager._stale_idx = 0
    req_ids = [f"r{i}" for i in range(num_reqs)]
    manager.req_states = SimpleNamespace(
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        num_computed_tokens_np=np.full(num_reqs, 300, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.full(num_reqs, 246, dtype=np.int32)),
    )
    manager._max_total_logits = 1 << 30
    manager.num_bonus_tokens = 1
    manager.row_cost = row_cost
    manager.dead_rows_tau = 0.20
    manager.calibrated_confidence = True
    return manager


def screen_budgets_and_phase_locking() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    req_ids_16 = [f"r{i}" for i in range(16)]
    num_tokens_16 = {rid: 6 for rid in req_ids_16}
    drafts_16 = {rid: [1, 2, 3, 4, 5] for rid in req_ids_16}

    # 1. prose-c16 medium step: unbounded (ceiling=1000) vs bounded (ceiling=40)
    prose_med = np.tile(
        np.array([[0.66, 0.48, 0.44, 0.42, 0.40]], dtype=np.float32), (16, 1)
    )
    mgr = make_manager(prose_med)
    mgr.high_row_cost_ceiling = 1000
    rows_unbounded_med = mgr.get_num_tokens(num_tokens_16, drafts_16)
    mgr.high_row_cost_ceiling = 40
    rows_bounded_med = mgr.get_num_tokens(num_tokens_16, drafts_16)
    records.append(
        {
            "case": "prose_c16_medium_step",
            "unbounded_rows": int(rows_unbounded_med),
            "bounded_rows": int(rows_bounded_med),
        }
    )

    # 2. prose-c16 high step: unbounded (ceiling=1000) vs bounded (ceiling=40)
    prose_hi = np.tile(
        np.array([[0.75, 0.62, 0.72, 0.42, 0.40]], dtype=np.float32), (16, 1)
    )
    mgr._stale_confidences = [SimpleNamespace(np=prose_hi)]
    mgr.high_row_cost_ceiling = 1000
    rows_unbounded_hi = mgr.get_num_tokens(num_tokens_16, drafts_16)
    mgr.high_row_cost_ceiling = 40
    rows_bounded_hi = mgr.get_num_tokens(num_tokens_16, drafts_16)
    records.append(
        {
            "case": "prose_c16_high_step",
            "unbounded_rows": int(rows_unbounded_hi),
            "bounded_rows": int(rows_bounded_hi),
        }
    )

    # 3. code-c16 high step: bounded preserves 3 drafts/req (64 rows)
    code_hi = np.tile(
        np.array([[0.86, 0.78, 0.74, 0.65, 0.60]], dtype=np.float32), (16, 1)
    )
    mgr._stale_confidences = [SimpleNamespace(np=code_hi)]
    rows_code_hi = mgr.get_num_tokens(num_tokens_16, drafts_16)
    records.append(
        {
            "case": "code_c16_high_step",
            "bounded_rows": int(rows_code_hi),
        }
    )

    # 4. 1+15 split wave convergence with 1-ULP BF16 even/odd noise
    step3_conf = np.ones((16, 5), dtype=np.float32)
    step3_conf[0] = [0.66, 0.54, 0.45, 0.40, 0.38]
    for i in range(1, 16):
        eps = 0.0015 if (i % 2 == 0) else 0.0
        step3_conf[i] = [0.72 + eps, 0.58 + eps, 0.48 + eps, 0.42 + eps, 0.39 + eps]
    mgr._stale_confidences = [SimpleNamespace(np=step3_conf)]
    pos = np.array([249] + [246] * 15, dtype=np.int32)
    rng = np.random.default_rng(42)
    true_accept = {
        p: int(rng.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15]))
        for p in range(240, 400)
    }
    steps_to_merge = 0
    for step in range(1, 15):
        mgr.req_states.num_computed_tokens_np = pos.copy()
        mgr.get_num_tokens(num_tokens_16, drafts_16)
        caps = [mgr._capacities_cpu_override[rid] for rid in req_ids_16]
        for i in range(16):
            pos[i] += 1 + min(caps[i], true_accept[int(pos[i])])
        if len(set(pos.tolist())) == 1:
            steps_to_merge = step
            break
    records.append(
        {
            "case": "phase_lock_1_plus_15_to_16",
            "steps_to_merge": int(steps_to_merge),
            "merged_pos": int(pos[0]),
            "unique_positions_after_merge": len(set(pos.tolist())),
        }
    )

    # 5. Idle-wakeup burst coalescing
    engine = object.__new__(EngineCoreProc)
    engine.input_queue = queue.Queue()
    engine.aborts_queue = queue.Queue()
    engine._idle_state_callbacks = []
    engine._loop_watchdog = None
    engine.process_input_queue_block = True
    engine.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=16)
    )
    admitted: list[int] = []
    engine.has_work = lambda: len(admitted) > 0
    engine.is_running = lambda: True
    engine._handle_client_request = lambda _kind, rid: admitted.append(rid)

    def feed_burst():
        for i in range(16):
            engine.input_queue.put(("add", i))
            time.sleep(0.0001)

    feeder = threading.Thread(target=feed_burst, daemon=True)
    feeder.start()
    engine._process_input_queue()
    feeder.join(timeout=1.0)
    records.append(
        {
            "case": "idle_wakeup_burst_coalesce_16",
            "admitted_count": len(admitted),
        }
    )
    return records


def screen_gpu_step_cost() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not torch.cuda.is_available():
        return records
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    H = 7168
    for tp in (4, 5, 6, 7, 8):
        inter_tp = ((2048 + tp - 1) // tp + 63) // 64 * 64
        q_b_n = ((24576 + tp - 1) // tp + 127) // 128 * 128
        expert_bytes_per_exp = int(H * inter_tp * 3 * 9 // 16)
        w_exp = torch.randint(
            0, 255, (64, expert_bytes_per_exp // 4), dtype=torch.int32, device=device
        )
        x40 = torch.randn((40, H), dtype=torch.bfloat16, device=device)
        x56 = torch.randn((56, H), dtype=torch.bfloat16, device=device)
        wq = torch.randn((H, q_b_n // 2), dtype=torch.bfloat16, device=device)
        idx_sync = torch.randint(0, 64, (14,), device=device)
        idx_shat = torch.randint(0, 64, (46,), device=device)

        def step_fn(x: torch.Tensor, idx: torch.Tensor):
            out = x @ wq
            exp_load = w_exp.index_select(0, idx).sum()
            return out, exp_load

        for _ in range(5):
            step_fn(x40, idx_sync)
            step_fn(x56, idx_shat)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(61 * 3):
            step_fn(x56, idx_shat)
        torch.cuda.synchronize()
        shat_ms = (time.perf_counter() - t0) * 1000.0 / 3.0

        t0 = time.perf_counter()
        for _ in range(61 * 3):
            step_fn(x40, idx_sync)
        torch.cuda.synchronize()
        sync_ms = (time.perf_counter() - t0) * 1000.0 / 3.0

        records.append(
            {
                "case": f"gb10_61layer_proxy_tp{tp}",
                "tp": tp,
                "inter_tp": inter_tp,
                "shattered_m56_46exp_ms": round(shat_ms, 3),
                "synced_m40_14exp_ms": round(sync_ms, 3),
                "speedup_ratio": round(shat_ms / sync_ms, 4),
            }
        )
    return records


def main() -> int:
    for rec in screen_budgets_and_phase_locking() + screen_gpu_step_cost():
        print(json.dumps(rec, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
