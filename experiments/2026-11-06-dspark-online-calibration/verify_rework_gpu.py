import json
import time
from types import SimpleNamespace
import numpy as np
import torch

from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    OnlineAcceptanceEstimator,
    _refit_kernel,
    triton,
)
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)


def make_manager(
    confidences: np.ndarray,
    row_cost: float = 0.035,
    dead_rows_tau: float = 0.20,
    calibrated: bool = True,
    knee_scale: float = 0.55,
) -> AdaptiveVerificationManager:
    num_reqs, num_steps = confidences.shape
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_steps
    manager.num_draft_steps = num_steps
    manager.step_steps = num_steps
    manager._stale_confidences = [SimpleNamespace(np=confidences.copy())]
    manager._stale_idx = 0
    manager.req_states = SimpleNamespace(
        req_id_to_index={f"r{i}": i for i in range(num_reqs)},
        num_computed_tokens_np=np.ones(num_reqs, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int32)),
    )
    manager._max_total_logits = 1 << 30
    manager.num_bonus_tokens = 1
    manager.row_cost = row_cost
    manager.high_row_cost_knee = 16
    manager.high_row_cost_scale = knee_scale
    manager.dead_rows_tau = dead_rows_tau
    manager.calibrated_confidence = calibrated
    manager._has_admitted_drafts = False
    return manager


def bench_kernels(device: torch.device) -> dict:
    num_steps = 5
    max_reqs = 64
    estimator = OnlineAcceptanceEstimator.for_confidence_head(
        max_reqs, num_steps, device
    )
    estimator.REFIT_INTERVAL = 1_000_000  # benchmark accumulate separately

    results = {}
    for num_reqs in (1, 8, 16):
        probs = torch.full((num_reqs, num_steps), 0.75, device=device, dtype=torch.float32)
        slots = torch.arange(num_reqs, device=device, dtype=torch.int32)
        idx_map = slots.repeat_interleave(num_steps)
        num_sampled = torch.full((num_reqs,), 3, device=device, dtype=torch.int32)
        num_rejected = torch.full((num_reqs,), 2, device=device, dtype=torch.int32)

        for _ in range(100):
            estimator.calibrate_confidence_probs(probs, idx_map)
            estimator.step(slots, num_sampled, num_rejected)
        torch.cuda.synchronize(device)

        iters = 2000
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            estimator.calibrate_confidence_probs(probs, idx_map)
        end.record()
        torch.cuda.synchronize(device)
        cal_us = (start.elapsed_time(end) * 1000.0) / iters

        start.record()
        for _ in range(iters):
            estimator.step(slots, num_sampled, num_rejected)
        end.record()
        torch.cuda.synchronize(device)
        acc_us = (start.elapsed_time(end) * 1000.0) / iters

        results[f"c{num_reqs}"] = {
            "calibrate_confidence_us": round(cal_us, 3),
            "accumulate_us": round(acc_us, 3),
        }

    # Benchmark _refit_kernel
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(100):
        _refit_kernel[(1,)](
            estimator.slope,
            estimator.intercepts,
            estimator.info,
            estimator.info.stride(0),
            estimator.grad,
            estimator.grad.stride(0),
            estimator.counts,
            NUM_SPECULATIVE_STEPS=num_steps,
            L2=estimator.L2,
            DAMPING=estimator.DAMPING_OBSERVATIONS,
            MAX_STEP=estimator.MAX_STEP,
            MAX_SLOPE=estimator.MAX_SLOPE,
            MAX_ABS_INTERCEPT=estimator.MAX_ABS_INTERCEPT,
            BLOCK=triton.next_power_of_2(num_steps),
        )
    torch.cuda.synchronize(device)
    iters = 2000
    start.record()
    for _ in range(iters):
        _refit_kernel[(1,)](
            estimator.slope,
            estimator.intercepts,
            estimator.info,
            estimator.info.stride(0),
            estimator.grad,
            estimator.grad.stride(0),
            estimator.counts,
            NUM_SPECULATIVE_STEPS=num_steps,
            L2=estimator.L2,
            DAMPING=estimator.DAMPING_OBSERVATIONS,
            MAX_STEP=estimator.MAX_STEP,
            MAX_SLOPE=estimator.MAX_SLOPE,
            MAX_ABS_INTERCEPT=estimator.MAX_ABS_INTERCEPT,
            BLOCK=triton.next_power_of_2(num_steps),
        )
    end.record()
    torch.cuda.synchronize(device)
    results["refit_us"] = round((start.elapsed_time(end) * 1000.0) / iters, 3)
    return results


def eval_calibration_and_policy(device: torch.device) -> dict:
    torch.manual_seed(20261106)
    num_steps = 5
    raw_mean = torch.tensor([0.78, 0.72, 0.71, 0.71, 0.73], device=device)
    true_cond_c1 = torch.tensor([0.6768, 0.5560, 0.3965, 0.4598, 0.6385], device=device)
    true_surv_c1 = torch.cumprod(true_cond_c1, dim=0)
    raw_surv_c1 = torch.cumprod(raw_mean, dim=0)

    estimator = OnlineAcceptanceEstimator.for_confidence_head(32, num_steps, device)
    estimator.REFIT_INTERVAL = 20
    slots = torch.arange(8, device=device, dtype=torch.int32)
    idx_map = slots.repeat_interleave(num_steps)

    for _ in range(200):
        probs = raw_mean.unsqueeze(0).expand(8, num_steps).clone()
        estimator.calibrate_confidence_probs(probs, idx_map)
        draws = torch.rand(8, num_steps, device=device) < true_cond_c1
        all_ok = draws.all(dim=1)
        first_fail = (~draws).int().argmax(dim=1)
        accepted = torch.where(all_ok, torch.full_like(first_fail, num_steps), first_fail)
        estimator.step(
            slots,
            (accepted + 1).to(torch.int32),
            (num_steps - accepted).to(torch.int32),
        )

    probe = raw_mean.unsqueeze(0).clone()
    estimator.calibrate_confidence_probs(probe, slots[:1])
    cal_surv_c1 = torch.cumprod(probe[0], dim=0)

    raw_mae_c1 = float((raw_surv_c1 - true_surv_c1).abs().mean().item())
    cal_mae_c1 = float((cal_surv_c1 - true_surv_c1).abs().mean().item())
    mae_ratio_c1 = raw_mae_c1 / max(cal_mae_c1, 1e-9)

    # Evaluate c1 admission under:
    # (a) old rc15 broken combo (calibrated probs + uncalibrated dead_rows_tau=0.25, row_cost=0.045)
    # (b) reworked combo (calibrated probs + calibrated_confidence=True, dead_rows_tau=0.20, row_cost=0.035)
    cal_np = probe.cpu().numpy()
    mgr_rc15 = make_manager(cal_np, row_cost=0.045, dead_rows_tau=0.25, calibrated=False)
    mgr_rc15.get_num_tokens({"r0": 6}, {"r0": [1, 2, 3, 4, 5]})
    rc15_c1_budget = mgr_rc15._batch_budget[2]

    mgr_rework = make_manager(cal_np, row_cost=0.035, dead_rows_tau=0.20, calibrated=True)
    mgr_rework.get_num_tokens({"r0": 6}, {"r0": [1, 2, 3, 4, 5]})
    rework_c1_budget = mgr_rework._batch_budget[2]

    # Evaluate c8 heterogeneous batch (4 high-confidence code/copy requests + 4 low-confidence creative requests)
    # True conditional acceptance for high-conf vs low-conf:
    raw_c8 = np.vstack(
        [
            np.tile([0.88, 0.84, 0.82, 0.80, 0.78], (4, 1)),
            np.tile([0.66, 0.58, 0.54, 0.52, 0.50], (4, 1)),
        ]
    ).astype(np.float32)
    raw_c8_t = torch.from_numpy(raw_c8).to(device)
    cal_c8_t = raw_c8_t.clone()
    estimator.calibrate_confidence_probs(cal_c8_t, slots)
    cal_c8 = cal_c8_t.cpu().numpy()
    true_surv_c8 = np.cumprod(cal_c8, axis=1)

    req_tokens = {f"r{i}": 6 for i in range(8)}
    req_drafts = {f"r{i}": [1, 2, 3, 4, 5] for i in range(8)}

    # Candidate 1107 baseline at c8: flat linear slope (knee_scale=1.0), uncalibrated probs
    mgr_base_c8 = make_manager(
        raw_c8, row_cost=0.035, dead_rows_tau=0.20, calibrated=False, knee_scale=1.0
    )
    mgr_base_c8.get_num_tokens(req_tokens, req_drafts)
    base_c8_budget = mgr_base_c8._batch_budget[2]
    # True expected tokens per unit true step cost (using 16-row hardware knee cost)
    hw_cost_mgr = make_manager(cal_c8, row_cost=0.035, knee_scale=0.55)
    raw_surv_c8 = np.cumprod(raw_c8, axis=1)
    base_order = np.argsort(raw_surv_c8.reshape(-1))[::-1]
    true_flat = true_surv_c8.reshape(-1)
    base_expected = 8.0 + float(true_flat[base_order[:base_c8_budget]].sum()) if base_c8_budget > 0 else 8.0
    base_hw_cost = float(hw_cost_mgr._step_cost(np.array([8 + base_c8_budget]))[0])
    base_efficiency = base_expected / base_hw_cost

    # Rework at c8: two-slope 16-row MMA knee (knee_scale=0.55) + calibrated probs + matched dead_rows_tau
    mgr_rework_c8 = make_manager(
        cal_c8, row_cost=0.035, dead_rows_tau=0.20, calibrated=True, knee_scale=0.55
    )
    mgr_rework_c8.get_num_tokens(req_tokens, req_drafts)
    rework_c8_budget = mgr_rework_c8._batch_budget[2]
    rework_order = np.argsort(true_flat)[::-1]
    rework_expected = 8.0 + float(true_flat[rework_order[:rework_c8_budget]].sum())
    rework_hw_cost = float(hw_cost_mgr._step_cost(np.array([8 + rework_c8_budget]))[0])
    rework_efficiency = rework_expected / rework_hw_cost

    return {
        "c1_raw_survival_mae": round(raw_mae_c1, 4),
        "c1_calibrated_survival_mae": round(cal_mae_c1, 4),
        "c1_mae_reduction_factor": round(mae_ratio_c1, 2),
        "c1_rc15_draft_budget": int(rc15_c1_budget),
        "c1_rework_draft_budget": int(rework_c1_budget),
        "c1_effective_dead_rows_tau": round(mgr_rework.effective_dead_rows_tau(), 5),
        "c8_baseline_draft_budget": int(base_c8_budget),
        "c8_rework_draft_budget": int(rework_c8_budget),
        "c8_baseline_tokens_per_cost": round(base_efficiency, 4),
        "c8_rework_tokens_per_cost": round(rework_efficiency, 4),
        "c8_efficiency_gain_pct": round(100.0 * (rework_efficiency / base_efficiency - 1.0), 2),
    }


if __name__ == "__main__":
    device = torch.device("cuda")
    out = {
        "device_name": torch.cuda.get_device_name(device),
        "kernel_latency_us": bench_kernels(device),
        "calibration_and_policy": eval_calibration_and_policy(device),
    }
    print(json.dumps(out, indent=2))
