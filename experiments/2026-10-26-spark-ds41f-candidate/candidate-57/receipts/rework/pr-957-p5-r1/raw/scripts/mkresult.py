#!/usr/bin/env python3
"""Builds result.json from raw/ (replay outputs, cells.json, pytest outputs, series check).
usage: mkresult.py <deliverable dir>"""
import hashlib
import json
import os
import sys

D = sys.argv[1]
R = os.path.join(D, "raw")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def tail(p):
    return open(p).read().strip().splitlines()[-1].strip("= ")


cells = json.load(open(os.path.join(R, "cells.json")))
rep = {f"{p}-seed{s}": json.load(open(os.path.join(R, "out", f"replay-{p}-seed{s}.json")))
       for p in ("986fit", "tp3fit") for s in (0, 1)}
r0 = rep["986fit-seed0"]
PATCH = "57-03-957-p5-perf-dsv41-tp3-decode-compressor-mla-and-spec-fusions.patch"


def rng(prof_list, c, arm, key, sub=None):
    vals = []
    for prof in prof_list:
        for cell in cells[prof][c][arm]:
            v = cell.get(key)
            if v is None:
                continue
            vals.append(v[sub] if sub is not None else v)
    return [min(vals), max(vals)] if vals else None


def summary(c, arm="p5@tp3"):
    out = {}
    for prof in ("986fit", "tp3fit"):
        lst = cells[prof][c][arm]
        out[prof] = [{k: v for k, v in x.items() if k not in ("profile", "concurrency", "arm")} for x in lst]
    return out


def headline(c, arm="p5@tp3"):
    h = {
        "draft_budget_per_step_c54": rng(["986fit"], c, arm, "draft_budget_per_step", 0),
        "draft_budget_per_step_p5": rng(["986fit"], c, arm, "draft_budget_per_step", 1),
        "steps_budget_differs_pct": rng(["986fit", "tp3fit"], c, arm, "steps_budget_differs_pct"),
        "tokens_per_step_pct": rng(["986fit", "tp3fit"], c, arm, "tokens_per_step_pct"),
        "grid_gpu_step_pct": rng(["986fit", "tp3fit"], c, arm, "grid_gpu_step_pct"),
        "grid_interp_gpu_step_pct": rng(["986fit", "tp3fit"], c, arm, "grid_interp_gpu_step_pct"),
        "throughput_grid_gpu_only_pct": rng(["986fit", "tp3fit"], c, arm, "throughput_grid_gpu_only_pct"),
        "served_step_pct_model_A": rng(["986fit", "tp3fit"], c, arm, "served_step_pct_model_A"),
        "throughput_model_A_pct": rng(["986fit", "tp3fit"], c, arm, "throughput_model_A_pct"),
        "throughput_model_B_tp4_regression_pct": rng(["986fit", "tp3fit"], c, arm, "throughput_model_B_tp4_regression_pct"),
    }
    m = []
    for prof in ("986fit", "tp3fit"):
        for x in cells[prof][c][arm]:
            if "throughput_model_M_pct" in x:
                m += list(x["throughput_model_M_pct"].values())
    if m:
        h["throughput_model_M_pct"] = [min(m), max(m)]
    return h


g3 = r0["grid_tp3"]; g4 = r0["grid_tp4_switched"]
res = {
    "pr": 957,
    "part": 5,
    "label": "957-p5",
    "round": 3,
    "candidate": "candidate-57",
    "testbed": "tp3-switched",
    "placement": "candidate-57's last patch, after 957-p2 and 957-p4 on candidate-54",
    "pr_head": "e464d364431a26b031a6e91f5dd610f429129c46",
    "source_patch": "experiments/2026-11-08-tp3-decode-recovery/vllm/0762-perf-dsv41-tp3-decode-compressor-mla-and-spec-fusions.patch (sha256 44412274ff72075c37afed64dbd9cb638cf604a34901b10f11b6fee128385e71); part 5 = its adaptive_verification.py hunks and the block appended to test_high_concurrency_row_cost_knee_admits_coalesced_drafts",
    "verdict": "not-defensible",
    "verdict_reason": (
        "Part 5 does change budgets at TP3, but the replay predicts no gain anywhere. It trims draft rows that "
        "candidate-50's tp3-switched grid prices below the tokens they carry. Every step-time model built on that "
        "grid predicts lower throughput at c1, c4 and c8, and at c1 and c8 the loss is beyond the 0.35-0.4% A/A floor. "
        "The scale drives c4 and c8: at c8, -1.2% to -1.3% on the grid step alone and -3.1% to -3.3% on the served step. "
        "The dead-row floor drives c1: -0.9% to -1.0% on the grid step and -1.0% to -1.2% on the served step. "
        "#986's blend calibrated on the 2026-10-18 row-cost A/B also predicts a loss: -0.3% to -1.2% at c1 and "
        "-0.2% to -1.3% at c8. Only #986's tp4-ring serving regression taken alone (model B) predicts a c1 gain "
        "(+1.0% to +1.1%). It prices a wasted row at 1.98 ms, against the TP3 grid's 0.17 ms a row, and the A/B-calibrated "
        "blend gives it at most 43% weight at c1. Recommendation (the coordinator decides): leave 957-p5 out of "
        "candidate-57. If it is carried, the no957p5 overlay supports an attribution control on the same image."
    ),
    "a_a_floor_pct": {"c16": 0.43, "c8": 0.39, "c1": 0.35,
                      "source": "candidate-47/receipts/rework/pr-986-r1 (A/A floor of the screen's paired decode)"},
    "clears_floor": {
        "c1": "yes, in the wrong direction: the grid predicts -0.9% to -1.2%. Only model B alone predicts +1.0%, and the A/B-calibrated blend predicts -0.3% to -1.2%",
        "c4": "no: -0.2% (grid step) to -0.5% (served step), at or inside the floor, and negative",
        "c8": "yes, in the wrong direction: -1.2% to -3.3% on the grid. Blend: -0.2% (central) to -1.3%",
    },
    "patch": {
        "file": PATCH,
        "sha256": sha(os.path.join(D, PATCH)),
        "form": "git format-patch; author Christopher Owen <3221756+christopherowen@users.noreply.github.com>, author date Sun, 8 Nov 2026 11:30:00 +0100 (the PR's, as 957-p2 and 957-p4 carry it); subject kept as the PR's (as the split parts keep it); rationale with a 'Round 3 rebase' section",
        "files": {"vllm/v1/worker/gpu/spec_decode/adaptive_verification.py": "+31/-6",
                  "tests/v1/spec_decode/test_adaptive_verification.py": "+79/-0"},
        "applies": "fresh clone at candidate-54's patch head 4e8133ff, git am 57-01 (957-p2), 57-02 (957-p4): tree 72b40d71 (= the selection's candidate-57 stack); git am this patch: tree 5016d620c2d4a6e208f5799c3ecc673222b253f4; git diff --check 4e8133ff..HEAD clean (raw/series-check.txt)",
        "applies_to_no957p5_overlay": "git apply --check passes on candidate-54's two files (no957p5/overlay/), which no other candidate-57 patch touches (957-p2 and 957-p4 touch neither)",
        "overlay": {
            "overlay/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py": sha(os.path.join(D, "overlay/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py")),
            "overlay/tests/v1/spec_decode/test_adaptive_verification.py": sha(os.path.join(D, "overlay/tests/v1/spec_decode/test_adaptive_verification.py")),
            "no957p5/overlay/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py": sha(os.path.join(D, "no957p5/overlay/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py")),
            "no957p5/overlay/tests/v1/spec_decode/test_adaptive_verification.py": sha(os.path.join(D, "no957p5/overlay/tests/v1/spec_decode/test_adaptive_verification.py")),
            "note": "overlay/ is candidate-57's two files with this patch (equal to the series tree 5016d620); no957p5/overlay/ is candidate-54's (= candidate-57's without part 5)",
        },
    },
    "base": {"name": "candidate-54", "patch_count": 671, "patch_head": "4e8133ff10a125b90753c2727f162e29fe37f30c",
             "tree": "5f021d9a7f01c03e2389dd3d8021759cfa19ea34",
             "candidate_57_stack_before_p5": "72b40d7102ab56384a68885ad3a61733afe1c5a2 (957-p2, 957-p4)",
             "workspace": "sp8:~/spark-lab/agents/round3-957p5 (vllm/ cloned --shared from round-55-57-select/c54-vllm, branch p5-final)"},
    "conflict": {
        "0789_971_r1": "_step_cost prices rows past HIGH_ROW_COST_KNEE (16) with single_req_tile_spill_scale = SINGLE_REQ_TILE_SPILL_SCALE (2.0) when num_sampling_requests <= 1, else high_row_cost_scale (HIGH_ROW_COST_SCALE 0.55); get_num_tokens passes num_sampling_requests",
        "part_5": "picks HIGH_ROW_COST_SCALE_TP3 (0.85) for self.high_row_cost_scale in __init__ when tp_size == 3 (written on 1107, where that scale priced every step) and floors the calibrated dead-row threshold at MIN_CALIBRATED_DEAD_ROWS_TAU_TP3 (0.12)",
    },
    "composition": {
        "decision": "HIGH_ROW_COST_SCALE_TP3 replaces only the multi-request scale (self.high_row_cost_scale, num_sampling_requests > 1). A single sampling request keeps #971 round 1's SINGLE_REQ_TILE_SPILL_SCALE = 2.0 on every TP size. At c1 the effective scale is therefore max(2.0, 0.85) = 2.0. No TP3-specific single-request scale is added.",
        "why": [
            "The two scales model different things. 0.55 and 0.85 are the shared-weight-tile discount for rows past the 16-row MMA tile when several requests coalesce. 2.0 is the single-request penalty for spilling the 17th row into a second MMA wave and the 24-row graph. #971 round 1 measured that step at TP3 (initial c1 copy step, 24-row graph to 16-row graph, -5.52%, Welch95 [-9.0, -7.6] ms) with the 2.0 scale.",
            "Part 5 was written on 1107, where one scale priced every step, so on its own base it replaced the c1 price too. Carried that way onto candidate-54, it would undo #971 round 1 at TP3. The fixed cell c1_copy_0.95x16 gives 15 drafts and 16 rows with this patch at TP3, and 16 drafts and 17 rows if 0.85 also priced single requests (raw/out/replay-*.json, deterministic).",
            "There is no TP3 evidence for a different single-request scale. A multiplied scale (2.0 x 0.85 / 0.55) would be a new, unmeasured constant.",
            "A c1 step with a prefill in the batch has two sampling requests, so it uses the multi-request branch, TP3's 0.85 included. That is #971 round 1's own definition, unchanged.",
        ],
    },
    "dead_row_floor": {
        "acts_at_recipe_tau_0.2": True,
        "why": "The recipe sets SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2 and adaptive_verification_row_cost 0.035, and the model runner passes calibrated_confidence = (estimator _refits > 0 and the confidence head is used). Once the estimator has refit, candidate-54's effective_dead_rows_tau is min(0.2, 2*0.035/(1+3*0.035)) = 0.0633. With part 5 at TP3 it is min(0.2, max(0.12, 0.0633)) = 0.12 (the replay's managers report 0.0633 and 0.1200). Before the first refit both return 0.2. So the floor is not inert; it raises the host admission cut and the device dead-row cut at TP3.",
        "kept": "yes. The deployment-only rule leaves out only what does nothing here, and the floor acts. It is also the whole of part 5's effect at c1, where the replay predicts a loss.",
        "where_it_binds": "c1 only, in effect: 22-29% of c1 steps change budget, 1.3-1.9% of c4 steps, about 0.01% of c8 steps. At c4 and c8 the budget's own cut is already above 0.12.",
    },
    "test_budgets_rederived_on_candidate_54": {
        "c8_medium [0.78, 0.72, 0.71, 0.71, 0.73] x 8, row cost 0.035, tau 0.2 calibrated": {"tp3": "24 drafts, 32 rows", "tp4": "32 drafts, 40 rows (unchanged)", "flat_scale_1.0": "24 drafts", "floor_binds": False},
        "c1_typical [0.677, 0.556, 0.396, 0.460, 0.638] (survival 0.677, 0.376, 0.149, 0.0686, 0.0437)": {"tp3": "3 drafts, 4 rows (position 3 below 0.12)", "tp4": "4 drafts, 5 rows (unchanged; 0.0686 >= 0.0633)"},
        "c1_copy 0.95 x 16": {"tp3": "15 drafts, 16 rows", "tp4": "15 drafts, 16 rows", "rejected_composition_0.85_at_c1": "16 drafts, 17 rows"},
        "c2_copy 0.95 x 16": {"tp3": "32 drafts, 34 rows", "tp4": "32 drafts, 34 rows"},
        "c4_medium x 4": {"tp3": "16 drafts, 20 rows", "tp4": "20 drafts, 24 rows"},
        "derived_by": "the test file, plus the replay's fixed cells run through each module's own __init__ (raw/out/replay-*.json 'deterministic'; raw/out/quick-budget.txt is the first numpy emulation)",
        "tests_added": [
            "test_high_concurrency_row_cost_knee_admits_coalesced_drafts: the TP3 block (24 drafts, 32 rows; effective tau 0.12) and a TP4 block (32 drafts, 40 rows)",
            "test_tp3_survival_floor_prunes_the_typical_c1_tail[tp4|tp3]: 4 drafts and 3 drafts",
            "test_tp3_constants_keep_the_single_request_tile_spill_cap: c1 copy 0.95 x 16 stays 15 drafts under the TP3 constants",
            "test_tp_size_selects_the_row_cost_constants[1|3|4]: the constants __init__ picks with get_tp_group patched (CUDA)",
        ],
    },
    "tests": {
        "file": "tests/v1/spec_decode/test_adaptive_verification.py, pytest -v --noconftest, spark-ds41f:candidate-50 with the candidate overlay (GPU container under the lock)",
        "candidate_57_with_p5_run1": tail(os.path.join(R, "out", "pytest-c57p5-run1.txt")),
        "candidate_57_with_p5_run2": tail(os.path.join(R, "out", "pytest-c57p5-run2.txt")),
        "candidate_54": tail(os.path.join(R, "out", "pytest-c54.txt")),
        "note": "6 new cases: 46 + 6 = 52; the one skip is the base's two-GPU distributed test",
    },
    "tp4_bit_for_bit": {
        "result": "unchanged",
        "evidence": [
            "Every replay step of every profile (c1, c4, c4-theta-c1, c8, c16; 20,000 steps each; two seeds; two acceptance profiles) gives the same budget with the patch's own __init__ at world size 4 as with candidate-54's: 0 differing steps (tp4_bit_for_bit in each raw/out/replay-*.json)",
            "The fixed cells are equal at TP4 (deterministic_equal true), effective tau 0.0633 on both, and __init__ at world size 4 picks scale 0.55, floor 0.0, spill 2.0",
            "candidate-54 has no TP dependence: c54@tp3 equals c54@tp4 on every step",
            "test_tp_size_selects_the_row_cost_constants[4-0.55-0.0] and [1-0.55-0.0] pass",
        ],
    },
    "replay": {
        "what": "a policy replay of the real AdaptiveVerificationManager. Candidate-54's adaptive_verification.py and the rebased part 5's were loaded by path in spark-ds41f:candidate-50 with candidate-54's 36 changed files mounted, so every vllm import is candidate-54's. Each manager was built through its own __init__ (small CUDA buffers) with get_tp_group patched to world size 3 or 4. get_num_tokens drove each step; one manager per arm persisted across the steps, so its idle-explore counter carries as in serving.",
        "recipe_inputs": "5 drafts, 16 verification steps (copy drafter on), row cost 0.035, SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2 (env), calibrated confidence (lambda: True), max_total_logits = get_max_chunk_logits(129280) (never binds)",
        "acceptance_profile": "#986 round 1's served profile (candidate-47/receipts/rework/pr-986-r1): its fitted two-component logit-normal confidence model per concurrency (theta from replay-seed0.json and replay-seed1.json), its step sampler (served requests a step plus one 21-token prefill at the served prompt rate), and calibrated acceptance (expected tokens = admitted survival). c4 has no served record: it uses c8's theta at half c8's requests and prompt rate, and c1's theta as a sensitivity (c4-theta-c1).",
        "sensitivity_profile": "#986's fit_conf, imported unchanged, refit to candidate-50's tp3-switched served counters (c8 two boots, single-stream c1). The c1 fit loss is 0.0016 / 0.0005, against #986's tp4 c1 fit at 0.0057 (raw/out/fit-tp3-seed*.txt). Same draws as the primary run.",
        "fidelity": {
            "tp3_served_counters_candidate_50": r0["served_tp3_counters"],
            "replay_c54_c8_coverage_survival_seed0": [r0["concurrency"]["c8"]["arms"]["c54@tp3"]["coverage"], r0["concurrency"]["c8"]["arms"]["c54@tp3"]["survival"], r0["concurrency"]["c8"]["arms"]["c54@tp3"]["tokens_mean"]],
            "replay_c54_c1_coverage_survival_seed0": [r0["concurrency"]["c1"]["arms"]["c54@tp3"]["coverage"], r0["concurrency"]["c1"]["arms"]["c54@tp3"]["survival"], r0["concurrency"]["c1"]["arms"]["c54@tp3"]["tokens_mean"]],
            "note": "TP3's served per-position acceptance matches tp4-ring's closely: c8 survival 0.677/0.372/0.184/0.098/0.053 against 0.679/0.367/0.178/0.093/0.051. That supports #986's profile at TP3.",
        },
        "step_time": {
            "grid": "candidate-50's tp3-switched screen grid, two boots (receipts/references/candidate-50-no1009-tp3-switched-screen{,-again}/grid.json), fitted at 4K: t = a + b n + c1 min(R,16) + c2 max(R-16,0); rows padded to the TP3 capture ladder [1,2,3,4,6,8,12,16,20,24,28,32,40,48] (eager above 48)",
            "grid_fit_tp3": g3["fit"],
            "grid_fit_tp3_rmse_ms": g3["fit_rmse_ms"],
            "grid_tp3_implied_knee_scale": g3["implied_knee_scale"],
            "grid_fit_tp4_switched_for_comparison": g4["fit"],
            "grid_tp4_switched_implied_knee_scale": g4["implied_knee_scale"],
            "interp_check": "piecewise-linear between the grid's own points at the same request count (decode steps the grid covers): c8 GPU step -2.77% to -2.83% against the fit's -2.45% to -2.59%",
            "model_A": "grid + constant served overhead, anchored so candidate-54's replayed mean equals candidate-50's tp3-switched served step: c1 1000/26.293 = 38.03 ms (single-stream), c8 1000/8.650 = 115.61 ms (screen c8, two boots); c4 overhead interpolated in requests (49.5 ms)",
            "model_B": "#986's tp4-ring per-batch serving regression (intercept + ms/token + ms/wasted row). A sensitivity only: it is TP4's and prices a wasted row at 1.98 ms at c1 and 0.46 ms at c8.",
            "model_M": "#986's blend w A + (1 - w) B, with its weights calibrated on the 2026-10-18 row-cost A/B (c1 w 0.57/0.86/1.0, c8 w 0.0/0.01/0.30). The weights were fitted with TP4's model A, so applying them at TP3 is an approximation.",
        },
        "arms": "c54@tp3, c54@tp4, p5@tp3, p5@tp4; decomposition p5-scale-only@tp3 (floor forced 0) and p5-floor-only@tp3 (scale forced 0.55)",
        "headline_p5_vs_c54_at_tp3": {
            "c1": headline("c1"),
            "c4": headline("c4"),
            "c8": headline("c8"),
        },
        "decomposition": {
            "c1": {"scale_only": headline("c1", "p5-scale-only@tp3"), "floor_only": headline("c1", "p5-floor-only@tp3")},
            "c4": {"scale_only": headline("c4", "p5-scale-only@tp3"), "floor_only": headline("c4", "p5-floor-only@tp3")},
            "c8": {"scale_only": headline("c8", "p5-scale-only@tp3"), "floor_only": headline("c8", "p5-floor-only@tp3")},
        },
        "padded_rows_histogram_seed0": {c: {"c54@tp3": r0["concurrency"][c]["arms"]["c54@tp3"]["padded_rows_histogram"],
                                            "p5@tp3": r0["concurrency"][c]["arms"]["p5@tp3"]["padded_rows_histogram"]} for c in ("c1", "c4", "c8")},
        "cells": {c: summary(c) for c in ("c1", "c4", "c4-theta-c1", "c8")},
        "files": {k: f"raw/out/replay-{k}.json" for k in rep},
    },
    "why_no_gain": [
        "The grid prices rows far below the budget's own price. At TP3 a row costs 0.170 ms up to 16 rows and 0.220 ms past 16, on an 18.6 ms fixed step: 0.9% and 1.2% of the step. The recipe's row cost is 3.5% of the step. At c8 (about 26 rows) candidate-54 already charges a row past 16 about 1.1% of its modelled step, against the grid's 0.9% of the GPU step and 0.19% of the served step. Part 5's 0.85 raises that to about 1.6%, so it trims rows whose survival (about 0.2-0.3) is worth more tokens than the 0.22 ms they cost.",
        "At c1 the floor rejects slots with survival between 0.063 and 0.12. On the grid, such a slot pays for itself above roughly 0.02 survival (tokens 2.3 x 0.17 ms / 19.3 ms GPU step), or 0.01 on the served step.",
        "The PR's premise, that TP3 rows past the knee cost more than TP4's, does not show in the grids. TP3-switched rows cost 0.170 and 0.220 ms; TP4-switched rows cost 0.255 and 0.272 ms (0.9% against 1.7% of the step). Neither grid shows the knee discount: the slope ratio past 16 is 1.29 at TP3 and 1.06 at TP4.",
        "At TP3 a c8 step of 40 rows runs the 40-row graph. The ladder holds 40, so the PR's '40 rows -> padded to 48' does not apply here.",
    ],
    "serving_judgment": "candidate-57's tp3-switched screen with single-stream and sampled companions. Predicted if part 5 is carried: c8 paired decode lower (the grid alone says 1.2% to 3.3%), single-stream c1 lower by about 1%, and temperature-0 text unchanged. A no957p5 control on the same image needs only no957p5/overlay/.",
    "not_run": [
        "No serving run and no step-time measurement of my own: the grid is the step-time evidence named in the brief, and a one-GPU sp8 profile cannot run TP3 shapes",
        "No copy-draft replay beyond the fixed cells: the screen's prompts give coverage past position 4 of 0.0002 at c8 and 0 at c1",
    ],
    "hosts_and_rules": "sp8 (spark-0016) only. Every container went through raw/scripts/run.sh: flock /tmp/spark-lab-gpu.lock, ~/spark-hold.json absent and docker ps empty checked inside the lock, each block logged in raw/out/commands.log. The GPU work was tiny CUDA buffers for the managers' __init__ and the unit tests. No push, commit to this repository, PR, image build, cluster or window operation. The laptop did file transfers and text work only (no series replay, no build).",
    "inputs": {
        "origin_main": "6de8656022eddfa1ef14c10c99ee78c3138493be (read with git show)",
        "pr_957_head": "e464d364431a26b031a6e91f5dd610f429129c46 (git fetch origin pull/957/head)",
        "round3_selection": "candidate-55/receipts/selection-proposal.json (rework_needed['957-p5'], candidates['57']); rebases/57-01 (sha256 67ed1f5f...), 57-02 (b8f0bf3a...)",
        "sha256": {
            "986 served_targets.json": "08cca565bc715e3dbc90d9a7729d6ebc939a7f8812d0beb4c49e6c5b1d2faaa5",
            "986 replay-seed0.json": "f2d6f31fbcdec11909ab2dbcdc88b8a5c382fd275dadf670fe2aa2422c2cbe1b",
            "986 replay-seed1.json": "d2811540d06f4f0624dbbfbeb1d9baa589c52cfb3cca2997b4aa38967ff709de",
            "986 replay_budget.py": "3d1c94f7563e16aa7b6945c6e58e643ca730f90edb4a3c25d68b70b08505cfa1",
            "candidate-50-no1009-tp3-switched-screen grid.json": "11fa55d8466093980e467033903a2fb17dd1638bd11a5445a59ea0ce4166f1ec",
            "candidate-50-no1009-tp3-switched-screen-again grid.json": "b047c4333ec255de1f9e455251fadfae28a9e51be40bbbabc80c202cc56aac34",
            "candidate-50-no1009-tp4-switched-screen grid.json": "c906c72ebaf1112846c71ad2a4ce57bf90df202c0837a16b17939eeaa2e7ed4c",
            "candidate-50-no1009-tp4-switched-screen-again grid.json": "9983297a2eb3703c4e03a396e7a6b373bb4c73a6efa2e6a87f0197cacaeee5f4",
            "candidate-50 tp3-switched c8.json": "b69624cc9755da920e37f277af0721280651edc5bc9e92f1e5c664b9e1ea9174",
            "candidate-50 tp3-switched-again c8.json": "7bac75fa29bcc5aee908b408a106117549cea59d4d50ad1f556deaa0daa18ab6",
            "candidate-50 tp3-switched single-stream c1.json": "79e62348c0a2734bb7a7adf09ff31679b63c984f27fb8427459ce662ed627aff",
            "av_c54.py (candidate-54 adaptive_verification.py)": sha(os.path.join(R, "in", "av_c54.py")),
            "av_p5.py (rebased)": sha(os.path.join(R, "in", "av_p5.py")),
        },
        "image": "spark-ds41f:candidate-50 (8c1ea1f62f62, vLLM tree d5778ab5) + candidate-54's 36 changed files (raw/in/changed-c54.txt) for the replay, or candidate-57-with-p5's 42 (raw/in/changed-c57p5.txt) for its tests; sp8 has no candidate-54 image",
    },
}
json.dump(res, open(os.path.join(D, "result.json"), "w"), indent=1)
print("ok", len(json.dumps(res)))
