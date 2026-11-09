"""result.json for the PR #986 rework, round 1, from the receipts in this directory."""
import json, sys
R = sys.argv[1]
rows = json.load(open(f"{R}/rows-a-step-compare.json"))
reg = json.load(open(f"{R}/batch-regression.json"))
tgt = json.load(open(f"{R}/served_targets.json"))
dec = json.load(open(f"{R}/pr986-screen-decomposition.json"))
rep = [json.load(open(f"{R}/replay-seed{s}.json")) for s in (0, 1)]
g = rep[0]["grid"]

def served(step):
    s = rows["steps"][step]; a, b = s["with986"]["mean"], s["no986"]["mean"]
    return {"no986": {k: round(b[k], 4) for k in ("rows", "tokens", "yield", "wasted")},
            "with986": {k: round(a[k], 4) for k in ("rows", "tokens", "yield", "wasted")},
            "change_pct": {k: round(v, 2) for k, v in s["with986_vs_no986_pct"].items() if k in ("rows", "tokens", "yield", "wasted", "steps_per_s")},
            "coverage_by_position_no986": [round(x, 4) for x in b["coverage"][:5]],
            "coverage_change_by_position": [round(x, 4) for x in s["with986_minus_no986"]["coverage"][:5]],
            "survival_by_position_no986": [round(x, 4) for x in b["survival"][:5]],
            "survival_change_by_position": [round(x, 4) for x in s["with986_minus_no986"]["survival"][:5]],
            "boot_spread_rows": [round(s["with986"]["boot_spread"]["rows"], 3), round(s["no986"]["boot_spread"]["rows"], 3)]}

def replay(conc, pol):
    out = []
    for r in rep:
        p = r["concurrency"][conc]["policies"][pol]; d = p["vs_1107"]
        out.append({"seed": r["seed"], "rows_pct": round(d["rows_pct"], 2), "tokens_pct": round(d["tokens_pct"], 2),
                    "steps_budget_differs": round(p["share_steps_budget_differs_from_1107"], 3),
                    "steps_above_64_rows": round(p["share_steps_rows_gt_64"], 4),
                    "throughput_pct": {"A_grid_plus_constant": round(d["throughput_model_A_pct"], 2),
                                       "B_served_batch_regression": round(d["throughput_model_B_pct"], 2),
                                       "M_calibrated_to_2026-10-18_row_cost_ab": {k: round(v, 2) for k, v in d["throughput_model_M_pct"].items()}}})
    return out

result = {
    "pr": 986, "round": 1, "candidate": "candidate-47",
    "verdict": "no-defensible-correction",
    "summary": (
        "In serving, #986 changes the draft budget only at c16, and by little: verified rows -0.87% (44.61 -> 44.22 a step, "
        "boot spread 0.06-0.08), tokens -0.44%, wasted rows -2.3%, coverage at positions 1-4 down 0.002-0.007. At c8 and c1 it "
        "picks the same rows (c1 identical by construction: one request, at most 6 rows, every #986 term is zero). The regimes its "
        "screen priced are absent from the release suite (contexts <= 285 tokens; c1 <= 6 rows; c8 <= 48 rows) or rare (c16: about "
        "8.5% of steps above 64 rows in a replay that reproduces the served change). Where it fires, the predicted saving was the "
        "screen's own latency model (a non-monotone TP1 dense table on zero tensors plus a hand-set 440 us/row MoE term above 64 "
        "rows), not a cost the tp4-ring engine grid shows: the grid has no context cost from 4K to 1M and prices 64 -> 96 rows at "
        "about what 1107's 0.55 slope already charges. A replay of the real 1107 and #986 modules over steps fitted to the served "
        "counters predicts |change| <= 0.16% at c16, <= 0.01% at c8 and 0 at c1 for #986 and for a knee re-calibrated to the grid, "
        "under every step-time model tried; the A/A floors are 0.43/0.39/0.35%. No correction of #986's terms can be confirmed in "
        "serving, and the context band, the one part serving would hit (production's 64-100K agent prompts), prices a cost that "
        "does not exist on tp4-ring."),
    "what_986_changes_in_serving": {"c16": served("c16"), "c8": served("c8"), "c1": served("c1"), "poisson": served("poisson"),
                                     "paired_decode_from_candidate_47_README": {"c16": "+0.38% (margin 0.64%)", "c8": "+0.20% (0.53%)", "c1": "+0.24% (0.34%)"},
                                     "note_c1": "the c1 budget is identical by construction, so c1's +0.24% paired (and the first pair's +0.18% 'better') is boot noise"},
    "why_the_modeled_gains_do_not_appear": {
        "regime_absent": {
            "context_band": "release prompts are 15-29 tokens with 256 output tokens (contexts <= 285); CONTEXT_ROW_COST_KNEE is 8,192, so the c8 32K-128K (+5.0-5.2%) and heterogeneous 2K+128K (+7.0%) regimes never occur",
            "c1_plateau": "the +1.1% c1 regime needs > 16 rows from one request (copy-draft bursts); served c1 verifies at most 6 rows (5 drafts; coverage at positions >= 5 is <= 0.0006)",
            "c8": "c8 holds at most 48 rows, so the 64-row knee and second-wave step cannot fire; only the 0.25/0.75-row offsets at 16/32 rows apply (replay: budget differs on 5.7% of steps, rows -0.01%)",
            "c16_second_wave": "the screen set all 16 requests to conditional confidences [0.90, 0.90, 0.90, 0.62, 0.50] (every step 80 rows); served c16 conditional acceptance is 0.72, 0.50, 0.47, 0.52, 0.55 and rows average 44.6; the replay puts 8.5-8.7% of 1107's c16 steps above 64 rows",
        },
        "prediction_wrong": {
            "c16_80_to_64_rows_saving_by_term_ms": dec["c16_second_wave_saturation_2k"]["saved_ms_by_term"],
            "c16_share_by_term": dec["c16_second_wave_saturation_2k"]["share_by_term"],
            "c8_32k_40_to_32_rows_saving_by_term_ms": dec["c8_med_accept_32k"]["saved_ms_by_term"],
            "c8_128k_40_to_32_rows_saving_by_term_ms": dec["c8_med_accept_128k"]["saved_ms_by_term"],
            "same_trim_by_context_in_its_own_latency_model": dec["c8_med_accept_same_trim_by_context"],
            "reading": ("the context-dependent indexer term is 7-15% of the modeled c8 saving; the rest is context-independent (dense table "
                        "R=32 -> 40 and the MoE term), and the screen's own latency model rewards the same 40 -> 32 trim by +4.3% at 2K, where "
                        "#986's cost model keeps 40 rows. The context band was a trigger, not a priced cost."),
            "pr986_dense_table_non_monotone_us": dec["dense_non_monotone"],
            "tp4_grid_context_effect_ms": g["context_effect"],
            "tp4_grid_fit_ms": {k: round(v, 3) for k, v in g["fit"].items()},
            "tp4_grid_fit_rmse_ms": round(g["fit_rmse_ms"], 3),
            "tp4_grid_in_1107_cost_units": {k: round(v, 3) for k, v in rep[0]["grid_knee_in_1107_units"].items()},
            "pricing_64_to_96_rows_in_row_units": {"tp4_grid": round(rep[0]["grid_knee_in_1107_units"]["jump_rows"] + 32 * rep[0]["grid_knee_in_1107_units"]["above_64_scale"], 1),
                                                   "1107": 17.6, "986": 34.0},
            "tile_steps": "no step of the TP1 table's size (+1.2 ms at 16 -> 17, +2.3 ms at 32 -> 40) shows in the grid: the 4x24, 8x24, 8x48 and 16x48 shapes sit +0.33, +0.15, +0.17 and -0.23 ms from one fit with a 0.27 ms/row slope from 17 to 64 rows (every 4K shape but 2x12 within 0.33 ms)",
        },
        "same_rows": "c1 identical; c8 differs on 5.7% of steps by about a row; c16 differs on 33% of steps, almost all from the 64-row knee and second-wave step (tile offsets alone: -0.03% rows)",
    },
    "replay": {
        "method": ("real 1107 and #986 adaptive_verification.py (get_num_tokens) over 20,000 steps per concurrency, two seeds; per-request "
                   "calibrated confidences from a two-component logit-normal fitted to candidate-47-no986's served coverage and survival by "
                   "position and tokens a step (c16 fit loss 2-4e-5, tokens 34.29-34.35 vs 34.29); decoding requests from TTFT/e2e; a 21-token "
                   "prefill on the served prompt-token share of steps; a numpy copy of 1107's rule matches the module on 100% of steps"),
        "validation_c16": {"served_rows_pct": -0.87, "served_tokens_pct": -0.44, "replay_rows_pct": [-1.00, -1.01], "replay_tokens_pct": [-0.50, -0.50]},
        "step_time_models": {
            "A": "tp4-ring engine grid fit (same image, four boots) + a constant served overhead (c16 97.9 ms, c8 66.0 ms, c1 13.1 ms)",
            "B": "served per-batch regression of step time on tokens and wasted rows (10 x 30 s batches x 4 boots a loop)",
            "M": "w A + (1 - w) B, w set so that row cost 0.020 vs 0.035 reproduces experiments/2026-10-18-dspark-linear-budget (c16 +0.79%, sampled c8 -0.28%, c1 +0.21%), bracketed +-0.5%",
        },
        "986": {c: replay(c, "986") for c in ("c16", "c8", "c1", "c1-ctx32k", "c1-ctx64k", "c1-ctx128k")},
        "986_terms_at_c16": {"64_row_knee_and_wave_only": replay("c16", "986-no-tile-steps"), "tile_steps_only": replay("c16", "986-no-64-knee")},
        "corrected_knee_from_tp4_grid": {c: replay(c, "1107+grid-64-knee") for c in ("c16", "c8", "c1")},
        "row_cost_0.020_check": {c: replay(c, "1107-row-cost-0.020") for c in ("c16", "c8", "c1")},
        "oracle_upper_bounds": {c: {"A": replay(c, "oracle-A"), "B": replay(c, "oracle-B"), "M": replay(c, "oracle-M-central")} for c in ("c16", "c8", "c1")},
        "reading": ("Every re-pricing of #986's terms, including the knee re-calibrated to the grid (jump 6.7 rows, 0.28x slope above 64), "
                    "predicts |change| <= 0.16% at c16, <= 0.01% at c8 and 0 at c1. The budget optimum itself cannot be signed from this "
                    "evidence: oracle A drafts 33% more rows at c16 (+5.2% under A, -5.1% under B), oracle B 19% fewer (+1.1% under B, "
                    "-8.4% under A); under M the c16 optimum drafts more (+0.4 to +2.3% depending on w), the direction TODO.md lists "
                    "as tried and not adopted (lowering the row cost). That is the recipe constant, a separate variable, not a correction "
                    "of #986."),
    },
    "serving_step_vs_grid": {
        "served_step_ms": {c: round(reg[c]["step_ms_mean"], 2) for c in ("c16", "c8", "c1")},
        "grid_step_at_served_rows_ms": {"c16": "~28 (16x48)", "c8": "~22 (8x24 to 8x48)", "c1": "~16 (1x3 to 1x6)"},
        "per_batch_regression": {c: {k: round(v, 3) for k, v in tgt["token_wasted_regression"][c].items()} for c in ("c16", "c8", "c1")},
        "reading": ("served steps are 1.8x (c1) to 4.5x (c16) the grid's; across 30 s batches c16 output tok/s stays within about 1% while "
                    "tokens a step vary +-4%, so step time moves with tokens a step (about 3 ms a token). The grid's dummy steps route "
                    "identical tokens (servebench/engine.py); real text spreads rows over many more experts. Draft-budget changes move "
                    "tokens a step and step time together, as the 2026-10-18 row-cost A/B found (+7-12% tokens a step, -7-11% steps/s, "
                    "same tok/s)."),
    },
    "correction": None,
    "patch": None,
    "serving_measurement_to_run": {
        "required": "none: drop #986 from this release (its own mark cannot be met by any re-pricing of its terms; predicted changes are below the A/A floor)",
        "optional_if_the_context_band_is_proposed_again": ("servebench closed loop, c1 and c2, --context 65536 (and 131072), 300 s windows, two boots each, "
                                                          "candidate-47 against candidate-47-no986 on the same image; replay predicts -0.26% (M; A -0.39%, "
                                                          "B +0.51%) at 64K and -0.47% (M; A -0.66%, B +0.76%) at 128K at c1, i.e. at or below the 0.35% floor"),
    },
    "host": {"dgx1": "read-only file copies; no GPU work", "sp3": "spark-0011, CPU-only replay in spark-ds41f:1107 (sha256:3f32a962ce04b7c8d85337cd427a8946b8b6bcaadd97cc8d8a8cd14e8d68a6ce), --gpus all only so vLLM imports resolve; flock /tmp/spark-lab-gpu.lock, docker ps empty and no hold file checked inside the lock"},
    "inputs": {"patch_0762_sha256": "e5ff525055fd02feae330e423c9882542b22d8ef56fd202b7d5e5ef5859884c4", "pr_head": "c00819eeb06b22493277c241ba4414a7d6415cba",
               "adaptive_verification_1107_sha256": "4560e7fefd05b3914c95b0fa574b2a4bd0cce8b05558d7772e8b3bede5bc4801",
               "adaptive_verification_986_sha256": "bf802343732a90e69b0d313a60bcb853b1385c4700227d9d5a895f7c7fb86a56"},
}
json.dump(result, open(f"{R}/result.json", "w"), indent=1)
print("ok", len(json.dumps(result)))
