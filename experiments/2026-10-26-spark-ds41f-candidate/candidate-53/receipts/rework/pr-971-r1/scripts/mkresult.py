#!/usr/bin/env python3
"""mkresult.py DIR: result.json for #971 round 1 from DIR/raw/analysis.json and the receipts."""
import hashlib, json, statistics as st, sys
D = sys.argv[1]
a = json.load(open(f"{D}/raw/analysis.json"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def cells(pair, tau="recipe"):
    out = {}
    for k, v in a[pair].items():
        if not k.startswith(f"tau_{tau}/"):
            continue
        A, C = pair.split("_")[1], pair.split("_")[3]
        out[k[len(f"tau_{tau}/"):]] = {
            f"{A}_shape_graph_x_live": v[f"{A}_shape_graph_x_live"], f"{C}_shape_graph_x_live": v[f"{C}_shape_graph_x_live"],
            f"{A}_ms": v["base"], f"{C}_ms": v["candidate"],
            f"{A}_mean_ms": round(st.mean(v["base"]), 3), f"{C}_mean_ms": round(st.mean(v["candidate"]), 3),
            "delta_pct": v["delta_pct"], "welch95_ms": v.get("welch95"), "boot_spread_ms": v["boot_spread"], "verdict": v["verdict"]}
    return out
aa = {k: {"ms": [v["base"], v["candidate"]], "delta_pct": v["delta_pct"], "verdict": v["verdict"]}
      for k, v in a["same_shape_across_arms"].items() if k.startswith("orig_vs_r1/")}
aa_abs = sorted(abs(v["delta_pct"]) for v in aa.values())
patch = f"{D}/0782-perf-spec-decode-copy-drafter-survival-and-tile-cap.patch"
b2r = cells("cells_base_to_r1"); o2r = cells("cells_orig_to_r1")
pol = {arm: a["policy"][f"{arm}-boot1"]["recipe"] for arm in ("base", "orig", "r1")}
for arm in pol:
    pol[arm] = {k: v for k, v in pol[arm].items() if k not in ("kind", "tau_name")}
result = {
 "pr": 971,
 "round": 1,
 "candidate": "candidate-53",
 "pr_head": "9ef22e1fcdf4c83d782d863dafbd51278362b6d5",
 "patch": "0782-perf-spec-decode-copy-drafter-survival-and-tile-cap.patch (round 1), git format-patch, author Christopher Owen <3221756+christopherowen@users.noreply.github.com>, author date Sun, 8 Nov 2026 12:45:00 +0000 (the PR's); rationale with a 'Round 1' section",
 "patch_sha256": sha(patch),
 "replaces": {"patch": "candidate-53/vllm/0782-perf-spec-decode-copy-drafter-survival-and-tile-cap.patch", "sha256": sha(f"{D}/inputs/0782-perf-spec-decode-copy-drafter-survival-and-tile-cap.patch")},
 "verdict": "corrected-meets-mark",
 "verdict_scope": "sp8-evaluable parts of the mark declared below; paired decode and temperature-0 text are the tp3-switched screen's and the determinism gate's",
 "reason": ("The corrected #971 keeps the copy_apply survival gate and the single-request tile cap and drops the zero-confidence dead-row floor. "
            "At the recipe's tau 0.2, on the image, same-session ABBA: the c1 initial copy step moves from the 24-row graph (17 rows) to the 16-row graph (16 rows), "
            f"{b2r['TP3/warm/initial_step']['delta_pct']}% at TP3 ({b2r['TP3/warm/initial_step']['base_mean_ms']} -> {b2r['TP3/warm/initial_step']['r1_mean_ms']} ms, "
            f"95% interval {b2r['TP3/warm/initial_step']['welch95_ms']} ms; {b2r['TP3/128MiB_wash/initial_step']['delta_pct']}% washed) and "
            f"{b2r['TP4/warm/initial_step']['delta_pct']}% at TP4 ({b2r['TP4/128MiB_wash/initial_step']['delta_pct']}% washed), all beyond the interval and the A/A spread; "
            "copy_apply admits 2- and 3-token copies (base: not); no cell is slower; the PR's tests pass twice (114 passed, 1 skipped). "
            "The dropped part changes nothing at tau 0.2: every policy output and all 24 recipe-tau dead-row cases are bit-identical between original #971 and round 1, "
            "and the shrunk step both choose (16-row graph, 6 unmasked rows) times the same at TP3 (+0.43% warm, +0.29% washed: same under the cell rule, as identical kernels on one shape must)."),
 "change": {
  "kept": [
   "tilelang_copy.copy_apply_kernel: the replacement gate compares the copy's expected yield with DSpark's cumulative survival (dspark_survival *= c_j; yield += survival) instead of the raw sum of conditional confidences (rationale 1), with the module docstring line",
   "adaptive_verification: SINGLE_REQ_TILE_SPILL_SCALE = 2.0, the manager's single_req_tile_spill_scale, _step_cost(rows, num_sampling_requests=...) using it above the 16-row knee for one sampling request and HIGH_ROW_COST_SCALE otherwise, and get_num_tokens passing num_sampling_requests (rationale 2)",
   "adaptive_verification.reallocate_drafts: for one request, capacities = draft_budget directly (skips assign_draft_budget, which the tile cap would otherwise newly launch on c1 steps). Equivalent: the image's assign_draft_budget admits exactly draft_budget slots for one request at every budget 1..16 with flat, shrunk-copy and mixed confidences, on all three arms (raw/ab-*.jsonl, assign_draft_budget_one_request)",
   "tests: test_copy_drafts.py's cumulative-survival reference and gate cases (unchanged); test_single_request_tile_spill_caps_16th_copy_draft_at_16_rows (the PR's tile-cap assertions, unchanged)"],
  "dropped": [
   "adaptive_verification: _ZERO_CONFIDENCE_LOG_TAU = -1e30 and dead_row_args' has_copy_tail branch (returning DeadRows with log_tau -1e30 on wide copy steps when tau <= 0); dead_row_args is back to the base's text",
   "model_runner.GPUModelRunner.declare_step_kernels: dead_rows = adaptive and (dead_rows_enabled() or self.copy_drafter is not None) and ...; back to the base's line, so the patch no longer touches model_runner.py",
   "tests: the dead_row_args block at the end of the tile-cap test (and '_and_masks_shrunk_tail' in its name); the copy-drafter case of test_prepare_kernels.py::test_dead_row_caps_are_declared_before_the_manager_exists, so the patch no longer touches test_prepare_kernels.py"],
  "separable": True,
  "why_separable": ("The dead-row floor is three self-contained edits (a constant, one branch of dead_row_args, one boolean in declare_step_kernels) that nothing else in the patch reads: "
                    "the survival gate is inside copy_apply_kernel, the tile cap inside _step_cost/get_num_tokens/reallocate_drafts, and neither calls dead_row_args or depends on declare_step_kernels' flag. "
                    "Removing them leaves every kept hunk byte-identical (raw/setup/0782-orig-vs-r1-patch-body.diff: only removed lines and hunk offsets differ), and 0783, 0784 and 0785 apply unchanged after it."),
  "diff_vs_original_candidate_53_tree": "raw/setup/c53-orig-vs-r1-tree.diff: 4 files, +5/-37 (tests/v1/spec_decode/test_adaptive_verification.py, tests/v1/worker/test_prepare_kernels.py, vllm/v1/worker/gpu/model_runner.py, vllm/v1/worker/gpu/spec_decode/adaptive_verification.py)",
  "patch_stat": "4 files, 102 insertions, 14 deletions (was 6 files, 138/18)"
 },
 "why_the_dropped_part_is_inert_here": {
  "recipe": "config/cluster.json sets SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2 in its top-level environment; no shape overrides it (tp4-ring, tp4..tp8-switched, tp3-triangle, tp3-switched, tp3-triangle-4k). The recipe's speculative config (dspark, enable_adaptive_verification, greedy drafts, row cost 0.035) turns the copy drafter on (SpeculativeConfig sets copy_drafts).",
  "code": [
   "dead_rows_enabled() is _DEAD_ROWS_TAU > 0: true at 0.2, so declare_step_kernels' '(dead_rows_enabled() or self.copy_drafter is not None)' equals the base's 'dead_rows_enabled()'.",
   "effective_dead_rows_tau() returns tau (0.2), or min(tau, 2 row_cost / (1 + 3 row_cost)) = 0.0633 once calibrated: positive either way. With tau > 0 the original dead_row_args returns None exactly when no drafts were admitted and otherwise DeadRows(..., log(tau)): the base's result. Its has_copy_tail branch and -1e30 floor are reached only at tau <= 0.",
   "GPUModelRunner calls dead_row_args on every adaptive step (no batch sharder), so at tau 0.2 the base already marks a shrunk copy's -1 tail (confidence 0, survival 0 < tau) as padding: the cut and mask the PR's tau-0 floor produced."],
  "measured_identity": {
   "policy_identical_r1_vs_orig": a["policy_identity_r1_vs_orig"],
   "dead_row_grid": {k: a["dead_row_grid"][k] for k in ("cases", "boot_to_boot_identical", "r1_vs_orig_differing_cases", "r1_vs_base_differing_cases", "orig_vs_base_differing_cases", "recipe_cases_r1_vs_orig_identical")},
   "dead_row_grid_note": "36 cases: tau mode (recipe 0.2 uncalibrated, recipe calibrated = 0.0633, zero) x step width (5, 16) x drafts admitted (yes, no) x draft rows (shrunk copy, full copy, low DSpark). Compared fields: DeadRows present, log_tau, cut, unmasked rows, the whole is_padding mask, input ids, logits indices, verified count. Round 1 equals the base in all 36; the original differs from both only in the three tau-0, 16-wide, admitted cases (where it masks the -1 tail: 6 unmasked rows, cut 5, instead of 16 rows).",
   "declare_step_kernels_dead_rows": {
    "base": [(r["dead_rows_enabled"], r["copy_drafter"], r["dead_rows"]) for r in a["declare_dead_rows"]["base-boot1"]],
    "orig": [(r["dead_rows_enabled"], r["copy_drafter"], r["dead_rows"]) for r in a["declare_dead_rows"]["orig-boot1"]],
    "r1": [(r["dead_rows_enabled"], r["copy_drafter"], r["dead_rows"]) for r in a["declare_dead_rows"]["r1-boot1"]],
    "note": "with the module's dead_rows_enabled() (tau 0.2) all three declare dead rows with or without a copy drafter; only with dead_rows_enabled forced False (tau 0) does the original differ"},
   "shrunk_step_timing_orig_vs_r1_tau_0.2": {k: o2r[k] for k in ("TP3/warm/shrunk_step", "TP3/128MiB_wash/shrunk_step", "TP4/warm/shrunk_step", "TP4/128MiB_wash/shrunk_step")},
   "shrunk_step_timing_note": "Both arms choose the same shape (16-row graph, 6 unmasked rows) and run identical kernels on it, so these cells are an A/A. TP4 warm reads -3.79% 'better' under the cell rule: that is the session's A/A spread (identical code on one shape), not the patch; TP3 warm and washed and TP4 washed read same."
  }
 },
 "mark": {
  "declared": ("sp8, on the image at the recipe's tau 0.2, 2 boots an arm, same-session ABBA against the base: the single-stream (c1) initial copy step faster beyond the interval at TP3 and TP4 "
               "(the 16-row graph instead of the 24-row graph); copy_apply admits the 2- and 3-token copies DSpark [0.70..0.50] blocks; no cell of the step bench slower; "
               "the post-reject shrunk step unchanged against original #971 (the dropped part inert). tp3-switched screen (coordinator): paired decode at 1 stream better and at 8 streams not worse; temperature-0 text unchanged."),
  "declared_output_change": "drafting and verification rows (copy continuation yield, single-request 16-row cap); temperature-0 text must not change; sampled realizations may change. The 'wide-step padding mask' is no longer part of the patch.",
  "dropped_from_the_mark": "the post-reject shrunk step faster beyond the interval (it exists only at tau 0)"
 },
 "parts": [
  {"part": "c1 initial copy step faster beyond the interval at TP3", "verdict": "pass",
   "detail": f"warm {b2r['TP3/warm/initial_step']['base_ms']} -> {b2r['TP3/warm/initial_step']['r1_ms']} ms ({b2r['TP3/warm/initial_step']['delta_pct']}%, Welch95 {b2r['TP3/warm/initial_step']['welch95_ms']} ms); 128 MiB wash {b2r['TP3/128MiB_wash/initial_step']['base_ms']} -> {b2r['TP3/128MiB_wash/initial_step']['r1_ms']} ({b2r['TP3/128MiB_wash/initial_step']['delta_pct']}%, {b2r['TP3/128MiB_wash/initial_step']['welch95_ms']}). Own-mark run: -5.11% / -5.29%."},
  {"part": "c1 initial copy step faster beyond the interval at TP4", "verdict": "pass",
   "detail": f"warm {b2r['TP4/warm/initial_step']['base_ms']} -> {b2r['TP4/warm/initial_step']['r1_ms']} ms ({b2r['TP4/warm/initial_step']['delta_pct']}%, {b2r['TP4/warm/initial_step']['welch95_ms']}); wash {b2r['TP4/128MiB_wash/initial_step']['base_ms']} -> {b2r['TP4/128MiB_wash/initial_step']['r1_ms']} ({b2r['TP4/128MiB_wash/initial_step']['delta_pct']}%, {b2r['TP4/128MiB_wash/initial_step']['welch95_ms']}). Own-mark run: -4.17% / -5.54%."},
  {"part": "copy_apply survival gate admits 2- and 3-token copies", "verdict": "pass",
   "detail": "image kernel, offered 2/3/4/8 at the 0.95 prior against DSpark [0.70, 0.65, 0.60, 0.55, 0.50]: base admits [no, no, yes, yes], r1 and original [yes, yes, yes, yes], every boot; the PR's own screen on r1 (2 boots) passes its gate asserts (cumulative survival 1.6532 against raw sum 3.0). copy_apply kernel, 8 requests: 2.91 / 2.92 us warm, 3.81 / 3.88 us washed."},
  {"part": "no slower cell", "verdict": "pass",
   "detail": "no base -> r1 cell is 'worse' at tau 0.2 (or at tau 0). The shrunk step at tau 0.2 (24x6 -> 16x6): TP3 -0.14% warm, +0.22% washed (Welch95 +0.02..+0.26 ms on 62.7 ms, inside the boots' spread; the A/A of identical code on 16x6 washed is +0.29%), TP4 -4.26% warm ('better' by the cell rule, within the A/A spread), -0.61% washed."},
  {"part": "dropped dead-row floor changes nothing at tau 0.2", "verdict": "pass",
   "detail": "policy outputs identical in all 4 boots; 24/24 recipe-tau dead-row cases bit-identical; declare_step_kernels flag identical; shrunk step on the shape both choose: TP3 +0.43% warm, +0.29% washed, TP4 -0.36% washed (same), TP4 warm -3.79% (an A/A: identical kernels and shape)."},
  {"part": "the PR's tests", "verdict": "pass", "detail": f"r1, -v --noconftest, two container runs: {a['pytest_r1']['boot1']}; {a['pytest_r1']['boot2']}. The same 114 passed and 1 skipped as the own-mark run on the original, with the tile-cap test renamed."},
  {"part": "paired decode at 1 stream better, at 8 streams not worse; temperature-0 text unchanged", "verdict": "cannot-evaluate-on-sp8",
   "detail": "multi-node: the tp3-switched screen with its single-stream and sampled companions, and the determinism gate. The dropped part was inert at the recipe's tau, so a decode measurement of original #971 at the recipe also describes round 1."}
 ],
 "measured": {
  "gate_copy_apply_offered_2_3_4_8": {arm: [x[1] for x in a["gate_copy_apply"][f"{arm}-boot1"]] for arm in ("base", "orig", "r1")},
  "policy_tau_0.2_boot1": pol,
  "cells_base_to_r1_tau_0.2": b2r,
  "cells_orig_to_r1_tau_0.2": o2r,
  "cells_base_to_r1_tau_0": cells("cells_base_to_r1", "pr"),
  "cells_base_to_r1_tau_0_note": "tau 0 is not served; shown for completeness. Without the floor, r1's shrunk step at tau 0 is its 16-row initial-step shape (16 unmasked rows): still faster than the base's 24x17, but not the original's 16x6.",
  "aa_identical_kernels_orig_vs_r1": aa,
  "aa_summary": f"orig vs r1 on identical shapes and kernels (16 cells): median |change| {st.median(aa_abs):.2f}%, max {max(aa_abs):.2f}%; the cell rule flags {sum(1 for v in aa.values() if v['verdict'] != 'same')} of 16 (two boots an arm). The initial-step gains (4.7-6.6%) exceed every A/A cell on the 24x17 and 16x16 shapes (max 1.78%).",
  "assign_draft_budget_one_request_equals_budget": a["assign_draft_budget_one_request"],
  "pr_script_r1": {k: {"gate": [(g["offered"], g["parent_admits"], g["candidate_admits"]) for g in v["gate"]], "copy_apply_kernel_us": v["copy_apply_kernel_us"], "stopped_at": "line 294, assert dead_args is not None (section 2: its make_manager forces dead_rows_tau = 0.0, so it asserts the dropped floor); its tile-cap asserts before it (c1 parent 17 tokens / budget 16, candidate 16 / 15) passed; its section 3 timing does not run"} for k, v in a["pr_script_r1"].items()},
  "pytest_r1": a["pytest_r1"]
 },
 "series": {
  "order": ["0781 #960", "0782 #971 round 1", "0783 #977", "0784 #984", "0785 #998 (candidate-53's current 0785, unchanged)"],
  "sha256": {"0781": "95387b8e53e255d6...", "0782_r1": sha(patch), "0783": "6ebb45904f7478ad...", "0784": "b82e1eecdb74042f...", "0785": "cbb393495e7a1ce5..."},
  "applies": "git am on round-51-53-base (0bffb886, tree 14c5a2dc) from a fresh clone, all five clean (raw/setup/series-check.txt)",
  "tree": "684f4eeaac18fda9a661da5f09ecc1c500cdd2d8 (original candidate-53: c0e25ff14f7cbf760973002447e50ba507b987f9)",
  "git_diff_check": "clean",
  "lock_note": "candidate-53's source.json and upstreams.lock.json (patch_head, expected_tree c0e25ff1, patchset_sha256) describe the original series; a candidate carrying round 1 needs bin/spark build prepare --lock on a dgx node (expected_tree 684f4eea)."
 },
 "overlay": {
  "files": ["overlay/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py", "overlay/vllm/v1/worker/gpu/model_runner.py", "overlay/tests/v1/spec_decode/test_adaptive_verification.py", "overlay/tests/v1/worker/test_prepare_kernels.py"],
  "what": "the four files in which the corrected candidate-53 tree (684f4eea) differs from the original (c0e25ff1), at their corrected content. Mounted over spark-ds41f:candidate-53 they give the corrected tree; model_runner.py and test_prepare_kernels.py are then the original candidate-53's minus #971's dead-row lines.",
  "measured_arm": "r1 = spark-ds41f:candidate-50 (8c1ea1f6) + the 32 files in which 684f4eea differs from candidate-50's tree d5778ab5 (raw/setup/changed-files-r1.txt), read-only; SHA-256 as seen in the container in raw/setup/verify-r1.txt"
 },
 "host": "sp8 (spark-0016, NVIDIA GB10)",
 "image": "spark-ds41f:candidate-50 sha256:8c1ea1f62f62c3e2adc03b645c091a5499cf7163101aca32875c894b96f155db (no candidate-53 image on sp8; every candidate-53 patch is Python/TileLang source, JIT-compiled)",
 "arms": {
  "base": "the image + candidate-50's 4 no1009/overlay files (ro): the own-mark run's base arm",
  "r1": "the image + the corrected candidate-53 tree's 32 changed files (ro)",
  "orig": "the image + the original candidate-53 tree's 33 changed files (ro), byte-identical to the own-mark run's c53 overlay",
  "check": "import2-*: each arm imports the changed modules from the mounted paths; module facts: base (no tile cap, no survival gate), r1 (SINGLE_REQ_TILE_SPILL_SCALE 2.0, survival gate, one-request capacity line; no -1e30 floor, declare_step_kernels unchanged), orig (all of r1 plus _ZERO_CONFIDENCE_LOG_TAU -1e30, the has_copy_tail branch and the copy-drafter dead-row declaration)"
 },
 "method": {
  "order": "one lock hold (q2.sh batch 'ab', 20:30:31-20:32:14Z): base b1, r1 b1, orig b1, orig b2, r1 b2, base b2 (ABBA for base/r1 and for r1/orig), a fresh container each; boot2 runs time the shapes in reverse order (the own-mark harness's rule)",
  "warm_up": "harness/warm.py (one untimed GEMM, then >= 200 ms of bf16 GEMMs) in each process, then 50 graph replays per shape before 40 timed; one priming run per arm before the session filled each arm's TileLang cache",
  "workload": "harness/ab971r1.py: sections 1-3 are the own-mark harness ab971.py unchanged (the PR's 58-layer step workload: one routed MoE layer + one dense GEMM per CUDA-graph replay, x58, TP3 inter 768 and TP4 inter 576, warm and 128 MiB wash); each arm's own copy_apply and AdaptiveVerificationManager pick the step's shape; sections 4-6 add the assign_draft_budget equivalence, the dead-row grid and the declare_step_kernels flag",
  "env": "-e SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2 on every A/B run (config/cluster.json)",
  "cell_rule": "the own-mark rule (scripts/analyze/common.py): better/worse only when both boots of one arm lie on one side of both boots of the other and the mean change exceeds the larger boot-to-boot change; Welch 95% interval over the 80 samples an arm"
 },
 "analysis": [
  "The miss was that the PR's -60% shrunk step exists only at tau 0, which its script forces and the recipe never uses. Reading the code confirms it is structural, not a measurement accident: at any positive tau the original dead_row_args reduces to the base's, and declare_step_kernels' new 'or self.copy_drafter is not None' is shadowed by dead_rows_enabled() being true. So the floor is dead code on every served shape, and step 1 of docs/release-benchmark.md (take only the patches that act on this deployment) leaves it out.",
  "The rest of the patch does not depend on it. The survival gate lives in copy_apply_kernel and the tile cap in _step_cost / get_num_tokens / reallocate_drafts; neither reads dead_row_args or the declared flag. The one-request 'scheduled_drafts[0] = draft_budget' line belongs to the tile cap: once the cap trims a c1 budget from 16 to 15, the base path would launch assign_draft_budget, and for one request that kernel admits exactly draft_budget slots (checked on all arms at every budget 1..16), so the line only skips a launch.",
  "Measured on the image in one 103-second session, the kept parts reproduce the own-mark run: the c1 initial step at the 0.95 prior goes from the 24-row graph to the 16-row graph, 5.5% faster at TP3 and 4.7% at TP4 warm (5.8% / 6.6% washed), far outside the A/A spread of identical kernels between arms on those shapes (at most 1.8%). The survival gate admits the 2- and 3-token copies the base rejects. No cell is slower.",
  "Against the original #971 at tau 0.2 the step policy is bit-identical (shapes, DeadRows, cut, padding mask, input ids, logits indices, verified counts in 24 recipe-tau cases, plus the declare flag); the only differences in the 36-case grid are the three tau-0 wide-step cases the floor was written for. The shrunk step's timing on the shape both arms choose is therefore an A/A: TP3 +0.43% / +0.29%, same.",
  "Caveat on the cell rule: with two boots an arm it flags 5 of 16 A/A cells of identical code between r1 and orig (largest -3.79% at TP4 warm 16x6). The TP4 warm shrunk-step 'better' (-4.26% vs base) is that noise, not a claim; the initial-step gains are 2.6x or more the largest A/A cell on their shapes.",
  "The tile cap's serving value (1-stream paired decode) and temperature-0 identity remain the tp3-switched screen's and gate's; since the dropped floor was inert at the recipe's tau, any recipe-tau decode result for original #971 also describes round 1."
 ],
 "notes": [
  "Candidate-53's README says #960, #971, #984 and #998 all edit model_runner.py; with round 1, #971 no longer does (its files: adaptive_verification.py, tilelang_copy.py and two test files). #977 still shares tilelang_copy.py and test_copy_drafts.py with it.",
  "receipts/selection.json would record the dropped part under left_out, with the step-1 reason: it acts only at SPARK_DS41F_DSPARK_DEAD_ROWS_TAU <= 0, which no shape of config/cluster.json serves.",
  "The PR's screen_copy_drafter_smooth.py cannot run to the end on round 1 by construction (it asserts the dropped tau-0 floor at line 294); the policy-driven harness carries its timing instead.",
  "q.sh's import step stopped at its last check (inspect.getsource on a TileLang JITImpl, rc=1 on all three arms; every earlier line printed). import2-* (harness/import_check.py) is the corrected check.",
  "q.sh waited behind other agents' long GPU jobs between our short ones, so it was stopped while prime-r1 waited for the lock (not started) and replaced by q2.sh, which holds the lock once per step (prime about 4 min, script 20 s, ab 103 s, pytest about 2 min)."
 ],
 "not_done": [
  "No push, commit, PR, image build, cluster or window operation, and no dgx use.",
  "Paired decode (1 and 8 streams), the sampled companion and temperature-0 identity: multi-node, the coordinator's tp3-switched screen and gate.",
  "bin/spark doctor, bin/spark build check and bin/spark build prepare --lock: no repository change is proposed here; the series was replayed on sp8 with git am."
 ],
 "receipts": {
  "patch": "0782-perf-spec-decode-copy-drafter-survival-and-tile-cap.patch",
  "overlay": "overlay/",
  "commands": "commands.txt (setup, trees, arms, queue, analysis; then every GPU-lock block as run.sh logged it)",
  "raw": "raw/ (every run's output: ab-*.jsonl/.txt, script-r1-boot*.txt, pytest-r1-boot*.txt, import*/prime*; raw/analysis.json; raw/setup: tree and patch diffs, file lists, SHA checks, series check, queue logs, the round-1 message)",
  "scripts": "scripts/ (make_r1.py, run.sh, q.sh, q2.sh, verify.sh, check_verify.py, harness/, analyze/, mkresult.py, msg-r1.txt)",
  "inputs": "inputs/ (candidate-53's 0781..0785 from origin/main db30946f52)",
  "sp8": "~/spark-lab/agents/candidate-53-rework-971 (vllm clone with tags orig-series, r1-series, r1-0782, branch r1-final; TileLang caches c/tl-<arm>)"
 }
}
json.dump(result, open(f"{D}/result.json", "w"), indent=1)
print("verdict", result["verdict"]); print(result["reason"]); print(result["measured"]["aa_summary"])
