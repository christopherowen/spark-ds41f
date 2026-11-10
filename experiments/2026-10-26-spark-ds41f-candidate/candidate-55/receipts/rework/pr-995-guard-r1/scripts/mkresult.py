#!/usr/bin/env python3
"""mkresult.py DIR: result.json from DIR/raw/analysis.json and the fixed facts of this rework."""
import hashlib, json, sys
D = sys.argv[1]
a = json.load(open(f"{D}/raw/analysis.json"))
patch = "0781-fix-spec_decode-guard-negative-slots-in-copy_alloc-a.patch"
sha = hashlib.sha256(open(f"{D}/{patch}", "rb").read()).hexdigest()
def cells(t, keys=("base", "cand", "delta_pct", "verdict")):
    return {k: {kk: v[kk] for kk in keys if kk in v} for k, v in t.items()}
r = {
 "pr": 995, "label": "995-guard", "kind": "fix (robustness)", "round": "candidate-55 (round 3), rework of the selection's rework_needed[995-guard]",
 "date": "2026-11-10",
 "verdict": "ready",
 "verdict_basis": [
  "Canary: test_negative_slot_bounds_guard_in_alloc_and_apply passes on the candidate arm in both boots (63 passed each; candidate-54's own suite: 62 passed), and fails on candidate-54's kernels (copy_alloc writes int_state[-1]: the canary row 888888 reads 0, 0).",
  "Bit-exact: 882 copy steps a boot (copy_match, copy_alloc, copy_apply; 126 cells; copy state carried over seven steps; a mid-prefill step; 5,892 copies taken and 120 refused) hash the same on all four boots of both arms.",
  "Copy-step bench (benchmark_copy_drafts.py --gate --cases copy, the copy step of #977's own-mark harness, unmodified): within budget on every boot of both arms; against candidate-54, ABBA in one session, the six boots an arm at 201 replays pooled: 24 of 24 cells same (-3.2%..+5.9%). Single ABBA sets flag 0-5 cells either way, a different cell each time (bimodal medians); no cell is flagged slower in two sets, and several flagged-slower cells are flagged faster in another set."
 ],
 "verdict_caveat": "The guard is not free at kernel level. With launch latency amortized (32 copy_alloc + copy_apply pairs in one CUDA graph, 4 boots an arm, ABBA twice) the candidate's two kernels are slower in all 24 cells by +0.004..+0.079 us a pair (mean +0.040 us on 3.07..3.39 us, +0.1%..+2.4%; minimum replay +0.05 us); 3 of 24 cells are beyond the pooled boot range, 9 and 10 in the two ABBA sets. That is about 40 ns of device time a decode step, below the copy-step bench's resolution (its boot-to-boot spread is 0.3-2 us) and about 1e-6 of a 33 ms c1 step; serving launches these kernels eagerly, and the eager timing shows no cell slower (3 faster). The mark names 'a copy-step bench' and points at #971's/#977's own-mark harness: on that bench nothing is slower. If the coordinator counts the 32-pair kernel bench as a copy-step bench, the mark is missed by that ~40 ns and the verdict becomes not-defensible (a robustness fix for a path no serving caller takes, at a small but real cost).",
 "serving_effect": {
  "any_serving_path_passes_negative_slot_to_copy_alloc_or_copy_apply": False,
  "statement": "No serving path passes a negative slot to copy_alloc or copy_apply. The guard is a robustness fix with no serving effect: serving output and copy state are bit-identical to candidate-54's, and the only cost is the guard's branches (about 40 ns of device time a step).",
  "caller_trace": [
   "Every caller of tilelang_copy.copy_alloc/copy_apply in the tree (git grep, candidate-54 5f021d9a): vllm/v1/worker/gpu/spec_decode/copy_drafts.py (CopyDrafter._prepare -> copy_alloc, line 179; CopyDrafter.propose -> copy_apply, line 233), benchmarks/kernels/benchmark_copy_drafts.py (idx = arange) and tests/v1/spec_decode/test_copy_drafts.py.",
   "CopyDrafter.prepare_alloc and propose are called only from GPUModelRunner.sample_tokens (model_runner.py 3687 and 3740), both with input_batch.idx_mapping, and only on the last PP rank (a non-last rank returns at 3455-3473).",
   "input_batch is execute_model_state.input_batch: either prepare_inputs (model_runner.py 1806-1807: async_tensor_h2d(batch_req_state.idx_mapping_np)), whose idx_mapping_np is req_states.req_id_to_index[req_id] for each scheduled request (1747-1748, dict __getitem__, no default), or InputBatch.make_dummy (input_batch.py 213-214: arange(num_reqs)). Slots come from RequestState.add_request: free_indices.pop(), free_indices = list(range(max_num_reqs)) (states.py 31, 115-117). maybe_restore_pcp_for_sampling returns the batch unchanged without prefill context parallelism (pcp_manager.py 837-839). idx_mapping is not padded to num_reqs_after_padding (a separate count).",
   "The copy path's only -1 entries: match_idx_mapping = np.where(incomplete_prefill, -1, idx_mapping_np) (model_runner.py 3671-3684), passed to copy_match alone (copy_drafts.py 171-178), which reads total_len[slot] only if slot >= 0 (tilelang_copy.py 235-238) and so writes no source for those rows; copy_alloc/copy_apply get the real slots for the same rows.",
   "Other -1 producers do not reach the copy kernels: PPHandler.get_prev_sampled_outputs (pp_utils.py 140) feeds postprocess_sampled only (model_runner.py 1589); tilelang_inputs' out_sample_idx_mapping -1 entries (tilelang_inputs.py 2509, 2747) are sampler-side.",
   "Warm-up: CopyDrafter.register_warmups -> tilelang_inputs.declare(copy_calls) -> prepare.declare -> tilelang.prepare: compiles, launches nothing.",
   "The copy kernels run eagerly in sample_tokens (no CUDA-graph capture around them)."
  ]
 },
 "patch": {
  "file": patch, "sha256": sha, "commit": "78ce3c6e3e7624675db7351421e253f44f449962", "tree": "f77fc74bb76e72fc9b33046c461bbcd0bb3d4589",
  "author": "Christopher Owen <3221756+christopherowen@users.noreply.github.com>", "date": "Sun, 8 Nov 2026 18:27:04 +0000 (the PR patch's)",
  "subject": "fix(spec_decode): guard negative slots in copy_alloc and copy_apply",
  "base": {"patch_head": "4e8133ff10a125b90753c2727f162e29fe37f30c", "tree": "5f021d9a7f01c03e2389dd3d8021759cfa19ea34", "series": "candidate-54/vllm/series (671 patches)"},
  "applies_as_0781": "git am --committer-date-is-author-date with the fixed committer on a fresh checkout of 4e8133ff reproduces commit 78ce3c6e, tree f77fc74b; git diff --check clean against 4e8133ff and against vLLM 19f2c20e (raw/amcheck.txt)",
  "pair_check_candidate_55": "with round-55-57-select/out/55 (1008, 978, 988-deadrow), git am in both orders: tree 06a2f3d420a70893a03a9d707e6fc9dd32e07f9b both ways; no shared file (raw/pair-check-55.txt, raw/pair-check-55-files.txt)",
  "numbering_note": "Named 0781 as asked: it applies first after candidate-54's series. candidate-54's own folder also has a 0781 (64k-chunked L2 prefetch); if candidate-55 numbers on from candidate-54's last patch, rename it 0792 (git format-patch --start-number 792); the content does not change.",
  "files": {"vllm/v1/worker/gpu/spec_decode/tilelang_copy.py": "+48 -37 (copy_alloc_kernel, copy_apply_kernel)", "tests/v1/spec_decode/test_copy_drafts.py": "+93"},
  "source": {"pr_head": "aa65752e9864badb8873de80b29cc9ea7211c1aa", "pr_patch": "experiments/2026-11-08-copy-drafter-wide-ramp-and-guards/vllm/0762-perf-spec-decode-copy-drafter-guards-and-wide-ramp.patch", "pr_patch_sha256": "b4040c07950344e3978f4e4fe3bcd58e513d84170c475ef51ae783f900228218", "split_995_alloc_apply_sha256": "eac25e5c9a5e72e847b55e9cfe3fc816a79cfdd0af8c886d06fc18a1fb139618", "split_3way_result_sha256": "3c150312aa85ca39fd409541187493973ee75ab4cea02bef9560ffc9c58e8b93 (reproduced)"}
 },
 "taken_from_995": [
  "tilelang_copy.py hunk 5 (copy_alloc_kernel: `if slot < 0:` writes chosen (-1, 0, 0), estimates PRIOR_HITS / PRIOR_TRIALS, alloc block; the rest moves into `else:`), every line as the PR has it except the doublings update's argument line",
  "tilelang_copy.py hunk 6 (copy_apply_kernel: `slot >= 0` added to the use_copy condition and to the pending-state reset), unchanged; applied 3-way, its context moved with 0789's survival loop",
  "test_negative_slot_bounds_guard_in_alloc_and_apply, its 91 lines unchanged, appended after test_copy_match_long_context_two_tier_scan (two blank lines before it; the PR had three)"
 ],
 "left_out_of_995": [
  "Cap decay (PR rationale 2): the else branch keeps the base's `T.if_then_else(accepted >= live, T.min(doublings + 1, MAX_DOUBLINGS), 0)`, re-indented; with it ref_select's and design_walk's decay changes and test_partial_acceptance_steps_down_cap_doublings_gently",
  "Hunk 1 (module docstring): it holds only the decay sentence, so dropping the sentence drops the hunk; the docstring is the base's",
  "copy_match anchor pruning, need_len tile bound and one-variable hit reduction (PR rationale 3, hunks 2-4): conflicts with #977's two-tier scan (0790) in three hunks; to be its own later experiment on the new base, measured with #977's bench_copy_match_long_context.py",
  "31/63-draft ramp (PR rationale 4): SPARK_DS41F_COPY_MAX_DRAFTS (vllm/config/speculative.py), greedy_rejection_kernel widths above 32 (tilelang_sampling.py), benchmark_copy_drafts.py --lookahead, test_wide_lookahead_31_and_63_greedy_verification_and_ramp: opt-in, the recipe keeps 16 drafts"
 ],
 "host": "sp8 (spark-0016, NVIDIA GB10), 2026-11-10T01:27Z..01:47Z; shared with another agent's jobs (never concurrent with ours: docker_ps_before none in all 36 blocks)",
 "arms": {
  "image": "spark-ds41f:candidate-50 sha256:8c1ea1f62f62c3e2adc03b645c091a5499cf7163101aca32875c894b96f155db (vllm.tree d5778ab5)",
  "base": "+ the 36 files that differ between d5778ab5 and candidate-54's tree 5f021d9a, mounted read-only (all modified, none added or removed; 2 are C sources of engram_rows, not on the copy path)",
  "cand": "+ the same 36 paths from the guard's tree f77fc74b: differs from base in tilelang_copy.py and test_copy_drafts.py only (raw/overlay-sha.txt; container-side hashes in raw/verify-*.txt; tilelang_copy imported from the mounted path, SHA-256 93173530 base / b38cf4c9 cand, in every bench boot)",
  "order": "ABBA in one session: base, cand, cand, base (q.sh, q2.sh, q3.sh), then base, cand, cand, base, base, cand, cand, base (q4.sh); a boot is a fresh container; separate TileLang caches per arm (removed afterwards)"
 },
 "tests": a["tests"],
 "identity": {
  "main": {k: a["identity"][k] for k in ("cells", "steps_hashed_per_boot", "all_cells_sha", "cells_differing_between_any_boots", "bit_exact", "coverage_base_boot1", "tilelang_copy_sha256")},
  "refusals": {k: a["identity_refusals"][k] for k in ("cells", "steps_hashed_per_boot", "all_cells_sha", "cells_differing_between_any_boots", "bit_exact", "coverage_base_boot1", "tilelang_copy_sha256")},
  "harness_note": "guard995.py's docstring says its DSpark confidences admit and refuse copies; in the run every offered copy was taken (a fresh copy state's 0.95 prior beats any DSpark yield), so the refusal branch whose condition the guard edits was covered by ids995b.py (q2.sh) instead: 120 refused row-steps, bit-exact on all four boots."
 },
 "bench": {
  "copy_step_gate_21_replays": {"what": "benchmark_copy_drafts.py --gate --cases copy, unmodified (CUDA-graph replay of copy_match + copy_alloc + copy_apply, 64 MiB flush, median of 21); boots 1-2 an arm", "budget": "0 slower on all 4 boots", "summary": a["gate"]["summary"], "cells": cells(a["gate"]["cells"])},
  "copy_step_harness_21_replays": {"what": "guard995.py: the same cells and launch, its own containers", "summary": a["timing"]["step_us"]["summary"], "cells": cells(a["timing"]["step_us"]["cells"])},
  "copy_step_confirm_201_replays": {"what": "confirm995.py: the gate's module unmodified, timer at 201 replays (as candidate-53's confirm977.py); three ABBA sets, boots 1-6 an arm", "sets": {k: {"summary": v["summary"], "flagged": cells(v["flagged"])} for k, v in a["confirm201"]["sets"].items()}, "pooled_6_boots_summary": a["confirm201"]["pooled_summary"], "pooled_cells": cells(a["confirm201"]["pooled_cells"], ("base", "cand", "delta_pct", "welch95", "verdict")), "budget": "0 slower on all 12 boots"},
  "alloc_apply_graph_201_replays": {"what": "guard995.py: copy_alloc + copy_apply alone, one pair a graph", "summary": a["timing"]["alloc_apply_us"]["summary"], "note": "medians sit on the event timer's quantum (8.192 us in most cells)"},
  "alloc_apply_eager": {"what": "guard995.py: 200 back-to-back eager copy_alloc + copy_apply launches, wall us a pair, median of 5 (the serving path's launch mode)", "summary": a["timing"]["eager_alloc_apply_us"]["summary"]},
  "alloc_apply_32_pairs_a_graph": {"what": "k32995.py: 32 pairs a CUDA graph, 51 replays, median / 32; boots 3-6 an arm (two ABBA sets)", "sets": {k: {"summary": v["summary"]} for k, v in a["k32_pair_us"]["sets"].items()}, "pooled_4_boots_summary": a["k32_pair_us"]["pooled_summary"], "pooled_cells": cells(a["k32_pair_us"]["pooled_cells"], ("base", "cand", "delta_pct", "welch95", "verdict")), "all_cells_delta": "candidate higher in 24/24 cells on the median (+0.004..+0.079 us, mean +0.040), the minimum (mean +0.050) and the graph median"},
  "cell_rule": "candidate-53 own marks (analyze/common.py): two boots an arm, 'worse' only when every candidate boot is above every base boot and |delta| exceeds the larger within-arm boot change; pooled: the same over all boots an arm, with the larger within-arm range, and a Welch 95% interval on the boot medians"
 },
 "files": {
  "patch": patch, "overlay": ["overlay/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py (sha256 b38cf4c9...)", "overlay/tests/v1/spec_decode/test_copy_drafts.py (sha256 356687ff...)"],
  "commands": "commands.txt", "raw": "raw/ (every output as run, raw/commands.txt one block a container, raw/analysis.json, raw/inputs, raw/setup)",
  "scripts": "scripts/ (setup.sh, run.sh, verify.sh, q.sh..q4.sh, harness/guard995.py, ids995b.py, confirm995.py, k32995.py, analyze.py, mkresult.py, msg-0781.txt); harness/warm.py is candidate-53's, unchanged (on sp8)",
  "sp8": "~/spark-lab/agents/round3-995-guard: vllm clone (branch guard-995 = 78ce3c6e), overlays, scripts, outputs, patch/"
 }
}
json.dump(r, open(f"{D}/result.json", "w"), indent=1)
print("ok", sha)
