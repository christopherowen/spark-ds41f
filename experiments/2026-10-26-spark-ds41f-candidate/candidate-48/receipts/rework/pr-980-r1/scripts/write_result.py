"""Write result.json for the PR #980 r1 rework from numbers.json plus the analysis."""
import hashlib
import json
import sys

D = sys.argv[1]
n = json.load(open(f"{D}/numbers.json"))
patch = f"{D}/0764-perf-spec-decode-elide-tp-confidence-broadcast-and-lock-sampling-seeds.patch"
patch_sha = hashlib.sha256(open(patch, "rb").read()).hexdigest()

ab = n["stepcost_in_process"]
pp = n["stepcost_per_process"]


def ab_comp(c):
    return {b: {
        "host_us_geomean": ab[b]["components"][c]["host_us_geomean"],
        "c48_minus_1107_host_us_mean_over_14_shapes": ab[b]["components"][c]["c48_minus_1107_host_us_mean_over_shapes"],
        "c48_minus_1107_host_us_range_over_shapes": ab[b]["components"][c]["c48_minus_1107_host_us_range"],
        "r1_minus_1107_host_us_mean_over_14_shapes": ab[b]["components"][c]["r1_minus_1107_host_us_mean_over_shapes"],
        "r1_minus_1107_host_us_range_over_shapes": ab[b]["components"][c]["r1_minus_1107_host_us_range"],
        "largest_per_shape_ci95_us": ab[b]["components"][c]["max_ci95_us"],
    } for b in ("boot1", "boot2")}


result = {
    "pr": 980,
    "round": 1,
    "pr_head": "11b169b0dd07de9aa45ad6b9ab9f0f58156fb822",
    "host": "sp2 (spark-0010, NVIDIA GB10), single GPU; hold file absent, docker idle, every GPU command under flock /tmp/spark-lab-gpu.lock",
    "images": {
        "candidate-48": "sha256:d6ffd842978849651e756ebd39e9bf2da7ddb03e746113013533029878485a24",
        "1107": "sha256:3f32a962ce04b7c8d85337cd427a8946b8b6bcaadd97cc8d8a8cd14e8d68a6ce",
    },
    "mark": [
        "keep the lockstep fix: 0/1000 rank mismatches at TP3 and TP4 on the PR's own 1,000-step screen; its tests pass",
        "add no per-step cost on the greedy decode path: single-GPU step cost of the worker's sampler and spec-decode post-step path against 1107's files, both boots, at the grid's shapes",
        "explain or fix the TP2 crash",
        "(coordinator follow-up, same round) derive _seed_rng from the configured seed, identical on every TP rank; keep a deterministic fallback when none is configured; test that two seeds differ and two ranks with one seed agree; re-run the harness-fixed lockstep screen at TP3/TP4, both boots, and the PR's tests",
    ],
    "verdict": "corrected-meets-mark",
    "reason": (
        "The engine grid's timed steps run none of #980's changed lines, so #980 cannot add the +0.27% "
        "(about 55 us per 21.1 ms step) as per-step work. The grid times _dummy_run: a target forward on "
        "zeroed dummy tokens plus DSparkSpeculator.propose(dummy_run=True). That path never samples "
        "(sampler.py, states.py), never calls record_confidences (adaptive_verification.py), and DSpark "
        "with its TileLang confidence head never calls OnlineAcceptanceEstimator.predict; DFlash's "
        "predict_from_stats only serves probabilistic vocab-parallel drafts, and the recipe drafts greedily. "
        "On the real greedy decode step, a line trace of the five post-step components finds exactly three "
        "changed lines executed, all in record_confidences (adaptive_verification.py 269, 270, 274: the "
        "tp_invariant getattr hoisted ahead of the TP check, plus a callable() test). In-process A/B/C "
        "timing puts the cost at 0.0 to 0.2 us per step, the same as the A/A between byte-identical code. "
        "The PR's -17 us broadcast saving does not apply to production either: DSparkSpeculator has "
        "exposed tp_invariant_confidence since 0568 and overrides the property #980 adds to "
        "DraftModelSpeculator, so the production broadcast was already elided in 1107. "
        "The correction (r1) drops that adaptive_verification.py hunk. Its callable() branch is dead: "
        "the model runner passes the speculator's value to the manager as a bool when it creates it "
        "(model_runner.initialize_kv_cache). With the hunk gone, record_confidences is byte-identical "
        "to 1107's, and the trace finds zero changed lines on the greedy path. "
        "The same round fixes #980's --seed regression: SamplingStates._seed_rng was "
        "np.random.default_rng(0) whatever --seed was, while 1107's unseeded request seeds came from "
        "the global NumPy RNG the worker seeds with set_random_seed(ModelConfig.seed). Now the model "
        "runner hands ModelConfig.seed to Sampler(seed=...) and SamplingStates(seed=...), which draws "
        "unseeded requests' seeds from np.random.default_rng(seed). ModelConfig.seed is the same on "
        "every TP rank, so lockstep holds; with no configured seed it falls back to vLLM's default 0, "
        "the value 1107 seeded with by default (and #980's default_rng(0)). A new test checks that two "
        "seeds give different unseeded seeds, that ranks with one seed agree however each perturbs the "
        "global RNG, that no seed equals seed 0, and that the Sampler forwards its seed. "
        "Lockstep is kept: the harness-fixed screen with configured seed 1234 shows 0/1000 in every "
        "candidate mode and history at TP3 and TP4, both boots, and before the seed change the PR's "
        "own screen showed the same with the control reproducing the PR's 263/1000 and 239/1000. The "
        "tests pass: 190 passed, 1 skipped, both boots. The TP2 crash is a bug in the screen harness, "
        "not in #980 (tp2_finding). "
        "Caveat for the 4-node confirmation: r1 keeps #980's sampling changes, which alter the tokens "
        "the startup sampler warmup leaves in req_states.last_sampled_tokens. Every grid dummy step "
        "feeds those tokens to the MoE draft model. So r1 may show the same grid offset as "
        "candidate-48 without adding any per-step work (grid_attribution)."
    ),
    "patch": {
        "file": "0764-perf-spec-decode-elide-tp-confidence-broadcast-and-lock-sampling-seeds.patch",
        "sha256": patch_sha,
        "author": "Christopher Owen <3221756+christopherowen@users.noreply.github.com>, Sun, 8 Nov 2026 15:19:42 +0000 (kept)",
        "change_from_pr_head": [
            "drops the vllm/v1/worker/gpu/spec_decode/adaptive_verification.py hunk: record_confidences is back to 1107's, byte-identical (rationale item 3)",
            "keys the unseeded-request generator to the configured seed: states.py adds unseeded_request_seed_rng(seed) = np.random.default_rng(0 if seed is None else seed mod 2**64) and SamplingStates(max_num_reqs, vocab_size, seed=None); sampler.py adds Sampler(..., seed=None) and passes it on; model_runner.py puts ModelConfig.seed (the value the worker passes to set_random_seed) in the sampler kwargs (rationale item 1)",
            "adds two tests: test_unseeded_request_seeds_follow_configured_seed_across_tp_ranks and test_sampler_hands_configured_seed_to_sampling_states",
            "acceptance_estimator.py, dflash/speculator.py and speculator.py are #980's unchanged",
        ],
        "files": [
            "tests/v1/spec_decode/test_adaptive_verification.py",
            "vllm/v1/worker/gpu/model_runner.py",
            "vllm/v1/worker/gpu/sample/sampler.py",
            "vllm/v1/worker/gpu/sample/states.py",
            "vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py",
            "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py",
            "vllm/v1/worker/gpu/spec_decode/speculator.py",
        ],
        "stack_check": {
            "base": "local-inference-lab/vllm 19f2c20ed4d6 + candidate-48 series 0001..0763 (643 active), git am with the fixed committer and --committer-date-is-author-date as spark_ds41f/build.py",
            "original_reproduced": {"patch_head": "bd28dbf3b67301bbf285dedfeb1853835f4e0a4e", "tree": "fe5bddc4df277d37838339fbed09576c3219e34d", "equals_candidate48_lock": True},
            "r1_in_place_of_0764_then_0765_0771": {"applies": True, "patch_head": "b180aa08ce3181664328abaf7b5b2a5d61463081", "tree": "21156b980a4322fa064eedddbe6d3b6169fa4433", "git_diff_check": "clean", "diff_vs_candidate48_tree": "adaptive_verification.py (back to 1107's), sample/states.py, sample/sampler.py, model_runner.py (+3 lines), tests/v1/spec_decode/test_adaptive_verification.py (+68 lines)"},
            "superseded_r1_noseed": {"patch_head": "91b1f432c5c103b1a06c2d2199c90c5f3851626a", "tree": "70fb6f9ad870df963f1c944a4fd3dab598d85662", "where": "superseded/r1-noseed/ (patch, overlay, overlay-tests): the files the step-cost benches and the earlier screens ran"},
            "note": "0762..0771 other than 0764 touch none of 0764's files, model_runner.py included",
        },
        "overlay": "overlay/vllm/v1/worker/gpu/...: seven files, #980's six (adaptive_verification.py equal to 1107's / no980's) plus model_runner.py (the candidate-48 image's plus the 3-line seed kwarg); a grid arm must mount all seven, because model_runner.py passes seed= to Sampler. overlay-tests/ holds the patch's test file",
    },
    "seed_change": {
        "problem": "#980's SamplingStates._seed_rng = np.random.default_rng(0) ignored --seed; in 1107 unseeded request seeds came from np.random.randint on the global RNG, which gpu_worker seeds with set_random_seed(self.model_config.seed), so --seed selected them",
        "fix": "ModelConfig.seed -> model runner sampler kwargs -> Sampler(seed=) -> SamplingStates(seed=) -> unseeded_request_seed_rng(seed) = np.random.default_rng(0 if seed is None else int(seed) % 2**64); add_request draws from it as before",
        "lockstep": "the generator depends only on ModelConfig.seed, which every TP rank reads from the same config, and on the order of add_request calls, which every rank shares; the global RNG plays no part",
        "fallback": "no configured seed (None, e.g. a SamplingStates built without one) is seed 0: vLLM's ModelConfig.seed default is 0, which is what 1107 seeded the global RNG with by default, and what #980's default_rng(0) used, so the default --seed keeps #980's seed stream",
        "not_identical_to_1107": "the stream for a given --seed differs from 1107's (a PCG64 generator of its own, not draws from the shared global MT19937), as #980's declared change to sampled realizations already allows",
        "tests": {
            "test_unseeded_request_seeds_follow_configured_seed_across_tp_ranks": "two ranks built with seed 1234 assign identical seeds to 8 unseeded requests after different global-RNG perturbations; seed 4321 gives a disjoint set (and agrees with itself); no seed equals seed 0",
            "test_sampler_hands_configured_seed_to_sampling_states": "Sampler(seed=7) twice gives the same first unseeded seed, and Sampler(seed=8) a different one",
            "model_runner_wiring": "not unit-tested: the three added lines put self.model_config.seed in the kwargs both Sampler and GPUWatermarkSampler (which forwards **kwargs) receive",
        },
        "per_step_effect": "none: the change touches SamplingStates.__init__ / add_request, Sampler.__init__ and the runner's construction only; the greedy-path line trace of the final files is in cause_of_added_step_cost.greedy_decode_step_trace.r1seed",
    },
    "cause_of_added_step_cost": {
        "finding": "none of #980's code runs in the grid's timed steps; on the real greedy decode step it adds at most about 0.2 us (3 lines in record_confidences), far below the 50-60 us attributed",
        "grid_path_static": {
            "engine_grid -> _dummy_run": "execute_model(dummy_run=True) with InputBatch.make_dummy (input_ids and positions zeroed), then speculator.propose(dummy_run=True); no sample_tokens, so no Sampler, no rejection sampler, no record_confidences (model_runner.py engine_grid / _dummy_run)",
            "DSpark propose": "_generate_draft -> _sample_sequential -> _sample_logits, which calls _maybe_predict_acceptance only when use_confidence_head is false (dspark/speculator.py 284); the recipe has the TileLang confidence head and draft_sample_method greedy (draft_logits None), so neither predict() nor DFlash.sample_draft/predict_from_stats runs; the draft step is CUDA-graph captured",
            "tp_invariant_confidence": "read once at manager construction (model_runner.initialize_kv_cache); DSparkSpeculator overrides the DraftModelSpeculator property #980 adds, so the production value and the broadcast decision are 1107's",
        },
        "greedy_decode_step_trace": n["executed_changed_lines"],
        "greedy_decode_step_trace_reading": "c48 = original #980 (3 lines in record_confidences); r1 = r1-noseed and r1seed = the final files: no changed line in any greedy-path component; sample_sampled, draft_predict and draft_vp_predict are the non-greedy paths the patch means to change",
        "step_cost_in_process_abc": {
            "method": "stepcost_ab.py: the three arms' copies of the changed modules loaded side by side in one process on spark-ds41f:candidate-48; each arm's objects built from its own copies; calls interleaved with the arm order rotated every iteration; each call timed alone on an idle GPU (host clock around the call); 1500 iterations x 3 arms per component and shape; slow outliers beyond median + 5 MAD dropped; TP4 emulated as the PR's screen does (fake rank-ordered TP group, parallel_state TP 4); shapes = the grid's 14 decode shapes (num_reqs 1,4,8,16 x 2 and 3 rows; 1,2,4,8,12,16 x 6 rows); max_num_reqs 16, 5 speculative steps, vocab 129280",
            "aa_controls": "r1's adaptive_verification.py is byte-identical to 1107's (record_conf r1 vs 1107); r1's other files are byte-identical to candidate-48's",
            "files": "measured with superseded/r1-noseed's files; the --seed change adds no per-step line (seed_change.per_step_effect)",
            "greedy_total": ab_comp("greedy_total"),
            "record_conf": ab_comp("record_conf"),
            "sample_greedy": ab_comp("sample_greedy"),
            "est_step": ab_comp("est_step"),
            "est_calibrate": ab_comp("est_calibrate"),
            "get_num_tokens": ab_comp("get_num_tokens"),
            "greedy_check": ab_comp("greedy_check"),
            "reading": "greedy post-step total (sampler eligibility reads, estimator step, calibration, record_confidences, get_num_tokens), about 125 us of host work per step: candidate-48 minus 1107 = +0.05 / +0.21 us (boot1/boot2, mean over shapes); r1 minus 1107 = +0.20 / +0.40 us, the same band as the A/A between byte-identical code. record_confidences: candidate-48 +0.14 / -0.01 us. That is at most 0.002% of a 21.1 ms grid step",
        },
        "step_cost_per_process": {
            "method": "stepcost.py: one fresh container per arm and boot (boot1 order 1107, c48, r1; boot2 r1, c48, 1107), the arm's six files mounted over the candidate-48 image, 9 rounds x 300 calls per component and shape, wall time per call to the end of its GPU work",
            "geomean_over_14_shapes": {c: pp[c] for c in ("greedy_total", "record_conf", "sample_greedy", "est_step", "est_calibrate", "get_num_tokens", "greedy_check")},
            "reading": "process-to-process variance dominates here: r1 vs 1107 differ by +11% on record_confidences with byte-identical code (a bimodal wait on the previous step's copy event), and greedy_total arms differ by 0.7-1.1% against 0.3-2.0% boot-to-boot spreads. It bounds any greedy-path cost to about 1 us per step; the in-process A/B/C bench resolves it",
        },
        "other_paths_not_greedy": {
            "sample_sampled_tp4": {"per_process_wall_geomean_us": pp["sample_sampled"]["geomean_wall_us"], "c48_vs_1107_pct": pp["sample_sampled"]["c48_vs_1107_pct"],
                                   "in_process_host": ab_comp("sample_sampled"),
                                   "reading": "#980 routes unseeded top-k/top-p rows at TP>1 through TileLang top-k/top-p + gumbel instead of FlashInfer: host time +18 us per call on average over the shapes (+25 us at 1 request, -13 us at 16 x 6 rows), and per-call wall time to the end of the GPU work 30% lower in geomean (equal at 1 request, up to about 40% lower at 8-16 requests x 3-6 rows); not the greedy path, and the release run found the sampled companion not worse"},
            "draft_predict": {"in_process_host": ab_comp("draft_predict"), "reading": "predict() with the scratch buffers: 0.2-0.3 us less host time (DraftModel/DFlash drafters without a confidence head)"},
            "draft_vp_predict": {"in_process_host": ab_comp("draft_vp_predict"), "reading": "DFlash vocab-parallel probabilistic drafts: predict_from_stats adds 0.42-0.51 us of host time per draft step (one more Python call); 1107's DFlash already fed the per-rank stats to _predict_kernel inline, so the PR's -11.4 us per draft step is against predict()'s full-vocab rescan, not against 1107; eager-only, as the draft step is CUDA-graph captured"},
        },
    },
    "grid_attribution": {
        "observation": "candidate-48 21.1043 / 21.1292 ms vs no980 21.0514 / 21.0667 ms (+0.27%); no1011 +0.16% vs candidate-48",
        "why_it_is_not_980_step_work": "per the static path and the trace above, no #980 line executes in the timed steps",
        "likely_mechanism": "the grid's dummy steps read state that earlier traffic leaves behind, and #980 changes that state. The startup sampler warmup (gpu/warmup.py: SamplingParams.for_sampler_warmup(), temperature 0.9, top-k 50, top-p 0.9, unseeded, slots 0..n-1) samples different tokens under #980: seeds come from SamplingStates._seed_rng instead of the global NumPy RNG, and at TP>1 the rows go through TileLang top-k/top-p + gumbel instead of FlashInfer. Those tokens stay in req_states.last_sampled_tokens. _dummy_run passes them to the drafter (model_runner.py 1210, last_sampled -> prepare_dflash_inputs), and the DSpark draft layers are MoE, so routing and the drafter's step time depend on them. Each arm's warmup is deterministic, which gives a per-arm offset that repeats across boots, the pattern the attribution showed. Any sampled traffic before the grid does the same, and so may KV the warmup wrote into dummy-context blocks",
        "measured_precondition_on_sp2": {
            "warmup_tokens": n["warmup_tokens"],
            "reading": "with TP4 emulated and the worker's seeding (set_random_seed(0)), the warmup's sampled tokens are identical across boots within each arm, and differ between 1107 (FlashInfer path, tokens hash 56c19316f1ab50f8) and candidate-48 / r1 (TileLang path, afe98f3c0d04d6b8)",
        },
        "not_verified": "the effect of these tokens on the 4-node grid's step time; it needs the model and was not run here",
        "position_note": "candidate-48 booted first in every triple (positions 1 and 4), and the no1011 overlay, which changes only FlashAttention-backend lines DS4.1 does not run, also measured 0.16% faster than candidate-48",
        "how_to_settle_on_the_grid": [
            "read forward_ms vs drafter_ms in the existing candidate-48 / no980 grid records: a delta in drafter_ms on code that does not run points at the drafter's inputs",
            "run arms r1 (overlay/, seven files), no980, and the two 2x2 overlays in diag-overlays/ (diag-r1seed-1107sampling: r1's spec-decode files with 1107's sampler.py and states.py and the image's model_runner.py; diag-1107-r1seedsampling: 1107's spec-decode files with r1's sampler.py, states.py and model_runner.py), all on the candidate-48 image, rotating arm order so no arm always boots first; if the delta follows sampler.py/states.py (the warmup tokens) and not the spec-decode files, it is the leftover-input effect, not engine speed",
            "at the recipe's default --seed 0 the corrected patch's warmup tokens equal r1-noseed's and candidate-48's (default_rng(0) either way), so the caveat holds for the final patch",
            "a harness fix that would make the grid's dummy inputs independent of history (resetting req_states.last_sampled_tokens / next_prefill_tokens, and the dummy context blocks, before timing) is outside this PR",
        ],
    },
    "lockstep_screen": {
        "final_files_harness_fixed_screen": {
            "script": "screen_zero_broadcast_sync_rowsfix_valid_seeded.py: the PR's screen with draft_sampled laid out as [num_logits], only the sampled entries rejection_sample writes compared (raw counts kept as unchecked_sampled_raw_steps), and --sampling-seed handed to SamplingStates as the configured seed (scripts/screen-valid-seeded-vs-pr.diff is the whole change); --sampling-seed 1234; the final seven files mounted on the candidate-48 image; same one-container layout; boot1 TP3 then TP4, boot2 TP4 then TP3",
            "runs": n["screen_r1seed"],
            "reading": "configured seed 1234: every candidate mode (greedy, sampled unseeded, dflash estimator) has 0/1000 in the draft, confidence, budget, capacities, dead-cut, sampled (written entries) and estimator histories at TP3 and TP4, both boots, with seeds equal across ranks, 0 broadcasts and 10 refits; the raw rows differ only in the uninitialized tail past num_sampled (0-6 steps at TP4, as without the seed); the control arm, which draws seeds from the global RNG, diverges as designed (264/1000 TP3, 240/1000 TP4 budget mismatches in this harness-fixed screen, against 263/239 in the PR's)",
            "verdict": "pass",
        },
        "script": "before the seed change: the PR's screen_zero_broadcast_sync.py, unchanged, --worker at TP3 and TP4, 1,000 steps, every rank a process in one container on sp2's GB10 (NCCL_MULTI_RANK_GPU_ENABLE=1, as the sp8 own-mark run), r1's six files mounted on the candidate-48 image; boot order as the PR's orchestrator (boot1 TP3 then TP4, boot2 TP4 then TP3)",
        "runs": n["screen_mark"],
        "reading": "every candidate mode (greedy, sampled unseeded, dflash estimator) has 0/1000 in the draft, confidence, budget, capacities, dead-cut, sampled and estimator histories, with 0 broadcasts, 10 refits and seeds equal across ranks; the control reproduces the PR's 263/1000 (TP3) and 239/1000 (TP4) budget mismatches and max confidence diffs of 0.29860765 / 0.51592845",
        "verdict": "pass",
    },
    "tests": {"command": "pytest --noconftest -v on the PR's six test files (README 3.1), the final seven files and test file mounted on the candidate-48 image", "boot1": n["pytest_r1seed"].get("boot1"), "boot2": n["pytest_r1seed"].get("boot2"), "new_tests": ["test_unseeded_request_seeds_follow_configured_seed_across_tp_ranks", "test_sampler_hands_configured_seed_to_sampling_states"], "before_seed_change": {"boot1": n["pytest"].get("boot1"), "boot2": n["pytest"].get("boot2")}, "verdict": "pass"},
    "tp2_finding": {
        "conclusion": "a screen-harness bug, not #980's and not the shared-GPU layout. The serving worker cannot hit it, at TP2, TP3 or TP4",
        "mechanism": "rejection_sample takes draft_sampled as [num_logits] and reads request r's draft i at row cu_num_logits[r] + i + 1 (rejection_sampler_utils.py: tl.load(draft_sampled_ptr + logit_idx + 1)); the worker passes input_ids[logits_indices] (InputBatch.logits_rows), which is [num_logits]. The screen passes draft_tokens.view(-1): num_reqs x num_steps = 40 entries where 48 are read, so the last request reads 8 int64 past the tensor, into its 512-byte allocator block's stale bytes (and every request's drafts are misaligned). The kernel clamps only negative ids (tl.maximum(0, ...)); a stale value at or above the vocab size indexes target and draft logits out of bounds. Whether it faults depends on what the allocator block held, which varies with world size, vocab size (4096 x world) and mode history: hence TP2's candidate_sampled_unseeded, and not TP3/TP4",
        "reproduced": {
            "unmodified_screen_tp2_on_sp2": {k: n["tp2"][k] for k in ("screen-tp2-c48-orig", "screen-tp2-r1-orig")},
            "deterministic_repro_no_tp": {"1107_image_no_980_code": n["tp2"].get("oob-repro-1107-image"), "candidate48_image": n["tp2"].get("oob-repro-c48-image"),
                                          "reading": "the screen's TP2 shapes in one process, rejection_sample only: [num_logits] layout runs; the 40-entry argument runs while the 8 entries past it are 0, and raises 'an illegal memory access' when they hold an out-of-vocab value, on the 1107 image too"},
        },
        "harness_fixed": {
            "rowsfix": "draft_sampled laid out as [num_logits] with each request's drafts at rows 1..num_steps (scripts/screen-rowsfix.diff)",
            "rowsfix_runs": {k: n["tp2"][k] for k in ("screen-tp2-c48-rowsfix", "screen-tp2-r1-rowsfix", "screen-tp3-r1-rowsfix", "screen-tp4-r1-rowsfix")},
            "rowsfix_reading": "no crash at TP2, TP3 or TP4; but the sampled history then differs on 7 (TP2) and 2 (TP4) steps while drafts, confidences, budgets, num_sampled and the estimator all match: rejection_sample returns sampled = new_empty(num_reqs, num_steps + 1) and writes row r's first num_sampled[r] entries only, and the screen records the whole row, uninitialized tail included",
            "valid_entries": "rowsfix + compare entries < num_sampled only, raw counts kept as unchecked_sampled_raw_steps (scripts/screen-rowsfix-valid.diff; scripts/screen-rowsfix-valid-vs-pr.diff is the whole change against the PR's screen)",
            "valid_runs": {k: n["tp2"][k] for k in ("screen-tp2-r1-valid", "screen-tp2-r1-valid-again", "screen-tp2-c48-valid", "screen-tp3-r1-valid", "screen-tp4-r1-valid")},
            "valid_reading": "0/1000 in every history for every candidate mode at TP2 (twice with r1, once with candidate-48's files), TP3 and TP4; only the uninitialized tail differs (0-6 steps), which the worker never reads",
        },
        "980_sampling_path": "#980's sampling changes feed rejection_sample only seed values and the choice of the TileLang top-k/top-p path; seeds and temperatures are indexed by request slot (in bounds), so they cannot produce an out-of-bounds token id",
    },
    "observations_outside_the_mark": [
        "the PR's predict_from_stats saving (-11.4 us per draft step) is against predict()'s rescan; against 1107's DFlash inline launch it is +0.4-0.5 us of host time (eager only)",
        "the screen's own pass criteria rest on the harness bug's luck at TP3/TP4 (misaligned drafts, 8 out-of-bounds reads per step); the harness-fixed screen in scripts/ is the sound one for future runs",
    ],
    "not_done": [
        "no 4-node grid run (the coordinator's confirmation, sp1..sp4)",
        "the screen ran with all ranks on one GB10, not across nodes; its latency records are not comparable to the PR's",
        "the warmup-token mechanism is supported by code reading and the sp2 precondition, not by a grid measurement",
    ],
    "receipts": {
        "commands": "commands.txt (setup and every run with UTC times and exit codes)",
        "raw": "raw/ (each run's output; stepcost-*.json, stepcost-ab-*.json, screen jsonl, trace-*.json, executed-changed-lines-*.json, stepcost-summary.json)",
        "numbers": "numbers.json (assemble_numbers.py over raw/)",
        "scripts": "scripts/",
        "superseded": "superseded/r1-noseed/: the first r1 (without the seed change): patch, overlay, overlay-tests",
    },
}
json.dump(result, open(f"{D}/result.json", "w"), indent=1)
print("ok", patch_sha)
