"""Update write_result.py for the --seed correction (r1 final = r1seed)."""
p = "write_result.py"
s = open(p).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


sub('''to 1107's, and the trace finds zero changed lines on the greedy path. Lockstep is kept: the "
        "PR's own screen with r1 shows 0/1000 in every candidate mode and history at TP3 and TP4, both "
        "boots, and the control reproduces the PR's 263/1000 and 239/1000. The tests pass, 188 passed "
        "and 1 skipped, both boots. The TP2 crash is a bug in the screen harness, not in #980 (tp2_finding). "''',
'''to 1107's, and the trace finds zero changed lines on the greedy path. "
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
        "not in #980 (tp2_finding). "''')

sub('''        "change_from_pr_head": "drops the vllm/v1/worker/gpu/spec_decode/adaptive_verification.py hunk (record_confidences back to 1107's, byte-identical); rationale item 3 of the message says so; the other five files and the test file are #980's unchanged",
        "files": [
            "tests/v1/spec_decode/test_adaptive_verification.py",
            "vllm/v1/worker/gpu/sample/sampler.py",''',
'''        "change_from_pr_head": [
            "drops the vllm/v1/worker/gpu/spec_decode/adaptive_verification.py hunk: record_confidences is back to 1107's, byte-identical (rationale item 3)",
            "keys the unseeded-request generator to the configured seed: states.py adds unseeded_request_seed_rng(seed) = np.random.default_rng(0 if seed is None else seed mod 2**64) and SamplingStates(max_num_reqs, vocab_size, seed=None); sampler.py adds Sampler(..., seed=None) and passes it on; model_runner.py puts ModelConfig.seed (the value the worker passes to set_random_seed) in the sampler kwargs (rationale item 1)",
            "adds two tests: test_unseeded_request_seeds_follow_configured_seed_across_tp_ranks and test_sampler_hands_configured_seed_to_sampling_states",
            "acceptance_estimator.py, dflash/speculator.py and speculator.py are #980's unchanged",
        ],
        "files": [
            "tests/v1/spec_decode/test_adaptive_verification.py",
            "vllm/v1/worker/gpu/model_runner.py",
            "vllm/v1/worker/gpu/sample/sampler.py",''')

sub('''            "r1_in_place_of_0764_then_0765_0771": {"applies": True, "patch_head": "91b1f432c5c103b1a06c2d2199c90c5f3851626a", "tree": "70fb6f9ad870df963f1c944a4fd3dab598d85662", "git_diff_check": "clean", "diff_vs_candidate48_tree": "adaptive_verification.py only (1 insertion, 4 deletions)"},
            "note": "0762..0771 other than 0764 touch none of 0764's files",''',
'''            "r1_in_place_of_0764_then_0765_0771": {"applies": True, "patch_head": "b180aa08ce3181664328abaf7b5b2a5d61463081", "tree": "21156b980a4322fa064eedddbe6d3b6169fa4433", "git_diff_check": "clean", "diff_vs_candidate48_tree": "adaptive_verification.py (back to 1107's), sample/states.py, sample/sampler.py, model_runner.py (+3 lines), tests/v1/spec_decode/test_adaptive_verification.py (+68 lines)"},
            "superseded_r1_noseed": {"patch_head": "91b1f432c5c103b1a06c2d2199c90c5f3851626a", "tree": "70fb6f9ad870df963f1c944a4fd3dab598d85662", "where": "superseded/r1-noseed/ (patch, overlay, overlay-tests): the files the step-cost benches and the earlier screens ran"},
            "note": "0762..0771 other than 0764 touch none of 0764's files, model_runner.py included",''')

sub('''        "overlay": "overlay/vllm/v1/worker/gpu/... (the six files; adaptive_verification.py equals 1107's / no980's), overlay-tests/ holds the patch's test file",''',
'''        "overlay": "overlay/vllm/v1/worker/gpu/...: seven files, #980's six (adaptive_verification.py equal to 1107's / no980's) plus model_runner.py (the candidate-48 image's plus the 3-line seed kwarg); a grid arm must mount all seven, because model_runner.py passes seed= to Sampler. overlay-tests/ holds the patch's test file",
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
        "per_step_effect": "none: the change touches SamplingStates.__init__ / add_request, Sampler.__init__ and the runner's construction only; the greedy-path line trace of the final files is in cause_of_added_step_cost.greedy_decode_step_trace.r1seed",''')

sub('''            "aa_controls": "r1's adaptive_verification.py is byte-identical to 1107's (record_conf r1 vs 1107); r1's other files are byte-identical to candidate-48's",''',
'''            "aa_controls": "r1's adaptive_verification.py is byte-identical to 1107's (record_conf r1 vs 1107); r1's other files are byte-identical to candidate-48's",
            "files": "measured with superseded/r1-noseed's files; the --seed change adds no per-step line (seed_change.per_step_effect)",''')

sub('''            "run arms r1 (overlay/), no980, and the two 2x2 overlays in diag-overlays/ (diag-r1-1107sampling: r1 with 1107's sampler.py and states.py; diag-1107-r1sampling: 1107's files with r1's sampler.py and states.py), all as overlays on the candidate-48 image, rotating arm order so no arm always boots first; if the delta follows sampler.py/states.py (the warmup tokens) and not the spec-decode files, it is the leftover-input effect, not engine speed",''',
'''            "run arms r1 (overlay/, seven files), no980, and the two 2x2 overlays in diag-overlays/ (diag-r1seed-1107sampling: r1's spec-decode files with 1107's sampler.py and states.py and the image's model_runner.py; diag-1107-r1seedsampling: 1107's spec-decode files with r1's sampler.py, states.py and model_runner.py), all on the candidate-48 image, rotating arm order so no arm always boots first; if the delta follows sampler.py/states.py (the warmup tokens) and not the spec-decode files, it is the leftover-input effect, not engine speed",
            "at the recipe's default --seed 0 the corrected patch's warmup tokens equal r1-noseed's and candidate-48's (default_rng(0) either way), so the caveat holds for the final patch",''')

sub('''    "lockstep_screen": {
        "script": "the PR's screen_zero_broadcast_sync.py, unchanged,''',
'''    "lockstep_screen": {
        "final_files_harness_fixed_screen": {
            "script": "screen_zero_broadcast_sync_rowsfix_valid_seeded.py: the PR's screen with draft_sampled laid out as [num_logits], only the sampled entries rejection_sample writes compared (raw counts kept as unchecked_sampled_raw_steps), and --sampling-seed handed to SamplingStates as the configured seed (scripts/screen-valid-seeded-vs-pr.diff is the whole change); --sampling-seed 1234; the final seven files mounted on the candidate-48 image; same one-container layout; boot1 TP3 then TP4, boot2 TP4 then TP3",
            "runs": n["screen_r1seed"],
            "reading": "SEED_READING",
            "verdict": "VERDICT_SEEDED",
        },
        "script": "before the seed change: the PR's screen_zero_broadcast_sync.py, unchanged,''')

sub('''    "tests": {"command": "pytest --noconftest -v on the PR's six test files (README 3.1), r1's six files and r1's test file mounted on the candidate-48 image", "boot1": n["pytest"].get("boot1"), "boot2": n["pytest"].get("boot2"), "verdict": "pass"},''',
'''    "tests": {"command": "pytest --noconftest -v on the PR's six test files (README 3.1), the final seven files and test file mounted on the candidate-48 image", "boot1": n["pytest_r1seed"].get("boot1"), "boot2": n["pytest_r1seed"].get("boot2"), "new_tests": ["test_unseeded_request_seeds_follow_configured_seed_across_tp_ranks", "test_sampler_hands_configured_seed_to_sampling_states"], "before_seed_change": {"boot1": n["pytest"].get("boot1"), "boot2": n["pytest"].get("boot2")}, "verdict": "pass"},''')

sub('''        "SamplingStates._seed_rng = np.random.default_rng(0) ignores --seed: unseeded request seeds came from the global NumPy RNG before, which the worker seeds with set_random_seed(model_config.seed) (gpu_worker.py), so they followed --seed; with #980 every server draws the same seed sequence whatever --seed is",
''', '')
sub('''        "scripts": "scripts/",''', '''        "scripts": "scripts/",
        "superseded": "superseded/r1-noseed/: the first r1 (without the seed change): patch, overlay, overlay-tests",''')
open(p, "w").write(s)
print("updated")
