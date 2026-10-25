#!/usr/bin/env python3
"""Write the kernel-lab bundles that run every TileLang port together.

usage: make_integration_bundles.py VLLM_REPO BASE HEAD [--overlay BUNDLE... | --serving]

Every file the vLLM branch changes between BASE (candidate 4's source, which image
-next-4 holds) and HEAD goes into each bundle: package files under overlay/vllm (the lab
mounts them over the image's tree), tests and benchmarks as mounts. Each bundle then runs
one share of the ports' GPU tests and gated benchmarks (bundles/integration-*), so each
family is tested with all the other ports in place. With ``--overlay``, only the package
files are written, as the overlay of each named bundle (a variant sweep that times the
branch's production kernels). With ``--serving``, the package files go to
bundles/serving/overlay and ports-tp4.json is the serving arm that mounts them over
candidate 4 (rc4/rc-tp4.json, image -next-4).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-4"

PYTEST = ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
          "-p", "prepare_fixtures"]

# name: (description, [(label, argv after python, timeout seconds[, env])][, candidate]):
# env sets a step's environment variables over the bundle's (None unsets one); candidate
# adds keys to the bundle's candidate.json (io_uring for the disk Engram reader).
PARTS = {
    "runner": (
        "the V2 runner's input preparation and token counts",
        [
            ("runner and sink tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-runner",
                "tests/v1/worker/test_gpu_tilelang_inputs.py",
                "tests/v1/worker/test_gpu_block_table.py",
                "tests/v1/worker/test_kv_block_zeroer.py",
                "tests/v1/worker/test_dead_rows.py",
                "tests/v1/worker/test_gpu_input_batch_v2.py",
                "tests/v1/worker/test_gpu_sink.py",
                "tests/v1/spec_decode/test_dflash_prepare_inputs.py",
                "tests/v1/worker/test_prepare_kernels.py",
                # The grammar bitmask in TileLang (structured output).
                "tests/v1/worker/test_gpu_tilelang_structured.py",
                # Captured outputs written in place (vLLM ed9dae5ae).
                "tests/v1/worker/test_graph_output_buffers.py",
                # The consumed drafts' gather, the new requests' row resets and the
                # constant offsets of a step without drafts (run 19).
                "tests/v1/worker/test_draft_tokens_handler.py",
                # A target without an MTP override buffer (DS4.1 since 58a492841).
                "tests/v1/worker/test_mtp_target_states.py",
                "tests/v1/spec_decode/test_adaptive_verification.py",
                # Prefill's glue: SP attention's front, CED's routing mask and slots,
                # NCCL's last-dimension all-gathers, CED's zeroed full rows (run 19).
                "tests/kernels/test_deepseek_v41_tilelang_columns.py",
                "tests/kernels/test_deepseek_v41_tilelang_ced.py",
                "tests/kernels/test_deepseek_v41_tilelang_collectives.py"], 3600),
            # Prompt logprobs under sink rejection (0266,
            # experiments/2026-10-24-prompt-logprobs-sink): scheduler tests, run without
            # the TileLang backend, whose aligned prefill chunks their test scheduler
            # (no long_prefill_token_threshold) does not configure.
            ("sink scheduler tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-sink-scheduler",
                "tests/v1/core/test_sink_draft_zero.py"], 1800,
             {"VLLM_DS41_KERNEL_BACKEND": None}),
            # Startup without unused kernels (2026-10-25): no B12X preparation unit or
            # temporary KV cache under TileLang, the runner warm-up without Triton's
            # top-k/top-p. The tests pick each backend themselves; the files' other
            # tests are B12X's, so the variable is unset. Run 23 found the attention
            # file's older B12X-path tests stale (bare modules missing _decode_capacity
            # and _main_width, fixtures that need the repository's conftest): only the
            # new tests run here.
            ("startup tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-startup",
                *(f"tests/models/test_deepseek_v4_1_attention.py::{name}" for name in (
                    "test_tilelang_attention_declares_no_b12x_unit",
                    "test_tilelang_binding_prepares_the_attention_staging",
                    "test_tilelang_context_kv_write_needs_no_helper_plan",
                    "test_b12x_cache_writes_use_the_helper_plan_of_their_kind")),
                "tests/models/test_deepseek_v4_1_compressor_prepare.py",
                "tests/model_executor/test_b12x_warmup.py",
                "tests/v1/worker/test_mixed_warmup_gate.py",
                "tests/v1/worker/test_gpu_warmup_blocks.py"], 3600,
             {"VLLM_DS41_KERNEL_BACKEND": None}),
            # The JIT monitor's hooks (its off mode among them) need their own
            # conftest, which resets the monitor's process-global state between tests.
            ("JIT monitor tests", ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider",
                                   "--confcutdir", "tests/jit_monitor",
                                   "--basetemp", "/c/pytest-integration-jit-monitor",
                                   "tests/jit_monitor/test_hooks.py"], 1200),
            ("runner benchmark", ["benchmarks/kernels/benchmark_runner_inputs_tilelang.py",
                                  "--gate"], 3600),
            ("token-count benchmark", ["benchmarks/kernels/benchmark_token_counts.py", "--gate",
                                       "--seed", "61027"], 1200),
        ],
    ),
    "sampling": (
        "the sampling family: Gumbel, temperature, sink detection, greedy reductions",
        [
            ("sampling tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-sampling",
                "tests/v1/worker/test_gpu_tilelang_sampling.py",
                "tests/v1/worker/test_gpu_gumbel_sample.py",
                "tests/v1/worker/test_gpu_sink.py",
                "tests/v1/worker/test_dspark_greedy_markov.py",
                "tests/v1/worker/gpu/test_greedy_verification.py",
                "tests/v1/worker/test_gpu_sampler_flags.py",
                "tests/v1/worker/test_prepare_kernels.py"], 5400),
            ("sampling benchmark", ["benchmarks/kernels/benchmark_tilelang_sampling.py",
                                    "--gate"], 3600),
            # The generation controls' TileLang kernels (dispatched since vLLM
            # 987ef4337 and 3bb713c4f): experiments/2026-10-24-sampling-controls.
            ("sampling-controls tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-controls",
                "tests/v1/worker/test_gpu_sampling_controls.py",
                "tests/v1/worker/test_gpu_thinking_budget.py",
                "tests/v1/worker/test_gpu_logit_bias.py",
                "tests/v1/worker/test_gpu_bad_words.py",
                "tests/v1/worker/test_gpu_rejection_sampler_chunking.py",
                "tests/v1/worker/test_gpu_rejection_sampler_i64.py"], 3600),
            ("sampling-controls benchmark",
             ["benchmarks/kernels/benchmark_sampling_controls.py", "--gate"], 1800),
            ("sampling-controls auxiliary benchmark",
             ["benchmarks/kernels/benchmark_sampling_auxiliary.py", "--gate"], 1800),
            # The sink detector's flags, cut and greedy alternates, and the grammar
            # bitmask, against the torch chains and the Triton kernel they replace.
            ("sink glue and grammar bitmask benchmark",
             ["benchmarks/kernels/benchmark_sink_glue.py", "--gate"], 1800),
            # Top-k / top-p filtering in TileLang (vLLM 914a15e80).
            ("top-k/top-p tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-topk-topp",
                "tests/v1/worker/test_gpu_tilelang_topk_topp.py"], 5400),
            ("top-k/top-p benchmark",
             ["benchmarks/kernels/benchmark_tilelang_topk_topp.py", "--gate"], 3600),
        ],
    ),
    "mhc": (
        "the mHC post and stream collapse, the fused decode router and the model forward",
        [
            ("mHC, router and model-forward tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-mhc",
                "tests/kernels/test_deepseek_v41_tilelang_mhc.py",
                # The fused decode router reads the padding mask and the input ids
                # (vLLM 23ae46804); the model forward without the MTP buffer and
                # with the sink column from the last collapse (58a492841, 5998c7916).
                "tests/kernels/moe/test_deepseek_v41_tilelang_router.py",
                "tests/models/test_deepseek_v4_1_ced_model.py",
                "tests/models/test_deepseek_v4_1_sp_prefill.py"], 3600),
            # Disk Engram (run 19): the hash and gate read the image sentinel from the
            # ids, the IDs land in layer planes in place, the columns all-gather
            # without copies, the reader's rows against b12x's lookup.
            ("Engram tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-engram",
                "tests/kernels/test_deepseek_v41_tilelang_engram.py",
                "tests/kernels/test_deepseek_v41_tilelang_engram_hash.py",
                "tests/kernels/test_deepseek_v41_engram_rows.py"], 3600,
             {"ENGRAM_ROWS_TEST_DIR": "/c/engram-rows-test"}),
            # The Engram model tests drive B12X's native hash and lookup (the masks
            # in their B12X form): the TileLang backend unset.
            ("Engram model tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-engram-model",
                "-o", "faulthandler_timeout=240",
                "tests/models/test_deepseek_v4_1_engram.py"], 3600,
             {"VLLM_DS41_KERNEL_BACKEND": None,
              "ENGRAM_ROWS_TEST_DIR": "/c/engram-rows-test"}),
            ("mHC benchmark", ["benchmarks/kernels/benchmark_deepseek_v41_mhc_post.py",
                               "--gate"], 3600),
            # The vision tower in TileLang and the image span's row copies (run 21):
            # bit for bit where B12X's order is defined, else against DeepSeek's
            # math within B12X's own error; the gate times the tower against B12X's.
            ("vision and image-span tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-vision",
                "tests/kernels/test_deepseek_v41_tilelang_vision.py",
                "tests/models/test_deepseek_v4_1_image_span.py"], 5400),
            ("vision benchmark", ["benchmarks/kernels/benchmark_tilelang_vision.py",
                                  "--gate"], 3600),
        ],
        {"io_uring": True},
    ),
    "misc": (
        "the embedding, DSpark's confidence head and context inputs, logprob ranks, the "
        "resampled token and the rejection sampler",
        [
            ("embedding, confidence, ranks and verification tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-misc",
                "tests/kernels/test_deepseek_v41_tilelang_embedding.py",
                "tests/kernels/test_deepseek_v41_tilelang_confidence.py",
                "tests/v1/worker/test_gpu_logprob_ranks.py",
                "tests/v1/spec_decode/test_rejection_sampler_utils.py",
                # The RMSNorm on the fused projection's halves, in place (vLLM 1e62f7ea6).
                "tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py::test_rmsnorm",
                "tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py"
                "::test_rmsnorm_of_a_fused_projections_half",
                # DSpark's context-graph inputs in one launch (vLLM 2b6d4613c).
                "tests/kernels/test_deepseek_v41_tilelang_dspark_context.py"], 3600),
            # Logprobs in TileLang (run 19): the token log-softmax, the top-k ids,
            # and the verified tokens' custom-id logprobs (0267).
            ("logprob tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-logprob",
                "tests/v1/worker/test_gpu_tilelang_logprob_softmax.py",
                "tests/v1/worker/test_gpu_tilelang_topk_ids.py",
                "tests/v1/worker/test_gpu_verified_logprob_token_ids.py"], 3600),
            ("logprob softmax benchmark",
             ["benchmarks/kernels/benchmark_tilelang_logprob_softmax.py", "--gate"], 1800),
            ("logprob top-k ids benchmark",
             ["benchmarks/kernels/benchmark_tilelang_topk_ids.py", "--gate"], 1800),
            # The rejection sampler's block verification in TileLang (vLLM 4c7261200).
            ("rejection-sampler tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-rejection",
                "tests/v1/spec_decode/test_tilelang_rejection.py"], 3600),
            # The unchanged image fails eight of these (vocabulary-parallel draft seeds;
            # experiments/2026-10-24-resampled-token): their own step, so they are told
            # apart from the ports'.
            ("vocabulary-parallel drafting tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-vocab-parallel",
                "tests/v1/spec_decode/test_vocab_parallel_drafting.py"], 1800),
            ("embedding and confidence benchmark",
             ["benchmarks/kernels/benchmark_deepseek_v41_tilelang_misc.py", "--gate"], 3600),
            ("logprob-rank benchmark", ["benchmarks/kernels/benchmark_logprob_ranks.py",
                                        "--gate", "--seed", "40391"], 1200),
            ("resampled-token benchmark", ["benchmarks/kernels/benchmark_resampled_token.py",
                                           "--gate", "--seed", "52807"], 600),
            ("resampled-token benchmark, mixed temperatures",
             ["benchmarks/kernels/benchmark_resampled_token.py", "--gate", "--seed", "18443",
              "--mixed"], 600),
            ("rejection-sampler benchmark",
             ["benchmarks/kernels/benchmark_tilelang_rejection.py", "--gate"], 1800),
        ],
    ),
}

RUN = '''#!/usr/bin/env python3
"""Run this bundle's steps in order; each step's exit status is printed, and the bundle
fails if any step failed. Every step runs whatever came before it."""
import json
import os
import subprocess
import sys

STEPS = json.load(open("/b/steps.json"))
codes = []
for label, argv, timeout, *rest in STEPS:
    print(f"=== {label}", flush=True)
    env = dict(os.environ)
    for name, value in (rest[0] if rest else {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    try:
        code = subprocess.run([sys.executable, *argv], timeout=timeout, env=env).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    print(f"=== {label}: exit {code}", flush=True)
    codes.append((label, code))
print(json.dumps({"steps": codes}), flush=True)
sys.exit(0 if all(code == 0 for _, code in codes) else 1)
'''


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def write_overlay(repo: str, head: str, changed: list[str], bundle: Path) -> None:
    overlay = bundle / "overlay"
    if overlay.exists():
        shutil.rmtree(overlay)
    for path in changed:
        if path.startswith("vllm/"):
            target = overlay / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.run(["git", "-C", repo, "show", f"{head}:{path}"],
                                              check=True, capture_output=True).stdout)


def main() -> None:
    repo, base, head = sys.argv[1:4]
    head_commit = git(repo, "rev-parse", "--short=9", head).strip()
    status = git(repo, "diff", "--name-status", base, head).split("\n")
    changed = [line.split("\t")[-1] for line in status if line]
    assert all(line.split("\t")[0] in ("A", "M") for line in status if line), status
    if sys.argv[4:5] == ["--serving"]:
        write_overlay(repo, head, changed, HERE / "bundles" / "serving")
        experiment = "{home}/projects/spark-ds41f/experiments/2026-10-24-tilelang-completion"
        mounts = [["{home}/projects/spark-ds41f/cache", "/cache", "rw"]]
        mounts += [[f"{experiment}/bundles/serving/overlay/{path}", f"{ROOT}/{path}", "ro"]
                   for path in changed if path.startswith("vllm/")]
        arm = {
            "notes": (f"Every TileLang port (vLLM branch tilelang-ports {head_commit}) mounted "
                      "over candidate 4 at tp4-ring: the serving check (outputs identical to "
                      "candidate 4 at temperature 0 and seeded temperature 1, no undeclared "
                      "kernel)."),
            "extends": {"config": "experiments/2026-10-23-next-release/rc4/rc-tp4.json"},
            "set": {"container": {"mounts": mounts}},
        }
        (HERE / "ports-tp4.json").write_text(json.dumps(arm, indent=1) + "\n")
        print(f"serving: {len(mounts) - 1} package files of {head_commit}")
        return
    if sys.argv[4:5] == ["--overlay"]:
        for name in sys.argv[5:]:
            write_overlay(repo, head, changed, HERE / "bundles" / name)
            print(f"{name}: overlay of {head_commit}")
        return
    fixtures = HERE / "bundles/sampling/prepare_fixtures.py"
    for part, (what, steps, *extra) in PARTS.items():
        bundle = HERE / "bundles" / f"integration-{part}"
        if bundle.exists():
            shutil.rmtree(bundle)
        (bundle / "overlay").mkdir(parents=True)
        mounts = [["prepare_fixtures.py", "/b/prepare_fixtures.py"],
                  ["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"]]
        for path in changed:
            content = subprocess.run(["git", "-C", repo, "show", f"{head}:{path}"], check=True,
                                     capture_output=True).stdout
            if path.startswith("vllm/"):
                target = bundle / "overlay" / path
            else:
                name = path.replace("/", "_")
                target = bundle / name
                mounts.append([name, f"{ROOT}/{path}"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        shutil.copy(fixtures, bundle / "prepare_fixtures.py")
        (bundle / "run.py").write_text(RUN)
        (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")
        candidate = {
            "description": (
                f"Every TileLang port together (vLLM branch tilelang-ports {head_commit}, "
                f"{len(changed)} files over candidate 4's source) on image -next-4: {what}, "
                "GPU tests then the gated benchmarks against the kernels the ports replace."),
            "image": IMAGE,
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TILELANG_CACHE_DIR": f"/c/tilelang-integration-{part}",
                "VLLM_DS41_KERNEL_BACKEND": "tilelang",
                "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x",
            },
            "workdir": "vllm",
            "argv": ["/b/run.py"],
            "verdict": "exit",
            "mounts": mounts,
            **(extra[0] if extra else {}),
        }
        (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
        print(f"{bundle.name}: {len(changed)} files, {len(steps)} steps")


if __name__ == "__main__":
    main()
